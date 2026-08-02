"""Бюджет вычислений: не больше 100 строгих GFLOPs на одно изображение.

Регламент задаёт лимит в классических FLOPs, где умножение-сложение (MAC)
считается за две операции. Популярные счётчики (fvcore, thop, ptflops) пишут в
выводе «FLOPs», а считают MACs, то есть вдвое меньше — и решение, собранное по
их числу, укладывается в лимит только на бумаге. Источником правды в спорных
случаях регламент называет `torch.utils.flop_counter.FlopCounterMode`, поэтому
здесь используется только он.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch.utils.flop_counter import FlopCounterMode

from .config import get_path

#: лимит регламента, строгие FLOPs на одно изображение
LIMIT_GFLOPS = 100.0

#: (нормализованная секция model, размер) -> GFLOPs. Сеть не зависит от секций
#: data.* и loss.*, поэтому свип по ним считается один раз, а не на каждой точке
_CACHE: dict[tuple[str, int], float] = {}


def count_gflops(model: torch.nn.Module, size: int, *, channels: int = 3) -> float:
    """Строгие GFLOPs одного forward на входе (1, channels, size, size).

    Модель считается там, где лежит: перекладывать её здесь нельзя, иначе
    вызывающий получил бы обратно испорченный объект.
    """
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, channels, size, size, device=device))
    return counter.get_total_flops() / 1e9


def config_gflops(cfg_model: Mapping[str, Any], size: int) -> float:
    """То же по секции `model` конфига: собрать сеть и посчитать её forward.

    Считается на `meta`-устройстве: FlopCounterMode работает на уровне
    диспетчера, поэтому настоящие тензоры ему не нужны — число получается то же,
    но без арифметики и без памяти (0.1 с против нескольких секунд на 768px).
    Предобученные веса не запрашиваются: на число операций они не влияют, а
    ходить за ними в сеть ради проверки бюджета незачем.
    """
    from .models import build_model

    spec = dict(cfg_model)
    spec["encoder_weights"] = None
    # у энкодеров с оконным вниманием размер входа зашит в маски внимания:
    # считать их на стороне, отличной от `img_size`, нельзя — они там просто
    # не собираются. Ведём `img_size` за размером, а не наоборот
    encoder_kwargs = dict(spec.get("encoder_kwargs") or {})
    if "img_size" in encoder_kwargs:
        encoder_kwargs["img_size"] = int(size)
        spec["encoder_kwargs"] = encoder_kwargs

    key = (json.dumps(spec, sort_keys=True, default=str), int(size))
    if key not in _CACHE:
        model = build_model(spec).eval().to("meta")
        _CACHE[key] = count_gflops(model, int(size))
    return _CACHE[key]


def inference_gflops(cfg: Mapping[str, Any], *, n_views: int = 1, n_models: int = 1) -> float:
    """Во что обходится ОДНО изображение целиком, а не один forward.

    Лимит регламента задан на изображение, поэтому TTA и ансамбль входят в счёт
    множителями: `--tta hflip,vflip` это два вида, сабмит по двум прогонам —
    две модели, и вместе они дают четырёхкратную стоимость.
    """
    size = int(get_path(cfg, "data.size", 512))
    return config_gflops(cfg.get("model", {}), size) * int(n_views) * int(n_models)


@dataclass(frozen=True)
class Verdict:
    """Влезает ли одно изображение в лимит — и почему проверка всё же прошла."""

    gflops: float
    limit: float
    size: int
    n_views: int = 1
    n_models: int = 1
    exempt: bool = False
    exempt_reason: str | None = None

    @property
    def within_limit(self) -> bool:
        """Настоящий ответ регламента, без всяких пометок."""
        return self.gflops <= self.limit

    @property
    def ok(self) -> bool:
        return self.within_limit or self.exempt

    @property
    def text(self) -> str:
        recipe = f"{self.size}px"
        if self.n_views > 1:
            recipe += f" x{self.n_views} видов TTA"
        if self.n_models > 1:
            recipe += f" x{self.n_models} моделей"
        head = f"{self.gflops:.1f} из {self.limit:.0f} GFLOPs на изображение ({recipe})"

        if self.within_limit:
            return f"{head} — в бюджете"
        excess = f"превышение в {self.gflops / self.limit:.2f} раза"
        if self.exempt:
            reason = self.exempt_reason or "причина не указана"
            return f"{head} — {excess}, но помечен budget.exempt: {reason}"
        return f"{head} — {excess}"

    def as_dict(self) -> dict:
        return {
            "gflops": round(self.gflops, 1),
            "limit_gflops": self.limit,
            "within_limit": self.within_limit,
            "exempt": self.exempt,
        }


def check(
    cfg: Mapping[str, Any],
    *,
    n_views: int = 1,
    n_models: int = 1,
    allow_exempt: bool = True,
    limit: float = LIMIT_GFLOPS,
) -> Verdict:
    """Вердикт по конфигу с учётом фактического рецепта инференса.

    `allow_exempt=False` выключает признание пометки `budget.exempt` — так
    проверяется сабмит, где никаких послаблений быть не может.
    """
    exempt = bool(get_path(cfg, "budget.exempt", False)) if allow_exempt else False
    return Verdict(
        gflops=inference_gflops(cfg, n_views=n_views, n_models=n_models),
        limit=float(limit),
        size=int(get_path(cfg, "data.size", 512)),
        n_views=int(n_views),
        n_models=int(n_models),
        exempt=exempt,
        exempt_reason=get_path(cfg, "budget.exempt_reason") if exempt else None,
    )


def check_submission(
    cfgs: Sequence[Mapping[str, Any]],
    *,
    n_views: int = 1,
    limit: float = LIMIT_GFLOPS,
) -> Verdict:
    """Бюджет готовой посылки — единственное место, где послаблений нет вообще.

    Складывается по моделям ансамбля, а не умножается: модели могут быть
    разными, и у каждой своя цена. `budget.exempt` здесь не читается: пометка
    описывает эксперимент, а посылка меряется настоящим лимитом регламента.
    """
    sizes = [int(get_path(cfg, "data.size", 512)) for cfg in cfgs]
    return Verdict(
        gflops=sum(inference_gflops(cfg, n_views=n_views) for cfg in cfgs),
        limit=float(limit),
        size=max(sizes) if sizes else 0,
        n_views=int(n_views),
        n_models=len(cfgs),
    )


def largest_fitting_size(
    cfg: Mapping[str, Any],
    *,
    n_views: int = 1,
    n_models: int = 1,
    limit: float = LIMIT_GFLOPS,
    step: int = 32,
) -> int | None:
    """Самая большая сторона входа, кратная `step`, которая ещё влезает в лимит.

    Отвечает на вопрос, который возникает сразу после отказа: «а на чём тогда
    учить». `None` значит, что не влезает даже минимальный вход — тогда дело не
    в разрешении, а в самой сети или в числе видов TTA.

    Полного двоичного поиска не нужно: FLOPs почти квадратичны по стороне, и
    из одного замера сразу получается близкая оценка, которую остаётся
    подвинуть на шаг-другой.
    """
    per_image = int(n_views) * int(n_models)
    model_cfg = cfg.get("model", {})

    def cost(side: int) -> float:
        return config_gflops(model_cfg, side) * per_image

    start = int(get_path(cfg, "data.size", 512))
    guess = int(start * math.sqrt(limit / cost(start)))
    size = max(step, guess - guess % step)

    while size >= step and cost(size) > limit:
        size -= step
    if size < step:
        return None
    while cost(size + step) <= limit:
        size += step
    return size


def rejection_text(cfg: Mapping[str, Any], verdict: Verdict) -> str:
    """Отказ вместе с ответом на вопрос, который возникает сразу следом.

    «Не влезает» без «а на чём тогда» заставляет подбирать размер вручную,
    перезапуская префлайт на каждой попытке.
    """
    fits = largest_fitting_size(
        cfg, n_views=verdict.n_views, n_models=verdict.n_models, limit=verdict.limit
    )
    tail = (f"в бюджет укладывается вход {fits}px" if fits
            else "в бюджет не укладывается ни одно разрешение — дело в самой сети")
    return (f"{verdict.text}; {tail}. Если прогон осознанно исследовательский и "
            "в сабмит не пойдёт, пометь его `budget.exempt: true`")
