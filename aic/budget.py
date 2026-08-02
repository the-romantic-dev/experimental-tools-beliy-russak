"""Бюджет вычислений: не больше 100 строгих GFLOPs на одно изображение.

Регламент задаёт лимит в классических FLOPs, где умножение-сложение (MAC)
считается за две операции. Популярные счётчики (fvcore, thop, ptflops) пишут в
выводе «FLOPs», а считают MACs, то есть вдвое меньше — и решение, собранное по
их числу, укладывается в лимит только на бумаге. Источником правды в спорных
случаях регламент называет `torch.utils.flop_counter.FlopCounterMode`, поэтому
здесь используется только он.

Считается ГОТОВАЯ модель, а не конфиг: как вы её собираете, библиотека не знает.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

#: лимит регламента, строгие FLOPs на одно изображение
LIMIT_GFLOPS = 100.0


def count_gflops(model, size: int, *, channels: int = 3) -> float:
    """Строгие GFLOPs одного forward на входе (1, channels, size, size).

    Модель считается там, где лежит: перекладывать её здесь нельзя, иначе
    вызывающий получил бы обратно испорченный объект. На `meta`-устройстве
    работает и даёт то же число, но без арифметики и без памяти — доли секунды
    против нескольких секунд на 768px.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, channels, size, size, device=device))
    return counter.get_total_flops() / 1e9


@dataclass(frozen=True)
class Verdict:
    """Влезает ли одно изображение в лимит."""

    gflops: float
    limit: float
    size: int
    n_views: int = 1
    n_models: int = 1

    @property
    def within_limit(self) -> bool:
        """Настоящий ответ регламента, без всяких пометок."""
        return self.gflops <= self.limit

    @property
    def ok(self) -> bool:
        return self.within_limit

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
        return f"{head} — превышение в {self.gflops / self.limit:.2f} раза"

    def as_dict(self) -> dict:
        return {
            "gflops": round(self.gflops, 1),
            "limit_gflops": self.limit,
            "within_limit": self.within_limit,
        }


def check(
    model,
    size: int,
    *,
    n_views: int = 1,
    limit: float = LIMIT_GFLOPS,
) -> Verdict:
    """Вердикт с учётом фактического рецепта инференса.

    Лимит регламента задан на ИЗОБРАЖЕНИЕ, а не на forward: TTA входит
    множителем, ансамбль — слагаемыми. `model` принимает и один модуль, и
    список: модели ансамбля могут быть разными, и у каждой своя цена, поэтому
    складывать их правильнее, чем умножать одну на количество.
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    per_view = sum(count_gflops(net, int(size)) for net in models)
    return Verdict(
        gflops=per_view * int(n_views),
        limit=float(limit),
        size=int(size),
        n_views=int(n_views),
        n_models=len(models),
    )


def largest_fitting_size(
    build: Callable[[int], object],
    *,
    start: int = 512,
    n_views: int = 1,
    n_models: int = 1,
    limit: float = LIMIT_GFLOPS,
    step: int = 32,
) -> int | None:
    """Самая большая сторона входа, кратная `step`, которая ещё влезает.

    Отвечает на вопрос, который возникает сразу после отказа: «а на чём тогда
    учить». `None` значит, что не влезает даже минимальный вход — тогда дело не
    в разрешении, а в самой сети или в числе видов TTA.

    Фабрика, а не готовая модель: у энкодеров с оконным вниманием размер входа
    зашит в маски внимания, и посчитать такую сеть на другой стороне нельзя —
    она там просто не собирается.

    Полного двоичного поиска не нужно: FLOPs почти квадратичны по стороне, и из
    одного замера получается близкая оценка, которую остаётся подвинуть на
    шаг-другой.
    """
    per_image = int(n_views) * int(n_models)

    def cost(side: int) -> float:
        return count_gflops(build(int(side)), int(side)) * per_image

    guess = int(start * math.sqrt(limit / cost(start)))
    size = max(step, guess - guess % step)

    while size >= step and cost(size) > limit:
        size -= step
    if size < step or cost(size) > limit:
        return None
    while cost(size + step) <= limit:
        size += step
    return size


def rejection_text(build: Callable[[int], object], verdict: Verdict) -> str:
    """Отказ вместе с ответом на вопрос, который возникает сразу следом.

    «Не влезает» без «а на чём тогда» заставляет подбирать размер вручную,
    перезапуская проверку на каждой попытке.
    """
    fits = largest_fitting_size(
        build,
        start=verdict.size,
        n_views=verdict.n_views,
        n_models=verdict.n_models,
        limit=verdict.limit,
    )
    tail = (f"в бюджет укладывается вход {fits}px" if fits
            else "в бюджет не укладывается ни одно разрешение — дело в самой сети")
    return f"{verdict.text}; {tail}"
