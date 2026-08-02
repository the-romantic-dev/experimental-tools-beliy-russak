"""Синтез подделок из чистых оригиналов.

Зачем. Весь недобор Dice сидит на мелких масках: в корзине «меньше 1% кадра»
Dice 0.135 и 71% полных промахов, у промахнутых кадров oracle-Dice 0.157 —
сигнала в вероятностной карте нет вообще. Это дефицит признаков, а не порога.
`e2-area` пробовал лечить его перевзвешиванием и проиграл 0.021: перевзвешивать
нечего, мелких кадров мало и они одни и те же. Синтез снимает ограничение —
маска известна точно, а **площадь вставки задаётся параметром**, то есть мелких
позитивов можно сделать столько, сколько нужно.

Все операции устроены одинаково: строится карта смешивания `alpha` (мягкая по
краю), считается «правленый» вариант той же области, и результат вклеивается
через `alpha`. Отсюда главный инвариант — **кадр меняется ровно там, где маска
ненулевая**, и он же возвращаемая GT-маска.

Порядок сжатия принципиален. `post_jpeg` применяется ПОСЛЕ правки, ко всему
кадру: если сжать до неё, граница вставки не попадёт в JPEG-сетку, и модель
выучит артефакт синтеза за одну эпоху вместо признаков подделки.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

#: чем правится кадр. Первые две операции приносят новое содержимое, вторые две
#: рвут только низкоуровневую статистику — ровно то, что модель сейчас
#: промахивает полностью
OPS = ("copy_move", "splice", "recompress", "local_blur")

_DONOR_OPS = frozenset({"splice"})


def needs_donor(op: str) -> bool:
    """Нужен ли операции второй кадр-источник."""
    return op in _DONOR_OPS


@dataclass(frozen=True)
class SynthSettings:
    """Ветка `data.synth` конфига.

    `fraction` — доля синтетических кадров СРЕДИ ПОЗИТИВОВ пула (0.3 значит, что
    каждый третий позитив синтетический), `area_range` — границы доли кадра под
    вставкой, разыгрывается лог-равномерно.
    """

    fraction: float
    area_range: tuple[float, float] = (0.001, 0.03)
    ops: tuple[str, ...] = OPS
    feather: tuple[int, int] = (1, 5)
    post_jpeg: tuple[int, int] | None = (70, 95)

    def __post_init__(self) -> None:
        unknown = [op for op in self.ops if op not in OPS]
        if unknown:
            raise ValueError(
                f"data.synth.ops: неизвестные операции {unknown}; "
                f"доступны {list(OPS)}"
            )
        if not self.ops:
            raise ValueError("data.synth.ops: список пуст, синтезировать нечем")
        low, high = self.area_range
        if not 0.0 < low <= high < 1.0:
            raise ValueError(
                f"data.synth.area_range: ожидается 0 < low <= high < 1, получено {self.area_range}"
            )
        if not 0.0 <= self.fraction < 1.0:
            raise ValueError(
                f"data.synth.fraction: доля среди позитивов, ожидается [0, 1), "
                f"получено {self.fraction}"
            )

    @classmethod
    def from_config(cls, cfg: dict | None) -> "SynthSettings | None":
        """`None`, если синтез выключен — пустой блок или нулевая доля."""
        if not cfg:
            return None
        fraction = float(cfg.get("fraction", 0.0) or 0.0)
        if fraction <= 0.0:
            return None

        def pair(key: str, default):
            value = cfg.get(key, default)
            if value is None:
                return None
            first, second = value
            return (type(default[0])(first), type(default[1])(second))

        defaults = cls(fraction=fraction)
        return cls(
            fraction=fraction,
            area_range=pair("area_range", defaults.area_range),
            ops=tuple(str(op) for op in cfg.get("ops", defaults.ops)),
            feather=pair("feather", defaults.feather),
            post_jpeg=pair("post_jpeg", defaults.post_jpeg),
        )


def pick_op(rng: np.random.Generator, settings: SynthSettings) -> str:
    return str(rng.choice(settings.ops))


def _jpeg(image: np.ndarray, quality: int) -> np.ndarray:
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:  # pragma: no cover — кодек в сборке cv2 есть всегда
        return image
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def _alpha_map(
    shape: tuple[int, int], rng: np.random.Generator, settings: SynthSettings
) -> np.ndarray:
    """Карта смешивания: жёсткая фигура заданной площади, размытая по краю."""
    height, width = shape
    low, high = settings.area_range
    target = float(np.exp(rng.uniform(math.log(low), math.log(high))))
    area_px = target * height * width

    ellipse = bool(rng.random() < 0.5)
    if ellipse:  # площадь эллипса — pi/4 от описанного прямоугольника
        area_px /= math.pi / 4.0
    ratio = float(np.exp(rng.uniform(math.log(0.5), math.log(2.0))))
    box_h = int(round(math.sqrt(area_px / ratio)))
    box_w = int(round(math.sqrt(area_px * ratio)))
    box_h = int(np.clip(box_h, 4, height))
    box_w = int(np.clip(box_w, 4, width))

    top = int(rng.integers(0, height - box_h + 1))
    left = int(rng.integers(0, width - box_w + 1))

    hard = np.zeros(shape, dtype=np.float32)
    if ellipse:
        cv2.ellipse(
            hard,
            center=(left + box_w // 2, top + box_h // 2),
            axes=(max(1, box_w // 2), max(1, box_h // 2)),
            angle=float(rng.uniform(0, 180)), startAngle=0, endAngle=360,
            color=1.0, thickness=-1,
        )
    else:
        hard[top:top + box_h, left:left + box_w] = 1.0

    sigma = float(rng.integers(settings.feather[0], settings.feather[1] + 1))
    if sigma <= 0:
        return hard
    ksize = int(2 * math.ceil(3 * sigma) + 1)
    return cv2.GaussianBlur(hard, (ksize, ksize), sigma)


def _bbox(alpha: np.ndarray) -> tuple[int, int, int, int]:
    rows = np.flatnonzero(alpha.any(axis=1))
    cols = np.flatnonzero(alpha.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _wrapped_crop(
    image: np.ndarray, top: int, bottom: int, left: int, right: int, shift: tuple[int, int]
) -> np.ndarray:
    """Кусок того же размера, взятый со сдвигом и заворотом по краям кадра."""
    height, width = image.shape[:2]
    rows = (np.arange(top, bottom) + shift[0]) % height
    cols = (np.arange(left, right) + shift[1]) % width
    return image[np.ix_(rows, cols)]


def _edited_region(
    image: np.ndarray,
    op: str,
    box: tuple[int, int, int, int],
    rng: np.random.Generator,
    donor: np.ndarray | None,
) -> np.ndarray:
    """Правленый вариант области `box` — считается только по ней, не по кадру."""
    top, bottom, left, right = box
    height, width = image.shape[:2]

    if op in {"copy_move", "splice"}:
        source = image
        if op == "splice":
            if donor is None:
                raise ValueError("операция splice требует donor — второй кадр-источник")
            source = donor
            if source.shape[:2] != image.shape[:2]:
                source = cv2.resize(source, (width, height), interpolation=cv2.INTER_AREA)
        # сдвиг не меньше стороны области, иначе вставка ляжет сама на себя
        shift = (
            int(rng.integers(bottom - top, height - (bottom - top) + 1)),
            int(rng.integers(right - left, width - (right - left) + 1)),
        )
        patch = _wrapped_crop(source, top, bottom, left, right, shift)
        if rng.random() < 0.5:
            patch = patch[:, ::-1]
        return np.ascontiguousarray(patch)

    region = image[top:bottom, left:right]
    if op == "recompress":
        return _jpeg(region, int(rng.integers(30, 71)))
    if op == "local_blur":
        sigma = float(rng.uniform(0.8, 2.5))
        return cv2.GaussianBlur(region, (0, 0), sigma)
    raise ValueError(f"неизвестная операция синтеза: {op}; доступны {list(OPS)}")


def synthesize(
    image: np.ndarray,
    rng: np.random.Generator,
    settings: SynthSettings,
    *,
    op: str | None = None,
    donor: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Чистый кадр -> (кадр с подделкой, маска правки).

    `image` — RGB uint8, маска возвращается float32 в [0, 1] и мягкая по краю:
    бинаризацией занимается датасет тем же `data.gt_binarize`, что и настоящие
    GT-маски, — чтобы синтетика и реальные кадры проходили один путь.
    """
    op = op or pick_op(rng, settings)
    if needs_donor(op) and donor is None:
        raise ValueError(f"операция {op} требует donor — второй кадр-источник")

    alpha = _alpha_map(image.shape[:2], rng, settings)
    top, bottom, left, right = _bbox(alpha)

    patch = _edited_region(image, op, (top, bottom, left, right), rng, donor)
    weights = alpha[top:bottom, left:right, None]
    region = image[top:bottom, left:right].astype(np.float32)

    out = image.copy()
    out[top:bottom, left:right] = np.round(
        region * (1.0 - weights) + patch.astype(np.float32) * weights
    ).astype(np.uint8)

    if settings.post_jpeg is not None:
        out = _jpeg(out, int(rng.integers(settings.post_jpeg[0], settings.post_jpeg[1] + 1)))

    return out, alpha
