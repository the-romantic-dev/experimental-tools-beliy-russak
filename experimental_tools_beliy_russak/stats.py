"""Достоверность прироста: сравнение прогона с эталоном.

Зачем модуль. Два одинаковых прогона на текущем бюджете расходятся на 0.011 AIC,
а типичное плечо даёт 0.004–0.012. Читать такие числа глазами нельзя, поэтому
сравнение с эталоном считается прямо во время прогона.

Здесь только арифметика: ни torch, ни обращения к диску (кроме `load_reference`).
Это единственная часть, которую надо покрыть тестами всерьёз, а тестировать её
можно, только если она не тянет за собой GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import EPS, FP_AREA_THRESHOLD, AICAccumulator, harmonic_aic


@dataclass(frozen=True)
class PerImage:
    """По-картиночные величины в фиксированной операционной точке."""

    #: Dice каждого кадра; для негативов бессмысленен и не используется
    dice: np.ndarray
    #: 1.0, если кадр даёт ложную тревогу по правилу 1% площади
    alarm: np.ndarray
    #: маска позитивов (в GT есть изменения)
    is_pos: np.ndarray


def per_image(accumulator: AICAccumulator, op: tuple[float, float, float]) -> PerImage:
    """Разложить накопленные гистограммы в по-картиночные величины.

    `op` — `(mask_threshold, cls_threshold, min_area)`. Средние по этим массивам
    обязаны совпадать с `accumulator.evaluate(*op)` до последнего знака: иначе
    вердикт будет считаться не по той метрике, по которой отбираются модели.
    """
    mask_threshold, cls_threshold, min_area = op
    pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = accumulator._tables()

    # индекс бина берётся ровно так же, как в AICAccumulator.sweep
    bin_index = int(np.clip(int(mask_threshold * accumulator.n_bins), 0, accumulator.n_bins - 1))
    pred = pred_counts[:, bin_index].astype(np.float64)
    inter = inter_counts[:, bin_index].astype(np.float64)
    area = pred / np.maximum(n_pixels, 1.0)

    keep = (cls_prob >= cls_threshold) & (area >= min_area)
    pred = np.where(keep, pred, 0.0)
    inter = np.where(keep, inter, 0.0)
    area = np.where(keep, area, 0.0)

    return PerImage(
        dice=2.0 * inter / (pred + gt_sum + EPS),
        alarm=(area >= FP_AREA_THRESHOLD).astype(np.float64),
        is_pos=gt_sum > 0,
    )


def align_by_stem(stems_a, stems_b) -> tuple[np.ndarray, np.ndarray]:
    """Индексы кадров, присутствующих в обоих наборах val.

    Сравнивать по позиции нельзя. У `f1-seed7` подвыборка позитивов бралась
    глобальным сидом, и с `f0-control-768` у него совпадает 1892 строки из 5664 —
    сравнение по индексу молча смешало бы разные кадры.
    """
    a = np.asarray(stems_a)
    b = np.asarray(stems_b)
    for name, values in (("первом", a), ("втором", b)):
        if len(np.unique(values)) != len(values):
            raise ValueError(f"stem'ы в {name} наборе повторяются — данные битые")

    common = np.intersect1d(a, b)
    order_a = np.argsort(a)
    order_b = np.argsort(b)
    idx_a = order_a[np.searchsorted(a, common, sorter=order_a)]
    idx_b = order_b[np.searchsorted(b, common, sorter=order_b)]
    return idx_a, idx_b


@dataclass(frozen=True)
class Boot:
    """Результат парного бутстрапа разницы AIC: плечо минус эталон."""

    delta: float
    lo: float
    hi: float
    sd: float


def _aic_of_draws(sample: PerImage, pos_idx: np.ndarray, neg_idx: np.ndarray) -> np.ndarray:
    """AIC для пачки бутстрап-выборок сразу; `*_idx` имеют форму (реплик, кадров)."""
    dice = sample.dice[pos_idx].mean(axis=1)
    fpr = sample.alarm[neg_idx].mean(axis=1)
    rest = 1.0 - fpr
    return 2.0 * dice * rest / np.maximum(dice + rest, 1e-12)


def paired_bootstrap(
    a: PerImage,
    b: PerImage,
    *,
    n: int = 2000,
    seed: int = 0,
    block: int = 250,
) -> Boot:
    """Парный бутстрап: обе выборки пересэмплируются ОДНИМИ И ТЕМИ ЖЕ индексами.

    Парность здесь не украшение: кадры общие, и совпадающая часть шума выборки
    сокращается. На независимых выборках интервал был бы вдвое шире и не мерил
    бы ничего полезного.

    Считается блоками по `block` реплик: матрица индексов на 2000 реплик и 5025
    позитивов заняла бы 160 МБ, блоками — около 10 МБ.
    """
    if a.dice.shape != b.dice.shape:
        raise ValueError("наборы кадров разной длины — сначала align_by_stem")
    if not np.array_equal(a.is_pos, b.is_pos):
        raise ValueError("разметка позитивов различается — сначала align_by_stem")

    pos = np.flatnonzero(a.is_pos)
    neg = np.flatnonzero(~a.is_pos)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("для AIC нужны и позитивы, и негативы")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    done = 0
    while done < n:
        size = min(block, n - done)
        pos_idx = rng.choice(pos, size=(size, pos.size), replace=True).astype(np.int32)
        neg_idx = rng.choice(neg, size=(size, neg.size), replace=True).astype(np.int32)
        diffs[done:done + size] = _aic_of_draws(b, pos_idx, neg_idx) - _aic_of_draws(a, pos_idx, neg_idx)
        done += size

    delta = (
        harmonic_aic(b.dice[pos].mean(), b.alarm[neg].mean())
        - harmonic_aic(a.dice[pos].mean(), a.alarm[neg].mean())
    )
    return Boot(
        delta=float(delta),
        lo=float(np.percentile(diffs, 2.5)),
        hi=float(np.percentile(diffs, 97.5)),
        sd=float(diffs.std(ddof=1)) if n > 1 else 0.0,
    )
