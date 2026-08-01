"""Статистика достоверности прироста.

Ошибка здесь тише, чем ошибка в метрике: неверный доверительный интервал не
ломает прогон, он просто убеждает, что шум — это результат.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest

from experimental_tools_beliy_russak.metrics import AICAccumulator
from experimental_tools_beliy_russak.stats import align_by_stem, per_image


def make_accumulator(seed: int = 0, n: int = 16) -> AICAccumulator:
    """Половина кадров — позитивы с квадратной маской, половина — негативы."""
    rng = np.random.default_rng(seed)
    probs = rng.random((n, 20, 20)).astype(np.float32)
    gts = np.zeros((n, 20, 20), dtype=np.float32)
    gts[::2, 5:15, 5:15] = 1.0
    cls = rng.random(n).astype(np.float32)

    accumulator = AICAccumulator(n_bins=256)
    accumulator.update(probs, gts, cls)
    return accumulator


@pytest.mark.parametrize("op", [(0.5, 0.0, 0.0), (0.25, 0.4, 0.0), (0.5, 0.0, 0.02)])
def test_per_image_reproduces_accumulator_evaluate(op):
    """Единственная проверка, которая ловит расхождение с настоящей метрикой."""
    accumulator = make_accumulator()
    expected = accumulator.evaluate(*op)
    got = per_image(accumulator, op)

    assert got.dice[got.is_pos].mean() == pytest.approx(expected.dice_pos, abs=1e-9)
    assert got.alarm[~got.is_pos].mean() == pytest.approx(expected.fpr_neg, abs=1e-9)
    assert got.is_pos.sum() == expected.n_pos
    assert (~got.is_pos).sum() == expected.n_neg


def test_align_by_stem_returns_intersection_in_matching_order():
    a = np.array(["x", "y", "z", "w"])
    b = np.array(["z", "w", "q", "x"])

    idx_a, idx_b = align_by_stem(a, b)

    assert list(a[idx_a]) == list(b[idx_b])
    assert sorted(a[idx_a]) == ["w", "x", "z"]


def test_align_by_stem_rejects_duplicates():
    with pytest.raises(ValueError, match="повторяются"):
        align_by_stem(np.array(["x", "x"]), np.array(["x"]))
