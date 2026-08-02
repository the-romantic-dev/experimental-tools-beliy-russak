"""Метрика AIC. Перенесена дословно, поэтому тесты стерегут инварианты."""

from __future__ import annotations

import numpy as np
import pytest

from aic.metric import (
    DEFAULT_AREA_GRID,
    DEFAULT_CLS_GRID,
    DEFAULT_MASK_GRID,
    AICAccumulator,
    dice_binary,
    harmonic_aic,
    score_masks,
)


def _mask(h, w, filled):
    m = np.zeros((h, w), dtype=bool)
    m[:filled] = True
    return m


def test_harmonic_aic_is_zero_when_a_component_is_zero():
    assert harmonic_aic(0.0, 0.0) == 0.0
    assert harmonic_aic(0.9, 1.0) == 0.0


def test_dice_of_identical_masks_is_one():
    m = _mask(10, 10, 4)
    assert dice_binary(m, m) == pytest.approx(1.0, abs=1e-4)


def test_score_masks_counts_the_one_percent_rule():
    """Негатив с предсказанием >= 1% площади — ложная тревога, меньше — нет."""
    gt_neg = np.zeros((100, 100), dtype=bool)
    small = np.zeros((100, 100), dtype=bool)
    small[0, :50] = True                      # 0.5% площади
    big = np.zeros((100, 100), dtype=bool)
    big[:2] = True                            # 2% площади
    gt_pos = _mask(100, 100, 10)

    res = score_masks([gt_pos, small, big], [gt_pos, gt_neg, gt_neg])
    assert res.n_pos == 1 and res.n_neg == 2
    assert res.fpr_neg == pytest.approx(0.5)


def test_tables_is_public_and_matches_sweep():
    """Разворот гистограмм доступен снаружи и согласован со свипом."""
    rng = np.random.default_rng(0)
    probs = rng.random((4, 8, 8)).astype(np.float32)
    gts = (rng.random((4, 8, 8)) > 0.7).astype(np.float32)
    acc = AICAccumulator(n_bins=64)
    acc.update(probs, gts)

    pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = acc.tables()
    assert pred_counts.shape == (4, 64)
    assert inter_counts.shape == (4, 64)
    assert gt_sum.shape == n_pixels.shape == cls_prob.shape == (4,)
    assert not hasattr(acc, "_tables")

    # |P_t| при t=0 — все пиксели кадра
    assert pred_counts[:, 0].tolist() == n_pixels.tolist()


def test_sweep_is_sorted_by_aic_descending():
    rng = np.random.default_rng(1)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((6, 8, 8)), (rng.random((6, 8, 8)) > 0.5).astype(np.float32))
    results = acc.sweep([0.2, 0.5, 0.8], DEFAULT_CLS_GRID[:2], DEFAULT_AREA_GRID[:2])
    aics = [r.aic for r in results]
    assert aics == sorted(aics, reverse=True)
    assert acc.best([0.2, 0.5, 0.8]).aic == max(r.aic for r in acc.sweep([0.2, 0.5, 0.8]))


def test_evaluate_agrees_with_sweep_of_one_point():
    rng = np.random.default_rng(2)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((5, 6, 6)), (rng.random((5, 6, 6)) > 0.6).astype(np.float32))
    direct = acc.evaluate(0.5, 0.0, 0.0)
    swept = acc.sweep([0.5], [0.0], [0.0])[0]
    assert direct.aic == swept.aic


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    acc = AICAccumulator(n_bins=16)
    acc.update(rng.random((3, 4, 4)), (rng.random((3, 4, 4)) > 0.5).astype(np.float32))
    path = tmp_path / "val.npz"
    acc.save(path)

    back = AICAccumulator.load(path)
    assert len(back) == len(acc)
    assert back.evaluate(0.5).aic == pytest.approx(acc.evaluate(0.5).aic)


def test_empty_accumulator_says_so():
    with pytest.raises(ValueError, match="пуст"):
        AICAccumulator().tables()


def test_default_grids_are_sane():
    assert min(DEFAULT_MASK_GRID) > 0.0 and max(DEFAULT_MASK_GRID) < 1.0
    assert DEFAULT_CLS_GRID[0] == 0.0
    assert DEFAULT_AREA_GRID[0] == 0.0
