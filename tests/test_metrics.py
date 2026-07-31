"""Проверка метрики AIC. Здесь ошибка дороже всего: неверная метрика тихо
уводит все эксперименты не туда, и заметно это станет только на лидерборде."""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest

from experimental_tools_beliy_russak.metrics import (
    FP_AREA_THRESHOLD,
    AICAccumulator,
    dice_binary,
    harmonic_aic,
    score_masks,
)


def make_mask(shape, box=None) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    if box:
        y0, y1, x0, x1 = box
        mask[y0:y1, x0:x1] = 255
    return mask


def test_perfect_positive_gives_aic_one():
    gt = make_mask((100, 100), (10, 50, 10, 50))
    result = score_masks([gt], [gt])
    assert result.dice_pos == pytest.approx(1.0, abs=1e-5)
    assert result.n_pos == 1 and result.n_neg == 0
    assert result.aic == pytest.approx(1.0, abs=1e-5)


def test_dice_matches_manual_computation():
    gt = make_mask((10, 10), (0, 5, 0, 10))    # 50 пикселей
    pred = make_mask((10, 10), (0, 10, 0, 5))  # 50 пикселей, пересечение 25
    assert dice_binary(pred > 0, gt > 0) == pytest.approx(2 * 25 / 100, abs=1e-6)


def test_negative_false_alarm_uses_one_percent_rule():
    gt = np.zeros((100, 100), dtype=np.uint8)

    just_under = make_mask((100, 100), (0, 99, 0, 1))  # 99 пикселей = 0.99%
    assert score_masks([just_under], [gt]).fpr_neg == 0.0

    just_over = make_mask((100, 100), (0, 100, 0, 1))  # 100 пикселей = 1.00%
    assert score_masks([just_over], [gt]).fpr_neg == 1.0


def test_empty_prediction_on_negative_is_not_penalised():
    gt = np.zeros((64, 64), dtype=np.uint8)
    result = score_masks([np.zeros_like(gt)], [gt])
    assert result.fpr_neg == 0.0
    # позитивов нет -> Dice_pos = 0 -> гармоническое среднее нулевое
    assert result.dice_pos == 0.0 and result.aic == 0.0


def test_harmonic_mean_punishes_either_component():
    assert harmonic_aic(0.9, 0.0) == pytest.approx(0.9473684, abs=1e-6)
    assert harmonic_aic(0.9, 1.0) == 0.0   # все негативы провалены
    assert harmonic_aic(0.0, 0.0) == 0.0   # сегментация провалена


def test_accumulator_matches_bruteforce_on_random_data():
    rng = np.random.default_rng(0)
    probs, gts = [], []
    for i in range(24):
        prob = rng.random((32, 40)).astype(np.float32)
        gt = np.zeros((32, 40), dtype=np.float32)
        if i % 4 != 0:  # каждый четвёртый кадр — негатив
            gt[rng.integers(0, 16):, rng.integers(0, 20):] = 1.0
        probs.append(prob)
        gts.append(gt)

    accumulator = AICAccumulator(n_bins=256)
    accumulator.update(np.stack(probs), np.stack(gts))

    for threshold in (0.25, 0.5, 0.75):
        expected = score_masks(
            [(p >= threshold) for p in probs],
            [(g > 0.5) for g in gts],
        )
        got = accumulator.evaluate(mask_threshold=threshold)
        assert got.dice_pos == pytest.approx(expected.dice_pos, abs=1e-6)
        assert got.fpr_neg == pytest.approx(expected.fpr_neg, abs=1e-9)
        assert got.aic == pytest.approx(expected.aic, abs=1e-6)


def test_cls_threshold_zeroes_predictions():
    rng = np.random.default_rng(1)
    probs = rng.random((8, 20, 20)).astype(np.float32)
    gts = np.zeros((8, 20, 20), dtype=np.float32)  # все негативы
    cls_probs = np.full(8, 0.2, dtype=np.float32)

    accumulator = AICAccumulator(n_bins=256)
    accumulator.update(probs, gts, cls_probs)

    assert accumulator.evaluate(0.5, cls_threshold=0.0).fpr_neg == 1.0
    # порог классификатора выше выданной уверенности -> маски обнуляются
    assert accumulator.evaluate(0.5, cls_threshold=0.5).fpr_neg == 0.0


def test_min_area_rule_suppresses_small_predictions():
    prob = np.zeros((1, 100, 100), dtype=np.float32)
    prob[0, :2, :100] = 0.9      # 2% кадра -> ложная тревога
    gt = np.zeros((1, 100, 100), dtype=np.float32)

    accumulator = AICAccumulator(n_bins=256)
    accumulator.update(prob, gt)
    assert accumulator.evaluate(0.5, min_area=0.0).fpr_neg == 1.0
    assert accumulator.evaluate(0.5, min_area=0.03).fpr_neg == 0.0


def test_sweep_returns_sorted_results_and_best_is_max():
    rng = np.random.default_rng(2)
    probs = rng.random((12, 24, 24)).astype(np.float32)
    gts = (rng.random((12, 24, 24)) > 0.7).astype(np.float32)

    accumulator = AICAccumulator(n_bins=64)
    accumulator.update(probs, gts)
    results = accumulator.sweep([0.3, 0.5, 0.7], [0.0], [0.0])
    assert [r.aic for r in results] == sorted([r.aic for r in results], reverse=True)
    assert accumulator.best([0.3, 0.5, 0.7]).aic == results[0].aic


def test_save_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    probs = rng.random((5, 16, 16)).astype(np.float32)
    gts = (rng.random((5, 16, 16)) > 0.5).astype(np.float32)

    accumulator = AICAccumulator(n_bins=64)
    accumulator.update(probs, gts)
    path = tmp_path / "oof.npz"
    accumulator.save(path)

    restored = AICAccumulator.load(path)
    assert len(restored) == len(accumulator)
    assert restored.evaluate(0.5).aic == pytest.approx(accumulator.evaluate(0.5).aic)


def test_fp_area_threshold_is_the_documented_one_percent():
    assert FP_AREA_THRESHOLD == 0.01
