"""Достоверность прироста. Арифметика перенесена — тесты стерегут её смысл."""

from __future__ import annotations

import numpy as np
import pytest

from aic.metric import AICAccumulator
from aic.runs import Eval
from aic.stats import (
    Boot,
    align_by_stem,
    compare,
    diverged_keys,
    gate_check,
    gate_metrics,
    paired_bootstrap,
    per_image,
    seeds_needed,
    verdict,
)


def _eval(n_pos=30, n_neg=10, quality=0.8, seed=0, stems=None):
    """Аккумулятор с управляемым качеством: доля покрытой маски GT.

    Именно доля покрытия, а не величина вероятности: при фиксированном пороге
    0.55 и 0.95 бинаризуются одинаково, и «плохое» плечо оказалось бы точной
    копией хорошего.
    """
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=64)
    probs, gts = [], []
    for i in range(n_pos + n_neg):
        gt = np.zeros((8, 8), dtype=np.float32)
        prob = rng.random((8, 8)).astype(np.float32) * 0.3
        if i < n_pos:
            gt[:4] = 1.0
            prob[:max(1, int(round(4 * quality)))] = 0.9
        probs.append(prob)
        gts.append(gt)
    acc.update(np.stack(probs), np.stack(gts))
    names = stems if stems is not None else [f"кадр{i}" for i in range(n_pos + n_neg)]
    return Eval(acc, np.array(names))


def test_per_image_means_match_the_accumulator():
    """Иначе вердикт считался бы не по той метрике, по которой отбирают модели."""
    ev = _eval()
    op = (0.5, 0.0, 0.0)
    sample = per_image(ev.acc, op)
    direct = ev.acc.evaluate(*op)
    assert sample.dice[sample.is_pos].mean() == pytest.approx(direct.dice_pos)
    assert sample.alarm[~sample.is_pos].mean() == pytest.approx(direct.fpr_neg)


def test_align_by_stem_finds_the_intersection():
    a = np.array(["a", "b", "c", "d"])
    b = np.array(["c", "a", "z"])
    idx_a, idx_b = align_by_stem(a, b)
    assert a[idx_a].tolist() == b[idx_b].tolist() == ["a", "c"]


def test_align_by_stem_rejects_duplicates():
    with pytest.raises(ValueError, match="повторяются"):
        align_by_stem(np.array(["a", "a"]), np.array(["a"]))


def test_paired_bootstrap_of_identical_samples_is_zero():
    ev = _eval()
    sample = per_image(ev.acc, (0.5, 0.0, 0.0))
    boot = paired_bootstrap(sample, sample, n=200, seed=0)
    assert boot.delta == pytest.approx(0.0, abs=1e-12)
    assert boot.lo == pytest.approx(0.0, abs=1e-12)
    assert boot.hi == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_is_reproducible():
    a = per_image(_eval(quality=0.6, seed=1).acc, (0.5, 0.0, 0.0))
    b = per_image(_eval(quality=0.9, seed=2).acc, (0.5, 0.0, 0.0))
    assert paired_bootstrap(a, b, n=300, seed=7) == paired_bootstrap(a, b, n=300, seed=7)


def test_paired_bootstrap_refuses_mismatched_samples():
    a = per_image(_eval(n_pos=30).acc, (0.5, 0.0, 0.0))
    b = per_image(_eval(n_pos=20).acc, (0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="разной длины"):
        paired_bootstrap(a, b, n=10)


def test_seeds_needed_grows_as_the_effect_shrinks():
    assert seeds_needed(0.02, 0.005) < seeds_needed(0.005, 0.005)
    assert seeds_needed(0.0, 0.005) is None


def test_seeds_needed_matches_the_two_sigma_rule():
    # k >= 8 * sigma^2 / delta^2
    assert seeds_needed(0.01, 0.005) == 2


def test_verdict_is_asymmetric_between_win_and_loss():
    """Заявка на выигрыш требует и пола шума, и чистого CI; провал — только пола."""
    sigma = 0.005
    floor = 2.0 * sigma * np.sqrt(2.0)

    dirty = Boot(delta=floor + 0.001, lo=-0.002, hi=0.02, sd=0.006)
    assert verdict(dirty.delta, dirty, sigma).label == "внутри шума обучения"

    clean = Boot(delta=floor + 0.001, lo=0.001, hi=0.02, sd=0.006)
    assert verdict(clean.delta, clean, sigma).label == "подтверждено"

    loss = Boot(delta=-floor - 0.001, lo=-0.03, hi=0.001, sd=0.006)
    assert verdict(loss.delta, loss, sigma).label == "хуже"


def test_verdict_without_sigma_refuses_to_judge():
    boot = Boot(delta=0.05, lo=0.01, hi=0.09, sd=0.02)
    assert verdict(0.05, boot, None).label == "пол шума не задан"


def test_verdict_on_diverged_keys_is_incomparable():
    boot = Boot(delta=0.05, lo=0.01, hi=0.09, sd=0.02)
    got = verdict(0.05, boot, 0.005, diverged=["train.epochs"])
    assert got.label == "несопоставимо"
    assert "train.epochs" in got.reason


def test_gate_holds_its_fire_before_the_threshold():
    curve = (np.array([8000.0, 16000.0]), np.array([0.4, 0.6]))
    gate = gate_check(curve, samples=8000, aic=0.01, after_samples=24000)
    assert gate.fired is False
    assert "рано судить" in gate.reason


def test_gate_fires_on_a_real_lag():
    curve = (np.array([8000.0, 16000.0, 24000.0]), np.array([0.4, 0.6, 0.7]))
    gate = gate_check(curve, samples=24000, aic=0.60, gate_delta=-0.05, after_samples=8000)
    assert gate.fired is True
    assert gate.ref_aic == pytest.approx(0.7)
    assert gate.delta == pytest.approx(-0.1)


def test_gate_interpolates_between_reference_points():
    curve = (np.array([8000.0, 24000.0]), np.array([0.4, 0.8]))
    gate = gate_check(curve, samples=16000, aic=0.6, after_samples=8000)
    assert gate.ref_aic == pytest.approx(0.6)
    assert gate.fired is False


def test_gate_outside_the_reference_curve_stays_silent():
    curve = (np.array([8000.0, 16000.0]), np.array([0.4, 0.6]))
    gate = gate_check(curve, samples=99999, aic=0.1, after_samples=1000)
    assert gate.fired is False
    assert "вне кривой" in gate.reason


def test_gate_metrics_of_none_is_empty():
    assert gate_metrics(None) == {}
    filled = gate_metrics(gate_check(
        (np.array([1000.0, 2000.0]), np.array([0.4, 0.5])), 2000, 0.45, after_samples=100
    ))
    assert set(filled) == {"ref/aic_at_samples", "ref/delta", "ref/gate"}


def test_diverged_keys_requires_an_explicit_list():
    a = {"train": {"epochs": 6}, "data": {"size": 768}}
    b = {"train": {"epochs": 12}, "data": {"size": 768}}
    assert diverged_keys(a, b, ["train.epochs", "data.size"]) == ["train.epochs"]
    assert diverged_keys(a, b, []) == []
    with pytest.raises(TypeError):
        diverged_keys(a, b)


def test_diverged_keys_treats_a_missing_path_as_none():
    assert diverged_keys({"a": 1}, {}, ["a"]) == ["a"]
    assert diverged_keys({}, {}, ["нет.такого"]) == []


def test_compare_against_itself_is_a_flat_zero():
    ev = _eval()
    cmp = compare(ev, ev, op=(0.5, 0.0, 0.0), train_sigma=0.005, bootstrap_n=200)
    assert cmp.delta_ref_op == pytest.approx(0.0, abs=1e-12)
    assert cmp.n_common == len(ev)
    assert cmp.verdict.label == "внутри шума обучения"


def test_compare_sees_a_better_arm():
    weak = _eval(quality=0.55, seed=1)
    strong = _eval(quality=0.95, seed=1)
    cmp = compare(strong, weak, op=(0.5, 0.0, 0.0), train_sigma=0.001, bootstrap_n=300)
    assert cmp.delta_ref_op > 0
    assert cmp.verdict.label in {"подтверждено", "внутри шума обучения"}


def test_compare_warns_about_a_partial_overlap():
    # кадры 0..19 позитивы, 20..27 негативы; в пересечение обязаны попасть и те,
    # и другие — иначе AIC на нём просто не определён
    a = _eval(n_pos=20, n_neg=8, stems=[f"кадр{i}" for i in range(28)])
    b_names = [f"кадр{i}" if (i < 10 or i >= 24) else f"чужой{i}" for i in range(28)]
    b = _eval(n_pos=20, n_neg=8, stems=b_names)

    cmp = compare(a, b, op=(0.5, 0.0, 0.0), bootstrap_n=100)
    assert cmp.n_common == 14
    assert cmp.n_pos == 10 and cmp.n_neg == 4
    assert "совпадает с текущим только" in cmp.warning


def test_compare_reports_both_operating_points():
    own = _eval(quality=0.9, seed=3)
    ref = _eval(quality=0.7, seed=3)
    cmp = compare(own, ref, op=(0.5, 0.0, 0.0), own_op=(0.3, 0.0, 0.0),
                  train_sigma=0.004, bootstrap_n=200)
    text = "\n".join(cmp.report("эталон"))
    assert "в точке эталона" in text
    assert "в своей тюненой точке" in text
    assert set(cmp.as_dict()) >= {"delta_ref_op", "delta_own_op", "ci_lo", "ci_hi", "label"}
