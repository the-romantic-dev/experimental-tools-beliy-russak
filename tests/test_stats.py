"""Статистика достоверности прироста.

Ошибка здесь тише, чем ошибка в метрике: неверный доверительный интервал не
ломает прогон, он просто убеждает, что шум — это результат.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest

from experimental_tools_beliy_russak.metrics import AICAccumulator, harmonic_aic
from experimental_tools_beliy_russak.stats import (
    Boot,
    PerImage,
    align_by_stem,
    comparable,
    gate_check,
    paired_bootstrap,
    per_image,
    seeds_needed,
    verdict,
)


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


def synthetic_pair(rng, n_pos: int, n_neg: int, gain: float):
    """Два прогона на одних кадрах: у второго Dice выше на `gain`."""
    is_pos = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    dice_a = np.clip(rng.normal(0.70, 0.30, n_pos + n_neg), 0.0, 1.0)
    dice_b = np.clip(dice_a + gain, 0.0, 1.0)
    alarm = (rng.random(n_pos + n_neg) < 0.07).astype(float)
    return (
        PerImage(dice=dice_a, alarm=alarm, is_pos=is_pos),
        PerImage(dice=dice_b, alarm=alarm.copy(), is_pos=is_pos),
    )


def test_bootstrap_delta_equals_direct_difference():
    rng = np.random.default_rng(0)
    a, b = synthetic_pair(rng, 400, 100, gain=0.05)
    pos, neg = a.is_pos, ~a.is_pos

    expected = (
        harmonic_aic(b.dice[pos].mean(), b.alarm[neg].mean())
        - harmonic_aic(a.dice[pos].mean(), a.alarm[neg].mean())
    )
    got = paired_bootstrap(a, b, n=200, seed=1)

    assert got.delta == pytest.approx(expected, abs=1e-12)
    assert got.lo < got.delta < got.hi


def test_bootstrap_of_identical_runs_is_zero():
    rng = np.random.default_rng(2)
    a, _ = synthetic_pair(rng, 300, 80, gain=0.0)
    got = paired_bootstrap(a, a, n=200, seed=3)

    assert got.delta == pytest.approx(0.0, abs=1e-12)
    assert got.sd == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_covers_truth_about_95_percent_of_the_time():
    """Покрытие: интервал строится по выборке val, истина — по «популяции».

    Ради этого теста модуль и существует. Если покрытие уедет, все вердикты
    станут враньём, а заметить это иначе нечем.
    """
    rng = np.random.default_rng(7)
    population = synthetic_pair(rng, 5000, 1200, gain=0.05)
    pop_a, pop_b = population
    pop_pos, pop_neg = np.flatnonzero(pop_a.is_pos), np.flatnonzero(~pop_a.is_pos)
    truth = (
        harmonic_aic(pop_b.dice[pop_pos].mean(), pop_b.alarm[pop_neg].mean())
        - harmonic_aic(pop_a.dice[pop_pos].mean(), pop_a.alarm[pop_neg].mean())
    )

    covered = 0
    trials = 100
    for trial in range(trials):
        draw = np.random.default_rng(1000 + trial)
        take_pos = draw.choice(pop_pos, 400, replace=False)
        take_neg = draw.choice(pop_neg, 120, replace=False)
        take = np.concatenate([take_pos, take_neg])
        sample = [
            PerImage(dice=p.dice[take], alarm=p.alarm[take], is_pos=p.is_pos[take])
            for p in population
        ]
        boot = paired_bootstrap(sample[0], sample[1], n=400, seed=trial)
        covered += int(boot.lo <= truth <= boot.hi)

    assert covered / trials >= 0.85


def test_bootstrap_rejects_mismatched_sets():
    rng = np.random.default_rng(4)
    a, _ = synthetic_pair(rng, 100, 20, gain=0.0)
    b, _ = synthetic_pair(rng, 90, 20, gain=0.0)
    with pytest.raises(ValueError, match="align_by_stem"):
        paired_bootstrap(a, b, n=50, seed=0)


def make_boot(delta: float, half_width: float = 0.003) -> Boot:
    return Boot(delta=delta, lo=delta - half_width, hi=delta + half_width, sd=half_width / 2)


def test_seeds_needed_matches_the_documented_formula():
    # k = ceil(8 * sigma^2 / delta^2); значения из спеки
    assert seeds_needed(0.005, 0.008) == 21
    assert seeds_needed(0.0068, 0.008) == 12
    assert seeds_needed(0.012, 0.008) == 4
    assert seeds_needed(0.0, 0.008) is None


def test_small_gain_is_called_noise_and_priced_in_seeds():
    got = verdict(0.005, make_boot(0.005), train_sigma=0.008)
    assert got.label == "внутри шума обучения"
    assert got.seeds_needed == 21


def test_large_gain_with_clean_interval_is_confirmed():
    got = verdict(0.05, make_boot(0.05), train_sigma=0.008)
    assert got.label == "подтверждено"
    assert got.seeds_needed is None


def test_large_gain_with_interval_over_zero_is_not_confirmed():
    """Порог по шуму обучения пройден, но выборка val сама по себе неубедительна."""
    got = verdict(0.05, make_boot(0.05, half_width=0.08), train_sigma=0.008)
    assert got.label == "внутри шума обучения"
    assert "CI" in got.reason


def test_large_loss_is_called_worse():
    got = verdict(-0.05, make_boot(-0.05), train_sigma=0.008)
    assert got.label == "хуже"


def test_missing_train_sigma_does_not_pretend_to_know():
    got = verdict(0.05, make_boot(0.05), train_sigma=None)
    assert got.label == "пол шума не задан"


def test_diverged_configs_block_the_verdict():
    got = verdict(0.05, make_boot(0.05), train_sigma=0.008, diverged=["data.epoch_size"])
    assert got.label == "несопоставимо"
    assert "data.epoch_size" in got.reason


def test_comparable_finds_the_key_that_diverged():
    base = {"data": {"size": 768, "epoch_size": 8000, "val_frac": 0.25, "val_seed": 42},
            "train": {"epochs": 6, "bs": 4, "accum_steps": 4, "fold": 0}}
    other = {"data": {"size": 768, "epoch_size": 4000, "val_frac": 0.25, "val_seed": 42},
             "train": {"epochs": 12, "bs": 4, "accum_steps": 4, "fold": 0}}

    ok, diverged = comparable(base, base)
    assert ok and diverged == []

    ok, diverged = comparable(base, other)
    assert not ok
    assert diverged == ["data.epoch_size", "train.epochs"]


#: кривая f0-control-768: 8000 показов на эпоху, val/aic_tuned по эпохам
F0_CURVE = (
    np.array([8000, 16000, 24000, 32000, 40000, 48000], dtype=float),
    np.array([0.3490, 0.5207, 0.6563, 0.7361, 0.7794, 0.8030]),
)


def test_gate_stays_silent_before_the_minimum_budget():
    """На 16k показов даже провальное плечо не снимаем: кривые ещё не разошлись."""
    got = gate_check(F0_CURVE, samples=16000, aic=0.20, gate_delta=-0.05, after_samples=24000)
    assert not got.fired
    assert got.delta is None


def test_gate_fires_on_the_real_effnetv2_numbers():
    """g3-effnetv2-s на 24k показов: 0.5970 против 0.6563 у эталона."""
    got = gate_check(F0_CURVE, samples=24000, aic=0.5970, gate_delta=-0.05, after_samples=24000)
    assert got.fired
    assert got.ref_aic == pytest.approx(0.6563)
    assert got.delta == pytest.approx(-0.0593, abs=1e-4)


def test_gate_lets_a_normal_arm_through():
    """f7-aux-all на 24k: 0.6771 против 0.6563 — выше эталона, снимать нечего."""
    got = gate_check(F0_CURVE, samples=24000, aic=0.6771, gate_delta=-0.05, after_samples=24000)
    assert not got.fired
    assert got.delta == pytest.approx(0.0208, abs=1e-4)


def test_gate_interpolates_between_reference_epochs():
    """Плечо с epoch_size=4000 попадает на 28k — середину между эпохами эталона."""
    got = gate_check(F0_CURVE, samples=28000, aic=0.70, gate_delta=-0.05, after_samples=24000)
    assert got.ref_aic == pytest.approx((0.6563 + 0.7361) / 2)
    assert not got.fired


def test_gate_refuses_to_judge_outside_the_reference_curve():
    got = gate_check(F0_CURVE, samples=96000, aic=0.10, gate_delta=-0.05, after_samples=24000)
    assert not got.fired
    assert got.delta is None
    assert "кривой эталона" in got.reason
