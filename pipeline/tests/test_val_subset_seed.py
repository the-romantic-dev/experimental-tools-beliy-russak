"""Состав валидации не должен зависеть от сида обучения.

Иначе прогон с другим `seed` меряется на другой подвыборке, и разница между ним
и опорным прогоном перестаёт быть оценкой шума обучения — она смешивается с
шумом выборки валидации. Ровно эту смесь и получило бы плечо f1_seed7, ради
которого замер пола шума затевается.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import pandas as pd
import pytest

from aic_pipeline.config import load_config
from aic_pipeline.train import _subset


def make_fold(n_pos: int = 400, n_neg: int = 40) -> pd.DataFrame:
    rows = [
        {"stem": f"p{i}", "is_negative": False, "mask_area": 0.2}
        for i in range(n_pos)
    ] + [
        {"stem": f"n{i}", "is_negative": True, "mask_area": 0.0}
        for i in range(n_neg)
    ]
    return pd.DataFrame(rows)


def selected(df, seed, frac=0.25):
    out = _subset(df, frac, None, seed, keep_negatives=True)
    return set(out["stem"])


def test_subset_depends_on_the_seed_it_is_given():
    """Базовое свойство, от которого и возникала проблема."""
    df = make_fold()
    assert selected(df, 42) != selected(df, 7)


def test_subset_is_deterministic_for_one_seed():
    df = make_fold()
    assert selected(df, 42) == selected(df, 42)


def test_all_negatives_survive_fractional_subsetting():
    """Негативы дают FPR_neg — половину метрики, их прореживать нельзя."""
    df = make_fold(n_pos=400, n_neg=40)
    out = _subset(df, 0.25, None, 42, keep_negatives=True)
    assert int(out["is_negative"].sum()) == 40
    assert len(out) == 40 + 100


def test_val_seed_is_defined_and_decoupled_in_base_config():
    cfg = load_config("f0_control")
    assert "val_seed" in cfg.data, "в data должен быть отдельный val_seed"
    assert cfg.data.val_seed == 42


@pytest.mark.parametrize("config_name", ["f0_control", "f1_seed7"])
def test_series_arms_share_one_validation_subset(config_name):
    """f0 и f1 отличаются сидом обучения, но обязаны меряться на одних кадрах."""
    reference = load_config("f0_control")
    cfg = load_config(config_name)

    df = make_fold()
    ref_rows = selected(df, int(reference.data.val_seed), reference.data.val_frac)
    arm_rows = selected(df, int(cfg.data.val_seed), cfg.data.val_frac)
    assert ref_rows == arm_rows


def test_changing_training_seed_does_not_move_validation():
    """Сид обучения и сид валидации — разные ручки."""
    f0 = load_config("f0_control")
    f1 = load_config("f1_seed7")

    assert f1.seed != f0.seed, "плечо обязано отличаться сидом обучения"
    assert f1.data.val_seed == f0.data.val_seed, "а валидация — совпадать"
