"""Врезка статистики в прогон: смотрим на то, что попадает в артефакты.

Обучение здесь не запускается — проверяются только те куски run(), которые
собирают поля метрик и summary.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

from aic_pipeline.config import load_config
from aic_pipeline.stats import Gate, gate_metrics, stats_settings


def test_gate_metrics_are_absent_when_stats_are_off():
    assert gate_metrics(None) == {}


def test_gate_metrics_carry_the_three_fields():
    gate = Gate(ref_aic=0.6563, delta=-0.0593, fired=True, reason="…")
    assert gate_metrics(gate) == {
        "ref/aic_at_samples": 0.6563,
        "ref/delta": -0.0593,
        "ref/gate": True,
    }


def test_gate_metrics_keep_none_when_gate_stayed_silent():
    gate = Gate(ref_aic=None, delta=None, fired=False, reason="рано судить")
    assert gate_metrics(gate) == {
        "ref/aic_at_samples": None,
        "ref/delta": None,
        "ref/gate": False,
    }


def test_default_config_leaves_the_machinery_off():
    """При stats.reference: null в metrics.jsonl не должно появиться ни одного поля."""
    assert stats_settings(load_config("_base")) is None
    assert gate_metrics(None) == {}
