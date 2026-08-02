"""Статистика прогонов: арифметика из `aic.stats`, знание конфига — здесь.

Бутстрап, вердикт и гейт живут в библиотеке: там они не тянут ни torch, ни
формат чужой папки. Здесь остаётся ровно то, чего библиотека решать не должна:

* какие ключи делают два прогона несопоставимыми. У каждого своя форма конфига,
  и список по умолчанию был бы тихой регламентацией способа обучения;
* разбор блока `stats` из конфига;
* подъём эталона с диска, включая ось показов — её даёт только
  `data.epoch_size`, то есть опять конфиг.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from aic.runs import Eval, Run
from aic.stats import (  # noqa: F401 — переэкспорт для вызывающего кода
    Boot,
    Comparison,
    Gate,
    PerImage,
    Verdict,
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

from .config import get_path as _get_path
from .workspace import runs_root

#: ключи, определяющие бюджет прогона. Сравнивать между собой можно только
#: прогоны, у которых они совпадают: иначе разница мерит бюджет, а не гипотезу
BUDGET_KEYS = (
    "data.size",
    "data.epoch_size",
    "data.val_frac",
    # val_limit и val_keep_negatives задают состав валидации наравне с val_frac:
    # разойдясь по ним, два прогона меряются на разных наборах кадров
    "data.val_limit",
    "data.val_keep_negatives",
    "data.val_seed",
    "train.epochs",
    "train.bs",
    "train.accum_steps",
    "train.fold",
)


def comparable(cfg: dict, cfg_ref: dict, keys=BUDGET_KEYS) -> tuple[bool, list[str]]:
    """Совпадают ли бюджеты двух прогонов; вторым — список разошедшихся ключей."""
    diverged = diverged_keys(cfg, cfg_ref, list(keys))
    return not diverged, diverged


def resolve_reference(name: str) -> Path:
    """Имя прогона или путь к его папке."""
    candidate = Path(name)
    return candidate if candidate.exists() else runs_root() / name


@dataclass(frozen=True)
class Reference:
    """Опорный прогон, поднятый с диска."""

    name: str
    eval: Eval
    op: tuple[float, float, float]
    cfg: dict
    curve: tuple[np.ndarray, np.ndarray]

    @property
    def accumulator(self):
        """Совместимость с кодом, который знал только про аккумулятор."""
        return self.eval.acc

    @property
    def stems(self) -> np.ndarray:
        return self.eval.stems


def load_reference(run_dir) -> Reference:
    """Единственная функция модуля, которая ходит на диск."""
    run = Run.open(run_dir)
    cfg = run.snapshot

    # Ось кривой — ЧИСЛО ПОКАЗОВ, а его даёт только data.epoch_size. Без него все
    # точки легли бы в ноль и `gate_check` на каждой эпохе отвечал бы «вне кривой»,
    # то есть гейт молча выключался бы на весь прогон. Пустая кривая означает ровно
    # то же самое, но об этом хотя бы можно сказать вслух один раз на старте.
    epoch_size = int(_get_path(cfg, "data.epoch_size") or 0)
    if epoch_size > 0 and "val/aic_tuned" in run.history.columns:
        curve = run.curve("val/aic_tuned", x=("step", epoch_size))
    else:
        curve = (np.zeros(0, dtype=float), np.zeros(0, dtype=float))

    return Reference(
        name=run.dir.name,
        eval=run.load_eval(),
        op=run.operating_point(),
        cfg=cfg,
        curve=curve,
    )


def compare_to_reference(
    accumulator,
    stems,
    cfg: dict,
    reference: Reference,
    *,
    own_op: tuple[float, float, float],
    train_sigma: float | None,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> Comparison:
    """Собрать финальное сравнение: дельты, интервал, вердикт.

    Отличие от библиотечного `compare` — только в том, что несопоставимость
    считается по `BUDGET_KEYS` этого слоя.
    """
    own = Eval(accumulator, np.asarray(stems))
    _, diverged = comparable(cfg, reference.cfg)
    return compare(
        own,
        reference.eval,
        op=reference.op,
        own_op=own_op,
        train_sigma=train_sigma,
        bootstrap_n=bootstrap_n,
        bootstrap_seed=bootstrap_seed,
        diverged=diverged,
    )


GATE_ACTIONS = ("warn", "stop")


@dataclass(frozen=True)
class StatsSettings:
    reference: str
    train_sigma: float | None
    bootstrap_n: int
    bootstrap_seed: int
    gate_delta: float
    gate_after_samples: int
    gate_action: str


def stats_settings(cfg) -> StatsSettings | None:
    """Разобрать блок `stats`. None означает «машинерия выключена»."""
    reference = _get_path(cfg, "stats.reference")
    if not reference:
        return None

    action = str(_get_path(cfg, "stats.gate_action") or "warn")
    if action not in GATE_ACTIONS:
        raise ValueError(f"stats.gate_action={action!r}, допустимо: {GATE_ACTIONS}")

    train_sigma = _get_path(cfg, "stats.train_sigma")
    bootstrap_n = _get_path(cfg, "stats.bootstrap_n")
    bootstrap_seed = _get_path(cfg, "stats.bootstrap_seed")
    gate_delta = _get_path(cfg, "stats.gate_delta")
    after = _get_path(cfg, "stats.gate_after_samples")

    return StatsSettings(
        reference=str(reference),
        train_sigma=None if train_sigma is None else float(train_sigma),
        bootstrap_n=2000 if bootstrap_n is None else int(bootstrap_n),
        bootstrap_seed=0 if bootstrap_seed is None else int(bootstrap_seed),
        gate_delta=-0.05 if gate_delta is None else float(gate_delta),
        gate_after_samples=24000 if after is None else int(after),
        gate_action=action,
    )
