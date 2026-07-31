"""Разбиение на фолды без утечки.

Один исходный кадр порождает несколько манипуляций (42.5k оригиналов -> 103.7k
строк). Если положить разные версии одного кадра в train и val, валидация
завысит качество. Поэтому режем `StratifiedGroupKFold` по `group_id`.

Страта = domain × generator × признак негатива × корзина по площади маски.
Это удерживает и доменный баланс, и долю негативов (от них зависит FPR_neg,
а значит и половина метрики) примерно одинаковыми во всех фолдах.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .indexing import load_index
from .workspace import split_path

AREA_BINS = (0.0, 0.01, 0.03, 0.08, 0.20, 1.01)


def area_bucket(area: float) -> int:
    return int(np.digitize(area, AREA_BINS[1:-1]))


def make_strata(df: pd.DataFrame) -> pd.Series:
    buckets = df["mask_area"].map(area_bucket)
    return (
        df["domain"].astype(str)
        + "|" + df["generator"].astype(str)
        + "|" + df["is_negative"].astype(int).astype(str)
        + "|" + buckets.astype(str)
    )


def make_folds(
    n_folds: int = 5,
    seed: int = 42,
    index_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    out_path = Path(out_path) if out_path is not None else split_path()
    df = load_index(index_path) if index_path else load_index()
    df = df[~df["broken"]].reset_index(drop=True)

    strata = make_strata(df)
    # редкие страты схлопываем, иначе StratifiedGroupKFold ругается
    counts = strata.value_counts()
    rare = set(counts[counts < n_folds].index)
    strata = strata.map(lambda s: "rare" if s in rare else s)

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=np.int8)
    for fold_id, (_, val_idx) in enumerate(
        splitter.split(df, y=strata, groups=df["group_id"])
    ):
        fold[val_idx] = fold_id

    df["fold"] = fold
    if (df["fold"] < 0).any():
        raise RuntimeError("часть строк не попала ни в один фолд")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_folds(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else split_path()
    if not Path(path).exists():
        raise FileNotFoundError(f"нет сплита {path}. Собери его: python cli.py split")
    return pd.read_parquet(path)


def check_leakage(df: pd.DataFrame) -> dict:
    """Ни одна group_id не должна встречаться в двух фолдах."""
    per_group = df.groupby("group_id")["fold"].nunique()
    leaked = per_group[per_group > 1]
    return {"leaked_groups": int(len(leaked)), "examples": leaked.index[:5].tolist()}


def summarize(df: pd.DataFrame) -> str:
    table = df.groupby("fold").agg(
        n=("stem", "size"),
        neg=("is_negative", "sum"),
        neg_frac=("is_negative", "mean"),
        groups=("group_id", "nunique"),
        area_med=("mask_area", "median"),
    )
    leak = check_leakage(df)
    return "\n".join(
        [
            table.to_string(),
            "",
            f"утечка групп между фолдами: {leak['leaked_groups']}"
            + (f" (например {leak['examples']})" if leak["leaked_groups"] else " — чисто"),
            "",
            "домены по фолдам:",
            pd.crosstab(df["fold"], df["domain"]).to_string(),
        ]
    )
