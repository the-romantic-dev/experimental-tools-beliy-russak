"""Отчёт по датасету: что лежит, в каких пропорциях, какие разрешения.

Нужен, чтобы решения про размер входа, аугментации и баланс негативов
принимались по цифрам, а не по ощущениям.
"""

from __future__ import annotations

import numpy as np

from aic.data import load_index
from ..workspace import index_path, split_path


def _section(title: str, body: str) -> str:
    return f"\n## {title}\n\n{body}\n"


def profile_dataset() -> str:
    df = load_index(index_path())
    parts = [f"# Профиль датасета\n\nстрок: {len(df)}, групп (исходных кадров): {df['group_id'].nunique()}"]

    negatives = df["is_negative"]
    parts.append(_section(
        "Позитивы и негативы",
        f"негативов (пустой GT): {int(negatives.sum())} ({negatives.mean() * 100:.2f}%)\n"
        f"позитивов: {int((~negatives).sum())}\n"
        f"уникальных чистых оригиналов в src: {df['orgl_path'].dropna().nunique()}\n\n"
        "Негативы формируют FPR_neg — вторую половину AIC. Их доля в трейне мала, "
        "поэтому в конфиге есть data.negative_fraction (баланс в батче) и "
        "data.extra_negatives (добор чистых оригиналов).",
    ))

    parts.append(_section("Домены", df.groupby("domain").agg(
        n=("stem", "size"),
        доля=("stem", lambda s: round(len(s) / len(df) * 100, 1)),
        негативов=("is_negative", "sum"),
        групп=("group_id", "nunique"),
        площадь_med=("mask_area", "median"),
    ).sort_values("n", ascending=False).to_string()))

    parts.append(_section("Генераторы манипуляций", df.groupby("generator").agg(
        n=("stem", "size"),
        доля=("stem", lambda s: round(len(s) / len(df) * 100, 1)),
        площадь_med=("mask_area", "median"),
        площадь_p90=("mask_area", lambda s: round(float(np.percentile(s, 90)), 4)),
    ).sort_values("n", ascending=False).to_string()))

    positive_area = df.loc[~negatives, "mask_area"]
    quantiles = positive_area.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    tiny = float((positive_area < 0.01).mean())
    parts.append(_section(
        "Площадь маски (только позитивы, доля кадра)",
        quantiles.round(4).to_string()
        + f"\n\nмасок меньше 1% кадра: {tiny * 100:.2f}% — на них Dice особенно хрупок, "
          "а порог бинаризации решает всё.",
    ))

    long_side = df[["height", "width"]].max(axis=1)
    parts.append(_section(
        "Разрешение",
        f"длинная сторона: медиана {long_side.median():.0f}, "
        f"p05 {long_side.quantile(0.05):.0f}, p95 {long_side.quantile(0.95):.0f}, "
        f"max {long_side.max():.0f}\n"
        f"кадров с длинной стороной > 1024: {int((long_side > 1024).sum())}\n"
        f"маска не совпадает по размеру с изображением: {int(df['size_mismatch'].sum())}",
    ))

    if split_path().exists():
        from aic.data import check_leakage, load_folds

        folds = load_folds(split_path())
        leak = check_leakage(folds)
        parts.append(_section(
            "Фолды",
            folds.groupby("fold").agg(
                n=("stem", "size"),
                негативов=("is_negative", "sum"),
                доля_нег=("is_negative", "mean"),
            ).round(4).to_string()
            + f"\n\nутечка групп между фолдами: {leak['leaked_groups']}",
        ))
    else:
        parts.append(_section("Фолды", "ещё не построены: python cli.py split"))

    return "".join(parts)


if __name__ == "__main__":
    print(profile_dataset())
