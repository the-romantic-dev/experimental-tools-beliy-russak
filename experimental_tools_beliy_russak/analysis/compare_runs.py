"""Сравнение прогонов: истории обучения, кривые, итоговая таблица.

Вынесено из ноутбуков, чтобы в них оставался только текст эксперимента и вызовы,
а логика жила в одном месте и покрывалась тестами.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd
import yaml

from ..metrics import DEFAULT_MASK_GRID, AICAccumulator
from ..workspace import runs_root


def resolve_run(run: str | Path) -> Path:
    """Принимает 'b1_fast' или 'runs/b1_fast' или полный путь."""
    path = Path(run)
    if path.is_dir():
        return path
    candidate = runs_root() / path.name
    if candidate.is_dir():
        return candidate
    raise FileNotFoundError(f"не нашёл прогон: {run}")


def load_history(run: str | Path) -> pd.DataFrame:
    """metrics.jsonl -> DataFrame по эпохам."""
    run_dir = resolve_run(run)
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    frame = pd.DataFrame(rows)
    frame.insert(0, "run", run_dir.name)
    return frame


def load_summary(run: str | Path) -> dict:
    run_dir = resolve_run(run)
    path = run_dir / "summary.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def compare_table(runs: Sequence[str | Path]) -> pd.DataFrame:
    """Итоги прогонов рядом: лучший AIC, его составляющие, пороги, время."""
    rows = []
    for run in runs:
        run_dir = resolve_run(run)
        history = load_history(run_dir)
        summary = load_summary(run_dir)
        best = summary.get("best") or {}

        config_path = run_dir / "config.yaml"
        config = (
            yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            if config_path.exists() else {}
        )

        rows.append({
            "run": run_dir.name,
            "encoder": config.get("model", {}).get("encoder"),
            "arch": config.get("model", {}).get("arch"),
            "size": config.get("data", {}).get("size"),
            "AIC": summary.get("best_aic"),
            "Dice_pos": best.get("dice_pos"),
            "FPR_neg": best.get("fpr_neg"),
            "thr": best.get("mask_threshold"),
            "cls_thr": best.get("cls_threshold"),
            "epochs": summary.get("epochs_done"),
            "мин/эпоху": round(history["train/epoch_time_s"].mean() / 60, 1)
            if "train/epoch_time_s" in history else None,
            "всего_мин": round(history["elapsed_s"].max() / 60, 1)
            if "elapsed_s" in history else None,
        })
    return pd.DataFrame(rows).sort_values("AIC", ascending=False, na_position="last")


DEFAULT_PANELS = (
    "train/loss+val/loss",   # расхождение этих двух — первый признак переобучения
    "val/aic_tuned",
    "val/dice_tuned",
    "val/fpr_tuned",
    "val/best_thr+val/best_cls_thr",
    "train/lr",
)

LINE_STYLES = ("-", "--", ":", "-.")


def plot_history(
    runs: Sequence[str | Path],
    metrics: Sequence[str] = DEFAULT_PANELS,
    figsize: tuple[float, float] = (14, 9),
):
    """Кривые по эпохам для нескольких прогонов на общих осях.

    В одной строке `metrics` можно перечислить несколько величин через `+` —
    они лягут на одну панель разными типами линии. Так train и val loss видно
    рядом, а не на разных картинках.
    """
    histories = {resolve_run(r).name: load_history(r) for r in runs}
    histories = {name: h for name, h in histories.items() if not h.empty}
    if not histories:
        raise ValueError("ни у одного прогона нет metrics.jsonl")

    colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    n = len(metrics)
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, 2, figsize=figsize, squeeze=False)

    style_marks = ("——", "– –", "· · ·", "–·–")
    for ax, spec in zip(axes.ravel(), metrics):
        parts = spec.split("+")
        for run_idx, (name, history) in enumerate(histories.items()):
            colour = colours[run_idx % len(colours)]
            for part_idx, metric in enumerate(parts):
                if metric not in history:
                    continue
                ax.plot(
                    history["step"], history[metric],
                    color=colour, linestyle=LINE_STYLES[part_idx % len(LINE_STYLES)],
                    marker="o", ms=3,
                    # цвет кодирует прогон, тип линии — величину; подписываем
                    # только первую, иначе легенда разрастается вдвое
                    label=name if part_idx == 0 else None,
                )
        title = spec if len(parts) == 1 else "   ".join(
            f"{m} {style_marks[i % len(style_marks)]}" for i, m in enumerate(parts)
        )
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("эпоха")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2 if len(histories) > 4 else 1)

    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def threshold_curve(run: str | Path, cls_threshold: float = 0.0, min_area: float = 0.0,
                    oof_name: str = "val.npz") -> pd.DataFrame:
    """AIC/Dice/FPR как функция порога бинаризации по сохранённой статистике."""
    run_dir = resolve_run(run)
    path = run_dir / "oof" / oof_name
    if not path.exists():
        raise FileNotFoundError(f"нет {path}")
    accumulator = AICAccumulator.load(path)
    results = accumulator.sweep(list(DEFAULT_MASK_GRID), [cls_threshold], [min_area])
    frame = pd.DataFrame([r.as_dict() for r in results])
    frame.insert(0, "run", run_dir.name)
    return frame.sort_values("mask_threshold").reset_index(drop=True)


def plot_threshold_curves(
    runs: Sequence[str | Path],
    cls_threshold: float = 0.0,
    min_area: float = 0.0,
    figsize: tuple[float, float] = (13, 4),
):
    """Как AIC и его составляющие зависят от порога — видно запас прочности."""
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for run in runs:
        frame = threshold_curve(run, cls_threshold, min_area)
        name = frame["run"].iloc[0]
        for ax, column in zip(axes, ("aic", "dice_pos", "fpr_neg")):
            ax.plot(frame["mask_threshold"], frame[column], label=name)

    for ax, column in zip(axes, ("AIC", "Dice_pos", "FPR_neg")):
        ax.set_title(column)
        ax.set_xlabel("порог бинаризации")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
