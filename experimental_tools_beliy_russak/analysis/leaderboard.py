"""Сравнение прогонов: одна таблица по всей папке runs/.

Показывает не весь конфиг, а только те параметры, которые между прогонами
РАЗЛИЧАЮТСЯ — иначе таблица становится нечитаемой уже на пятом эксперименте.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from ..config import flatten
from ..workspace import runs_root

HIDE_PREFIXES = ("_source", "name", "seed")


def _read_run(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.yaml"
    if not config_path.exists():
        return None

    row: dict = {"run": run_dir.name}
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        best = summary.get("best") or {}
        row.update({
            "best_aic": summary.get("best_aic"),
            "dice_pos": best.get("dice_pos"),
            "fpr_neg": best.get("fpr_neg"),
            "thr": best.get("mask_threshold"),
            "cls_thr": best.get("cls_threshold"),
            "min_area": best.get("min_area"),
            "epochs": summary.get("epochs_done"),
        })
    else:
        row["best_aic"] = None

    metrics_path = run_dir / "metrics.jsonl"
    if metrics_path.exists():
        lines = [l for l in metrics_path.read_text(encoding="utf-8").splitlines() if l]
        if lines:
            last = json.loads(lines[-1])
            row["last_epoch"] = last.get("step")
            row["elapsed"] = round(last.get("elapsed_s", 0) / 60, 1)
            if row.get("best_aic") is None:
                row["best_aic"] = max(
                    (json.loads(l).get("val/aic_tuned", 0) or 0) for l in lines
                )

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    row["_cfg"] = flatten(cfg)
    return row


def leaderboard(sort_by: str = "best_aic", only_diff: bool = True) -> pd.DataFrame:
    runs = runs_root()
    if not runs.exists():
        return pd.DataFrame()

    rows = [r for r in (_read_run(d) for d in sorted(runs.iterdir()) if d.is_dir()) if r]
    if not rows:
        return pd.DataFrame()

    configs = [row.pop("_cfg") for row in rows]
    keys = sorted({k for cfg in configs for k in cfg})
    if only_diff:
        keys = [
            k for k in keys
            if not k.startswith(HIDE_PREFIXES)
            and len({str(cfg.get(k)) for cfg in configs}) > 1
        ]

    for row, cfg in zip(rows, configs):
        for key in keys:
            row[key.split(".")[-1] if key.count(".") else key] = cfg.get(key)

    table = pd.DataFrame(rows)
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False, na_position="last")
    return table.round(4).reset_index(drop=True)


if __name__ == "__main__":
    print(leaderboard().to_string(index=False))
