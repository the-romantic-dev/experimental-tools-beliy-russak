"""Логирование прогона: консоль + metrics.jsonl + metrics.csv + TensorBoard.

JSONL — источник правды (дописывается построчно, переживает падение процесса),
CSV — чтобы открыть в Excel, TensorBoard — чтобы смотреть кривые.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd


class RunLogger:
    def __init__(self, run_dir: Path, use_tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.run_dir / "metrics.jsonl"
        self.csv_path = self.run_dir / "metrics.csv"
        self.log_path = self.run_dir / "train.log"
        self.start = time.time()
        self.writer = None

        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except Exception as exc:  # tensorboard не обязателен
                self.info(f"TensorBoard отключён: {exc}")

    def info(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log_metrics(self, step: int, metrics: dict[str, Any], prefix: str = "") -> None:
        record = {"step": step, "elapsed_s": round(time.time() - self.start, 1)}
        for key, value in metrics.items():
            record[f"{prefix}{key}" if prefix else key] = value

        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"step", "elapsed_s"}:
                    self.writer.add_scalar(key.replace("/", "_").replace("_", "/", 1), value, step)
            self.writer.flush()

    def flush_csv(self) -> None:
        if not self.jsonl_path.exists():
            return
        rows = [json.loads(line) for line in self.jsonl_path.read_text(encoding="utf-8").splitlines() if line]
        pd.DataFrame(rows).to_csv(self.csv_path, index=False)

    def log_hparams(self, flat_cfg: dict, metrics: dict) -> None:
        if self.writer is None:
            return
        clean = {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                 for k, v in flat_cfg.items()}
        numeric = {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        try:
            self.writer.add_hparams(clean, numeric)
        except Exception:
            pass

    def close(self) -> None:
        self.flush_csv()
        if self.writer is not None:
            self.writer.close()


def read_metrics(run_dir: Path) -> pd.DataFrame:
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return pd.DataFrame(rows)
