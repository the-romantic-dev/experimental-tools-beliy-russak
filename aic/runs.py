"""Папка прогона: ручка, которая пишет и читает документированный формат.

Формат тот же, что был у `train.run`, и меняться не будет — на нём стоят все
накопленные прогоны:

    <прогон>/
      config.yaml          непрозрачный снапшот: что вы захотели запомнить
      metrics.jsonl        источник правды, дописывается построчно
      metrics.csv          то же для Excel, перезаписывается на close()
      train.log            текстовый журнал
      summary.json         итоги, дописываются в несколько заходов
      ckpt/                чекпоинты
      oof/val.npz          гистограммы валидации
      oof/val_rows.parquet строки валидации в том же порядке
      tb/                  TensorBoard
      preds/               картинки для глаз

Библиотека не заглядывает в `config.yaml`: что там лежит — дело того, кто писал.

Писатели — глаголы (`log`, `save_*`), читатели — свойства (`history`,
`snapshot`, `summary`). Иначе `snapshot` пришлось бы делать и методом, и
свойством одновременно.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

#: подпапки, которые создаются сразу: пути на них ссылаются до первой записи
SUBDIRS = ("ckpt", "oof", "tb", "preds")


class Run:
    """Ручка на папку прогона."""

    def __init__(self, run_dir: str | Path, *, tensorboard: bool = False) -> None:
        self.dir = Path(run_dir)
        self.jsonl_path = self.dir / "metrics.jsonl"
        self.csv_path = self.dir / "metrics.csv"
        self.log_path = self.dir / "train.log"
        self.summary_path = self.dir / "summary.json"
        self.start = time.time()
        self.writer = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.dir / "tb"))
            except Exception as exc:  # tensorboard не обязателен
                self.info(f"TensorBoard отключён: {exc}")

    # --- создание и открытие ---------------------------------------------

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        name: str,
        *,
        resume: bool = False,
        tensorboard: bool = True,
    ) -> "Run":
        """`<runs_root>/<name>`; занятое имя без `resume` получает суффикс времени.

        Затирать чужую папку нельзя ни при каких обстоятельствах: в ней лежат
        часы обучения, и восстановить их неоткуда.
        """
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = runs_root / name
        if run_dir.exists() and not resume:
            run_dir = runs_root / f"{name}__{time.strftime('%m%d-%H%M%S')}"
        for sub in SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return cls(run_dir, tensorboard=tensorboard)

    @classmethod
    def open(cls, run_dir: str | Path) -> "Run":
        """Чужая папка на чтение. TensorBoard не поднимается."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"нет папки прогона {run_dir}")
        return cls(run_dir, tensorboard=False)

    def __repr__(self) -> str:
        return f"Run({str(self.dir)!r})"

    # --- запись ------------------------------------------------------------

    def info(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log(self, step: int, metrics: Mapping[str, Any], prefix: str = "") -> None:
        """Строка в `metrics.jsonl` (+ TensorBoard). `step` и `elapsed_s` свои.

        Ось показов библиотека не считает: если сравнение прогонов пойдёт по
        числу показов, а не по номеру эпохи, логируйте `samples` обычной
        метрикой — см. `Run.curve`.
        """
        record: dict[str, Any] = {
            "step": int(step),
            "elapsed_s": round(time.time() - self.start, 1),
        }
        for key, value in metrics.items():
            record[f"{prefix}{key}" if prefix else key] = value

        self.dir.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"step", "elapsed_s"}:
                    self.writer.add_scalar(key.replace("/", "_").replace("_", "/", 1), value, step)
            self.writer.flush()

    def save_snapshot(self, obj: Mapping[str, Any], name: str = "config.yaml") -> Path:
        """Записать снапшот как есть. Библиотека внутрь не смотрит."""
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(dict(obj), fh, allow_unicode=True, sort_keys=False)
        return path

    def save_summary(self, patch: Mapping[str, Any]) -> Path:
        """Дописать в `summary.json`, слив по ВЕРХНЕМУ уровню ключей.

        Мерж, а не перезапись: сводку заполняют в несколько заходов — обучение,
        потом калибровка, потом бюджет. Вложенные словари заменяются целиком:
        сливать `best` по частям некому и незачем, а частичный `best` был бы
        опаснее отсутствующего.
        """
        current = self.summary
        current.update(dict(patch))
        self.summary_path.parent.mkdir(parents=True, exist_ok=True)
        self.summary_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.summary_path

    def flush_csv(self) -> None:
        if not self.jsonl_path.exists():
            return
        self._history_frame().to_csv(self.csv_path, index=False)

    def close(self) -> None:
        self.flush_csv()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- чтение -------------------------------------------------------------

    def _history_frame(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        rows = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)

    @property
    def summary(self) -> dict:
        """`summary.json` или пустой словарь."""
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text(encoding="utf-8")) or {}
