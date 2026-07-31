"""Подбор постобработки по сохранённой статистике валидации.

Работает с runs/<name>/oof/val.npz — там лежат сжатые гистограммы вероятностей
по каждому кадру. Перебор сетки (порог маски × порог классификатора × минимальная
площадь) занимает секунды и НЕ требует ни модели, ни GPU, ни повторного прогона.

Почему это отдельный шаг: половина AIC — это (1 - FPR_neg), а FPR срабатывает
по грубому правилу «площадь >= 1% кадра». Оптимум по порогу для Dice и оптимум
для FPR почти никогда не совпадают, и найти компромисс перебором дешевле,
чем пытаться попасть в него лоссом.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import (
    DEFAULT_AREA_GRID,
    DEFAULT_CLS_GRID,
    DEFAULT_MASK_GRID,
    AICAccumulator,
    AICResult,
)


def calibrate(
    oof_path: str | Path,
    mask_grid=DEFAULT_MASK_GRID,
    cls_grid=DEFAULT_CLS_GRID,
    area_grid=DEFAULT_AREA_GRID,
    top_k: int = 10,
) -> tuple[AICResult, pd.DataFrame]:
    accumulator = AICAccumulator.load(oof_path)
    results = accumulator.sweep(list(mask_grid), list(cls_grid), list(area_grid))
    table = pd.DataFrame([r.as_dict() for r in results[:top_k]])
    return results[0], table


def calibrate_run(run_dir: str | Path, **kwargs) -> dict:
    run_dir = Path(run_dir)
    oof_path = run_dir / "oof" / "val.npz"
    if not oof_path.exists():
        raise FileNotFoundError(
            f"нет {oof_path}. Калибровка возможна только после обучения с валидацией."
        )
    best, table = calibrate(oof_path, **kwargs)
    payload = best.as_dict()
    (run_dir / "calib.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {"best": payload, "table": table}


def threshold_curve(oof_path: str | Path, cls_threshold: float = 0.0, min_area: float = 0.0):
    """AIC/Dice/FPR как функция порога бинаризации — удобно смотреть глазами."""
    accumulator = AICAccumulator.load(oof_path)
    results = accumulator.sweep(
        list(DEFAULT_MASK_GRID), [cls_threshold], [min_area]
    )
    frame = pd.DataFrame([r.as_dict() for r in results])
    return frame.sort_values("mask_threshold").reset_index(drop=True)
