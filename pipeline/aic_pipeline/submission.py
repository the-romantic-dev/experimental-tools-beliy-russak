"""Сборка и проверка сабмита.

Формат по условию задачи:

    submission.zip
      submission.csv          колонки img_path, prediction_path
      predictions/*.png       одноканальные PNG, значения 0/255,
                              размер совпадает с входным изображением

Пути в `test.csv` заданы относительно папки, где лежит сам CSV, и
`prediction_path` в шаблоне устроен как `predictions/<stem>_pred.png`.
Шаблон переиспользуется как есть — угадывать правило именования не нужно,
организаторы его уже задали.

Механика регламента — чтение test.csv, запись, проверка, zip — живёт в
`aic.submit`. Здесь остаётся то, что знает про чекпоинты и конфиги: загрузка
моделей ансамбля, сверка их входов, гейт по бюджету и разрешение порогов из
`calib.json` прогонов.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from aic.data import imwrite
from aic.metric import FP_AREA_THRESHOLD
from aic.submit import REQUIRED_COLUMNS, pack_zip  # noqa: F401 — переэкспорт
from aic.submit import load_test_table as _load_test_table
from aic.submit import validate as _validate

from . import budget
from .datasets import PredictDataset
from .inference import load_checkpoint, postprocess, predict_stream
from .transforms import build_transform
from .utils import pick_device
from .workspace import workspace

#: `template=None` значит «шаблон не нужен», поэтому «не задано» нужен отдельный
#: маркер — иначе эти два случая не различить
DEFAULT = "<из воркспейса>"


# ---------------------------------------------------------------------------
# таблица тестовой выборки
# ---------------------------------------------------------------------------

def load_test_table(
    test_csv: str | Path | None = None,
    template: str | Path | None = DEFAULT,
) -> tuple[pd.DataFrame, Path]:
    """`aic.submit.load_test_table` с путями по умолчанию из воркспейса."""
    if test_csv is None:
        test_csv = workspace().test_csv
    if template is DEFAULT:
        template = workspace().submission_template
    return _load_test_table(test_csv, template)


# ---------------------------------------------------------------------------
# сборка
# ---------------------------------------------------------------------------

def _resolve_thresholds(
    run_dirs: Sequence[Path],
    mask_threshold: float | None,
    cls_threshold: float | None,
    min_area: float | None,
) -> tuple[dict, list[str]]:
    """Пороги из calib.json прогонов; явно переданные значения побеждают."""
    notes: list[str] = []
    calibs = []
    for run_dir in run_dirs:
        path = Path(run_dir) / "calib.json"
        if path.exists():
            calibs.append(json.loads(path.read_text(encoding="utf-8")))
        else:
            notes.append(f"нет {path} — калибровка для этого прогона не делалась")

    def pick(name: str, explicit: float | None, fallback: float) -> float:
        if explicit is not None:
            return float(explicit)
        values = [c[name] for c in calibs if name in c]
        if not values:
            return fallback
        if len(values) > 1 and len(set(values)) > 1:
            notes.append(
                f"{name}: у прогонов разные значения {values}, взято среднее. "
                "Для ансамбля пороги лучше задать явно — усреднение вероятностей "
                "меняет калибровку"
            )
        return float(np.mean(values))

    return (
        {
            "mask_threshold": pick("mask_threshold", mask_threshold, 0.5),
            "cls_threshold": pick("cls_threshold", cls_threshold, 0.0),
            "min_area": pick("min_area", min_area, 0.0),
        },
        notes,
    )


def build_submission(
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    *,
    test_csv: str | Path | None = None,
    template: str | Path | None = DEFAULT,
    checkpoint: str = "best",
    mask_threshold: float | None = None,
    cls_threshold: float | None = None,
    min_area: float | None = None,
    tta: Sequence[str] = ("none",),
    batch_size: int = 8,
    num_workers: int = 6,
    limit: int | None = None,
    use_ema: bool = True,
    device: str | torch.device | None = None,
    progress=print,
) -> dict:
    run_dirs = [Path(r) for r in run_dirs]
    out_dir = Path(out_dir)
    table, root = load_test_table(test_csv, template)
    if limit:
        table = table.head(limit).copy()

    device = pick_device("auto") if device is None else torch.device(device)
    models, cfgs, sizes, val_modes, amps = [], [], set(), set(), set()
    for run_dir in run_dirs:
        ckpt = Path(checkpoint) if str(checkpoint).endswith(".pt") \
            else run_dir / "ckpt" / f"{checkpoint}.pt"
        if not ckpt.exists():
            raise FileNotFoundError(f"нет чекпоинта {ckpt}")
        model, cfg, _ = load_checkpoint(ckpt, device, use_ema=use_ema)
        models.append(model)
        cfgs.append(cfg)
        sizes.add(int(cfg.data.get("size", 512)))
        val_modes.add(str(cfg.data.get("val_mode", "resize")))
        amps.add(str(cfg.train.get("amp", "fp16")))
        progress(f"загружен {ckpt}")

    if len(sizes) > 1 or len(val_modes) > 1:
        raise ValueError(
            f"модели ансамбля требуют одинакового входа, а тут size={sorted(sizes)} "
            f"val_mode={sorted(val_modes)}. Собери сабмиты по отдельности."
        )
    # amp у моделей ансамбля может отличаться: он меняет только точность
    # вычисления forward, а не форму входа, так что берём любой
    size, val_mode, amp = sizes.pop(), val_modes.pop(), amps.pop()

    # Бюджет вычислений — до первого кадра. Считается настоящий рецепт: столько
    # моделей, сколько в ансамбле, столько видов, сколько задано в TTA. Пометка
    # `budget.exempt`, которой разрешён исследовательский прогон, здесь не
    # действует: посылка вне лимита это ноль за весь этап.
    verdict = budget.check_submission(cfgs, n_views=len(tuple(tta)))
    if not verdict.ok:
        raise ValueError(
            f"сабмит не укладывается в бюджет: {verdict.text}. "
            "Уменьшай вход, убирай TTA или собирай прогоны по отдельности"
        )
    progress(f"бюджет: {verdict.text}")

    thresholds, notes = _resolve_thresholds(run_dirs, mask_threshold, cls_threshold, min_area)
    for note in notes:
        progress(f"ВНИМАНИЕ: {note}")
    progress(f"пороги: {thresholds}")

    paths = [root / p for p in table["img_path"]]
    transform = build_transform({"size": size, "val_mode": val_mode}, train=False)
    loader = DataLoader(
        PredictDataset(paths, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for rel in table["prediction_path"]:
        (out_dir / rel).parent.mkdir(parents=True, exist_ok=True)

    areas = np.zeros(len(table), dtype=np.float64)
    cls_probs = np.zeros(len(table), dtype=np.float64)
    done = 0
    started = time.perf_counter()
    for item in predict_stream(models, loader, device, tta=tuple(tta), amp=amp, val_mode=val_mode):
        idx = item["index"]
        mask = postprocess(item["prob"], item["cls_prob"], **thresholds)
        imwrite(out_dir / table["prediction_path"].iloc[idx], mask)
        areas[idx] = float(np.count_nonzero(mask)) / mask.size
        cls_probs[idx] = item["cls_prob"]
        done += 1
        if done % 250 == 0:
            progress(f"  {done}/{len(table)}")
    elapsed = time.perf_counter() - started

    table.to_csv(out_dir / "submission.csv", index=False)

    non_empty = areas > 0
    flagged = areas >= FP_AREA_THRESHOLD
    stats = {
        "n_images": len(table),
        "n_models": len(models),
        "gflops_per_image": round(verdict.gflops, 1),
        # Регламент даёт 50 ms на кадр на H100. Здесь это не доказательство, а
        # единственный доступный сигнал: считается весь путь от чтения файла до
        # записи PNG, на том железе, где собирали. Батч и число воркеров на
        # число влияют, поэтому сравнивать его имеет смысл только с самим собой.
        "ms_per_image": round(1000 * elapsed / max(1, len(table)), 1),
        "device": str(device),
        "thresholds": thresholds,
        "empty_masks": int((~non_empty).sum()),
        "empty_share": round(float((~non_empty).mean()), 4),
        "flagged_share": round(float(flagged.mean()), 4),
        "area_median_nonempty": round(float(np.median(areas[non_empty])), 4) if non_empty.any() else 0.0,
        "cls_prob_median": round(float(np.median(cls_probs)), 4),
        "out_dir": str(out_dir),
    }
    return stats


def validate_submission(
    path: str | Path,
    test_csv: str | Path | None = None,
    max_problems: int = 20,
    check_sizes: bool = True,
    partial: bool = False,
) -> dict:
    """`aic.submit.validate` с путём к test.csv по умолчанию из воркспейса."""
    return _validate(
        path,
        workspace().test_csv if test_csv is None else test_csv,
        max_problems=max_problems,
        check_sizes=check_sizes,
        partial=partial,
    )
