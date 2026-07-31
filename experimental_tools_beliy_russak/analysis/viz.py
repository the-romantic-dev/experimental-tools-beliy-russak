"""Визуальная отладка: вход / GT / предсказание одной картинкой.

Цифры говорят, что метрика просела, но не говорят почему. Сетка худших по Dice
кадров обычно отвечает на это за пару секунд: не тот масштаб, размытые границы,
модель рисует поверх текстуры, негативы ловят шум и так далее.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ..datasets import SegDataset, _read_image, _read_mask  # noqa: E402
from ..inference import load_checkpoint, predict_stream  # noqa: E402
from ..metrics import dice_binary  # noqa: E402
from ..workspace import resolve  # noqa: E402
from ..transforms import build_transform  # noqa: E402
from ..utils import pick_device  # noqa: E402


def _overlay(image: np.ndarray, mask: np.ndarray, color=(255, 64, 64), alpha=0.45) -> np.ndarray:
    out = image.astype(np.float32).copy()
    layer = np.zeros_like(out)
    layer[mask > 0] = color
    blend = mask[..., None] > 0
    out = np.where(blend, out * (1 - alpha) + layer * alpha, out)
    return out.astype(np.uint8)


def make_grid(
    run_dir: str | Path,
    n: int = 12,
    out: Path | None = None,
    checkpoint: str = "best",
    worst: bool = True,
    scan: int = 240,
    batch_size: int = 8,
    threshold: float | None = None,
    num_workers: int = 4,
) -> Path:
    from torch.utils.data import DataLoader

    run_dir = Path(run_dir)
    ckpt = Path(checkpoint) if str(checkpoint).endswith(".pt") else run_dir / "ckpt" / f"{checkpoint}.pt"
    rows_path = run_dir / "oof" / "val_rows.parquet"
    if not rows_path.exists():
        raise FileNotFoundError(f"нет {rows_path}: прогон не доучился до первой валидации")

    calib_path = run_dir / "calib.json"
    if threshold is None:
        import json

        threshold = (
            json.loads(calib_path.read_text(encoding="utf-8")).get("mask_threshold", 0.5)
            if calib_path.exists() else 0.5
        )

    df = pd.read_parquet(rows_path)
    df = df.sample(n=min(scan, len(df)), random_state=0).reset_index(drop=True)

    device = pick_device("auto")
    model, cfg, _ = load_checkpoint(ckpt, device)
    dataset = SegDataset(
        df, build_transform(cfg.data, train=False),
        source=cfg.data.get("source", "raw"),
        cache_size=int(cfg.data.get("cache_size", 768)),
        gt_binarize=cfg.data.get("gt_binarize", 0.5),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    items = []
    for item in predict_stream(
        model, loader, device,
        amp=str(cfg.train.get("amp", "fp16")),
        val_mode=str(cfg.data.get("val_mode", "resize")),
    ):
        row = df.iloc[item["index"]]
        gt = _read_mask(resolve(row["gt_path"]), item["prob"].shape[:2]) >= 128
        pred = item["prob"] >= threshold
        score = dice_binary(pred, gt) if gt.any() else (1.0 - float(pred.mean() >= 0.01))
        items.append({"row": row, "pred": pred, "gt": gt, "score": score,
                      "cls": item["cls_prob"]})

    items.sort(key=lambda d: d["score"], reverse=not worst)
    items = items[:n]

    fig, axes = plt.subplots(len(items), 3, figsize=(11, 3.4 * len(items)))
    axes = np.atleast_2d(axes)
    for ax_row, item in zip(axes, items):
        image = _read_image(resolve(item["row"]["chng_path"]))
        image = cv2.resize(image, (item["pred"].shape[1], item["pred"].shape[0]))
        ax_row[0].imshow(image)
        ax_row[0].set_title(
            f"{item['row']['domain']}/{item['row']['generator']}  cls={item['cls']:.2f}", fontsize=9
        )
        ax_row[1].imshow(_overlay(image, item["gt"].astype(np.uint8), (64, 220, 64)))
        ax_row[1].set_title("GT", fontsize=9)
        ax_row[2].imshow(_overlay(image, item["pred"].astype(np.uint8)))
        ax_row[2].set_title(f"pred  dice={item['score']:.3f}", fontsize=9)
        for ax in ax_row:
            ax.axis("off")

    fig.suptitle(
        f"{run_dir.name} — {'худшие' if worst else 'лучшие'} {len(items)} @ thr={threshold:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    out = Path(out) if out else run_dir / f"viz_{'worst' if worst else 'best'}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out
