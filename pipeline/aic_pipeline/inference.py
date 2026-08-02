"""Инференс: чекпоинт -> вероятности -> бинарные PNG-маски исходного размера.

Формат сабмита по условию: одноканальный PNG, 0 — фон, 255 — манипуляция,
размер как у входного изображения. Возврат к исходному размеру делается здесь,
а не в даталоадере, чтобы метрика считалась ровно на том, что уйдёт в сабмит.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import Cfg
from .datasets import PredictDataset, SegDataset
from .imageio import imwrite
from .metrics import AICAccumulator
from .models import build_model
from .transforms import build_transform
from .utils import pick_device

TTA_OPS = {
    "none": (lambda t: t, lambda t: t),
    "hflip": (lambda t: torch.flip(t, dims=[-1]), lambda t: torch.flip(t, dims=[-1])),
    "vflip": (lambda t: torch.flip(t, dims=[-2]), lambda t: torch.flip(t, dims=[-2])),
    "hvflip": (
        lambda t: torch.flip(t, dims=[-2, -1]),
        lambda t: torch.flip(t, dims=[-2, -1]),
    ),
}


def load_checkpoint(path: str | Path, device: torch.device | None = None, use_ema: bool = True):
    """Возвращает (model, cfg, calib). `calib` — пороги, подобранные при обучении."""
    device = device or pick_device("auto")
    state = torch.load(str(path), map_location=device, weights_only=False)
    cfg = Cfg(state["cfg"])
    model = build_model(cfg.model)

    weights = state.get("ema") if (use_ema and state.get("ema")) else state["model"]
    model.load_state_dict(weights)
    model.to(device, memory_format=torch.channels_last).eval()
    return model, cfg, state.get("calib", {})


@torch.no_grad()
def _forward_tta(model, images: torch.Tensor, tta: Sequence[str], amp: str):
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)
    prob_sum, cls_sum = None, None
    for op_name in tta:
        forward_op, inverse_op = TTA_OPS[op_name]
        # autocast("cuda") на CPU-тензорах не даёт ничего, кроме предупреждения:
        # смешанная точность есть только на карте
        with torch.amp.autocast("cuda", dtype=dtype,
                                enabled=dtype is not None and images.is_cuda):
            outputs = model(forward_op(images))
        probs = torch.sigmoid(inverse_op(outputs["logits"]).float())
        cls = torch.sigmoid(outputs["cls_logits"].float()).reshape(-1)
        prob_sum = probs if prob_sum is None else prob_sum + probs
        cls_sum = cls if cls_sum is None else cls_sum + cls
    return prob_sum / len(tta), cls_sum / len(tta)


def _to_original(prob: torch.Tensor, orig_h: int, orig_w: int, val_mode: str) -> np.ndarray:
    """(1, S, S) на модельной сетке -> (H, W) в исходном разрешении."""
    size = prob.shape[-1]
    if val_mode == "pad":
        scale = size / float(max(orig_h, orig_w))
        h = max(1, int(round(orig_h * scale)))
        w = max(1, int(round(orig_w * scale)))
        prob = prob[..., :h, :w]
    resized = F.interpolate(
        prob.unsqueeze(0), size=(orig_h, orig_w), mode="bilinear", align_corners=False
    )
    return resized.squeeze(0).squeeze(0).cpu().numpy()


@torch.no_grad()
def predict_stream(
    model,
    loader: DataLoader,
    device: torch.device,
    *,
    tta: Sequence[str] = ("none",),
    amp: str = "fp16",
    val_mode: str = "resize",
) -> Iterator[dict]:
    """Отдаёт по одному кадру: вероятностная карта исходного размера + cls-вероятность.

    `model` — одна модель или список моделей. Список означает ансамбль:
    вероятности усредняются ДО бинаризации, что и требуется — усреднять готовые
    бинарные маски заметно хуже. Все модели должны работать на одном входе,
    потому что батч через них прогоняется общий.
    """
    models = list(model) if isinstance(model, (list, tuple)) else [model]
    for net in models:
        net.eval()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        probs, cls_probs = None, None
        for net in models:
            net_probs, net_cls = _forward_tta(net, images, tta, amp)
            probs = net_probs if probs is None else probs + net_probs
            cls_probs = net_cls if cls_probs is None else cls_probs + net_cls
        probs, cls_probs = probs / len(models), cls_probs / len(models)

        for i in range(images.shape[0]):
            yield {
                "prob": _to_original(
                    probs[i], int(batch["orig_h"][i]), int(batch["orig_w"][i]), val_mode
                ),
                "cls_prob": float(cls_probs[i]),
                "index": int(batch["index"][i]),
                "name": batch["name"][i] if "name" in batch else None,
            }


def postprocess(
    prob: np.ndarray,
    cls_prob: float,
    *,
    mask_threshold: float = 0.5,
    cls_threshold: float = 0.0,
    min_area: float = 0.0,
) -> np.ndarray:
    """Вероятности -> uint8 маска 0/255 с применением всех правил постобработки."""
    mask = (prob >= mask_threshold)
    if cls_prob < cls_threshold:
        mask[:] = False
    elif min_area > 0 and mask.mean() < min_area:
        mask[:] = False
    return (mask.astype(np.uint8)) * 255


def predict_folder(
    checkpoint: str | Path,
    image_dir: str | Path,
    out_dir: str | Path,
    *,
    mask_threshold: float = 0.5,
    cls_threshold: float = 0.0,
    min_area: float = 0.0,
    tta: Sequence[str] = ("none",),
    batch_size: int = 8,
    num_workers: int = 6,
    limit: int | None = None,
    use_ema: bool = True,
) -> dict:
    device = pick_device("auto")
    model, cfg, _ = load_checkpoint(checkpoint, device, use_ema=use_ema)

    paths = sorted(
        p for p in Path(image_dir).rglob("*")
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if limit:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"в {image_dir} не найдено изображений")

    transform = build_transform(cfg.data, train=False)
    loader = DataLoader(
        PredictDataset(paths, transform), batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_positive = 0
    for item in predict_stream(
        model, loader, device, tta=tta, amp=str(cfg.train.get("amp", "fp16")),
        val_mode=str(cfg.data.get("val_mode", "resize")),
    ):
        mask = postprocess(
            item["prob"], item["cls_prob"],
            mask_threshold=mask_threshold, cls_threshold=cls_threshold, min_area=min_area,
        )
        n_positive += int(mask.any())
        imwrite(out_dir / f"{paths[item['index']].stem}.png", mask)

    return {"n_images": len(paths), "n_with_mask": n_positive, "out_dir": str(out_dir)}


@torch.no_grad()
def evaluate_full_res(
    checkpoint: str | Path,
    df,
    *,
    tta: Sequence[str] = ("none",),
    batch_size: int = 8,
    num_workers: int = 6,
    n_bins: int = 256,
    use_ema: bool = True,
) -> AICAccumulator:
    """Честная метрика: маски сравниваются в ИСХОДНОМ разрешении, а не на 512x512.

    Валидация в train.py считает метрику на модельной сетке — это быстро и
    годится для сравнения эпох, но итоговое число может отличаться. Перед
    сабмитом сверяйся этой функцией.
    """
    device = pick_device("auto")
    model, cfg, _ = load_checkpoint(checkpoint, device, use_ema=use_ema)

    dataset = SegDataset(
        df, build_transform(cfg.data, train=False),
        # Именно raw, даже если прогон учился с `data.source: cache`. Кадры в кэше
        # ужаты до cache_size, и «исходным разрешением» тут оказалось бы разрешение
        # кэша: маска возвращалась бы к нему, GT ужимался бы под неё, и вся затея
        # с честной сверкой перед сабмитом мерила бы не то, что уйдёт в сабмит.
        source="raw",
        gt_binarize=cfg.data.get("gt_binarize", 0.5),
        extra_negatives=0.0,
    )
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    accumulator = AICAccumulator(n_bins=n_bins)
    val_mode = str(cfg.data.get("val_mode", "resize"))
    amp = str(cfg.train.get("amp", "fp16"))

    from .datasets import _read_mask
    from .workspace import resolve

    for item in predict_stream(model, loader, device, tta=tta, amp=amp, val_mode=val_mode):
        row = dataset.df.iloc[item["index"]]  # SegDataset кладёт в index номер строки df
        prob = item["prob"]
        gt = _read_mask(resolve(row["gt_path"]), prob.shape[:2])
        gt_bin = (gt >= 128).astype(np.float32)
        accumulator.update(
            prob[None, ...], gt_bin[None, ...], np.array([item["cls_prob"]], dtype=np.float32)
        )
    return accumulator
