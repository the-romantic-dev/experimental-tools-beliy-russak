"""Worker-safe dataset for notebook segmentation experiments.

Jupyter notebooks on Windows cannot reliably use ``DataLoader(num_workers > 0)``
when the Dataset class is declared inside an IPython cell: worker processes are
started with spawn and need to import the class from a real module. This module
keeps the lightweight ``aic`` notebook path, but makes the data pipeline
importable so workers can prefetch batches.
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from aic import data as aic_data
from aic.paths import Workspace

cv2.setNumThreads(0)

GT_POSITIVE_VALUE = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resize_pair(image: np.ndarray, mask: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (size, size), interpolation=cv2.INTER_LINEAR)
    return image, (mask >= 0.5).astype(np.float32)


def random_resized_crop(
    rng: np.random.Generator,
    image: np.ndarray,
    mask: np.ndarray,
    size: int,
    scale: tuple[float, float],
    ratio: tuple[float, float] = (0.75, 1.3333),
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    for _ in range(10):
        target = h * w * rng.uniform(*scale)
        aspect = math.exp(rng.uniform(math.log(ratio[0]), math.log(ratio[1])))
        crop_w = int(round(math.sqrt(target * aspect)))
        crop_h = int(round(math.sqrt(target / aspect)))
        if crop_w <= w and crop_h <= h:
            x0 = int(rng.integers(0, w - crop_w + 1))
            y0 = int(rng.integers(0, h - crop_h + 1))
            image = image[y0:y0 + crop_h, x0:x0 + crop_w]
            mask = mask[y0:y0 + crop_h, x0:x0 + crop_w]
            break
    return resize_pair(image, mask, size)


def random_flips(
    rng: np.random.Generator,
    image: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        image, mask = image[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.2:
        image, mask = image[::-1], mask[::-1]
    turns = int(rng.integers(0, 4))
    if turns:
        image, mask = np.rot90(image, turns), np.rot90(mask, turns)
    return np.ascontiguousarray(image), np.ascontiguousarray(mask)


def photometric(rng: np.random.Generator, image: np.ndarray) -> np.ndarray:
    if rng.random() < 0.5:
        gain = 1.0 + rng.uniform(-0.12, 0.12)
        bias = rng.uniform(-12.0, 12.0)
        image = np.clip(image.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)

    if rng.random() < 0.3:
        quality = int(rng.integers(60, 101))
        ok, buffer = cv2.imencode(
            ".jpg",
            np.ascontiguousarray(image[:, :, ::-1]),
            [cv2.IMWRITE_JPEG_QUALITY, quality],
        )
        if ok:
            image = np.ascontiguousarray(cv2.imdecode(buffer, cv2.IMREAD_COLOR)[:, :, ::-1])

    if rng.random() < 0.2:
        if rng.random() < 0.5:
            noise = rng.normal(0.0, rng.uniform(2.0, 8.0), image.shape)
            image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        else:
            kernel = 3 if rng.random() < 0.5 else 5
            image = cv2.GaussianBlur(image, (kernel, kernel), 0)
    return np.ascontiguousarray(image)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    x = image.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


class ForgeryDataset(Dataset):
    """Notebook-compatible image/mask dataset that can be spawned by DataLoader."""

    def __init__(
        self,
        rows,
        *,
        size: int,
        train: bool,
        workspace: str | Path | Workspace | None = None,
        cache_size: int = 0,
        crop_scale: tuple[float, float] = (0.35, 1.0),
        extra_negatives: float = 0.0,
        seed: int = 0,
    ) -> None:
        self.size = int(size)
        self.train = bool(train)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self.crop_scale = tuple(float(x) for x in crop_scale)
        self.ws = workspace if isinstance(workspace, Workspace) else Workspace.find(workspace)
        self._rng: np.random.Generator | None = None

        self.records = [
            (row.chng_path, row.gt_path, bool(row.is_negative))
            for row in rows.itertuples()
        ]

        if self.train and extra_negatives > 0 and "orgl_path" in rows.columns:
            originals = rows["orgl_path"].dropna().unique().tolist()
            n_extra = min(int(round(len(rows) * float(extra_negatives))), len(originals))
            chosen = np.random.default_rng(seed).choice(originals, n_extra, replace=False)
            self.records += [(str(path), None, True) for path in chosen]

        self.is_negative = np.array([record[2] for record in self.records], dtype=bool)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            worker_id = info.id if info else 0
            self._rng = np.random.default_rng([self.seed, worker_id])
        return self._rng

    def _path(self, rel_path: str, is_mask: bool) -> Path:
        if self.cache_size:
            cached = aic_data.cached_path(self.ws, rel_path, self.cache_size, is_mask)
            if cached.exists():
                return cached
        return self.ws.resolve(rel_path)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        img_rel, gt_rel, _is_negative = self.records[index]

        image = aic_data.imread(self._path(img_rel, False))
        if image is None:
            raise FileNotFoundError(f"cannot read image: {img_rel}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if gt_rel is None:
            mask = np.zeros(image.shape[:2], dtype=np.float32)
        else:
            raw = aic_data.imread(self._path(gt_rel, True), cv2.IMREAD_GRAYSCALE)
            if raw is None:
                raise FileNotFoundError(f"cannot read mask: {gt_rel}")
            mask = (raw >= GT_POSITIVE_VALUE).astype(np.float32)

        if self.train:
            image, mask = random_resized_crop(self.rng, image, mask, self.size, self.crop_scale)
            image, mask = random_flips(self.rng, image, mask)
            image = photometric(self.rng, image)
        else:
            image, mask = resize_pair(image, mask, self.size)

        return {
            "image": to_tensor(image),
            "mask": torch.from_numpy(mask).unsqueeze(0),
            "cls": torch.tensor([float(mask.max() > 0)], dtype=torch.float32),
        }
