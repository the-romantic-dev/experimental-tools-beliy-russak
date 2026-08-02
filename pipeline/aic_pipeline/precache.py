"""Ресайз-кэш датасета для быстрых эпох.

103.7k изображений в исходном разрешении упираются в диск и в JPEG-декод.
Кэш один раз ужимает всё до `max_side` и складывает рядом, сохраняя структуру
путей — так что подмена источника сводится к смене корня.

Важная деталь для forensics-задачи: файлы, которые и так меньше `max_side`,
копируются БЕЗ перекодирования. Повторное JPEG-сжатие затирает ровно те
артефакты, по которым модель ловит подделку. По той же причине качество
по умолчанию высокое (95), а маски всегда PNG (без потерь).
"""

from __future__ import annotations

import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

from .workspace import resolve, workspace


def cache_root(max_side: int) -> Path:
    return workspace().cache / f"s{max_side}"


def cached_path(rel_path: str, max_side: int, is_mask: bool) -> Path:
    rel = Path(str(rel_path).replace("\\", "/"))
    if is_mask:
        rel = rel.with_suffix(".png")
    return cache_root(max_side) / rel


def _process_one(args: tuple[str, int, bool, int]) -> tuple[str, str]:
    """Возвращает (rel_path, status) где status = written|copied|skipped|error."""
    import cv2

    from .imageio import imread, imwrite

    rel_path, max_side, is_mask, quality = args
    src = resolve(rel_path)
    dst = cached_path(rel_path, max_side, is_mask)
    if dst.exists():
        return rel_path, "skipped"
    dst.parent.mkdir(parents=True, exist_ok=True)

    flag = cv2.IMREAD_GRAYSCALE if is_mask else cv2.IMREAD_COLOR
    image = imread(src, flag)
    if image is None:
        return rel_path, "error"

    h, w = image.shape[:2]
    scale = max_side / float(max(h, w))
    if scale >= 1.0:
        if is_mask and src.suffix.lower() == ".png":
            shutil.copyfile(src, dst)
            return rel_path, "copied"
        if not is_mask and src.suffix.lower() == dst.suffix.lower():
            shutil.copyfile(src, dst)
            return rel_path, "copied"

    if scale < 1.0:
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    if is_mask:
        imwrite(dst, image, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        imwrite(dst, image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return rel_path, "written"


def build_cache(
    df: pd.DataFrame,
    max_side: int = 768,
    workers: int = 12,
    quality: int = 95,
    include_originals: bool = True,
) -> dict[str, int]:
    """Прогоняет через кэш все пути из индекса (изображения, маски, оригиналы)."""
    tasks: list[tuple[str, int, bool, int]] = []
    seen: set[tuple[str, bool]] = set()

    def add(rel_path, is_mask: bool) -> None:
        if not isinstance(rel_path, str) or not rel_path:
            return
        key = (rel_path, is_mask)
        if key in seen:
            return
        seen.add(key)
        tasks.append((rel_path, max_side, is_mask, quality))

    for rel in df["chng_path"]:
        add(rel, False)
    for rel in df["gt_path"]:
        add(rel, True)
    if include_originals and "orgl_path" in df.columns:
        for rel in df["orgl_path"]:
            add(rel, False)

    stats = {"written": 0, "copied": 0, "skipped": 0, "error": 0}
    chunk = max(64, len(tasks) // (workers * 16) or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = 0
        for _, status in pool.map(_process_one, tasks, chunksize=chunk):
            stats[status] += 1
            done += 1
            if done % 10000 == 0:
                print(f"  {done}/{len(tasks)}  {stats}", flush=True)
    return stats


def cache_size_gb(max_side: int) -> float:
    root = cache_root(max_side)
    if not root.exists():
        return 0.0
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return total / 1e9
