"""Построение индекса датасета: train.csv -> artifacts/index.parquet.

Из train.csv напрямую нельзя ни разбить данные без утечки, ни понять, где
негативы. Поэтому один раз проходим по всем маскам и собираем таблицу:

    chng_path, gt_path, orgl_path, domain, generator, group_id,
    height, width, mask_area, is_negative, img_h, img_w, size_mismatch

Ключевое:
* `group_id` — идентификатор ИСХОДНОГО кадра. Один оригинал порождает до
  нескольких манипуляций, поэтому сплит обязан быть групповым по этому полю.
* `is_negative` — в GT нет изменений. Такие кадры формируют FPR_neg в метрике.
"""

from __future__ import annotations

import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from .workspace import index_path, resolve, train_csv

HASH_PREFIX = re.compile(r"^[0-9a-f]{12}_")
# префикс источника есть в имени изменённого файла, но не в имени оригинала:
# "openimages_bd3895b63372e1f2_removeanything..." против "bd3895b63372e1f2.jpg".
# Без снятия префикса один и тот же кадр получал два разных group_id.
DATASET_PREFIX = re.compile(r"^(coco|raise|openimages)_")

# токены, после которых в имени файла начинается описание манипуляции
METHOD_TOKENS = (
    "powerpaint", "brushnet", "hdpainter", "removeanything",
    "inpaintanything", "inpainted", "lama", "sd2",
)
SIZE_TOKENS = ("small", "medium", "large")
CUT_TOKENS = set(METHOD_TOKENS) | set(SIZE_TOKENS)

GT_POSITIVE_VALUE = 128  # маски мягкие (0..255), бинаризуем GT по этому порогу


def strip_hash(name: str) -> str:
    return HASH_PREFIX.sub("", name)


def stem_of(path: str) -> str:
    return strip_hash(Path(str(path)).name).rsplit(".", 1)[0]


def parse_domain(stem: str) -> str:
    if stem.startswith("coco_"):
        return "coco"
    if stem.startswith("raise_"):
        return "raise"
    if stem.startswith("openimages_"):
        return "openimages"
    if re.match(r"^D\d{2}_", stem):
        return "vision"
    if re.match(r"^\d+$", stem):
        return "plain"
    if re.match(r"^\d+_", stem):
        return "numid"
    return "other"


def parse_generator(stem: str) -> str:
    tokens = stem.split("_")
    for token in tokens:
        base = token.split("-")[0]
        if base in METHOD_TOKENS:
            return base
    return "none"


def parse_group_id(chng_stem: str, orgl_path: str | float | None) -> str:
    """Идентификатор исходного кадра — общий для всех его манипуляций."""
    if isinstance(orgl_path, str) and orgl_path:
        return DATASET_PREFIX.sub("", stem_of(orgl_path))

    tokens = chng_stem.split("_")
    cut = len(tokens)
    for i, token in enumerate(tokens):
        if token.split("-")[0] in CUT_TOKENS:
            cut = i
            break
    base = "_".join(tokens[:cut]) or chng_stem
    return DATASET_PREFIX.sub("", base)


def _probe(args: tuple[int, str, str]) -> tuple[int, int, int, float, int, int]:
    """Читает маску и заголовок изображения. Выполняется в отдельном процессе."""
    import cv2
    from PIL import Image

    from .imageio import imread

    row_id, gt_rel, img_rel = args
    mask = imread(resolve(gt_rel), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return row_id, 0, 0, -1.0, 0, 0
    height, width = mask.shape[:2]
    area = float(np.count_nonzero(mask >= GT_POSITIVE_VALUE)) / float(mask.size)

    try:
        with Image.open(resolve(img_rel)) as im:
            img_w, img_h = im.size
    except Exception:
        img_w = img_h = 0
    return row_id, height, width, area, img_h, img_w


def build_index(
    csv_path: Path | None = None,
    out_path: Path | None = None,
    workers: int = 12,
    limit: int | None = None,
) -> pd.DataFrame:
    # пути берутся из воркспейса при вызове, а не при импорте модуля: иначе
    # set_workspace() уже ничего не изменит
    csv_path = Path(csv_path) if csv_path is not None else train_csv()
    out_path = Path(out_path) if out_path is not None else index_path()
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit).copy()

    df = df.rename(
        columns={
            "orgl_img_path": "orgl_path",
            "chng_img_path": "chng_path",
            "gt_path": "gt_path",
        }
    )
    df["orgl_path"] = df["orgl_path"].astype("object").where(df["orgl_path"].notna(), None)

    df["stem"] = df["chng_path"].map(stem_of)
    df["domain"] = df["stem"].map(parse_domain)
    df["generator"] = df["stem"].map(parse_generator)
    df["group_id"] = [
        parse_group_id(stem, orgl) for stem, orgl in zip(df["stem"], df["orgl_path"])
    ]

    tasks = [
        (i, gt, img)
        for i, (gt, img) in enumerate(zip(df["gt_path"], df["chng_path"]))
    ]
    heights = np.zeros(len(df), dtype=np.int32)
    widths = np.zeros(len(df), dtype=np.int32)
    areas = np.zeros(len(df), dtype=np.float32)
    img_hs = np.zeros(len(df), dtype=np.int32)
    img_ws = np.zeros(len(df), dtype=np.int32)

    chunk = max(64, len(tasks) // (workers * 16) or 1)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = 0
        for row_id, h, w, area, ih, iw in pool.map(_probe, tasks, chunksize=chunk):
            heights[row_id], widths[row_id], areas[row_id] = h, w, area
            img_hs[row_id], img_ws[row_id] = ih, iw
            done += 1
            if done % 5000 == 0:
                print(f"  просканировано {done}/{len(tasks)}", flush=True)

    df["height"], df["width"] = heights, widths
    df["mask_area"] = areas
    df["img_h"], df["img_w"] = img_hs, img_ws
    df["size_mismatch"] = (df["height"] != df["img_h"]) | (df["width"] != df["img_w"])
    # ровно ноль, а не `<= 0`: у нечитаемой маски площадь -1, и она не негатив,
    # а битая строка — иначе она попадала бы в счётчик негативов отчёта
    df["is_negative"] = df["mask_area"] == 0.0
    df["broken"] = df["mask_area"] < 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def load_index(path: Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else index_path()
    if not Path(path).exists():
        raise FileNotFoundError(
            f"нет индекса {path}. Собери его: python cli.py index"
        )
    return pd.read_parquet(path)


def summarize(df: pd.DataFrame) -> str:
    lines = [
        f"строк: {len(df)}",
        f"уникальных group_id: {df['group_id'].nunique()}",
        f"негативов (пустой GT): {int(df['is_negative'].sum())} "
        f"({df['is_negative'].mean() * 100:.1f}%)",
        f"битых масок: {int(df['broken'].sum())}",
        f"маска != изображение по размеру: {int(df['size_mismatch'].sum())}",
        "",
        "по доменам:",
        df.groupby("domain").agg(
            n=("stem", "size"),
            neg=("is_negative", "sum"),
            area_med=("mask_area", "median"),
            groups=("group_id", "nunique"),
        ).to_string(),
        "",
        "по генераторам:",
        df.groupby("generator").agg(
            n=("stem", "size"),
            neg=("is_negative", "sum"),
            area_med=("mask_area", "median"),
        ).to_string(),
    ]
    return "\n".join(lines)
