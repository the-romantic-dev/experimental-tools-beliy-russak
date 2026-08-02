"""Данные задачи: чтение, разбор имён, индекс, фолды, предкэш.

Специфично для датасета AI Challenge, но ничего не решает за вас в обучении:
здесь только то, из чего каждый строит свой Dataset.

Пути берутся из переданного `Workspace`, а не из глобального состояния — иначе
порядок импортов начинал бы влиять на то, какие файлы прочитаются.
"""

from __future__ import annotations

import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .paths import Workspace



# --------------------------------------------------------------------------
# чтение и запись
# --------------------------------------------------------------------------

def imread(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Аналог cv2.imread. Возвращает None, если файла нет или он не декодируется."""
    path = Path(path)
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except (OSError, FileNotFoundError):
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def imwrite(path: str | Path, image: np.ndarray, params: Sequence[int] | None = None) -> bool:
    """Аналог cv2.imwrite. Формат определяется расширением пути."""
    path = Path(path)
    ok, buffer = cv2.imencode(path.suffix, image, list(params or []))
    if not ok:
        return False
    buffer.tofile(str(path))
    return True


# --------------------------------------------------------------------------
# индекс датасета
# --------------------------------------------------------------------------

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


def _probe(args: tuple[int, str, str, str]) -> tuple[int, int, int, float, int, int]:
    """Читает маску и заголовок изображения. Выполняется в отдельном процессе.

    Корень датасета приезжает аргументом, а не берётся из глобального состояния:
    дочерний процесс стартует заново, и никакого общего объекта в нём бы не было.
    """
    import cv2
    from PIL import Image

    row_id, gt_rel, img_rel, dataset_root = args
    root = Path(dataset_root)
    mask = imread(root / str(gt_rel).replace("\\", "/"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return row_id, 0, 0, -1.0, 0, 0
    height, width = mask.shape[:2]
    area = float(np.count_nonzero(mask >= GT_POSITIVE_VALUE)) / float(mask.size)

    try:
        with Image.open(root / str(img_rel).replace("\\", "/")) as im:
            img_w, img_h = im.size
    except Exception:
        img_w = img_h = 0
    return row_id, height, width, area, img_h, img_w


def build_index(
    ws: Workspace,
    *,
    csv_path: str | Path | None = None,
    out_path: str | Path | None = None,
    workers: int = 12,
    limit: int | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path) if csv_path is not None else ws.train_csv
    out_path = Path(out_path) if out_path is not None else ws.index_path
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

    root = str(ws.dataset_root)
    tasks = [
        (i, gt, img, root)
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


def load_index(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"нет индекса {path}. Собери его: aic.data.build_index(ws)"
        )
    return pd.read_parquet(path)


def summarize_index(df: pd.DataFrame) -> str:
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


# --------------------------------------------------------------------------
# фолды без утечки
# --------------------------------------------------------------------------

AREA_BINS = (0.0, 0.01, 0.03, 0.08, 0.20, 1.01)


def area_bucket(area: float) -> int:
    return int(np.digitize(area, AREA_BINS[1:-1]))


def make_strata(df: pd.DataFrame) -> pd.Series:
    buckets = df["mask_area"].map(area_bucket)
    return (
        df["domain"].astype(str)
        + "|" + df["generator"].astype(str)
        + "|" + df["is_negative"].astype(int).astype(str)
        + "|" + buckets.astype(str)
    )


def make_folds(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Групповой стратифицированный сплит по `group_id`.

    Групповой обязательно: один оригинал порождает несколько манипуляций, и,
    разъехавшись по фолдам, они дают утечку — модель видит тот же кадр.

    Индекс принимается таблицей, а не читается с диска: тот, кто зовёт, её и
    так держит в руках.
    """
    df = df[~df["broken"]].reset_index(drop=True)

    strata = make_strata(df)
    # редкие страты схлопываем, иначе StratifiedGroupKFold ругается
    counts = strata.value_counts()
    rare = set(counts[counts < n_folds].index)
    strata = strata.map(lambda s: "rare" if s in rare else s)

    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold = np.full(len(df), -1, dtype=np.int8)
    for fold_id, (_, val_idx) in enumerate(
        splitter.split(df, y=strata, groups=df["group_id"])
    ):
        fold[val_idx] = fold_id

    df["fold"] = fold
    if (df["fold"] < 0).any():
        raise RuntimeError("часть строк не попала ни в один фолд")

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df


def load_folds(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"нет сплита {path}. Собери его: aic.data.make_folds(df, out_path=...)"
        )
    return pd.read_parquet(path)


def check_leakage(df: pd.DataFrame) -> dict:
    """Ни одна group_id не должна встречаться в двух фолдах."""
    per_group = df.groupby("group_id")["fold"].nunique()
    leaked = per_group[per_group > 1]
    return {"leaked_groups": int(len(leaked)), "examples": leaked.index[:5].tolist()}


def summarize_folds(df: pd.DataFrame) -> str:
    table = df.groupby("fold").agg(
        n=("stem", "size"),
        neg=("is_negative", "sum"),
        neg_frac=("is_negative", "mean"),
        groups=("group_id", "nunique"),
        area_med=("mask_area", "median"),
    )
    leak = check_leakage(df)
    return "\n".join(
        [
            table.to_string(),
            "",
            f"утечка групп между фолдами: {leak['leaked_groups']}"
            + (f" (например {leak['examples']})" if leak["leaked_groups"] else " — чисто"),
            "",
            "домены по фолдам:",
            pd.crosstab(df["fold"], df["domain"]).to_string(),
        ]
    )


# --------------------------------------------------------------------------
# предкэш
# --------------------------------------------------------------------------

def cache_root(ws: Workspace, max_side: int) -> Path:
    return ws.cache / f"s{max_side}"


def cached_path(ws: Workspace, rel_path: str, max_side: int, is_mask: bool) -> Path:
    """Маски всегда PNG: JPEG дал бы значения между 0 и 255, и GT перестал бы
    быть GT."""
    rel = Path(str(rel_path).replace("\\", "/"))
    if is_mask:
        rel = rel.with_suffix(".png")
    return cache_root(ws, max_side) / rel


def _process_one(args: tuple[str, int, bool, int, str, str]) -> tuple[str, str]:
    """Возвращает (rel_path, status) где status = written|copied|skipped|error.

    Корни приезжают аргументом по той же причине, что и в `_probe`: дочерний
    процесс глобального состояния родителя не наследует.
    """
    import cv2

    rel_path, max_side, is_mask, quality, dataset_root, cache_dir = args
    rel = Path(str(rel_path).replace("\\", "/"))
    src = Path(dataset_root) / rel
    dst = Path(cache_dir) / (rel.with_suffix(".png") if is_mask else rel)
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
    ws: Workspace,
    df: pd.DataFrame,
    *,
    max_side: int = 768,
    workers: int = 12,
    quality: int = 95,
    include_originals: bool = True,
) -> dict[str, int]:
    """Прогоняет через кэш все пути из индекса (изображения, маски, оригиналы)."""
    tasks: list[tuple[str, int, bool, int, str, str]] = []
    seen: set[tuple[str, bool]] = set()
    roots = (str(ws.dataset_root), str(cache_root(ws, max_side)))

    def add(rel_path, is_mask: bool) -> None:
        if not isinstance(rel_path, str) or not rel_path:
            return
        key = (rel_path, is_mask)
        if key in seen:
            return
        seen.add(key)
        tasks.append((rel_path, max_side, is_mask, quality, *roots))

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


def cache_size_gb(ws: Workspace, max_side: int) -> float:
    root = cache_root(ws, max_side)
    if not root.exists():
        return 0.0
    total = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    return total / 1e9
