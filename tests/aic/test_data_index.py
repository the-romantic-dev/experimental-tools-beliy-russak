"""Разбор имён датасета и сборка индекса."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from aic.data import (
    build_index,
    imread,
    imwrite,
    load_index,
    parse_domain,
    parse_generator,
    parse_group_id,
    stem_of,
    strip_hash,
    summarize_index,
)
from aic.paths import Workspace


def test_strip_hash_removes_only_a_real_hash_prefix():
    assert strip_hash("bd3895b63372_кадр.jpg") == "кадр.jpg"
    assert strip_hash("кадр.jpg") == "кадр.jpg"


def test_stem_of_drops_dir_hash_and_suffix():
    assert stem_of("stage1/train/img/bd3895b63372_coco_1234.jpg") == "coco_1234"


@pytest.mark.parametrize(
    "stem,domain",
    [
        ("coco_1234", "coco"),
        ("raise_1234", "raise"),
        ("openimages_abc", "openimages"),
        ("D01_something", "vision"),
        ("4242", "plain"),
        ("42_something", "numid"),
        ("что-то-своё", "other"),
    ],
)
def test_parse_domain(stem, domain):
    assert parse_domain(stem) == domain


def test_parse_generator_finds_the_method_token():
    assert parse_generator("coco_123_powerpaint-v2_large") == "powerpaint"
    assert parse_generator("coco_123") == "none"


def test_group_id_comes_from_the_original_when_it_is_known():
    """Префикс источника есть у изменённого файла и нет у оригинала."""
    assert parse_group_id("openimages_bd3895_removeanything", "bd3895.jpg") == "bd3895"


def test_group_id_is_derived_by_cutting_the_method_token():
    assert parse_group_id("coco_1234_powerpaint_large", None) == "1234"


def test_group_id_is_shared_by_manipulations_of_one_frame():
    """На этом стоит групповой сплит: разъедется — будет утечка."""
    a = parse_group_id("coco_1234_powerpaint_large", None)
    b = parse_group_id("coco_1234_lama_small", None)
    assert a == b


def test_imwrite_and_imread_roundtrip_a_cyrillic_path(tmp_path):
    """cv2.imwrite на путях с кириллицей молча возвращает False — обёртка нужна."""
    path = tmp_path / "папка с пробелом" / "маска.png"
    path.parent.mkdir(parents=True)
    image = np.zeros((4, 6), dtype=np.uint8)
    image[0] = 255
    assert imwrite(path, image) is True
    back = imread(path, cv2.IMREAD_GRAYSCALE)
    assert back is not None
    assert np.array_equal(back, image)


def test_imread_of_a_missing_file_is_none(tmp_path):
    assert imread(tmp_path / "нет.png") is None


def _dataset(tmp_path):
    """Крошечный датасет: два позитива одного оригинала и один негатив."""
    ws = Workspace(tmp_path)
    stage = ws.dataset_root / "stage1"
    (stage / "train" / "img").mkdir(parents=True)
    (stage / "train" / "gt").mkdir(parents=True)

    rows = []
    for name, filled in [("coco_1_powerpaint", 3), ("coco_1_lama", 5), ("coco_2", 0)]:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[:filled] = 255
        imwrite(stage / "train" / "img" / f"{name}.png", img)
        imwrite(stage / "train" / "gt" / f"{name}.png", gt)
        rows.append({
            "chng_img_path": f"stage1/train/img/{name}.png",
            "gt_path": f"stage1/train/gt/{name}.png",
            "orgl_img_path": None,
        })
    ws.train_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ws.train_csv, index=False)
    return ws


def test_build_index_collects_sizes_areas_and_groups(tmp_path):
    ws = _dataset(tmp_path)
    df = build_index(ws, workers=1)

    assert len(df) == 3
    assert set(df["stem"]) == {"coco_1_powerpaint", "coco_1_lama", "coco_2"}
    assert df["height"].tolist() == [10, 10, 10]
    assert int(df["is_negative"].sum()) == 1
    assert not df["broken"].any()
    assert df[df["stem"] != "coco_2"]["group_id"].nunique() == 1


def test_build_index_writes_and_reads_back(tmp_path):
    ws = _dataset(tmp_path)
    build_index(ws, workers=1)
    assert ws.index_path.exists()
    assert len(load_index(ws.index_path)) == 3


def test_load_index_of_a_missing_file_tells_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path / "нет.parquet")


def test_build_index_marks_a_broken_mask(tmp_path):
    ws = _dataset(tmp_path)
    (ws.dataset_root / "stage1" / "train" / "gt" / "coco_2.png").write_bytes(b"not an image")
    df = build_index(ws, workers=1)
    assert int(df["broken"].sum()) == 1
    # битая строка не считается негативом: у неё площадь -1, а не 0
    assert int(df["is_negative"].sum()) == 0


def test_summarize_index_mentions_the_counts(tmp_path):
    ws = _dataset(tmp_path)
    text = summarize_index(build_index(ws, workers=1))
    assert "строк: 3" in text
