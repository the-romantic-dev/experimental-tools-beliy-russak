"""Фолды без утечки и предкэш."""

from __future__ import annotations

import numpy as np
import pandas as pd

from aic.data import (
    area_bucket,
    build_cache,
    cache_size_gb,
    cached_path,
    check_leakage,
    imread,
    imwrite,
    load_folds,
    make_folds,
    summarize_folds,
)
from aic.paths import Workspace


def _index(n_groups=20, per_group=3):
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            rows.append({
                "stem": f"кадр_{g}_{k}",
                "group_id": f"группа{g}",
                "domain": "coco" if g % 2 else "raise",
                "generator": "lama" if k else "none",
                "mask_area": 0.0 if k == 0 else 0.02 * (k + 1),
                "is_negative": k == 0,
                "broken": False,
                "chng_path": f"img/{g}_{k}.png",
                "gt_path": f"gt/{g}_{k}.png",
                "orgl_path": None,
            })
    return pd.DataFrame(rows)


def test_area_bucket_is_monotonic():
    assert area_bucket(0.0) <= area_bucket(0.02) <= area_bucket(0.5)


def test_every_row_lands_in_exactly_one_fold():
    folds = make_folds(_index(), n_folds=5, seed=0)
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
    assert (folds["fold"] >= 0).all()
    assert len(folds) == 60


def test_a_group_never_splits_across_folds():
    """Главное свойство: иначе модель видит тот же исходный кадр в валидации."""
    folds = make_folds(_index(), n_folds=5, seed=0)
    per_group = folds.groupby("group_id")["fold"].nunique()
    assert (per_group == 1).all()
    assert check_leakage(folds)["leaked_groups"] == 0


def test_folds_are_reproducible_for_a_seed():
    a = make_folds(_index(), n_folds=5, seed=7)["fold"].tolist()
    b = make_folds(_index(), n_folds=5, seed=7)["fold"].tolist()
    assert a == b


def test_broken_rows_are_dropped():
    df = _index()
    df.loc[0, "broken"] = True
    assert len(make_folds(df, n_folds=5, seed=0)) == len(df) - 1


def test_make_folds_writes_only_when_asked(tmp_path):
    df = _index()
    make_folds(df, n_folds=5, seed=0)
    assert not list(tmp_path.iterdir())

    out = tmp_path / "folds.parquet"
    make_folds(df, n_folds=5, seed=0, out_path=out)
    assert len(load_folds(out)) == 60


def test_summarize_folds_mentions_the_split():
    text = summarize_folds(make_folds(_index(), n_folds=5, seed=0))
    assert "fold" in text.lower()


def test_cached_path_mirrors_the_relative_layout(tmp_path):
    ws = Workspace(tmp_path)
    got = cached_path(ws, "stage1/train/img/a.jpg", 768, is_mask=False)
    assert got == ws.cache / "s768" / "stage1" / "train" / "img" / "a.jpg"


def test_masks_are_cached_as_png(tmp_path):
    """JPEG на маске дал бы значения между 0 и 255 — GT перестал бы быть GT."""
    ws = Workspace(tmp_path)
    got = cached_path(ws, "stage1/train/gt/a.jpg", 768, is_mask=True)
    assert got.suffix == ".png"


def test_build_cache_shrinks_the_long_side(tmp_path):
    ws = Workspace(tmp_path)
    src = ws.dataset_root / "img"
    src.mkdir(parents=True)
    imwrite(src / "большая.png", np.zeros((400, 200, 3), dtype=np.uint8))
    gt = ws.dataset_root / "gt"
    gt.mkdir(parents=True)
    imwrite(gt / "большая.png", np.zeros((400, 200), dtype=np.uint8))

    df = pd.DataFrame([{"chng_path": "img/большая.png", "gt_path": "gt/большая.png",
                        "orgl_path": None}])
    stats = build_cache(ws, df, max_side=100, workers=1)
    assert stats["error"] == 0

    cached = imread(cached_path(ws, "img/большая.png", 100, is_mask=False))
    assert cached is not None
    assert max(cached.shape[:2]) == 100


def test_build_cache_skips_what_is_already_there(tmp_path):
    ws = Workspace(tmp_path)
    (ws.dataset_root / "img").mkdir(parents=True)
    imwrite(ws.dataset_root / "img" / "кадр.png", np.zeros((50, 50, 3), dtype=np.uint8))
    (ws.dataset_root / "gt").mkdir(parents=True)
    imwrite(ws.dataset_root / "gt" / "кадр.png", np.zeros((50, 50), dtype=np.uint8))

    df = pd.DataFrame([{"chng_path": "img/кадр.png", "gt_path": "gt/кадр.png",
                        "orgl_path": None}])
    build_cache(ws, df, max_side=32, workers=1)
    again = build_cache(ws, df, max_side=32, workers=1)
    assert again["skipped"] == 2


def test_cache_size_of_an_absent_cache_is_zero(tmp_path):
    assert cache_size_gb(Workspace(tmp_path), 768) == 0.0
