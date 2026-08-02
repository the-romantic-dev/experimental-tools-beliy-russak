"""Сборка и проверка сабмита — на синтетике, без модели и без тестовых данных.

Валидатор существует ради того, чтобы ловить ошибки формата до загрузки,
поэтому каждый вид поломки проверяется отдельно.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import zipfile

import cv2
import numpy as np
import pandas as pd
import pytest

from aic_pipeline.imageio import imread, imwrite
from aic_pipeline.submission import (
    build_submission,
    load_test_table,
    pack_zip,
    validate_submission,
)


def test_imageio_survives_non_ascii_paths(tmp_path):
    """cv2.imread/imwrite на Windows молча ломаются на кириллице в пути —
    именно поэтому весь файловый ввод-вывод идёт через модуль imageio."""
    directory = tmp_path / "путь с кириллицей"
    directory.mkdir()
    target = directory / "маска.png"

    mask = np.zeros((7, 5), np.uint8)
    mask[:3] = 255
    assert imwrite(target, mask) is True
    assert target.exists() and target.stat().st_size > 0

    restored = imread(target, cv2.IMREAD_GRAYSCALE)
    assert restored is not None
    assert np.array_equal(restored, mask)


def test_imread_returns_none_for_missing_file(tmp_path):
    assert imread(tmp_path / "нет-такого.png") is None


@pytest.fixture
def testset(tmp_path):
    """Мини-«тестовая выборка»: 3 картинки разного размера + test.csv + шаблон."""
    root = tmp_path / "test_stage1"
    (root / "test_stage1_img").mkdir(parents=True)
    sizes = {"a": (40, 60), "b": (30, 30), "c": (50, 20)}
    for stem, (h, w) in sizes.items():
        imwrite(root / "test_stage1_img" / f"{stem}.jpg",
                np.full((h, w, 3), 128, np.uint8))

    rows = [f"test_stage1_img/{s}.jpg" for s in sizes]
    pd.DataFrame({"img_path": rows}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({
        "img_path": rows,
        "prediction_path": [f"predictions/{s}_pred.png" for s in sizes],
    }).to_csv(root / "submission.csv", index=False)
    return root, sizes


def write_submission(out_dir, root, sizes, *, values=255, area_rows=None,
                     skip=(), three_channel=(), wrong_size=()):
    (out_dir / "predictions").mkdir(parents=True, exist_ok=True)
    rows = []
    for stem, (h, w) in sizes.items():
        rel = f"predictions/{stem}_pred.png"
        rows.append({"img_path": f"test_stage1_img/{stem}.jpg", "prediction_path": rel})
        if stem in skip:
            continue
        shape = (h + 5, w) if stem in wrong_size else (h, w)
        mask = np.zeros(shape, np.uint8)
        filled = area_rows if area_rows is not None else shape[0] // 2
        mask[:filled, :] = values
        if stem in three_channel:
            mask = np.stack([mask] * 3, axis=-1)
        imwrite(out_dir / rel, mask)
    pd.DataFrame(rows).to_csv(out_dir / "submission.csv", index=False)
    return out_dir


@pytest.fixture
def checkpoint(tmp_path):
    """Настоящий чекпоинт самой дешёвой модели репозитория, со случайными весами.

    Качество здесь ни при чём: проверяется механика сборки, а веса нужны только
    чтобы `load_checkpoint` отработал как на настоящем прогоне.
    """
    import torch

    from aic_pipeline.config import load_config
    from aic_pipeline.models import build_model

    made: list[str] = []

    def make(*overrides: str):
        cfg = load_config("smoke", ["model.encoder_weights=null", *overrides])
        run_dir = tmp_path / f"прогон{len(made)}"
        made.append(run_dir.name)
        (run_dir / "ckpt").mkdir(parents=True)
        torch.save(
            {"cfg": dict(cfg), "model": build_model(cfg.model).state_dict()},
            run_dir / "ckpt" / "best.pt",
        )
        return run_dir

    return make


def build(run_dirs, out_dir, testset_root, **kwargs):
    """Сборка на CPU: тесты не должны занимать карту."""
    return build_submission(
        run_dirs, out_dir,
        test_csv=testset_root / "test.csv",
        template=testset_root / "submission.csv",
        device="cpu", num_workers=0, progress=lambda *_: None,
        **kwargs,
    )


def test_build_submission_refuses_a_recipe_over_the_flops_budget(testset, tmp_path, checkpoint):
    """Посылка вне лимита — ноль за весь этап, собирать её нельзя ни при каких пометках."""
    root, _ = testset
    run_dir = checkpoint("data.size=1024", "budget.exempt=true")

    with pytest.raises(ValueError, match="GFLOPs"):
        build([run_dir], tmp_path / "out", root)
    assert not (tmp_path / "out" / "submission.csv").exists()


def test_build_submission_counts_tta_views_against_the_budget(testset, tmp_path, checkpoint):
    """Сам по себе конфиг влезает, а четыре вида TTA — уже нет."""
    root, _ = testset
    run_dir = checkpoint("data.size=512")

    build([run_dir], tmp_path / "ok", root)
    with pytest.raises(ValueError, match="GFLOPs"):
        build([run_dir], tmp_path / "out", root, tta=("none", "hflip", "vflip", "hvflip"))


def test_build_submission_counts_the_whole_ensemble(testset, tmp_path, checkpoint):
    """Каждый прогон по отдельности в бюджете, а ансамбль из них — уже нет."""
    root, _ = testset
    first, second = checkpoint("data.size=576"), checkpoint("data.size=576")

    build([first], tmp_path / "ok", root)
    with pytest.raises(ValueError, match="GFLOPs"):
        build([first, second], tmp_path / "out", root)


def test_build_submission_reports_milliseconds_per_image(testset, tmp_path, checkpoint):
    """Лимит 50 ms на кадр — число должно появляться само, а не по отдельной просьбе."""
    root, _ = testset
    stats = build([checkpoint()], tmp_path / "out", root)

    assert stats["ms_per_image"] > 0
    assert stats["n_images"] == 3


def test_load_test_table_reuses_template_naming(testset):
    root, _ = testset
    table, resolved_root = load_test_table(root / "test.csv", root / "submission.csv")
    assert list(table.columns) == ["img_path", "prediction_path"]
    assert resolved_root == root
    assert table["prediction_path"].iloc[0] == "predictions/a_pred.png"


def test_load_test_table_falls_back_to_default_naming(testset):
    root, _ = testset
    table, _ = load_test_table(root / "test.csv", template=None)
    assert table["prediction_path"].tolist() == [
        "predictions/a_pred.png", "predictions/b_pred.png", "predictions/c_pred.png"
    ]


def test_valid_submission_passes(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes)
    report = validate_submission(out, test_csv=root / "test.csv")
    assert report["ok"], report["problems"]
    assert report["n_rows"] == 3 and report["n_checked"] == 3


def test_detects_missing_mask_file(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes, skip=("b",))
    report = validate_submission(out, test_csv=root / "test.csv")
    assert not report["ok"]
    assert any("не найден" in p for p in report["problems"])


def test_detects_wrong_mask_size(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes, wrong_size=("c",))
    report = validate_submission(out, test_csv=root / "test.csv")
    assert not report["ok"]
    assert any("размер маски" in p for p in report["problems"])


def test_detects_non_binary_values(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes, values=1)
    report = validate_submission(out, test_csv=root / "test.csv")
    assert not report["ok"]
    assert any("0/255" in p for p in report["problems"])


def test_detects_three_channel_mask(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes, three_channel=("a",))
    report = validate_submission(out, test_csv=root / "test.csv")
    assert not report["ok"]
    assert any("одноканальная" in p for p in report["problems"])


def test_detects_missing_rows(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes)
    table = pd.read_csv(out / "submission.csv").head(2)
    table.to_csv(out / "submission.csv", index=False)
    report = validate_submission(out, test_csv=root / "test.csv")
    assert not report["ok"]
    assert any("нет предсказаний" in p for p in report["problems"])


def test_reports_area_statistics(testset, tmp_path):
    """empty_share и flagged_share — ранний сигнал по FPR_neg."""
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes, area_rows=0)
    report = validate_submission(out, test_csv=root / "test.csv")
    assert report["ok"], report["problems"]
    assert report["empty_share"] == 1.0
    assert report["flagged_share"] == 0.0


def test_zip_has_flat_structure_and_validates(testset, tmp_path):
    root, sizes = testset
    out = write_submission(tmp_path / "sub", root, sizes)
    archive = pack_zip(out)

    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
    assert "submission.csv" in names
    assert all(n == "submission.csv" or n.startswith("predictions/") for n in names)

    report = validate_submission(archive, test_csv=root / "test.csv")
    assert report["ok"], report["problems"]


def test_missing_csv_in_archive_is_reported(tmp_path, testset):
    root, _ = testset
    broken = tmp_path / "broken.zip"
    with zipfile.ZipFile(broken, "w") as zf:
        zf.writestr("predictions/a_pred.png", b"not a png")
    report = validate_submission(broken, test_csv=root / "test.csv")
    assert not report["ok"]
    assert "submission.csv" in report["problems"][0]
