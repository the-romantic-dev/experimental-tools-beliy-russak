"""Сабмит: механика регламента плюс раннер над своей функцией."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

# `aic` до торча — см. комментарий в tests/aic/test_budget.py
from aic.data import imread, imwrite
from aic.submit import (
    load_test_table,
    pack_zip,
    postprocess,
    predict_folder,
    to_original,
    validate,
    write,
)

torch = pytest.importorskip("torch", reason="раннер требует torch")


def _test_set(tmp_path, n=4, size=(10, 12)):
    """Папка теста с test.csv и картинками."""
    root = tmp_path / "test_stage1"
    (root / "img").mkdir(parents=True)
    rows = []
    for i in range(n):
        rel = f"img/кадр{i}.png"
        imwrite(root / rel, np.zeros((*size, 3), dtype=np.uint8))
        rows.append({"img_path": rel})
    pd.DataFrame(rows).to_csv(root / "test.csv", index=False)
    return root


def test_load_test_table_derives_prediction_paths(tmp_path):
    root = _test_set(tmp_path)
    table, got_root = load_test_table(root / "test.csv")
    assert got_root == root
    assert list(table.columns) == ["img_path", "prediction_path"]
    assert table["prediction_path"].iloc[0] == "predictions/кадр0_pred.png"


def test_load_test_table_prefers_the_organisers_template(tmp_path):
    """Правило именования задано организаторами — угадывать его не надо."""
    root = _test_set(tmp_path, n=2)
    pd.DataFrame({
        "img_path": ["img/кадр0.png", "img/кадр1.png"],
        "prediction_path": ["preds/нулевой.png", "preds/первый.png"],
    }).to_csv(root / "submission.csv", index=False)

    table, _ = load_test_table(root / "test.csv", root / "submission.csv")
    assert table["prediction_path"].tolist() == ["preds/нулевой.png", "preds/первый.png"]


def test_load_test_table_rejects_an_incomplete_template(tmp_path):
    root = _test_set(tmp_path, n=2)
    pd.DataFrame({
        "img_path": ["img/кадр0.png"], "prediction_path": ["preds/a.png"],
    }).to_csv(root / "submission.csv", index=False)
    with pytest.raises(ValueError, match="не покрывает"):
        load_test_table(root / "test.csv", root / "submission.csv")


def test_postprocess_makes_a_zero_or_255_mask():
    prob = np.array([[0.1, 0.9], [0.6, 0.2]])
    mask = postprocess(prob, 1.0, mask_threshold=0.5)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask.dtype == np.uint8
    assert mask[0, 1] == 255 and mask[0, 0] == 0


def test_postprocess_blanks_the_frame_below_the_cls_threshold():
    prob = np.ones((4, 4))
    assert postprocess(prob, 0.1, mask_threshold=0.5, cls_threshold=0.5).max() == 0


def test_postprocess_blanks_a_mask_under_min_area():
    prob = np.zeros((10, 10))
    prob[0, 0] = 1.0                      # 1% площади
    assert postprocess(prob, 1.0, mask_threshold=0.5, min_area=0.05).max() == 0
    assert postprocess(prob, 1.0, mask_threshold=0.5, min_area=0.005).max() == 255


def test_to_original_returns_the_input_resolution():
    prob = torch.zeros(1, 8, 8)
    assert to_original(prob, 13, 21).shape == (13, 21)


def test_to_original_crops_the_padding_first():
    """В режиме pad правая и нижняя части сетки — паддинг, а не картинка."""
    prob = torch.zeros(1, 8, 8)
    prob[:, :4, :] = 1.0
    out = to_original(prob, 40, 80, val_mode="pad")
    assert out.shape == (40, 80)
    assert out.mean() == pytest.approx(1.0, abs=1e-6)


def test_predict_folder_calls_your_function_and_returns_stems(tmp_path):
    root = _test_set(tmp_path, n=3, size=(10, 12))
    paths = sorted((root / "img").glob("*.png"))
    seen = []

    def predict_fn(batch):
        seen.append(tuple(batch.shape))
        return torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 0.8)

    items = list(predict_folder(paths, predict_fn, size=16, batch_size=2))
    assert [stem for stem, _, _ in items] == ["кадр0", "кадр1", "кадр2"]
    assert all(prob.shape == (10, 12) for _, prob, _ in items)
    assert seen and all(shape[1:] == (3, 16, 16) for shape in seen)


def test_predict_folder_averages_tta_views(tmp_path):
    root = _test_set(tmp_path, n=1)
    paths = sorted((root / "img").glob("*.png"))
    calls = []

    def predict_fn(batch):
        calls.append(1)
        return torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 0.5)

    list(predict_folder(paths, predict_fn, size=16, tta=("none", "hflip")))
    assert len(calls) == 2


def test_predict_folder_rejects_an_unknown_tta(tmp_path):
    root = _test_set(tmp_path, n=1)
    paths = sorted((root / "img").glob("*.png"))
    with pytest.raises(ValueError, match="неизвестные виды TTA"):
        list(predict_folder(paths, lambda b: b, size=16, tta=("вверхногами",)))


def test_predict_folder_accepts_a_cls_probability(tmp_path):
    root = _test_set(tmp_path, n=1)
    paths = sorted((root / "img").glob("*.png"))

    def predict_fn(batch):
        probs = torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 0.9)
        return probs, torch.full((batch.shape[0],), 0.25)

    (_, _, cls_prob), = list(predict_folder(paths, predict_fn, size=16))
    assert cls_prob == pytest.approx(0.25)


def test_write_and_validate_a_clean_submission(tmp_path):
    root = _test_set(tmp_path, n=3, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    items = [(f"кадр{i}", np.full((10, 12), 0.9), 1.0) for i in range(3)]

    out_dir = tmp_path / "сабмит"
    stats = write(out_dir, items, table=table, mask_threshold=0.5)
    assert stats["n_images"] == 3
    assert (out_dir / "submission.csv").exists()

    report = validate(out_dir, root / "test.csv")
    assert report["ok"] is True, report["problems"]
    assert report["n_checked"] == 3


def test_write_without_a_table_uses_the_default_naming(tmp_path):
    out_dir = tmp_path / "без-таблицы"
    write(out_dir, [("кадр0", np.zeros((4, 4)), 1.0)])
    assert (out_dir / "predictions" / "кадр0_pred.png").exists()
    assert not (out_dir / "submission.csv").exists()


def test_validate_catches_a_wrong_mask_size(tmp_path):
    root = _test_set(tmp_path, n=1, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "кривой"
    write(out_dir, [("кадр0", np.zeros((5, 5)), 1.0)], table=table)

    report = validate(out_dir, root / "test.csv")
    assert report["ok"] is False
    assert any("размер маски" in p for p in report["problems"])


def test_validate_catches_a_missing_frame(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "неполный"
    write(out_dir, [("кадр0", np.zeros((10, 12)), 1.0)], table=table.head(1))

    report = validate(out_dir, root / "test.csv")
    assert report["ok"] is False
    assert any("нет предсказаний" in p for p in report["problems"])


def test_validate_accepts_a_partial_submission_when_asked(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "частичный"
    write(out_dir, [("кадр0", np.zeros((10, 12)), 1.0)], table=table.head(1))
    assert validate(out_dir, root / "test.csv", partial=True)["ok"] is True


def test_validate_catches_values_other_than_0_and_255(tmp_path):
    root = _test_set(tmp_path, n=1, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "серый"
    write(out_dir, [("кадр0", np.zeros((10, 12)), 1.0)], table=table)
    imwrite(out_dir / table["prediction_path"].iloc[0],
            np.full((10, 12), 128, dtype=np.uint8))

    report = validate(out_dir, root / "test.csv")
    assert any("не только 0/255" in p for p in report["problems"])


def test_pack_zip_puts_the_csv_in_the_archive_root(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "архив"
    write(out_dir, [(f"кадр{i}", np.zeros((10, 12)), 1.0) for i in range(2)], table=table)

    zip_path = pack_zip(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "submission.csv" in names
    assert sum(n.endswith(".png") for n in names) == 2


def test_validate_reads_a_zip_the_same_way(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "зип"
    write(out_dir, [(f"кадр{i}", np.zeros((10, 12)), 1.0) for i in range(2)], table=table)
    assert validate(pack_zip(out_dir), root / "test.csv")["ok"] is True
