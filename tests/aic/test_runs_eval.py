"""Eval, операционная точка и чекпоинты."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aic.metric import AICAccumulator
from aic.runs import Eval, Run

torch = pytest.importorskip("torch", reason="чекпоинты требуют torch")


def _acc(n=4, seed=0):
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((n, 6, 6)), (rng.random((n, 6, 6)) > 0.6).astype(np.float32))
    return acc


def _rows(n=4):
    return pd.DataFrame({"stem": [f"кадр{i}" for i in range(n)], "fold": [0] * n})


def test_eval_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="кадров"):
        Eval(_acc(4), np.array(["a", "b"]))


def test_eval_rejects_duplicate_stems():
    with pytest.raises(ValueError, match="повторя"):
        Eval(_acc(2), np.array(["a", "a"]))


def test_eval_best_delegates_to_the_accumulator():
    acc = _acc(5)
    ev = Eval(acc, np.array([f"к{i}" for i in range(5)]))
    assert ev.best([0.3, 0.5]).aic == acc.best([0.3, 0.5]).aic
    assert len(ev) == 5


def test_save_and_load_eval_roundtrip(tmp_path):
    run = Run.create(tmp_path, "оценка", tensorboard=False)
    acc, rows = _acc(4), _rows(4)
    run.save_eval(acc, rows)
    run.close()

    assert (run.dir / "oof" / "val.npz").exists()
    assert (run.dir / "oof" / "val_rows.parquet").exists()

    ev = Run.open(run.dir).load_eval()
    assert len(ev) == 4
    assert ev.stems.tolist() == rows["stem"].tolist()
    assert ev.best([0.5]).aic == pytest.approx(acc.best([0.5]).aic)


def test_save_eval_under_another_name(tmp_path):
    run = Run.create(tmp_path, "полное", tensorboard=False)
    run.save_eval(_acc(3), _rows(3), name="val_fullres")
    run.close()
    assert (run.dir / "oof" / "val_fullres.npz").exists()
    assert len(Run.open(run.dir).load_eval("val_fullres")) == 3


def test_save_eval_refuses_mismatched_rows(tmp_path):
    """Иначе stem'ы молча разъехались бы с кадрами и сравнение мерило бы не то."""
    run = Run.create(tmp_path, "рассинхрон", tensorboard=False)
    with pytest.raises(ValueError, match="кадров"):
        run.save_eval(_acc(4), _rows(3))
    run.close()


def test_save_eval_requires_a_stem_column(tmp_path):
    run = Run.create(tmp_path, "без-stem", tensorboard=False)
    with pytest.raises(ValueError, match="stem"):
        run.save_eval(_acc(2), pd.DataFrame({"fold": [0, 0]}))
    run.close()


def test_operating_point_comes_from_the_summary(tmp_path):
    run = Run.create(tmp_path, "точка", tensorboard=False)
    run.save_summary(
        {"best": {"mask_threshold": 0.275, "cls_threshold": 0.5, "min_area": 0.03}}
    )
    run.save_eval(_acc(4), _rows(4))
    run.close()
    assert Run.open(run.dir).operating_point() == (0.275, 0.5, 0.03)


def test_operating_point_falls_back_to_a_sweep(tmp_path):
    run = Run.create(tmp_path, "без-сводки", tensorboard=False)
    run.save_eval(_acc(6), _rows(6))
    run.close()

    reopened = Run.open(run.dir)
    thr, cls_thr, area = reopened.operating_point()
    assert 0.0 < thr < 1.0
    best = reopened.load_eval().best()
    assert (thr, cls_thr, area) == (best.mask_threshold, best.cls_threshold, best.min_area)


def test_load_eval_of_a_missing_file_says_so(tmp_path):
    run = Run.create(tmp_path, "нет-оценки", tensorboard=False)
    run.close()
    with pytest.raises(FileNotFoundError, match="val.npz"):
        Run.open(run.dir).load_eval()


def test_save_state_is_atomic_and_leaves_no_temp(tmp_path):
    run = Run.create(tmp_path, "чекпоинт", tensorboard=False)
    run.save_state({"model": {"w": torch.zeros(2)}, "epoch": 3})
    run.close()
    assert (run.dir / "ckpt" / "last.pt").exists()
    assert not list((run.dir / "ckpt").glob("*.tmp"))


def test_load_state_returns_whatever_was_put_in(tmp_path):
    run = Run.create(tmp_path, "состояние", tensorboard=False)
    run.save_state({"эпоха": 7, "своё": [1, 2, 3]}, name="best.pt")
    run.close()
    state = Run.open(run.dir).load_state("best.pt")
    assert state["эпоха"] == 7
    assert state["своё"] == [1, 2, 3]


def test_save_state_overwrites_in_place(tmp_path):
    run = Run.create(tmp_path, "перезапись", tensorboard=False)
    run.save_state({"epoch": 1})
    run.save_state({"epoch": 2})
    run.close()
    assert Run.open(run.dir).load_state()["epoch"] == 2
    assert len(list((run.dir / "ckpt").iterdir())) == 1


def test_load_state_on_a_missing_file_says_so(tmp_path):
    run = Run.create(tmp_path, "нет-чекпоинта", tensorboard=False)
    run.close()
    with pytest.raises(FileNotFoundError, match="last.pt"):
        Run.open(run.dir).load_state()
