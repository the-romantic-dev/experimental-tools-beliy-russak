"""Разбор валидации по корзинам. Арифметика — одна, из AICAccumulator.tables()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aic.analysis import OofView
from aic.metric import AICAccumulator
from aic.runs import Eval, Run


def _view(n_pos=12, n_neg=6, seed=0):
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=64)
    probs, gts = [], []
    for i in range(n_pos + n_neg):
        gt = np.zeros((10, 10), dtype=np.float32)
        prob = rng.random((10, 10)).astype(np.float32) * 0.2
        if i < n_pos:
            filled = 1 + i % 5          # от 1% до 5% площади
            gt[:filled] = 1.0
            prob[:filled] = 0.9
        probs.append(prob)
        gts.append(gt)
    acc.update(np.stack(probs), np.stack(gts))

    rows = pd.DataFrame({
        "stem": [f"кадр{i}" for i in range(n_pos + n_neg)],
        "domain": ["a" if i % 2 else "b" for i in range(n_pos + n_neg)],
        "generator": ["g1"] * (n_pos + n_neg),
    })
    return OofView(Eval(acc, rows["stem"].to_numpy()), rows)


def test_tables_come_from_the_accumulator_not_a_second_copy():
    """Гарантия, что дубль убран, а не переехал."""
    view = _view()
    pred, inter, gt_sum, n_pixels, cls_prob = view.ev.acc.tables()
    assert np.array_equal(view.pred, pred)
    assert np.array_equal(view.inter, inter)
    assert np.array_equal(view.area, pred / n_pixels[:, None])


def test_area_buckets_cover_every_positive_frame():
    view = _view()
    table = view.by_area((0.5, 0.0, 0.0))
    assert table["n"].sum() == int(view.is_pos.sum())
    assert {"n", "dice", "miss_share", "bucket"} <= set(table.columns)


def test_by_column_splits_on_a_row_field():
    view = _view()
    table = view.by_column("domain", (0.5, 0.0, 0.0))
    assert set(table.index) == {"a", "b"}
    assert table["n"].sum() == len(view.ev)


def test_by_column_without_rows_says_so():
    acc = AICAccumulator(n_bins=32)
    rng = np.random.default_rng(0)
    acc.update(rng.random((3, 4, 4)), (rng.random((3, 4, 4)) > 0.5).astype(np.float32))
    view = OofView(Eval(acc, np.array(["a", "b", "c"])))
    with pytest.raises(ValueError, match="строк валидации"):
        view.by_column("domain", (0.5, 0.0, 0.0))


def test_ceilings_are_at_least_the_current_score():
    view = _view()
    op = (0.5, 0.0, 0.0)
    ceilings = view.ceilings(op)
    current = view.ev.acc.evaluate(*op).aic
    assert ceilings["current"] == pytest.approx(current)
    assert ceilings["perfect_cls"] >= current - 1e-9
    assert ceilings["perfect_thr"] >= current - 1e-9


def test_at_respects_the_cls_threshold():
    """Кадр ниже порога классификатора обнуляется целиком."""
    view = _view()
    dice, area = view.at((0.5, 1.1, 0.0))     # порог выше любой вероятности
    assert float(dice.max()) == 0.0
    assert float(area.max()) == 0.0


def test_from_run_reads_the_folder(tmp_path):
    rng = np.random.default_rng(1)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((4, 6, 6)), (rng.random((4, 6, 6)) > 0.6).astype(np.float32))
    rows = pd.DataFrame({"stem": [f"к{i}" for i in range(4)], "domain": list("aabb")})

    run = Run.create(tmp_path, "разбор", tensorboard=False)
    run.save_eval(acc, rows)
    run.close()

    view = OofView.from_run(run.dir)
    assert len(view.ev) == 4
    assert set(view.by_column("domain", (0.5, 0.0, 0.0)).index) == {"a", "b"}
