"""Разбор валидации по корзинам: цифры должны сходиться с прямым счётом.

Отчёт — единственное, чем оцениваются e1/e2/e3 (агрегатный AIC не показывает,
за чей счёт вырос), поэтому его арифметика проверяется на синтетике, где
правильный ответ известен заранее.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import numpy as np
import pandas as pd
import pytest

from aic_pipeline.metrics import AICAccumulator
from aic_pipeline.analysis.oof_report import MISS_DICE, OofView, by_area, compare, report
from aic_pipeline.analysis.oof_report import _frame


def _make_run(tmp_path, probs, gts, cls_probs, rows=None, name="run"):
    run_dir = tmp_path / name
    (run_dir / "oof").mkdir(parents=True)
    accumulator = AICAccumulator(n_bins=64)
    accumulator.update(np.asarray(probs), np.asarray(gts), np.asarray(cls_probs, dtype=np.float32))
    accumulator.save(run_dir / "oof" / "val.npz")
    if rows is not None:
        pd.DataFrame(rows).to_parquet(run_dir / "oof" / "val_rows.parquet", index=False)
    return run_dir


@pytest.fixture
def simple_run(tmp_path):
    """Три кадра: точное попадание, полный промах, чистый негатив."""
    side = 20
    hit_gt = np.zeros((side, side), dtype=np.float32)
    hit_gt[:10] = 1.0                     # 50% площади
    hit_prob = hit_gt * 0.9

    miss_gt = np.zeros((side, side), dtype=np.float32)
    miss_gt[:2, :2] = 1.0                 # 1% площади
    miss_prob = np.zeros((side, side), dtype=np.float32)

    neg_gt = np.zeros((side, side), dtype=np.float32)
    neg_prob = np.zeros((side, side), dtype=np.float32)

    return _make_run(
        tmp_path,
        probs=[hit_prob, miss_prob, neg_prob],
        gts=[hit_gt, miss_gt, neg_gt],
        cls_probs=[0.99, 0.2, 0.01],
        rows=[
            {"domain": "coco", "generator": "brushnet"},
            {"domain": "plain", "generator": "none"},
            {"domain": "plain", "generator": "none"},
        ],
    )


def test_view_restores_dice_and_area_from_histograms(simple_run):
    view = OofView(simple_run)
    assert list(view.is_pos) == [True, True, False]

    k = view.bin_of(0.5)
    assert view.dice[0, k] == pytest.approx(1.0, abs=1e-3)   # точное попадание
    assert view.dice[1, k] == pytest.approx(0.0, abs=1e-6)   # полный промах
    assert view.area[0, k] == pytest.approx(0.5, abs=1e-3)


def test_area_buckets_split_by_gt_size(simple_run):
    view = OofView(simple_run)
    table = _frame(view, 0.5, 0.0)
    buckets = by_area(table)

    assert int(buckets["n"].sum()) == 2       # только позитивы
    small = buckets[buckets.index.map(lambda b: b.right <= 0.01)]
    assert float(small["dice"].iloc[0]) == pytest.approx(0.0)
    assert float(small["miss"].iloc[0]) == pytest.approx(1.0)


def test_miss_counts_frames_killed_by_the_gate(simple_run):
    """Кадр, обнулённый классификатором, — тоже промах: в сабмите он уйдёт
    пустым, и Dice по нему будет нулевой."""
    view = OofView(simple_run)
    ungated = _frame(view, 0.5, cls_threshold=0.0)
    gated = _frame(view, 0.5, cls_threshold=0.5)   # убивает второй кадр (cls 0.2)

    assert int(ungated.miss.sum()) == 1
    assert int(gated.miss.sum()) == 1
    assert float(gated.loc[1, "dice"]) < MISS_DICE


def test_operating_point_includes_min_area_from_calib(tmp_path):
    """Отчёт обязан считать AIC в той же точке, в которой отбиралась модель.
    Раньше min_area из calib.json терялся, и «текущий» сценарий в `ceilings`
    расходился с best_aic прогона ровно на вклад правила по площади."""
    import json

    side = 100
    gt = np.zeros((side, side), dtype=np.float32)
    prob = np.zeros((side, side), dtype=np.float32)
    prob[:2] = 0.9                        # 2% кадра на негативе -> ложная тревога
    run_dir = _make_run(tmp_path, probs=[prob], gts=[gt], cls_probs=[0.9], name="area")
    (run_dir / "calib.json").write_text(
        json.dumps({"mask_threshold": 0.5, "cls_threshold": 0.0, "min_area": 0.03}),
        encoding="utf-8",
    )

    view = OofView(run_dir)
    assert view.best_thresholds() == (0.5, 0.0, 0.03)
    assert view.score(0.5, 0.0, 0.0)["fpr_neg"] == pytest.approx(1.0)
    assert view.score(*view.best_thresholds())["fpr_neg"] == pytest.approx(0.0)


def test_old_calib_without_min_area_still_reads(simple_run):
    """Прогоны, посчитанные до появления ключа, не должны падать."""
    import json

    (simple_run / "calib.json").write_text(
        json.dumps({"mask_threshold": 0.4, "cls_threshold": 0.1}), encoding="utf-8"
    )
    assert OofView(simple_run).best_thresholds() == (0.4, 0.1, 0.0)


def test_ceilings_are_monotonic(simple_run):
    from aic_pipeline.analysis.oof_report import ceilings

    table = ceilings(OofView(simple_run))
    assert list(table["сценарий"])[0].startswith("текущий")
    # каждый следующий сценарий снимает ещё одно ограничение, хуже быть не может
    assert table["dice_pos"].is_monotonic_increasing


def test_report_renders_without_row_metadata(tmp_path):
    """val_rows.parquet может отсутствовать (старые прогоны) — отчёт обязан
    строиться и без разрезов по домену."""
    gt = np.zeros((1, 8, 8), dtype=np.float32)
    gt[0, :4] = 1.0
    run_dir = _make_run(tmp_path, probs=gt * 0.8, gts=gt, cls_probs=[0.9], name="bare")

    text = report(run_dir)
    assert "по площади GT" in text
    assert "по domain" not in text


def test_compare_puts_runs_side_by_side(simple_run, tmp_path):
    side = 20
    gt = np.zeros((side, side), dtype=np.float32)
    gt[:10] = 1.0
    other = _make_run(tmp_path, probs=[gt * 0.9], gts=[gt], cls_probs=[0.9], name="other")

    table = compare([simple_run, other])
    assert list(table["run"]) == ["run", "other"]
    assert {"aic", "dice", "fpr", "miss%"} <= set(table.columns)
    assert any(column.startswith("dice") and column != "dice" for column in table.columns)
