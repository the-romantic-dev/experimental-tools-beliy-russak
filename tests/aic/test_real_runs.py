"""Совместимость с накопленными прогонами. Без этих тестов переезд бессмыслен.

Тесты пропускаются, если папки runs/ нет: библиотека ставится и без воркспейса.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic.analysis import OofView, leaderboard
from aic.runs import Run
from aic.stats import compare, per_image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = REPO_ROOT / "runs"


def _runs_with_eval() -> list[Path]:
    if not RUNS.is_dir():
        return []
    return sorted(d for d in RUNS.iterdir() if (d / "oof" / "val.npz").exists())


pytestmark = pytest.mark.skipif(not _runs_with_eval(), reason="в репозитории нет прогонов")


def test_every_run_dir_opens():
    """Ни одна из накопленных папок не должна отвалиться на чтении."""
    for run_dir in _runs_with_eval():
        run = Run.open(run_dir)
        assert isinstance(run.snapshot, dict)
        assert isinstance(run.summary, dict)
        assert not run.history.empty, run_dir.name


def test_eval_loads_and_lengths_agree():
    for run_dir in _runs_with_eval():
        ev = Run.open(run_dir).load_eval()
        assert len(ev) == ev.stems.size
        assert len(np.unique(ev.stems)) == ev.stems.size, run_dir.name


def test_best_aic_reproduces_the_stored_summary():
    """Главная проверка: числа не поехали."""
    checked = 0
    for run_dir in _runs_with_eval():
        run = Run.open(run_dir)
        stored = (run.summary or {}).get("best")
        if not stored:
            continue
        op = run.operating_point()
        got = run.load_eval().acc.evaluate(*op)
        assert got.aic == pytest.approx(stored["aic"], abs=1e-9), run_dir.name
        assert got.dice_pos == pytest.approx(stored["dice_pos"], abs=1e-9), run_dir.name
        assert got.fpr_neg == pytest.approx(stored["fpr_neg"], abs=1e-9), run_dir.name
        checked += 1
    assert checked > 0, "ни у одного прогона нет summary.best — проверять нечего"


def test_per_image_means_match_the_summary():
    """Вердикт обязан считаться по той же метрике, по которой отбирают модели."""
    for run_dir in _runs_with_eval()[:3]:
        run = Run.open(run_dir)
        op = run.operating_point()
        ev = run.load_eval()
        sample = per_image(ev.acc, op)
        direct = ev.acc.evaluate(*op)
        assert sample.dice[sample.is_pos].mean() == pytest.approx(direct.dice_pos, abs=1e-9)
        assert sample.alarm[~sample.is_pos].mean() == pytest.approx(direct.fpr_neg, abs=1e-9)


def test_comparing_a_run_with_itself_is_zero():
    run_dir = _runs_with_eval()[0]
    run = Run.open(run_dir)
    ev = run.load_eval()
    cmp = compare(ev, ev, op=run.operating_point(), bootstrap_n=200)
    assert cmp.delta_ref_op == pytest.approx(0.0, abs=1e-12)
    assert cmp.n_common == len(ev)


def test_curve_from_step_matches_the_old_axis():
    """Ось показов из номера эпохи: (step + 1) * epoch_size, как считал load_reference."""
    for run_dir in _runs_with_eval():
        run = Run.open(run_dir)
        epoch_size = (run.snapshot.get("data") or {}).get("epoch_size")
        if not epoch_size or "val/aic_tuned" not in run.history.columns:
            continue
        xs, ys = run.curve("val/aic_tuned", x=("step", int(epoch_size)))
        assert xs[0] == pytest.approx(float(epoch_size))
        assert xs.size == ys.size > 0
        return
    pytest.skip("ни у одного прогона нет data.epoch_size в снапшоте")


def test_oof_view_opens_a_real_run():
    for run_dir in _runs_with_eval():
        if not (run_dir / "oof" / "val_rows.parquet").exists():
            continue
        view = OofView.from_run(run_dir)
        op = Run.open(run_dir).operating_point()
        buckets = view.by_area(op)
        assert buckets["n"].sum() == int(view.is_pos.sum())
        ceilings = view.ceilings(op)
        assert ceilings["perfect_cls"] >= ceilings["current"] - 1e-9
        assert ceilings["perfect_thr"] >= ceilings["current"] - 1e-9
        return
    pytest.skip("ни у одного прогона нет val_rows.parquet")


def test_leaderboard_covers_the_whole_runs_folder():
    board = leaderboard(RUNS)
    assert not board.empty
    assert "run" in board.columns
    names = set(board["run"])
    for run_dir in _runs_with_eval():
        assert run_dir.name in names, run_dir.name
