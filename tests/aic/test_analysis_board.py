"""Лидерборд по папке runs/. Читает только то, что прогон после себя оставил."""

from __future__ import annotations

import pytest

from aic.analysis import compare_table, flatten, history, leaderboard
from aic.runs import Run


def _run(root, name, *, aic, snapshot, epochs=6):
    run = Run.create(root, name, tensorboard=False)
    run.save_snapshot(snapshot)
    for step in range(epochs):
        run.log(step, {"val/aic_tuned": aic * (step + 1) / epochs, "samples": 8000 * (step + 1)})
    run.save_summary({
        "run": name,
        "best_aic": aic,
        "best": {"dice_pos": 0.7, "fpr_neg": 0.07, "mask_threshold": 0.275,
                 "cls_threshold": 0.5, "min_area": 0.0},
        "epochs_done": epochs,
    })
    run.close()
    return run.dir


def test_flatten_uses_dotted_paths():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    assert flatten({}) == {}


def test_leaderboard_sorts_by_best_aic(tmp_path):
    _run(tmp_path, "слабый", aic=0.60, snapshot={"data": {"size": 512}})
    _run(tmp_path, "сильный", aic=0.80, snapshot={"data": {"size": 768}})
    board = leaderboard(tmp_path)
    assert list(board["run"]) == ["сильный", "слабый"]
    assert board["best_aic"].tolist() == [0.80, 0.60]


def test_leaderboard_shows_only_differing_keys(tmp_path):
    _run(tmp_path, "а", aic=0.6, snapshot={"data": {"size": 768}, "train": {"lr": 1e-4}})
    _run(tmp_path, "б", aic=0.7, snapshot={"data": {"size": 768}, "train": {"lr": 3e-4}})
    board = leaderboard(tmp_path)
    assert "lr" in board.columns
    assert "size" not in board.columns


def test_leaderboard_keeps_everything_when_asked(tmp_path):
    _run(tmp_path, "а", aic=0.6, snapshot={"data": {"size": 768}})
    _run(tmp_path, "б", aic=0.7, snapshot={"data": {"size": 768}})
    board = leaderboard(tmp_path, only_diff=False)
    assert "size" in board.columns


def test_leaderboard_of_an_empty_root_is_an_empty_frame(tmp_path):
    assert leaderboard(tmp_path).empty
    assert leaderboard(tmp_path / "нет-такой").empty


def test_leaderboard_skips_dirs_without_a_snapshot(tmp_path):
    _run(tmp_path, "настоящий", aic=0.6, snapshot={"a": 1})
    (tmp_path / "мусор").mkdir()
    assert list(leaderboard(tmp_path)["run"]) == ["настоящий"]


def test_leaderboard_falls_back_to_the_log_without_a_summary(tmp_path):
    run = Run.create(tmp_path, "без-сводки", tensorboard=False)
    run.save_snapshot({"a": 1})
    run.log(0, {"val/aic_tuned": 0.42})
    run.close()
    board = leaderboard(tmp_path)
    assert board["best_aic"].iloc[0] == pytest.approx(0.42)


def test_leaderboard_disambiguates_colliding_leaf_names(tmp_path):
    """`loss.seg.bce` и `loss.area.small_seg.bce` не должны схлопнуться в bce."""
    _run(tmp_path, "а", aic=0.6, snapshot={
        "loss": {"seg": {"bce": 1.0}, "area": {"small_seg": {"bce": 0.0}}}})
    _run(tmp_path, "б", aic=0.7, snapshot={
        "loss": {"seg": {"bce": 2.0}, "area": {"small_seg": {"bce": 3.0}}}})
    board = leaderboard(tmp_path)
    assert "loss.seg.bce" in board.columns
    assert "loss.area.small_seg.bce" in board.columns


def test_leaderboard_marks_a_run_over_budget(tmp_path):
    run_dir = _run(tmp_path, "жирный", aic=0.6, snapshot={"a": 1})
    Run.open(run_dir).save_summary({"budget": {"gflops": 220.0, "within_limit": False}})
    board = leaderboard(tmp_path)
    assert bool(board["over_budget"].iloc[0]) is True
    assert board["gflops"].iloc[0] == 220.0


def test_compare_table_puts_runs_side_by_side(tmp_path):
    a = _run(tmp_path, "а", aic=0.6, snapshot={"train": {"lr": 1e-4, "bs": 8}})
    b = _run(tmp_path, "б", aic=0.7, snapshot={"train": {"lr": 3e-4, "bs": 8}})
    table = compare_table([a, b])
    assert list(table.columns) == ["а", "б"]
    assert "train.lr" in table.index
    assert "train.bs" not in table.index


def test_history_stacks_curves_of_several_runs(tmp_path):
    a = _run(tmp_path, "а", aic=0.6, epochs=3, snapshot={"x": 1})
    b = _run(tmp_path, "б", aic=0.8, epochs=3, snapshot={"x": 2})
    frame = history([a, b])
    assert set(frame["run"]) == {"а", "б"}
    assert len(frame) == 6
    assert {"step", "val/aic_tuned", "samples"} <= set(frame.columns)


def test_history_of_runs_without_the_metric_is_empty(tmp_path):
    run = Run.create(tmp_path, "пусто", tensorboard=False)
    run.save_snapshot({"a": 1})
    run.log(0, {"train/loss": 1.0})
    run.close()
    assert history([run.dir]).empty
