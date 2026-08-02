"""Чтение папки прогона, включая чужой и старый."""

from __future__ import annotations

import pytest

from aic.runs import Run


def _make(tmp_path, rows, snapshot=None):
    run = Run.create(tmp_path, "прогон", tensorboard=False)
    for step, metrics in enumerate(rows):
        run.log(step, metrics)
    if snapshot is not None:
        run.save_snapshot(snapshot)
    run.close()
    return Run.open(run.dir)


def test_history_reads_every_row(tmp_path):
    run = _make(tmp_path, [{"val/aic_tuned": 0.3}, {"val/aic_tuned": 0.6}])
    assert list(run.history["val/aic_tuned"]) == [0.3, 0.6]
    assert list(run.history["step"]) == [0, 1]


def test_history_of_a_run_without_metrics_is_empty(tmp_path):
    run = Run.create(tmp_path, "пусто", tensorboard=False)
    run.close()
    assert Run.open(run.dir).history.empty


def test_snapshot_comes_back_unchanged(tmp_path):
    payload = {"модель": "unet", "epoch_size": 8000}
    run = _make(tmp_path, [{"a": 1}], snapshot=payload)
    assert run.snapshot == payload


def test_snapshot_of_a_run_without_config_is_empty(tmp_path):
    run = Run.create(tmp_path, "без-конфига", tensorboard=False)
    run.close()
    assert Run.open(run.dir).snapshot == {}


def test_open_on_a_missing_dir_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="нет папки прогона"):
        Run.open(tmp_path / "такой-нет")


def test_curve_uses_a_logged_column(tmp_path):
    run = _make(
        tmp_path,
        [
            {"val/aic_tuned": 0.3, "samples": 8000},
            {"val/aic_tuned": 0.6, "samples": 16000},
        ],
    )
    xs, ys = run.curve("val/aic_tuned")
    assert xs.tolist() == [8000.0, 16000.0]
    assert ys.tolist() == [0.3, 0.6]


def test_curve_can_derive_the_axis_from_step(tmp_path):
    """Старые прогоны samples не логировали — множитель передаёт вызывающий."""
    run = _make(tmp_path, [{"val/aic_tuned": 0.3}, {"val/aic_tuned": 0.6}])
    xs, ys = run.curve("val/aic_tuned", x=("step", 8000))
    assert xs.tolist() == [8000.0, 16000.0]
    assert ys.tolist() == [0.3, 0.6]


def test_curve_skips_rows_without_the_metric(tmp_path):
    """Валидация может идти не на каждом шаге — такие строки в кривую не входят."""
    run = _make(
        tmp_path,
        [
            {"train/loss": 1.0, "samples": 8000},
            {"val/aic_tuned": 0.6, "samples": 16000},
        ],
    )
    xs, ys = run.curve("val/aic_tuned")
    assert xs.tolist() == [16000.0]
    assert ys.tolist() == [0.6]


def test_curve_without_the_axis_column_explains_how_to_fix_it(tmp_path):
    run = _make(tmp_path, [{"val/aic_tuned": 0.3}])
    with pytest.raises(KeyError) as err:
        run.curve("val/aic_tuned")
    message = str(err.value)
    assert "samples" in message
    assert 'x=("step"' in message or "x=('step'" in message


def test_curve_without_the_metric_column_says_which_one(tmp_path):
    run = _make(tmp_path, [{"samples": 8000}])
    with pytest.raises(KeyError, match="val/aic_tuned"):
        run.curve("val/aic_tuned")


def test_curve_of_an_empty_history_is_empty(tmp_path):
    run = Run.create(tmp_path, "пусто", tensorboard=False)
    run.close()
    xs, ys = Run.open(run.dir).curve("val/aic_tuned")
    assert xs.size == 0 and ys.size == 0
