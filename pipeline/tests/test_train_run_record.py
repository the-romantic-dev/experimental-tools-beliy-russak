"""Прогон обязан оставлять ось показов — иначе гейт по эталону нечем питать.

Обучение здесь не запускается: проверяется только то, что прогон пишет после
себя, и берутся ровно те части train.py, которые за это отвечают.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401 — переменные среды до торча

from aic.runs import Run
from aic_pipeline.train import epoch_samples, open_run_record


def test_epoch_samples_prefers_the_explicit_epoch_size():
    assert epoch_samples({"data": {"epoch_size": 8000}}, n_train_rows=99999) == 8000


def test_epoch_samples_falls_back_to_the_dataset_size():
    assert epoch_samples({"data": {}}, n_train_rows=1234) == 1234
    assert epoch_samples({}, n_train_rows=1234) == 1234


def test_open_run_record_snapshots_the_config(tmp_path):
    cfg = {"data": {"epoch_size": 100}, "train": {"lr": 3e-4}, "tensorboard": False}
    run = open_run_record(tmp_path, "проба", cfg)
    run.close()
    assert Run.open(run.dir).snapshot == cfg


def test_run_record_logs_samples_on_every_epoch(tmp_path):
    """Ось из лога, а не из конфига: чужой прогон читается без знания epoch_size."""
    cfg = {"data": {"epoch_size": 100}, "tensorboard": False}
    run = open_run_record(tmp_path, "ось", cfg)
    per_epoch = epoch_samples(cfg, n_train_rows=0)
    for epoch in range(3):
        run.log(epoch, {"val/aic_tuned": 0.1 * epoch, "samples": (epoch + 1) * per_epoch})
    run.close()

    xs, ys = Run.open(run.dir).curve("val/aic_tuned")
    assert xs.tolist() == [100.0, 200.0, 300.0]
    assert ys.tolist() == [0.0, 0.1, 0.2]


def test_a_second_run_of_the_same_name_does_not_overwrite(tmp_path):
    first = open_run_record(tmp_path, "занято", {"tensorboard": False})
    first.close()
    second = open_run_record(tmp_path, "занято", {"tensorboard": False})
    second.close()
    assert second.dir != first.dir
