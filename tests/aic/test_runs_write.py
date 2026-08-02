"""Папка прогона на запись. Формат не меняется — на нём стоят 30 прогонов."""

from __future__ import annotations

import json

import yaml

from aic.runs import Run


def test_create_makes_the_standard_subdirs(tmp_path):
    run = Run.create(tmp_path, "проба", tensorboard=False)
    assert run.dir == tmp_path / "проба"
    for sub in ("ckpt", "oof", "tb", "preds"):
        assert (run.dir / sub).is_dir()
    run.close()


def test_create_does_not_overwrite_an_existing_run(tmp_path):
    first = Run.create(tmp_path, "занято", tensorboard=False)
    first.close()
    second = Run.create(tmp_path, "занято", tensorboard=False)
    assert second.dir != first.dir
    assert second.dir.name.startswith("занято__")
    second.close()


def test_resume_reuses_the_same_dir(tmp_path):
    first = Run.create(tmp_path, "продолжаем", tensorboard=False)
    first.close()
    second = Run.create(tmp_path, "продолжаем", resume=True, tensorboard=False)
    assert second.dir == first.dir
    second.close()


def test_resume_on_a_missing_dir_just_creates_it(tmp_path):
    run = Run.create(tmp_path, "нового-нет", resume=True, tensorboard=False)
    assert run.dir.is_dir()
    run.close()


def test_log_appends_jsonl_with_step_and_elapsed(tmp_path):
    run = Run.create(tmp_path, "лог", tensorboard=False)
    run.log(0, {"train/loss": 1.5})
    run.log(1, {"train/loss": 1.2, "samples": 8000})
    run.close()

    lines = (run.dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert [r["step"] for r in rows] == [0, 1]
    assert all("elapsed_s" in r for r in rows)
    assert rows[1]["samples"] == 8000


def test_log_prefix_applies_to_metric_names_only(tmp_path):
    run = Run.create(tmp_path, "префикс", tensorboard=False)
    run.log(0, {"loss": 1.0}, prefix="val/")
    run.close()
    row = json.loads((run.dir / "metrics.jsonl").read_text(encoding="utf-8").strip())
    assert row["val/loss"] == 1.0
    assert row["step"] == 0


def test_close_writes_the_csv_mirror(tmp_path):
    run = Run.create(tmp_path, "csv", tensorboard=False)
    run.log(0, {"a": 1})
    run.close()
    text = (run.dir / "metrics.csv").read_text(encoding="utf-8")
    assert "step" in text and "a" in text


def test_save_snapshot_writes_yaml_verbatim(tmp_path):
    run = Run.create(tmp_path, "снапшот", tensorboard=False)
    payload = {"модель": "unet", "lr": 3e-4, "вложенное": {"a": [1, 2]}}
    run.save_snapshot(payload)
    run.close()
    back = yaml.safe_load((run.dir / "config.yaml").read_text(encoding="utf-8"))
    assert back == payload


def test_save_summary_merges_top_level_keys(tmp_path):
    run = Run.create(tmp_path, "сводка", tensorboard=False)
    run.save_summary({"run": "сводка", "best_aic": 0.5})
    run.save_summary({"budget": {"gflops": 42.0}})
    run.save_summary({"best_aic": 0.8})
    run.close()

    summary = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {"run": "сводка", "best_aic": 0.8, "budget": {"gflops": 42.0}}


def test_save_summary_replaces_nested_dicts_whole(tmp_path):
    run = Run.create(tmp_path, "вложенное", tensorboard=False)
    run.save_summary({"best": {"aic": 0.5, "thr": 0.3}})
    run.save_summary({"best": {"aic": 0.9}})
    run.close()
    summary = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["best"] == {"aic": 0.9}


def test_info_goes_to_console_and_file(tmp_path, capsys):
    run = Run.create(tmp_path, "сообщения", tensorboard=False)
    run.info("эпоха 3 из 12")
    run.close()
    assert "эпоха 3 из 12" in capsys.readouterr().out
    assert "эпоха 3 из 12" in (run.dir / "train.log").read_text(encoding="utf-8")


def test_run_works_as_a_context_manager(tmp_path):
    with Run.create(tmp_path, "контекст", tensorboard=False) as run:
        run.log(0, {"a": 1})
    assert (run.dir / "metrics.csv").exists()
