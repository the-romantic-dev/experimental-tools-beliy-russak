"""Очередь экспериментов: раскрытие плана, префлайт и семантика прогона.

Обучение здесь не запускается ни разу — `train.run` подменяется заглушкой.
Проверяется ровно то, ради чего очередь и нужна: порядок сохраняется, упавший
прогон не роняет остальные, готовые пропускаются.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import json
import shutil

import pytest
import yaml

from experimental_tools_beliy_russak.config import apply_override
from experimental_tools_beliy_russak.plans import (
    PlannedRun,
    describe,
    load_plan,
    preflight,
    run_plan,
)


def _write_plan(tmp_path, data: dict, name: str = "plan.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return path


# --- раскрытие плана --------------------------------------------------------


def test_plain_list_keeps_order(tmp_path):
    path = _write_plan(tmp_path, {"name": "s", "runs": [
        {"config": "baseline"}, {"config": "fast_cache_384"}, {"config": "smoke"},
    ]})
    plan_name, queue = load_plan(path)
    assert plan_name == "s"
    assert [r.config for r in queue] == ["baseline", "fast_cache_384", "smoke"]
    # имя берётся из самого конфига, как при обычном train -c
    assert [r.name for r in queue] == ["baseline-unet-convnext-512", "fast-384", "smoke"]


def test_bare_string_is_a_valid_run(tmp_path):
    path = _write_plan(tmp_path, {"runs": ["baseline"]})
    _, queue = load_plan(path)
    assert queue[0].config == "baseline"


def test_grid_expands_to_cartesian_product(tmp_path):
    path = _write_plan(tmp_path, {"runs": [{
        "config": "baseline", "name": "sweep",
        "grid": {"data.small_area_fraction": [0.3, 0.5],
                 "loss.area.small_weight": [1.0, 2.0]},
    }]})
    _, queue = load_plan(path)

    assert len(queue) == 4
    assert len({r.name for r in queue}) == 4
    assert queue[0].name == "sweep-small_area_fraction03-small_weight10"
    assert queue[0].overrides == {"data.small_area_fraction": 0.3, "loss.area.small_weight": 1.0}


def test_defaults_apply_everywhere_but_lose_to_explicit_set(tmp_path):
    path = _write_plan(tmp_path, {
        "defaults": {"train.epochs": 4},
        "runs": [
            {"config": "baseline"},
            {"config": "fast_cache_384", "set": {"train.epochs": 9}},
        ],
    })
    _, queue = load_plan(path)
    assert queue[0].overrides["train.epochs"] == 4
    assert queue[1].overrides["train.epochs"] == 9


def test_duplicate_names_are_rejected(tmp_path):
    """Два прогона с одним именем затёрли бы результаты друг друга."""
    path = _write_plan(tmp_path, {"runs": [{"config": "baseline"}, {"config": "baseline"}]})
    with pytest.raises(ValueError, match="имена прогонов повторяются"):
        load_plan(path)


def test_empty_plan_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="нет ни одного прогона"):
        load_plan(_write_plan(tmp_path, {"name": "empty"}))


def test_run_without_config_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="не задан `config`"):
        load_plan(_write_plan(tmp_path, {"runs": [{"name": "безымянный"}]}))


def test_missing_plan_file_says_where_it_looked():
    with pytest.raises(FileNotFoundError, match="план не найден"):
        load_plan("такого-плана-нет")


# --- переопределения доезжают в конфиг без искажений -------------------------


@pytest.mark.parametrize("value", [0.3, 1, True, None, "medium", [0.35, 1.0], {}])
def test_overrides_survive_the_round_trip(value):
    """План хранит значения как объекты, а train принимает строки `key=value`.
    Сериализация обязана возвращать ровно то же значение, иначе прогон тихо
    поедет не с теми параметрами."""
    run = PlannedRun("baseline", "проба", {"data.aug": value})
    cfg: dict = {}
    for override in run.as_cli_overrides():
        apply_override(cfg, override)
    assert cfg["data"]["aug"] == value
    assert cfg["name"] == "проба"


# --- префлайт ---------------------------------------------------------------


def test_preflight_passes_for_shipped_configs(tmp_path):
    path = _write_plan(tmp_path, {"runs": [{"config": "smoke"}]})
    _, queue = load_plan(path)
    assert preflight(queue) == []


def test_preflight_catches_a_broken_config(tmp_path):
    """Смысл префлайта: опечатка в пятом прогоне вылезает до старта очереди,
    а не через четыре часа."""
    path = _write_plan(tmp_path, {"runs": [
        {"config": "smoke", "name": "ок"},
        {"config": "smoke", "name": "битый", "set": {"model.encoder": "tu-нет-такого"}},
    ]})
    _, queue = load_plan(path)
    problems = preflight(queue)

    assert len(problems) == 1
    assert "битый" in problems[0]


def test_preflight_rejects_a_config_over_the_flops_budget(tmp_path):
    """Прогон вне бюджета получил бы 0 за этап — тратить на него ночь незачем."""
    path = _write_plan(tmp_path, {"runs": [
        {"config": "smoke", "name": "ок"},
        {"config": "smoke", "name": "жирный", "set": {"data.size": 1024}},
    ]})
    _, queue = load_plan(path)
    problems = preflight(queue)

    assert len(problems) == 1
    assert "жирный" in problems[0]
    assert "GFLOPs" in problems[0]
    # отказ бесполезен без ответа на вопрос «а на чём тогда учить»
    assert "px" in problems[0]


def test_preflight_checks_the_budget_of_every_point_of_a_size_sweep(tmp_path):
    """Дедупликация сборки моделей не должна прятать разницу в data.size.

    Сеть у всех точек свипа одна, поэтому собирается она один раз — но стоимость
    изображения у них разная, и проверять её надо у каждой.
    """
    path = _write_plan(tmp_path, {"runs": [{
        "config": "smoke", "name": "свип", "grid": {"data.size": [256, 1024]},
    }]})
    _, queue = load_plan(path)
    problems = preflight(queue)

    assert len(problems) == 1
    assert "size1024" in problems[0]


def test_preflight_lets_an_exempt_config_through(tmp_path):
    path = _write_plan(tmp_path, {"runs": [{
        "config": "smoke", "name": "исследование",
        "set": {"data.size": 1024, "budget.exempt": True,
                "budget.exempt_reason": "проверка гипотезы, в сабмит не пойдёт"},
    }]})
    _, queue = load_plan(path)
    assert preflight(queue) == []


# --- прогон очереди ---------------------------------------------------------


@pytest.fixture
def queue_env(tmp_path, monkeypatch):
    """Отдельный воркспейс во временной папке плюс заглушка вместо обучения.

    Конфиги копируются из настоящего `configs/`: очередь их действительно
    загружает и собирает по ним модели на префлайте. А вот `runs/` должен быть
    временным, иначе тест засорял бы настоящую папку прогонов.
    """
    import experimental_tools_beliy_russak.train as train_module
    from experimental_tools_beliy_russak.workspace import configs_root, use_workspace

    shutil.copytree(configs_root(), tmp_path / "configs")
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    calls: list[str] = []

    def fake_run(cfg, resume=None):
        name = cfg.get("name")
        calls.append(name)
        if "падать" in str(name):
            raise RuntimeError("синтетическое падение")
        return {"run": name, "best_aic": 0.5, "epochs_done": 1}

    monkeypatch.setattr(train_module, "run", fake_run)
    with use_workspace(tmp_path):
        yield runs_root, calls


def test_queue_runs_everything_in_order(tmp_path, queue_env):
    _, calls = queue_env
    path = _write_plan(tmp_path, {"name": "q", "runs": [
        {"config": "smoke", "name": "первый"},
        {"config": "smoke", "name": "второй"},
    ]})

    report = run_plan(path, logger=lambda *_: None)
    assert calls == ["первый", "второй"]
    assert [item["status"] for item in report["runs"]] == ["ok", "ok"]


def test_failed_run_does_not_stop_the_queue(tmp_path, queue_env):
    """Одна опечатка в середине плана не должна стоить всей ночи."""
    _, calls = queue_env
    path = _write_plan(tmp_path, {"name": "q", "runs": [
        {"config": "smoke", "name": "падать-тут"},
        {"config": "smoke", "name": "но-этот-должен-посчитаться"},
    ]})

    report = run_plan(path, logger=lambda *_: None)
    assert len(calls) == 2
    assert [item["status"] for item in report["runs"]] == ["failed", "ok"]
    assert "синтетическое падение" in report["runs"][0]["error"]


def test_stop_on_error_aborts_the_rest(tmp_path, queue_env):
    _, calls = queue_env
    path = _write_plan(tmp_path, {"name": "q", "runs": [
        {"config": "smoke", "name": "падать-тут"},
        {"config": "smoke", "name": "второй"},
    ]})

    run_plan(path, stop_on_error=True, logger=lambda *_: None)
    assert calls == ["падать-тут"]


def test_finished_runs_are_skipped_so_a_plan_can_be_resumed(tmp_path, queue_env):
    runs_root, calls = queue_env
    done = runs_root / "первый"
    done.mkdir()
    (done / "summary.json").write_text(json.dumps({"best_aic": 0.9}), encoding="utf-8")

    path = _write_plan(tmp_path, {"name": "q", "runs": [
        {"config": "smoke", "name": "первый"},
        {"config": "smoke", "name": "второй"},
    ]})

    report = run_plan(path, logger=lambda *_: None)
    assert calls == ["второй"]
    assert report["runs"][0]["status"] == "skipped"
    assert report["runs"][0]["best_aic"] == 0.9


def test_force_recomputes_finished_runs(tmp_path, queue_env):
    runs_root, calls = queue_env
    done = runs_root / "первый"
    done.mkdir()
    (done / "summary.json").write_text(json.dumps({"best_aic": 0.9}), encoding="utf-8")

    path = _write_plan(tmp_path, {"name": "q", "runs": [{"config": "smoke", "name": "первый"}]})
    run_plan(path, skip_done=False, logger=lambda *_: None)
    assert calls == ["первый"]


def test_only_filters_the_queue(tmp_path, queue_env):
    _, calls = queue_env
    path = _write_plan(tmp_path, {"name": "q", "runs": [
        {"config": "smoke", "name": "первый"},
        {"config": "smoke", "name": "второй"},
    ]})

    run_plan(path, only=["второй"], logger=lambda *_: None)
    assert calls == ["второй"]


def test_only_that_matches_nothing_is_an_error(tmp_path, queue_env):
    path = _write_plan(tmp_path, {"name": "q", "runs": [{"config": "smoke", "name": "первый"}]})
    with pytest.raises(ValueError, match="не выбрал ни одного прогона"):
        run_plan(path, only=["нет-такого"], logger=lambda *_: None)


def test_plan_report_lands_next_to_the_runs(tmp_path, queue_env):
    runs_root, _ = queue_env
    path = _write_plan(tmp_path, {"name": "мойплан", "runs": [
        {"config": "smoke", "name": "первый"},
    ]})

    run_plan(path, logger=lambda *_: None)
    saved = json.loads((runs_root / "_plan_мойплан.json").read_text(encoding="utf-8"))
    assert saved["plan"] == "мойплан"
    assert saved["runs"][0]["name"] == "первый"


def test_describe_marks_finished_runs(tmp_path, queue_env):
    runs_root, _ = queue_env
    (runs_root / "первый").mkdir()
    (runs_root / "первый" / "summary.json").write_text("{}", encoding="utf-8")

    path = _write_plan(tmp_path, {"name": "q", "runs": [
        {"config": "smoke", "name": "первый"},
        {"config": "smoke", "name": "второй"},
    ]})
    _, queue = load_plan(path)
    text = describe("q", queue)

    assert "первый" in text and "готов" in text
    assert "будет запущен" in text
