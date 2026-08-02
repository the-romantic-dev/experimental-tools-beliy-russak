"""Таблица прогонов: `aic board`.

Отчётный код, но врёт он тихо. Колонка, названная коротким последним куском
пути, может совпасть у двух разных ключей конфига — и тогда таблица показывает
значение одного параметра под именем другого, а решение принимается по ней.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import json

from aic_pipeline.analysis.leaderboard import leaderboard
from aic_pipeline.workspace import use_workspace


def write_run(root, name: str, config: str, best_aic: float, budget: dict | None = None) -> None:
    run_dir = root / "runs" / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(config, encoding="utf-8")
    summary = {"run": name, "best_aic": best_aic, "epochs_done": 2,
               "best": {"dice_pos": 0.7, "fpr_neg": 0.05, "mask_threshold": 0.4,
                        "cls_threshold": 0.0, "min_area": 0.0}}
    if budget is not None:
        summary["budget"] = budget
    (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")


def test_board_sorts_by_aic_and_shows_only_diverging_keys(tmp_path):
    write_run(tmp_path, "armA", "model:\n  encoder: tu-resnet34\n  arch: unet\n", 0.61)
    write_run(tmp_path, "armB", "model:\n  encoder: tu-convnext_tiny\n  arch: unet\n", 0.74)

    with use_workspace(tmp_path):
        table = leaderboard()

    assert table["run"].tolist() == ["armB", "armA"]
    assert table["encoder"].tolist() == ["tu-convnext_tiny", "tu-resnet34"]
    assert "arch" not in table.columns   # одинаков у обоих — в таблице не нужен


def test_columns_with_the_same_last_key_stay_distinguishable(tmp_path):
    """`loss.seg.bce` и `loss.area.small_seg.bce` дают один и тот же последний
    кусок: под общим именем в таблицу попадал только один из них."""
    write_run(
        tmp_path, "armA",
        "loss:\n  seg: {bce: 1.0}\n  area:\n    small_seg: {bce: 0.5}\n", 0.60,
    )
    write_run(
        tmp_path, "armB",
        "loss:\n  seg: {bce: 2.0}\n  area:\n    small_seg: {bce: 3.0}\n", 0.70,
    )

    with use_workspace(tmp_path):
        table = leaderboard()

    assert table.loc[table.run == "armB", "loss.seg.bce"].iloc[0] == 2.0
    assert table.loc[table.run == "armB", "loss.area.small_seg.bce"].iloc[0] == 3.0


def test_board_shows_what_each_run_costs_per_image(tmp_path):
    """Прирост, купленный выходом за лимит, должен быть виден рядом с приростом.

    Иначе «768 даёт +0.02 AIC» читается как готовый вывод, хотя эта настройка в
    посылку не пройдёт вообще.
    """
    write_run(tmp_path, "жирный", "data:\n  size: 768\n", 0.74,
              budget={"gflops": 160.6, "limit_gflops": 100.0,
                      "within_limit": False, "exempt": True})
    write_run(tmp_path, "зачётный", "data:\n  size: 512\n", 0.71,
              budget={"gflops": 71.4, "limit_gflops": 100.0,
                      "within_limit": True, "exempt": False})

    with use_workspace(tmp_path):
        table = leaderboard()

    assert table.loc[table.run == "жирный", "gflops"].iloc[0] == 160.6
    assert bool(table.loc[table.run == "жирный", "over_budget"].iloc[0]) is True
    assert bool(table.loc[table.run == "зачётный", "over_budget"].iloc[0]) is False


def test_board_survives_runs_from_before_the_budget_check(tmp_path):
    write_run(tmp_path, "старый", "data:\n  size: 512\n", 0.7)
    with use_workspace(tmp_path):
        table = leaderboard()
    assert table["gflops"].isna().all()


def test_board_is_empty_without_runs(tmp_path):
    (tmp_path / "runs").mkdir()
    with use_workspace(tmp_path):
        assert leaderboard().empty
