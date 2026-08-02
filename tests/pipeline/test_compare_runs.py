"""Сравнение прогонов и интроспекция энкодеров — то, на чём стоят ноутбуки."""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import json

import matplotlib
import pytest

matplotlib.use("Agg")

from experimental_tools_beliy_russak.models import describe_encoder  # noqa: E402
from experimental_tools_beliy_russak.analysis.compare_runs import (  # noqa: E402
    compare_table,
    load_history,
    load_summary,
    plot_history,
    resolve_run,
)


@pytest.fixture
def fake_run(tmp_path):
    """Минимальная папка прогона: config.yaml + metrics.jsonl + summary.json."""
    def make(name: str, aic_values: list[float], encoder: str = "tu-resnet34"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        (run_dir / "config.yaml").write_text(
            "model:\n"
            f"  encoder: {encoder}\n"
            "  arch: unet\n"
            "data:\n"
            "  size: 512\n",
            encoding="utf-8",
        )
        lines = []
        for epoch, value in enumerate(aic_values):
            lines.append(json.dumps({
                "step": epoch,
                "elapsed_s": 600 * (epoch + 1),
                "train/loss": 1.0 - 0.1 * epoch,
                "train/epoch_time_s": 540,
                "val/aic_tuned": value,
                "val/dice_tuned": value - 0.05,
                "val/fpr_tuned": 0.05,
            }))
        (run_dir / "metrics.jsonl").write_text("\n".join(lines), encoding="utf-8")
        (run_dir / "summary.json").write_text(json.dumps({
            "run": name,
            "best_aic": max(aic_values),
            "best": {
                "aic": max(aic_values), "dice_pos": max(aic_values) - 0.05,
                "fpr_neg": 0.05, "mask_threshold": 0.3, "cls_threshold": 0.7,
                "min_area": 0.0,
            },
            "epochs_done": len(aic_values),
        }), encoding="utf-8")
        return run_dir
    return make


def test_load_history_reads_all_epochs(fake_run):
    run_dir = fake_run("armA", [0.6, 0.7, 0.75])
    history = load_history(run_dir)
    assert len(history) == 3
    assert history["run"].unique().tolist() == ["armA"]
    assert history["val/aic_tuned"].max() == pytest.approx(0.75)


def test_load_history_of_run_without_metrics_is_empty(tmp_path):
    (tmp_path / "empty").mkdir()
    assert load_history(tmp_path / "empty").empty


def test_load_summary(fake_run):
    run_dir = fake_run("armA", [0.6, 0.8])
    assert load_summary(run_dir)["best_aic"] == pytest.approx(0.8)


def test_compare_table_sorts_by_aic_and_pulls_config(fake_run):
    a = fake_run("armA", [0.60, 0.70], encoder="tu-convnext_tiny")
    b = fake_run("armB", [0.65, 0.82], encoder="tu-resnet34")

    table = compare_table([a, b])
    assert table["run"].tolist() == ["armB", "armA"]          # лучший сверху
    assert table.iloc[0]["encoder"] == "tu-resnet34"
    assert table.iloc[0]["AIC"] == pytest.approx(0.82)
    assert table.iloc[0]["мин/эпоху"] == pytest.approx(9.0)   # 540 c


def test_resolve_run_rejects_unknown(tmp_path):
    with pytest.raises(FileNotFoundError, match="не нашёл прогон"):
        resolve_run(tmp_path / "нет-такого")


def test_plot_history_handles_single_and_multiple_runs(fake_run):
    a = fake_run("armA", [0.6, 0.7])
    b = fake_run("armB", [0.65, 0.8])
    assert plot_history([a]) is not None
    assert plot_history([a, b], metrics=("val/aic_tuned",)) is not None


# --- интроспекция энкодеров: основание гипотезы в ноутбуке ------------------

def test_convnext_has_no_stride2_features():
    """Ровно то, что проверяет эксперимент 1: у ConvNeXt карты на stride 2 нет,
    поэтому два последних блока U-Net-декодера остаются без skip-connection."""
    info = describe_encoder("tu-convnext_tiny")
    assert info["out_channels"][1] == 0
    assert info["skip_channels"][-2:] == [0, 0]
    assert info["blocks_without_skip"] == 2


def test_resnet_feeds_every_decoder_block_but_the_last():
    info = describe_encoder("tu-resnet34")
    assert info["out_channels"][1] == 64
    assert info["blocks_without_skip"] == 1
    assert all(s > 0 for s in info["skip_channels"][:-1])


def test_describe_encoder_reports_parameter_counts():
    info = describe_encoder("tu-resnet34")
    assert 0 < info["encoder_m"] < info["total_m"]
