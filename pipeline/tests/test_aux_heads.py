"""Вспомогательные головы: цели, контракт выхода, вклад в лосс."""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import numpy as np
import pytest
import torch

from aic_pipeline.geometry import (
    denormalize_area,
    mask_geometry,
    normalize_area,
    normalize_components,
)
from aic_pipeline.losses import build_loss
from aic_pipeline.models import build_model
from aic_pipeline.models.aux_heads import AUX_HEADS, parse_aux_spec


def box_mask(h=40, w=40, y0=0, y1=10, x0=0, x1=10) -> torch.Tensor:
    mask = torch.zeros(1, h, w)
    mask[0, y0:y1, x0:x1] = 1.0
    return mask


# --- цели ------------------------------------------------------------------

def test_normalize_area_maps_zero_and_full_frame_to_unit_range():
    assert normalize_area(0.0) == pytest.approx(0.0, abs=1e-9)
    assert normalize_area(1.0) == pytest.approx(1.0, abs=1e-6)


def test_normalize_area_spreads_small_masks_apart():
    """Смысл логарифма: мелкие маски не должны схлопываться в одну точку.

    В линейной шкале 0.001 и 0.01 отличались бы на 0.009 при диапазоне 0.77 —
    то есть MSE их бы не различал."""
    gap_small = normalize_area(0.01) - normalize_area(0.001)
    gap_large = normalize_area(0.5) - normalize_area(0.491)
    assert gap_small > 10 * gap_large


def test_area_normalization_roundtrips():
    for area in (0.0, 0.0004, 0.01, 0.16, 0.765):
        assert denormalize_area(normalize_area(area)) == pytest.approx(area, abs=1e-6)


def test_normalize_area_works_on_tensors():
    values = torch.tensor([0.0, 0.01, 1.0])
    out = normalize_area(values)
    assert out.shape == values.shape
    assert out[0] == pytest.approx(0.0, abs=1e-6)
    assert out[2] == pytest.approx(1.0, abs=1e-5)


def test_normalize_components_is_monotone_and_bounded():
    assert normalize_components(0) == 0.0
    assert normalize_components(1) < normalize_components(5) < normalize_components(16)
    assert normalize_components(16) == pytest.approx(1.0, abs=1e-6)


def test_geometry_of_a_corner_box():
    geom = mask_geometry(box_mask(40, 40, 0, 10, 0, 10), 40, 40)
    assert geom["area"].item() == pytest.approx(100 / 1600)
    assert geom["border"].item() == 1.0            # касается верха и левого края
    assert geom["components"].item() == pytest.approx(normalize_components(1))
    assert geom["geom_valid"].item() == 1.0
    # центр квадрата 0..9 это 4.5, нормируем на 39
    assert geom["centroid"][0].item() == pytest.approx(4.5 / 39, abs=1e-3)


def test_geometry_of_an_interior_box_does_not_touch_border():
    geom = mask_geometry(box_mask(40, 40, 10, 20, 10, 20), 40, 40)
    assert geom["border"].item() == 0.0


def test_geometry_counts_disconnected_pieces():
    mask = torch.zeros(1, 40, 40)
    mask[0, 5:10, 5:10] = 1.0
    mask[0, 25:30, 25:30] = 1.0
    geom = mask_geometry(mask, 40, 40)
    assert geom["components"].item() == pytest.approx(normalize_components(2))


def test_geometry_of_empty_mask_is_neutral_and_flagged_invalid():
    geom = mask_geometry(torch.zeros(1, 32, 32), 32, 32)
    assert geom["geom_valid"].item() == 0.0
    assert geom["area"].item() == 0.0
    assert geom["centroid"].tolist() == [0.5, 0.5]


def test_geometry_ignores_padding_outside_the_valid_region():
    """В режиме pad паддинг занимает до трети холста и не должен считаться."""
    mask = torch.zeros(1, 40, 40)
    mask[0, 0:10, 0:20] = 1.0
    full = mask_geometry(mask, 40, 40)
    cropped = mask_geometry(mask, 20, 20)
    assert cropped["area"].item() > full["area"].item()


# --- реестр и конфиг -------------------------------------------------------

def test_parse_aux_spec_drops_zero_weights():
    assert parse_aux_spec({"area": 0.1, "border": 0.0}) == {"area": 0.1}


def test_parse_aux_spec_accepts_a_plain_list():
    assert parse_aux_spec(["area", "border"]) == {"area": 0.1, "border": 0.1}


def test_parse_aux_spec_rejects_unknown_head():
    with pytest.raises(ValueError, match="неизвестная aux-голова"):
        parse_aux_spec({"perimetr": 0.1})


def test_parse_aux_spec_of_nothing_is_empty():
    assert parse_aux_spec(None) == {} and parse_aux_spec({}) == {}


# --- модель ----------------------------------------------------------------

def tiny_model(aux):
    return build_model({
        "backend": "smp", "arch": "unet", "encoder": "tu-resnet18",
        "encoder_weights": None, "cls_head": "aux", "aux_heads": aux,
    }).eval()


def test_model_emits_every_requested_head_with_right_shape():
    model = tiny_model({"area": 0.1, "border": 0.05, "centroid": 0.05, "components": 0.05})
    out = model(torch.randn(2, 3, 128, 128))

    assert out["logits"].shape == (2, 1, 128, 128)
    assert out["cls_logits"].shape == (2, 1)
    for name, spec in AUX_HEADS.items():
        assert out[f"aux_{name}"].shape == (2, spec.out_dim)


def test_model_without_aux_keeps_the_old_contract():
    out = tiny_model(None)(torch.randn(1, 3, 128, 128))
    assert set(out) == {"logits", "cls_logits"}


def test_aux_heads_start_at_zero():
    """На первом шаге головы не тянут представление никуда — как и зануленная
    проекция в fuse: gate и восстановленный стем."""
    model = tiny_model({"area": 0.1})
    out = model(torch.randn(2, 3, 128, 128))
    assert torch.allclose(out["aux_area"], torch.zeros_like(out["aux_area"]))


def test_model_survives_deepcopy_used_by_ema():
    """ModelEma копирует модель через deepcopy; хук на энкодере не должен
    утаскивать за собой всю модель и не должен разъезжаться с копией."""
    from copy import deepcopy

    model = tiny_model({"area": 0.1})
    clone = deepcopy(model).eval()
    x = torch.randn(1, 3, 128, 128)
    assert torch.allclose(model(x)["aux_area"], clone(x)["aux_area"])


# --- лосс ------------------------------------------------------------------

def make_batch(batch=2, size=32, empty=False):
    mask = torch.zeros(batch, 1, size, size)
    if not empty:
        mask[:, :, :8, :8] = 1.0
    area = mask.flatten(1).mean(dim=1, keepdim=True)
    return {
        "mask": mask,
        "label": (area > 0).float(),
        "area": area,
        "aux_border": torch.ones(batch, 1),
        "aux_centroid": torch.full((batch, 2), 0.1),
        "aux_components": torch.full((batch, 1), normalize_components(1)),
        "geom_valid": (area > 0).float(),
    }


def make_outputs(batch=2, size=32):
    return {
        "logits": torch.randn(batch, 1, size, size),
        "cls_logits": torch.randn(batch, 1),
        "aux_area": torch.randn(batch, 1),
        "aux_border": torch.randn(batch, 1),
        "aux_centroid": torch.randn(batch, 2),
        "aux_components": torch.randn(batch, 1),
    }


def test_loss_reports_every_aux_component():
    criterion = build_loss(
        {"seg": {"bce": 1.0, "dice": 1.0}, "cls_weight": 0.3},
        {"area": 0.1, "border": 0.05, "centroid": 0.05, "components": 0.05},
    )
    total, stats = criterion(make_outputs(), make_batch())
    for name in AUX_HEADS:
        assert f"aux_{name}" in stats
    assert torch.isfinite(total)


def test_aux_weight_actually_changes_the_total():
    outputs, batch = make_outputs(), make_batch()
    base = build_loss({"seg": {"bce": 1.0}}, None)(outputs, batch)[0]
    with_aux = build_loss({"seg": {"bce": 1.0}}, {"area": 0.5})(outputs, batch)[0]
    assert not torch.isclose(base, with_aux)


def test_positives_only_heads_ignore_clean_frames():
    """Для чистого кадра геометрии нет; лосс по ней обязан быть нулевым,
    а не учить модель предсказывать заглушку «центр кадра»."""
    criterion = build_loss({"seg": {"bce": 1.0}}, {"centroid": 1.0})
    _, stats = criterion(make_outputs(), make_batch(empty=True))
    assert stats["aux_centroid"] == pytest.approx(0.0)


def test_area_head_does_learn_from_clean_frames():
    """Площадь — исключение: у чистого кадра она честно нулевая, и это ровно
    тот сигнал, который проверяет FPR_neg."""
    criterion = build_loss({"seg": {"bce": 1.0}}, {"area": 1.0})
    _, stats = criterion(make_outputs(), make_batch(empty=True))
    assert stats["aux_area"] > 0.0


def test_dataset_emits_aux_targets_only_when_asked(tmp_path):
    """Сквозная проводка: датасет -> батч -> лосс. connectedComponents стоит
    ~1.5 мс на сэмпл, поэтому без запроса геометрия считаться не должна."""
    import numpy as np
    import pandas as pd

    from aic_pipeline.datasets import SegDataset
    from aic_pipeline.imageio import imwrite
    from aic_pipeline.transforms import build_val_transform
    from aic_pipeline import workspace

    root = tmp_path / "ds"
    (root / "img").mkdir(parents=True)
    (root / "mask").mkdir(parents=True)
    imwrite(root / "img" / "a.jpg", np.full((40, 60, 3), 120, np.uint8))
    mask = np.zeros((40, 60), np.uint8)
    mask[5:15, 5:25] = 255
    imwrite(root / "mask" / "a.png", mask)

    df = pd.DataFrame([{
        "chng_path": "img/a.jpg", "gt_path": "mask/a.png",
        "is_negative": False, "mask_area": 0.08, "orgl_path": None,
    }])

    original = workspace.dataset_root
    workspace.dataset_root = lambda: root
    try:
        import aic_pipeline.datasets as ds_module
        original_resolve = ds_module.resolve
        ds_module.resolve = lambda rel: root / str(rel)
        try:
            plain = SegDataset(df, build_val_transform(size=32))[0]
            rich = SegDataset(df, build_val_transform(size=32), aux_targets=True)[0]
        finally:
            ds_module.resolve = original_resolve
    finally:
        workspace.dataset_root = original

    assert not any(key.startswith("aux_") for key in plain)
    assert {"aux_border", "aux_centroid", "aux_components", "geom_valid"} <= set(rich)
    assert rich["geom_valid"].item() == 1.0
    assert rich["aux_centroid"].shape == (2,)


def test_loss_skips_heads_that_the_model_did_not_produce():
    """Старый чекпоинт без голов не должен ронять обучение с новым конфигом."""
    outputs = {"logits": torch.randn(2, 1, 32, 32), "cls_logits": torch.randn(2, 1)}
    total, stats = build_loss({"seg": {"bce": 1.0}}, {"area": 0.1})(outputs, make_batch())
    assert torch.isfinite(total)
    assert "aux_area" not in stats
