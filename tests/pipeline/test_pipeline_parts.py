"""Контракты модели, лоссов, датасета и постобработки — на синтетике, без данных."""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest
import torch

from experimental_tools_beliy_russak.datasets import build_balanced_sampler, build_sampler
from experimental_tools_beliy_russak.inference import postprocess
from experimental_tools_beliy_russak.losses import build_loss, soft_dice_loss, tversky_loss
from experimental_tools_beliy_russak.models import build_model
from experimental_tools_beliy_russak.transforms import build_train_transform, build_val_transform


@pytest.fixture(scope="module")
def tiny_model():
    return build_model({
        "backend": "smp", "arch": "unet",
        "encoder": "tu-resnet18", "encoder_weights": None, "cls_head": "aux",
    })


def test_model_output_contract(tiny_model):
    out = tiny_model(torch.randn(2, 3, 128, 128))
    assert set(out) == {"logits", "cls_logits"}
    assert out["logits"].shape == (2, 1, 128, 128)
    assert out["cls_logits"].shape == (2, 1)


def test_logits_cls_head_backend_also_matches_contract():
    model = build_model({
        "backend": "smp", "arch": "fpn",
        "encoder": "tu-resnet18", "encoder_weights": None, "cls_head": "from_logits",
    })
    out = model(torch.randn(1, 3, 128, 128))
    assert out["logits"].shape == (1, 1, 128, 128)
    assert out["cls_logits"].shape == (1, 1)


def test_dice_loss_is_zero_for_two_empty_masks():
    """Иначе каждый чистый кадр становится постоянным штрафом и модель
    учится рисовать маску даже там, где её нет."""
    logits = torch.full((2, 1, 8, 8), -20.0)
    targets = torch.zeros(2, 1, 8, 8)
    assert float(soft_dice_loss(logits, targets)) == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_is_zero_for_perfect_prediction():
    targets = torch.zeros(1, 1, 8, 8)
    targets[..., :4, :4] = 1.0
    logits = torch.where(targets > 0, torch.tensor(20.0), torch.tensor(-20.0))
    assert float(soft_dice_loss(logits, targets)) == pytest.approx(0.0, abs=1e-3)


def test_tversky_beta_penalises_false_negatives_more():
    targets = torch.zeros(1, 1, 10, 10)
    targets[..., :5, :] = 1.0
    miss = torch.full((1, 1, 10, 10), -5.0)          # всё пропустили
    over = torch.full((1, 1, 10, 10), 5.0)           # закрасили весь кадр
    recall_biased = (0.1, 0.9)
    assert float(tversky_loss(miss, targets, *recall_biased)) > float(
        tversky_loss(over, targets, *recall_biased)
    )


def test_combined_loss_reports_all_components():
    criterion = build_loss({
        "seg": {"bce": 1.0, "dice": 1.0, "focal": 0.5}, "cls_weight": 0.3,
    })
    outputs = {"logits": torch.randn(2, 1, 8, 8), "cls_logits": torch.randn(2, 1)}
    batch = {"mask": torch.zeros(2, 1, 8, 8), "label": torch.zeros(2, 1)}
    total, parts = criterion(outputs, batch)
    assert set(parts) == {"bce", "dice", "focal", "cls"}
    assert torch.isfinite(total)


def test_loss_rejects_empty_configuration():
    with pytest.raises(ValueError, match="ни одной компоненты"):
        build_loss({"seg": {"bce": 0.0, "dice": 0.0}})


def test_transforms_produce_expected_shapes():
    image = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
    mask = np.zeros((300, 400), dtype=np.float32)

    train = build_train_transform(size=192, preset="heavy")(image=image, mask=mask)
    assert train["image"].shape == (3, 192, 192)
    assert train["mask"].shape == (192, 192)

    for mode in ("resize", "pad"):
        val = build_val_transform(size=192, mode=mode)(image=image, mask=mask)
        assert val["image"].shape == (3, 192, 192)


def test_pad_mode_keeps_content_in_top_left_corner():
    """От этого зависит обратное преобразование в inference._to_original."""
    image = np.full((100, 400, 3), 255, dtype=np.uint8)
    mask = np.zeros((100, 400), dtype=np.float32)
    out = build_val_transform(size=200, mode="pad")(image=image, mask=mask)["image"]
    assert out[:, :40, :].abs().sum() > 0                 # контент сверху
    # снизу — ровный паддинг: внутри канала одно значение (после Normalize
    # нулевой пиксель превращается в свою константу для каждого канала)
    assert out[:, 120:, :].std(dim=(1, 2)).max() < 1e-4


def test_postprocess_applies_all_rules():
    prob = np.zeros((100, 100), dtype=np.float32)
    prob[:20, :20] = 0.9  # 4% кадра

    assert postprocess(prob, 1.0, mask_threshold=0.5).sum() == 400 * 255
    assert postprocess(prob, 1.0, mask_threshold=0.95).sum() == 0
    assert postprocess(prob, 0.1, mask_threshold=0.5, cls_threshold=0.5).sum() == 0
    assert postprocess(prob, 1.0, mask_threshold=0.5, min_area=0.05).sum() == 0
    assert set(np.unique(postprocess(prob, 1.0))) <= {0, 255}


class FakeDataset:
    is_negative = np.array([True] * 10 + [False] * 90)

    def __len__(self):
        return 100


def test_balanced_sampler_hits_requested_negative_share():
    sampler = build_balanced_sampler(FakeDataset(), negative_fraction=0.25, num_samples=20000)
    drawn = np.array(list(sampler))
    share = FakeDataset.is_negative[drawn].mean()
    assert share == pytest.approx(0.25, abs=0.02)


def test_epoch_size_sets_epoch_length_without_touching_the_pool():
    """epoch_size — это длина эпохи, а не урезание датасета: сэмплирование
    идёт с возвращением по всем 100 элементам."""
    sampler = build_sampler(FakeDataset(), {"negative_fraction": 0.25, "epoch_size": 30})
    drawn = list(sampler)
    assert len(drawn) == 30
    assert max(drawn) <= 99

    long_run = list(build_sampler(FakeDataset(), {"negative_fraction": 0.25, "epoch_size": 5000}))
    assert len(set(long_run)) == 100  # за длинную эпоху всплывает весь пул


def test_epoch_size_works_without_negative_balancing():
    """Регресс: раньше epoch_size молча игнорировался при negative_fraction=null."""
    sampler = build_sampler(FakeDataset(), {"negative_fraction": None, "epoch_size": 42})
    assert len(list(sampler)) == 42


def test_no_sampler_when_neither_option_is_set():
    assert build_sampler(FakeDataset(), {}) is None
    assert build_sampler(FakeDataset(), {"negative_fraction": None, "epoch_size": None}) is None
