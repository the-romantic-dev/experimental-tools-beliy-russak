"""Термы `losses_s3.py`: ML/CAML и серия L.

Терм, который всегда отдаёт ноль или NaN, неотличим на глаз от терма, которому
нечего лечить, — отсюда прямые числовые проверки формул, а не только "не упало".
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import torch
import pytest

from aic_pipeline.losses_s3 import (
    caml_loss,
    compute_loss,
    multiplicative_loss,
    soft_dice_from_probs,
)


def random_batch(bs=2, size=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(bs, 1, size, size, generator=g)
    targets = (torch.rand(bs, 1, size, size, generator=g) > 0.5).float()
    return logits, targets


# --- multiplicative_loss -----------------------------------------------------

def test_multiplicative_loss_is_scalar_and_finite():
    logits, targets = random_batch()
    loss = multiplicative_loss(logits, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_multiplicative_loss_on_empty_mask_is_finite():
    logits, _ = random_batch()
    targets = torch.zeros_like(logits)
    loss = multiplicative_loss(logits, targets)
    assert torch.isfinite(loss)


def test_multiplicative_loss_vanishes_when_the_model_is_confidently_right():
    _, targets = random_batch()
    logits = (targets * 2.0 - 1.0) * 20.0  # сигмоида даёт ~0/~1 точно по GT
    loss = multiplicative_loss(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_multiplicative_loss_matches_dice_times_ce_by_hand():
    """ML — произведение, не сумма: считаем то же самое отдельно и сверяем."""
    logits, targets = random_batch(size=8)
    epsilon = 1e-7
    probs = torch.sigmoid(logits).clamp(epsilon, 1.0 - epsilon)
    ce = -(targets * probs.log() + (1.0 - targets) * (1.0 - probs).log()).mean()
    dice = soft_dice_from_probs(probs, targets, smooth=epsilon)
    expected = dice * ce
    assert multiplicative_loss(logits, targets, epsilon).item() == pytest.approx(
        expected.item(), abs=1e-5
    )


def test_multiplicative_loss_gradient_is_finite():
    logits, targets = random_batch()
    logits = logits.requires_grad_(True)
    multiplicative_loss(logits, targets).backward()
    assert torch.isfinite(logits.grad).all()


# --- caml_loss ----------------------------------------------------------------

def test_caml_loss_is_scalar_and_finite():
    logits, targets = random_batch()
    loss = caml_loss(logits, targets)
    assert loss.shape == ()
    assert torch.isfinite(loss)


def test_caml_loss_on_empty_mask_is_finite():
    logits, _ = random_batch()
    targets = torch.zeros_like(logits)
    loss = caml_loss(logits, targets)
    assert torch.isfinite(loss)


def test_caml_loss_vanishes_when_the_model_is_confidently_right():
    _, targets = random_batch()
    logits = (targets * 2.0 - 1.0) * 20.0
    loss = caml_loss(logits, targets)
    assert loss.item() == pytest.approx(0.0, abs=1e-3)


def test_caml_loss_gradient_is_finite():
    logits, targets = random_batch()
    logits = logits.requires_grad_(True)
    caml_loss(logits, targets).backward()
    assert torch.isfinite(logits.grad).all()


def test_caml_loss_gradient_does_not_flow_through_alpha():
    """Без `.detach()` на p̄/D/alpha обучение раскачивается (см. докстринг
    `caml_loss`). Эталон — та же формула, но alpha явно зафиксирована питоновским
    числом до backward: градиент через неё физически пройти не может. Если
    когда-нибудь `.detach()` в `caml_loss` потеряется, эти градиенты разойдутся.
    """
    logits, targets = random_batch(size=8)
    logits = logits.requires_grad_(True)
    caml_loss(logits, targets).backward()
    grad_actual = logits.grad.clone()

    reference = logits.detach().clone().requires_grad_(True)
    epsilon = 1e-7
    probs = torch.sigmoid(reference).clamp(epsilon, 1.0 - epsilon)
    ce = -(targets * probs.log() + (1.0 - targets) * (-probs).log1p()).mean()
    dice = soft_dice_from_probs(probs, targets, smooth=epsilon)
    with torch.no_grad():
        alpha = float((1.0 - probs.mean()) ** (1.0 - dice))
    loss_ref = dice * ce.clamp(min=1e-6) ** alpha
    loss_ref.backward()

    assert torch.allclose(grad_actual, reference.grad, atol=1e-6)


# --- compute_loss integration --------------------------------------------------

def make_outputs(bs=4, size=640, seed=1):
    g = torch.Generator().manual_seed(seed)
    return {
        "logits": torch.randn(bs, 1, size, size, generator=g),
        "cls_logits": torch.randn(bs, 1, generator=g),
    }


def make_batch(bs=4, size=640, seed=2):
    g = torch.Generator().manual_seed(seed)
    mask = (torch.rand(bs, 1, size, size, generator=g) > 0.5).float()
    label = (mask.flatten(1).amax(1, keepdim=True) > 0).float()
    return {"mask": mask, "label": label}


def test_compute_loss_defaults_to_bce_plus_dice():
    total, parts = compute_loss(make_outputs(bs=2, size=32), make_batch(bs=2, size=32), {})
    assert torch.isfinite(total)
    assert {"bce", "dice", "cls"} <= set(parts)
    assert "ml" not in parts and "caml" not in parts


def test_compute_loss_with_multiplicative_flag():
    cfg = {"use_multiplicative": True}
    total, parts = compute_loss(make_outputs(bs=4, size=640), make_batch(bs=4, size=640), cfg)
    assert torch.isfinite(total)
    assert "ml" in parts and "cls" in parts
    assert "bce" not in parts and "dice" not in parts and "caml" not in parts


def test_compute_loss_with_caml_flag():
    cfg = {"use_caml": True}
    total, parts = compute_loss(make_outputs(bs=4, size=640), make_batch(bs=4, size=640), cfg)
    assert torch.isfinite(total)
    assert "caml" in parts and "cls" in parts
    assert "bce" not in parts and "dice" not in parts and "ml" not in parts


def test_compute_loss_prefers_caml_when_both_flags_are_set():
    cfg = {"use_caml": True, "use_multiplicative": True}
    _, parts = compute_loss(make_outputs(bs=2, size=32), make_batch(bs=2, size=32), cfg)
    assert "caml" in parts and "ml" not in parts


def test_compute_loss_gate_stays_additive_under_caml():
    """Гейт (`cls`, вес 0.3) не входит в произведение ML/CAML — он всегда
    прибавляется отдельно, поэтому total не может быть меньше 0.3*cls при
    неотрицательном seg-лоссе."""
    outputs, batch = make_outputs(bs=2, size=32), make_batch(bs=2, size=32)
    total, parts = compute_loss(outputs, batch, {"use_caml": True})
    assert total.item() >= 0.3 * parts["cls"].item() - 1e-5
