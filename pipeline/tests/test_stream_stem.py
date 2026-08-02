"""Стем при `fuse: input` обязан стартовать эквивалентно RGB-baseline.

Первопричина провала прогона e3-srm-input (AIC 0.614 против 0.761 у контроля):
timm адаптирует предобученный стем под 9 входных каналов, копируя блок RGB-весов
три раза и деля всё на 1/3. Отклик на RGB оказывается втрое слабее, а две трети
веса стема получают шумовой остаток, пропущенный через цветовые фильтры ImageNet.
Замер показал расхождение признаков энкодера с baseline на 37% ещё до обучения.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import pytest
import torch
import torch.nn as nn

from aic_pipeline.models import build_model, find_stem_conv, restore_pretrained_stem
from aic_pipeline.streams import InputFusion


def emulate_timm_adaptation(weight3: torch.Tensor, in_channels: int) -> torch.Tensor:
    """Ровно то, что делает timm.adapt_input_conv для in_chans != 3."""
    repeat = -(-in_channels // 3)
    weight = weight3.repeat(1, repeat, 1, 1)[:, :in_channels]
    return weight * (3.0 / in_channels)


def test_restore_recovers_rgb_block_and_zeroes_the_rest():
    torch.manual_seed(0)
    original = torch.randn(16, 3, 4, 4)

    conv = nn.Conv2d(9, 16, 4, bias=False)
    with torch.no_grad():
        conv.weight.copy_(emulate_timm_adaptation(original, 9))

    encoder = nn.Sequential(conv)
    encoder.out_channels = [9]

    assert restore_pretrained_stem(encoder, in_channels=9) is True
    weight = conv.weight.data
    assert torch.allclose(weight[:, :3], original, atol=1e-6)
    assert torch.count_nonzero(weight[:, 3:]) == 0


def test_restore_is_exact_for_non_multiple_channel_counts():
    torch.manual_seed(1)
    original = torch.randn(8, 3, 3, 3)
    conv = nn.Conv2d(7, 8, 3, bias=False)
    with torch.no_grad():
        conv.weight.copy_(emulate_timm_adaptation(original, 7))

    encoder = nn.Sequential(conv)
    assert restore_pretrained_stem(encoder, in_channels=7) is True
    assert torch.allclose(conv.weight.data[:, :3], original, atol=1e-6)
    assert torch.count_nonzero(conv.weight.data[:, 3:]) == 0


def test_find_stem_conv_picks_the_layer_that_sees_the_input():
    """Ищем по факту вызова, а не по имени и не по порядку в modules()."""
    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.later = nn.Conv2d(4, 4, 3, padding=1)   # зарегистрирован первым
            self.stem = nn.Conv2d(9, 4, 3, padding=1)    # но вызывается позже

        def forward(self, x):
            return self.later(self.stem(x))

    net = Net()
    assert find_stem_conv(net, in_channels=9) is net.stem


def test_find_stem_conv_returns_none_when_nothing_matches():
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    assert find_stem_conv(net, in_channels=9) is None


def test_restore_reports_false_when_stem_not_found():
    net = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1))
    assert restore_pretrained_stem(net, in_channels=9) is False


def test_stream_model_without_pretrained_weights_still_builds():
    model = build_model({
        "backend": "smp", "arch": "unet", "encoder": "tu-resnet18",
        "encoder_weights": None, "cls_head": "aux",
        "stream": "srm_bayar", "fuse": "input",
    }).eval()
    out = model(torch.randn(1, 3, 128, 128))
    assert out["logits"].shape == (1, 1, 128, 128)
    assert out["cls_logits"].shape == (1, 1)


@pytest.mark.parametrize("encoder", ["tu-resnet18"])
def test_stream_model_starts_equal_to_rgb_baseline(encoder):
    """Ключевая проверка: на шаге 0 поток не должен менять признаки энкодера.

    Веса предобученные, поэтому тест требует сети при первом запуске; если их
    нет — пропускаем, а не падаем.
    """
    import segmentation_models_pytorch as smp

    try:
        baseline = smp.create_model(
            "unet", encoder, encoder_weights="imagenet", in_channels=3, classes=1
        ).eval()
    except Exception as exc:  # нет сети или кэша весов
        pytest.skip(f"предобученные веса недоступны: {exc}")

    model = build_model({
        "backend": "smp", "arch": "unet", "encoder": encoder,
        "encoder_weights": "imagenet", "cls_head": "aux",
        "stream": "srm_bayar", "fuse": "input",
    }).eval()

    torch.manual_seed(0)
    x = torch.randn(2, 3, 128, 128)
    fusion = InputFusion(("srm", "bayar")).eval()

    with torch.no_grad():
        expected = [f for f in baseline.encoder(x) if f.numel()][1:]
        got = [f for f in model.core.encoder(fusion(x)) if f.numel()][1:]

    assert len(expected) == len(got) and expected
    for stage, (a, b) in enumerate(zip(expected, got)):
        assert torch.allclose(a, b, atol=1e-5), f"стадия {stage} разошлась с baseline"
