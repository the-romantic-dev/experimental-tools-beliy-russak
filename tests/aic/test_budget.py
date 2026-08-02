"""Бюджет вычислений: строгие FLOPs, а не MACs."""

from __future__ import annotations

import pytest

# `aic` импортируется ПЕРВЫМ и до торча — иначе интерпретатор просто падает.
# В conda-среде `challenges` две копии OpenMP, и torch, потянувший numpy раньше,
# чем выставлены переменные среды из aic/__init__.py, обрывает процесс на
# numpy.blas_fpe_check без всякого исключения. Порядок здесь — не стиль.
from aic.budget import (  # noqa: E402
    LIMIT_GFLOPS,
    check,
    count_gflops,
    largest_fitting_size,
    rejection_text,
)

torch = pytest.importorskip("torch", reason="бюджет требует torch")
import torch.nn as nn  # noqa: E402


class Conv(nn.Module):
    """Одна свёртка: её FLOPs считаются на бумаге и проверяются точно."""

    def __init__(self, cin=3, cout=16, k=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)

    def forward(self, x):
        return self.conv(x)


def test_count_gflops_counts_strict_flops_not_macs():
    """MAC = две операции. Счётчики, пишущие MACs, дали бы вдвое меньше."""
    model = Conv(3, 16, 3).eval()
    size = 64
    macs = 3 * 16 * 3 * 3 * size * size
    assert count_gflops(model, size) == pytest.approx(2 * macs / 1e9, rel=1e-6)


def test_count_gflops_scales_quadratically_with_side():
    model = Conv().eval()
    assert count_gflops(model, 128) == pytest.approx(4 * count_gflops(model, 64), rel=1e-6)


def test_count_gflops_works_on_the_meta_device():
    """FlopCounterMode работает на уровне диспетчера: настоящих тензоров не надо."""
    model = Conv().eval().to("meta")
    assert count_gflops(model, 64) > 0


def test_check_says_within_limit_for_a_tiny_model():
    verdict = check(Conv().eval(), 64)
    assert verdict.within_limit is True
    assert verdict.ok is True
    assert "в бюджете" in verdict.text
    assert verdict.limit == LIMIT_GFLOPS


def test_tta_multiplies_the_cost():
    model = Conv().eval()
    one = check(model, 64)
    two = check(model, 64, n_views=2)
    assert two.gflops == pytest.approx(2 * one.gflops)
    assert "x2 видов TTA" in two.text


def test_an_ensemble_is_summed_not_multiplied():
    """Модели ансамбля могут быть разными, и у каждой своя цена."""
    small, big = Conv(3, 8, 3).eval(), Conv(3, 32, 3).eval()
    together = check([small, big], 64)
    assert together.gflops == pytest.approx(
        count_gflops(small, 64) + count_gflops(big, 64)
    )
    assert together.n_models == 2
    assert "x2 моделей" in together.text


def test_check_rejects_what_does_not_fit():
    verdict = check(Conv(3, 512, 7).eval(), 512, limit=1.0)
    assert verdict.within_limit is False
    assert verdict.ok is False
    assert "превышение" in verdict.text


def test_as_dict_is_json_ready():
    payload = check(Conv().eval(), 64).as_dict()
    assert set(payload) == {"gflops", "limit_gflops", "within_limit"}


def test_largest_fitting_size_finds_a_multiple_of_step():
    def build(size):
        return Conv(3, 64, 3).eval()

    got = largest_fitting_size(build, start=512, limit=5.0)
    assert got is not None and got % 32 == 0
    assert count_gflops(build(got), got) <= 5.0
    assert count_gflops(build(got + 32), got + 32) > 5.0


def test_largest_fitting_size_is_none_when_nothing_fits():
    assert largest_fitting_size(lambda size: Conv(3, 512, 7).eval(), limit=1e-9) is None


def test_the_builder_receives_the_size():
    """Энкодеры с оконным вниманием собираются только под свой img_size."""
    seen = []

    def build(size):
        seen.append(size)
        return Conv().eval()

    largest_fitting_size(build, start=256, limit=5.0)
    assert seen and all(isinstance(s, int) for s in seen)


def test_rejection_text_answers_the_next_question():
    verdict = check(Conv(3, 256, 5).eval(), 512, limit=2.0)
    text = rejection_text(lambda size: Conv(3, 256, 5).eval(), verdict)
    assert "превышение" in text
    assert "укладывается вход" in text or "ни одно разрешение" in text
