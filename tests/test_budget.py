"""Бюджет вычислений: не больше 100 строгих GFLOPs на одно изображение.

Главная ловушка регламента вынесена в первый же тест. fvcore, thop и ptflops
пишут в выводе «FLOPs», а считают MACs — вдвое меньше. Решение, собранное по
их числу, укладывается в лимит только на бумаге. Поэтому здесь проверяется не
«счётчик что-то посчитал», а ровно то, что он даёт 2 FLOPs на одно
умножение-сложение.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import pytest
import torch

from experimental_tools_beliy_russak.budget import (
    LIMIT_GFLOPS, check, check_submission, config_gflops, count_gflops,
    inference_gflops, largest_fitting_size, rejection_text,
)
from experimental_tools_beliy_russak.config import load_config
from experimental_tools_beliy_russak.models import build_model


def smoke_config(*overrides: str):
    """Самый дешёвый конфиг репозитория; веса не тянем из сети."""
    return load_config("smoke", ["model.encoder_weights=null", *overrides])


def test_counter_reports_strict_flops_not_macs():
    conv = torch.nn.Conv2d(3, 8, kernel_size=3, padding=1, bias=False)
    macs = 3 * 8 * 3 * 3 * 32 * 32
    assert count_gflops(conv, size=32) == pytest.approx(2 * macs / 1e9)


def test_counting_from_config_matches_counting_the_built_model():
    """Счёт по конфигу идёт на meta-устройстве — число обязано совпасть с настоящим."""
    cfg = smoke_config()
    built = build_model(cfg.model).eval()
    assert config_gflops(cfg.model, 256) == pytest.approx(count_gflops(built, 256))


def test_inference_cost_takes_the_input_size_from_the_config():
    assert inference_gflops(smoke_config()) == pytest.approx(
        config_gflops(smoke_config().model, 256)
    )


def test_tta_and_ensemble_multiply_the_cost():
    """Лимит задан на изображение, а не на forward: четыре вида TTA стоят вчетверо."""
    cfg = smoke_config()
    one = inference_gflops(cfg)
    assert inference_gflops(cfg, n_views=4) == pytest.approx(4 * one)
    assert inference_gflops(cfg, n_models=2) == pytest.approx(2 * one)
    assert inference_gflops(cfg, n_views=2, n_models=3) == pytest.approx(6 * one)


# --- вердикт ----------------------------------------------------------------

def test_verdict_separates_a_config_inside_the_limit_from_one_outside():
    assert check(smoke_config()).ok

    verdict = check(smoke_config("data.size=1024"))
    assert not verdict.ok
    assert verdict.gflops > LIMIT_GFLOPS


def test_tta_can_push_a_config_that_fits_alone_out_of_the_budget():
    cfg = smoke_config("data.size=512")
    assert check(cfg).ok
    assert not check(cfg, n_views=4).ok


def test_exempt_lets_a_research_config_run_but_never_reaches_a_submission():
    """`budget.exempt` — пометка «прогон исследовательский», а не поднятый лимит.

    Сабмит проверяется против настоящих 100 GFLOPs всегда: иначе пометка,
    поставленная ради одного эксперимента, однажды тихо уехала бы в посылку.
    """
    cfg = smoke_config("data.size=1024", "budget.exempt=true")

    assert check(cfg).ok
    assert check(cfg).exempt
    assert not check(cfg, allow_exempt=False).ok


# --- какой размер входа ещё влезает -----------------------------------------

def test_largest_fitting_size_is_the_biggest_multiple_of_32_inside_the_budget():
    cfg = smoke_config()
    size = largest_fitting_size(cfg)

    assert size % 32 == 0
    assert config_gflops(cfg.model, size) <= LIMIT_GFLOPS
    assert config_gflops(cfg.model, size + 32) > LIMIT_GFLOPS


def test_largest_fitting_size_shrinks_when_tta_is_paid_for():
    cfg = smoke_config()
    assert largest_fitting_size(cfg, n_views=4) < largest_fitting_size(cfg)


def test_largest_fitting_size_is_none_when_nothing_fits():
    """Сеть, которая не влезает даже на минимальном входе, должна сказать это прямо."""
    assert largest_fitting_size(smoke_config(), n_views=10_000) is None


def test_rejection_says_what_input_would_fit():
    """Отказ без ответа «а на чём тогда» заставляет подбирать размер вручную."""
    cfg = smoke_config("data.size=1024")
    text = rejection_text(cfg, check(cfg))

    assert f"{largest_fitting_size(cfg)}px" in text
    assert "budget.exempt" in text


# --- гейт на входе в обучение -----------------------------------------------

def test_train_refuses_an_over_budget_config_before_touching_anything(tmp_path):
    """Отказ обязан случиться до данных, до карты и до создания папки прогона.

    Иначе проверка стоила бы загрузки датасета, а в `runs/` копились бы пустые
    папки прогонов, которые никогда не стартовали.
    """
    from experimental_tools_beliy_russak.train import run
    from experimental_tools_beliy_russak.workspace import use_workspace

    cfg = smoke_config("data.size=1024", "name=не-должен-появиться")
    with use_workspace(tmp_path):
        with pytest.raises(ValueError, match="GFLOPs"):
            run(cfg)
        assert not (tmp_path / "runs").exists()


def test_verdict_carries_the_fields_that_land_in_summary():
    """`board` читает бюджет из summary.json — состав полей часть контракта."""
    assert set(check(smoke_config()).as_dict()) == {
        "gflops", "limit_gflops", "within_limit", "exempt",
    }


# --- бюджет готовой посылки -------------------------------------------------

def test_submission_budget_sums_the_models_of_the_ensemble():
    """Модели ансамбля могут быть разными, поэтому именно сумма, а не множитель."""
    cheap, dear = smoke_config(), smoke_config("data.size=384")
    assert check_submission([cheap, dear]).gflops == pytest.approx(
        inference_gflops(cheap) + inference_gflops(dear)
    )


def test_submission_budget_never_honours_the_exempt_mark():
    """Пометка про эксперимент не должна тихо уехать в посылку."""
    cfg = smoke_config("data.size=1024", "budget.exempt=true")
    assert check(cfg).ok
    assert not check_submission([cfg]).ok


def test_submission_budget_counts_tta_views():
    cfg = smoke_config("data.size=512")
    assert check_submission([cfg]).ok
    assert not check_submission([cfg], n_views=4).ok
