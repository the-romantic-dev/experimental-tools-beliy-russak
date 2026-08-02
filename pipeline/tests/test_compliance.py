"""Ограничения регламента, которые проверяются не арифметикой, а структурой кода.

Бюджет FLOPs считается и лежит в test_budget.py. Здесь два других запрета:

* данные, кроме предоставленных. Автоматического вердикта тут быть не может —
  вопрос трактовки, считается ли предобучение на ImageNet «использованием
  стороннего датасета». Что можно сделать машинно — не дать источнику весов
  спрятаться: у timm набор, на котором предобучали, зашит в тег имени, и по
  строке `encoder_weights: imagenet` не видно, что за ней стоит LVD-1689M;
* тестовая выборка в обучении. Здесь вердикт как раз однозначный, и держится он
  на том, что пути к тесту знает только код сборки посылки.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

from pathlib import Path

import pytest

from aic_pipeline.compliance import (
    modules_touching_test_set, pretrained_sources,
)

# --- откуда берутся веса ----------------------------------------------------


def test_pretrained_sources_exposes_the_dataset_hidden_in_a_timm_tag():
    """`convnext_tiny.dinov3_lvd1689m` предобучен НЕ на ImageNet.

    В конфиге при этом написано `encoder_weights: imagenet`, и по нему разница
    не видна вообще — а для «нельзя использовать другие датасеты» она решающая.
    """
    sources = pretrained_sources({
        "backend": "smp", "encoder": "tu-convnext_tiny.dinov3_lvd1689m",
        "encoder_weights": "imagenet",
    })
    assert any("dinov3_lvd1689m" in source for source in sources)


def test_pretrained_sources_reports_plain_imagenet_too():
    sources = pretrained_sources({
        "backend": "smp", "encoder": "tu-resnet34", "encoder_weights": "imagenet",
    })
    assert any("imagenet" in source for source in sources)


def test_pretrained_sources_is_empty_for_a_network_trained_from_scratch():
    assert pretrained_sources({
        "backend": "smp", "encoder": "tu-resnet18", "encoder_weights": None,
    }) == []


def test_pretrained_sources_reports_the_huggingface_checkpoint():
    """У hf-бэкенда веса приезжают всегда, `encoder_weights` тут ни при чём."""
    sources = pretrained_sources({
        "backend": "hf", "hf_name": "nvidia/mit-b0", "encoder_weights": None,
    })
    assert any("nvidia/mit-b0" in source for source in sources)


# --- тестовая выборка не участвует в обучении -------------------------------


#: пути к тесту нужны ровно трём местам: тому, кто их знает, тому, кто собирает
#: посылку, и командам CLI. Любой четвёртый модуль — повод разбираться
ALLOWED = {"workspace.py", "submission.py", "cli.py", "compliance.py"}


def test_only_the_submission_path_knows_where_the_test_set_lies():
    """Псевдо-разметка запрещена регламентом, и цена нарушения — ноль за этап.

    Проверяется не намерение, а достижимость: если про тестовые пути знает
    только сборка посылки, то ни индексация, ни фолды, ни даталоадер физически
    не могут до них добраться.
    """
    touching = {path.name for path in modules_touching_test_set()}
    assert touching <= ALLOWED, (
        f"тестовую выборку упоминают лишние модули: {sorted(touching - ALLOWED)}. "
        "Если это обучающий путь — регламент запрещает такое прямо"
    )


def test_the_leak_check_actually_looks_at_the_package(tmp_path):
    """Тест выше зелёный и когда проверка ничего не нашла — это надо исключить."""
    assert modules_touching_test_set(), "проверка не нашла даже submission.py"
