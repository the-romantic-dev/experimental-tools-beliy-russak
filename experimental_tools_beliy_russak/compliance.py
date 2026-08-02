"""Ограничения регламента, кроме бюджета вычислений (тот живёт в `budget.py`).

Здесь два запрета, устроенных совсем по-разному.

**Данные, кроме предоставленных.** Машинного вердикта тут быть не может: считать
ли предобучение на ImageNet «использованием стороннего датасета» — вопрос
трактовки, и задавать его надо организаторам, а не коду. Что делается машинно —
источник весов не даёт спрятаться. У timm набор, на котором предобучали, зашит
в тег имени модели, поэтому `tu-convnext_tiny.dinov3_lvd1689m` при
`encoder_weights: imagenet` на самом деле приносит веса от LVD-1689M, и по
конфигу этого не видно.

**Тестовая выборка в обучении.** Здесь вердикт однозначный, и держится он не на
намерении, а на достижимости: пути к тесту знает только код сборки посылки, и
обучающий путь физически не может до них добраться.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

#: как называются пути к тестовой выборке в `workspace.py`
TEST_SET_NAMES = ("test_csv", "test_root", "test_img_dir", "submission_template")

_TEST_SET_PATTERN = re.compile("|".join(TEST_SET_NAMES))


def pretrained_sources(cfg_model: Mapping[str, Any]) -> list[str]:
    """Внешние веса, которые притянет эта модель. Пусто = сеть учится с нуля.

    Возвращает описания вида `tu-convnext_tiny.dinov3_lvd1689m -> dinov3_lvd1689m`,
    где справа стоит набор, на котором предобучали.
    """
    sources: list[str] = []

    encoder = str(cfg_model.get("encoder") or "")
    weights = cfg_model.get("encoder_weights")
    if weights and encoder:
        # `convnext_tiny.dinov3_lvd1689m` — всё, что после точки, у timm называет
        # конкретный набор весов, и он важнее строки `imagenet` из конфига
        dataset = encoder.split(".", 1)[1] if "." in encoder else str(weights)
        sources.append(f"{encoder} -> {dataset}")

    if str(cfg_model.get("backend", "")).lower() == "hf":
        # у hf-бэкенда `from_pretrained` тянет веса всегда, независимо от
        # `encoder_weights` — иначе этот источник в инвентаре бы не появился
        sources.append(f"{cfg_model.get('hf_name')} -> веса чекпоинта Hugging Face")

    return sources


def package_root() -> Path:
    return Path(__file__).resolve().parent


def modules_touching_test_set() -> list[Path]:
    """Модули пакета, которые вообще упоминают пути к тестовой выборке.

    Проверка текстовая, и это осознанно: она отвечает на вопрос «может ли этот
    код дотянуться до теста», а не «дотягивается ли прямо сейчас». Обучающий
    модуль, в котором появилось само имя `test_csv`, уже повод разбираться,
    даже если вызов пока закомментирован.
    """
    found = []
    for path in sorted(package_root().rglob("*.py")):
        if _TEST_SET_PATTERN.search(path.read_text(encoding="utf-8")):
            found.append(path)
    return found
