"""Общая настройка тестов.

Воркспейсом всегда считается корень репозитория. Без этого результат зависел бы
от того, из какой папки запущен pytest: `workspace()` по умолчанию ищет
ближайшую вверх папку с `configs/`, а из чужой директории не нашёл бы ничего.

Прибивается на уровне модуля, а не фикстурой: `test_configs_build.py` собирает
список конфигов при импорте, то есть до того, как отработала бы любая фикстура.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from aic_pipeline.workspace import set_workspace

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

set_workspace(REPO_ROOT)


@pytest.fixture(autouse=True)
def _isolate_workspace():
    """Тест, подменивший воркспейс, не должен утащить подмену в следующий."""
    yield
    set_workspace(REPO_ROOT)


@pytest.fixture(autouse=True)
def _isolate_registries():
    """То же для реестров: зарегистрированная в тесте компонента не живёт дальше.

    Иначе `@register_loss("мой")` из одного теста меняла бы вывод `aic registry`
    в другом, и порядок тестов начал бы влиять на результат.
    """
    from aic_pipeline.registry import sandbox

    with sandbox():
        yield
