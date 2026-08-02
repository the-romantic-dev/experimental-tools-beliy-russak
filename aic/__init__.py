"""`aic` — инструменты для задачи AI Challenge: метрика, статистика, бюджет.

Библиотека не регламентирует способ обучения. Цикл, модель, лоссы и загрузка
данных — ваши; здесь то, что нужно всем и одинаково:

    import aic

    ws = aic.Workspace.find()
    run = aic.Run.create(ws.runs, "my-unet-768")
    run.log(epoch, {"val/aic_tuned": 0.71, "samples": 8000})

Две вещи этот модуль обязан сделать раньше остальных.

**Переменные среды.** Выставляются до первого импорта torch, numpy и cv2: в
conda-среде `challenges` живут две копии OpenMP, и при совместном импорте torch
и numpy интерпретатор падает с OMP: Error #15. Любой вход в пакет проходит через
этот файл первым.

**Ленивость.** Публичные имена отдаются через `__getattr__` (PEP 562). Жадный
импорт потянул бы pandas и cv2 на каждое обращение, а `budget` — ещё и torch,
которого в среде может не быть вовсе.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# albumentations иначе ходит в сеть на каждом импорте и печатает баннер
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# cv2 внутри воркеров плодит потоки и душит CPU
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import importlib  # noqa: E402 — только после переменных среды

__version__ = "0.1.0"

#: имя на верхнем уровне -> "модуль" или "модуль:имя_внутри"
_EXPORTS: dict[str, str] = {
    # где что лежит
    "Workspace": "paths",
    # метрика
    "AICAccumulator": "metric",
    "AICResult": "metric",
    "score_masks": "metric",
    "harmonic_aic": "metric",
    "dice_binary": "metric",
    # прогон
    "Run": "runs",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(
            f"в {__name__} нет имени {name!r}; доступны: {', '.join(__all__)}"
        )
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attribute or name)
    globals()[name] = value  # второй доступ уже без importlib
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
