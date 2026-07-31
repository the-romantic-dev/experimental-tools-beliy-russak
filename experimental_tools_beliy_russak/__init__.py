"""Тулкит экспериментов «Белый русак» — сегментация подделанных областей.

Соглашение об импорте:

    import experimental_tools_beliy_russak as etbr

    cfg = etbr.load_config("baseline", ["train.lr=1e-4"])
    etbr.set_workspace("D:/projects/next-competition")

Свои лоссы, аугментации и оптимизаторы добавляются без правки библиотеки —
файлом `aic_plugins.py` в корне воркспейса, см. `registry.py` и `docs/EXTENDING.md`.

Две вещи, которые этот модуль обязан сделать раньше всех остальных.

**Переменные среды.** Они выставляются до первого импорта torch/numpy/cv2:
в conda-среде `challenges` живут две копии OpenMP, и при совместном импорте
torch и numpy интерпретатор падает с OMP: Error #15. Поэтому любой вход в
пакет — хоть `etbr.load_config`, хоть `from etbr.train import run` — проходит
через этот файл первым.

**Ленивость.** Публичные имена отдаются через `__getattr__` (PEP 562), а не
импортируются сразу. Жадный импорт потянул бы torch, timm, smp и albumentations
и превратил бы `import etbr` из мгновенного в десятисекундный — а половине
задач (посмотреть лидерборд, разобрать конфиг) torch вообще не нужен.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
# albumentations иначе ходит в сеть на каждом импорте и печатает баннер
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
# cv2 внутри DataLoader-воркеров плодит потоки и душит CPU
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")

import importlib  # noqa: E402 — только после переменных среды

__version__ = "0.1.0"

#: имя на верхнем уровне -> "модуль" или "модуль:имя_внутри"
_EXPORTS: dict[str, str] = {
    # где что лежит
    "Workspace": "workspace",
    "current_workspace": "workspace:workspace",
    "set_workspace": "workspace",
    "use_workspace": "workspace",
    "find_workspace_root": "workspace",
    # точки расширения: свои компоненты из aic_plugins.py воркспейса
    "register_loss": "registry",
    "register_aug": "registry",
    "register_val_mode": "registry",
    "register_optimizer": "registry",
    "register_scheduler": "registry",
    "register_backend": "registry",
    "load_plugins": "registry",
    "Registry": "registry",
    # конфиги
    "Cfg": "config",
    "load_config": "config",
    "save_config": "config",
    "apply_override": "config",
    "config_hash": "config",
    "flatten": "config",
    # очередь экспериментов
    "PlannedRun": "plans",
    "load_plan": "plans",
    "run_plan": "plans",
    "preflight": "plans",
    "describe_plan": "plans:describe",
    # прогон
    "RunLogger": "logging_utils",
    "read_metrics": "logging_utils",
    "seed_everything": "utils",
    "pick_device": "utils",
    "make_run_dir": "utils",
    "AverageMeter": "utils",
    "ModelEma": "utils",
    "train_run": "train:run",
    # метрика
    "AICAccumulator": "metrics",
    "AICResult": "metrics",
    "score_masks": "metrics",
    "harmonic_aic": "metrics",
    "dice_binary": "metrics",
    # данные
    "load_index": "indexing",
    "load_folds": "splits",
    "SegDataset": "datasets",
    # модель
    "build_model": "models",
    "build_loss": "losses",
    # анализ и сабмит
    "leaderboard": "analysis.leaderboard",
    "compare_table": "analysis.compare_runs",
    "plot_history": "analysis.compare_runs",
    "build_submission": "submission",
    "validate_submission": "submission",
}

__all__ = ["__version__", *sorted(_EXPORTS)]


def __getattr__(name: str):
    target = _EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"в {__name__} нет имени {name!r}; доступны: {', '.join(__all__)}")
    module_name, _, attribute = target.partition(":")
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attribute or name)
    globals()[name] = value  # второй доступ уже без importlib
    return value


def __dir__() -> list[str]:
    return sorted({*globals(), *_EXPORTS})
