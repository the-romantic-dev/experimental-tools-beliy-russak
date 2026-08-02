"""Чем именно посчитан прогон: коммит, версии, командная строка.

Регламент требует, чтобы код воспроизводил отправленный `submission.csv`.
Снапшота конфига для этого мало: тот же конфиг на другой версии timm даёт другую
сеть (набор стадий `features_only` менялся), на другой версии smp — другой
декодер, а на другом коммите тулкита — другой лосс. Через месяц после сабмита
восстановить это по памяти нельзя, поэтому снимок кладётся рядом с прогоном
сразу, в `env.json`.

Состояние git читается у КОДА, а не у текущей папки: пакет ставится через
`pip install -e` и вполне может лежать не там, откуда запущена команда.
"""

from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

#: версии именно этих пакетов меняют форму сети или численность результата
TRACKED_PACKAGES = (
    "torch", "torchvision", "timm", "segmentation_models_pytorch",
    "albumentations", "numpy", "opencv-python-headless", "pandas",
)


def _package_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in TRACKED_PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _git(*args: str) -> str | None:
    """Вывод git в папке пакета; `None`, если это не репозиторий или git нет."""
    try:
        return subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), *args],
            stderr=subprocess.DEVNULL, text=True, timeout=10,
        ).strip()
    except Exception:  # noqa: BLE001 — годится любая причина: нет git, нет репозитория
        return None


def _git_state() -> dict:
    commit = _git("rev-parse", "HEAD")
    if commit is None:
        return {"commit": None, "branch": None, "dirty": None}
    status = _git("status", "--porcelain")
    return {
        "commit": commit,
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        # грязная рабочая копия значит, что коммита недостаточно для
        # воспроизведения — про это лучше знать до сабмита, а не после
        "dirty": bool(status),
    }


def environment_info() -> dict:
    """Всё, что понадобится, чтобы повторить этот прогон через месяц."""
    return {
        "started": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "command": " ".join(sys.argv),
        "packages": _package_versions(),
        "git": _git_state(),
    }


def write_environment(run_dir: str | Path) -> Path:
    """Положить снимок среды в `<run_dir>/env.json`."""
    path = Path(run_dir) / "env.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(environment_info(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return path
