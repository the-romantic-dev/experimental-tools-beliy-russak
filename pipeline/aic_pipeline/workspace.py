"""Воркспейс пайплайна: синглтон поверх `aic.Workspace` плюс конфиги и планы.

Все пути считает библиотека. Здесь остаётся то, чего в ней нет намеренно:

* синглтон. Библиотека глобального состояния не держит, а CLI без него неудобен:
  `aic train -c baseline` не должен требовать путь в каждой команде. Это
  свойство ЭТОГО слоя, и живёт оно здесь;
* `configs/` и `plans/` — места, о которых знает только пайплайн;
* поиск корня по `configs/`, а не по `data/`: без конфигов команда
  `aic train -c ...` всё равно ничего не сделает.

Пути отдаются функциями, а не константами. Константу вида `RUNS_ROOT` вызывающий
код забрал бы через `from .workspace import RUNS_ROOT` один раз на импорте, и
`set_workspace()` после этого уже ничего бы не изменил.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from aic.paths import WORKSPACE_ENV, Workspace as _Workspace

#: по наличию этой папки воркспейс пайплайна и опознаётся при поиске вверх
MARKER = "configs"


class Workspace(_Workspace):
    """`aic.Workspace` плюс места, о которых знает только пайплайн."""

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def plans(self) -> Path:
        return self.root / "plans"


def find_workspace_root(start: str | Path | None = None) -> Path:
    """Ближайшая вверх папка с `configs/`; если такой нет — сама `start`."""
    return Workspace.find(start, marker=MARKER).root


_current: Workspace | None = None


def workspace() -> Workspace:
    """Текущий воркспейс. Результат кэшируется до следующего `set_workspace`."""
    global _current
    if _current is None:
        env_root = os.environ.get(WORKSPACE_ENV)
        _current = Workspace(env_root) if env_root else Workspace(find_workspace_root())
    return _current


def set_workspace(root: str | Path | None) -> Workspace:
    """Задать корень явно. `None` сбрасывает к автоопределению."""
    global _current
    _current = None if root is None else Workspace(root)
    return workspace()


@contextlib.contextmanager
def use_workspace(root: str | Path):
    """Временно переключить воркспейс — для тестов и для работы с чужой папкой."""
    global _current
    previous = _current
    try:
        yield set_workspace(root)
    finally:
        _current = previous


# --- сокращения, чтобы вызывающий код не писал workspace() каждый раз ------

def project_root() -> Path:
    return workspace().root


def data_root() -> Path:
    return workspace().data


def dataset_root() -> Path:
    return workspace().dataset_root


def train_csv() -> Path:
    return workspace().train_csv


def src_dir() -> Path:
    return workspace().src_dir


def cache_root() -> Path:
    return workspace().cache


def test_root() -> Path:
    return workspace().test_root


def test_csv() -> Path:
    return workspace().test_csv


def submission_template() -> Path:
    return workspace().submission_template


def test_img_dir() -> Path:
    return workspace().test_img_dir


def artifacts_root() -> Path:
    return workspace().artifacts


def index_path() -> Path:
    return workspace().index_path


def split_path() -> Path:
    return workspace().split_path


def runs_root() -> Path:
    return workspace().runs


def submissions_root() -> Path:
    return workspace().submissions


def configs_root() -> Path:
    return workspace().configs


def plans_root() -> Path:
    return workspace().plans


def resolve(rel_path: str, root: Path | None = None) -> Path:
    return workspace().resolve(rel_path, root)


def ensure_dirs() -> None:
    workspace().ensure_dirs()
