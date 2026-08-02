"""Единая точка правды про то, где что лежит.

Воркспейс — это папка с данными, конфигами и прогонами. Раньше она совпадала с
папкой, из которой импортировался код, и путь считался как `<пакет>/..`. Теперь
код — установленная библиотека и может лежать где угодно, поэтому корень ищется
отдельно от него:

1. явно заданный `set_workspace(path)`;
2. переменная среды `AIC_WORKSPACE`;
3. ближайшая вверх от текущей папки директория, внутри которой есть `configs/`;
4. текущая рабочая папка.

Пункт 3 покрывает обычную работу в клоне репозитория: и из корня, и из
`notebooks/` находится один и тот же воркспейс.

Пути отдаются функциями, а не константами. Константу вида `RUNS_ROOT` вызывающий
код забрал бы через `from .workspace import RUNS_ROOT` один раз на импорте, и
`set_workspace()` после этого уже ничего бы не изменил.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

WORKSPACE_ENV = "AIC_WORKSPACE"
#: по наличию этой папки воркспейс и опознаётся при поиске вверх
MARKER = "configs"


class Workspace:
    """Корень воркспейса и все производные от него пути."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Workspace) and self.root == other.root

    # --- данные ---------------------------------------------------------

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def dataset_root(self) -> Path:
        # в CSV пути записаны относительно этой папки: "stage1/train/img/....jpg"
        return self.data / "train_stage1"

    @property
    def train_csv(self) -> Path:
        return self.dataset_root / "stage1" / "train.csv"

    @property
    def src_dir(self) -> Path:
        return self.dataset_root / "stage1" / "train" / "src"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    # тестовая выборка: пути внутри test.csv заданы относительно папки с CSV
    @property
    def test_root(self) -> Path:
        return self.data / "test_stage1" / "test_stage1"

    @property
    def test_csv(self) -> Path:
        return self.test_root / "test.csv"

    @property
    def submission_template(self) -> Path:
        return self.test_root / "submission.csv"

    @property
    def test_img_dir(self) -> Path:
        return self.test_root / "test_stage1_img"

    # --- результаты работы ----------------------------------------------

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def index_path(self) -> Path:
        return self.artifacts / "index.parquet"

    @property
    def split_path(self) -> Path:
        return self.artifacts / "folds.parquet"

    @property
    def runs(self) -> Path:
        return self.root / "runs"

    @property
    def submissions(self) -> Path:
        return self.root / "submissions"

    # --- описания экспериментов -----------------------------------------

    @property
    def configs(self) -> Path:
        return self.root / "configs"

    @property
    def plans(self) -> Path:
        return self.root / "plans"

    # --- операции --------------------------------------------------------

    def resolve(self, rel_path: str, root: Path | None = None) -> Path:
        """Путь из CSV -> абсолютный путь на диске."""
        base = self.dataset_root if root is None else Path(root)
        return base / str(rel_path).replace("\\", "/")

    def ensure_dirs(self) -> None:
        for directory in (self.artifacts, self.runs, self.cache, self.submissions):
            directory.mkdir(parents=True, exist_ok=True)


def find_workspace_root(start: str | Path | None = None) -> Path:
    """Ближайшая вверх папка с `configs/`; если такой нет — сама `start`."""
    current = Path(start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / MARKER).is_dir():
            return candidate
    return current


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
