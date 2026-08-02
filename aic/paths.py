"""Где что лежит. Обычный объект: создают и передают аргументом.

Раньше это был синглтон с двадцатью функциями-сокращениями (`runs_root()`,
`train_csv()`, ...) и `set_workspace()` для подмены. Работать с двумя папками
в одном процессе он не давал, а порядок импортов начинал влиять на результат.

    ws = aic.Workspace("D:/projects/aiijc")
    ws = aic.Workspace.find()      # AIC_WORKSPACE, иначе поиск вверх по data/
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

WORKSPACE_ENV = "AIC_WORKSPACE"
#: по наличию этой папки воркспейс и опознаётся при поиске вверх. Раньше маркером
#: были `configs/` — теперь конфиги принадлежат пайплайну, а не воркспейсу
MARKER = "data"


@dataclass(frozen=True)
class Workspace:
    """Корень воркспейса и все производные от него пути."""

    root: Path

    def __init__(self, root: str | Path) -> None:
        object.__setattr__(self, "root", Path(root).resolve())

    def __repr__(self) -> str:
        return f"Workspace({str(self.root)!r})"

    @classmethod
    def find(cls, start: str | Path | None = None, *, marker: str = MARKER) -> "Workspace":
        """Переменная среды, иначе ближайшая вверх папка с `marker`, иначе `start`.

        Поиск не поднимается до домашней папки и выше. Иначе один `~/data`
        (а он есть у многих) делает воркспейсом весь домашний каталог: из
        любого ноутбука без своего `data/` рядом `find()` молча вернул бы `~`,
        и дальше `ws.train_csv` указывал бы в несуществующий путь. Ошибка
        всплыла бы только на чтении файла и выглядела бы как проблема с
        данными, а не с определением корня.
        """
        env_root = os.environ.get(WORKSPACE_ENV)
        if env_root:
            return cls(env_root)

        current = Path(start or Path.cwd()).resolve()
        try:
            home = Path.home().resolve()
        except (RuntimeError, OSError):  # домашней папки может не быть вовсе
            home = None

        for candidate in (current, *current.parents):
            if candidate == home:
                break
            if (candidate / marker).is_dir():
                return cls(candidate)
        return cls(current)

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

    # --- операции --------------------------------------------------------

    def resolve(self, rel_path: str, root: Path | None = None) -> Path:
        """Путь из CSV -> абсолютный путь на диске."""
        base = self.dataset_root if root is None else Path(root)
        return base / str(rel_path).replace("\\", "/")

    def ensure_dirs(self) -> None:
        for directory in (self.artifacts, self.runs, self.cache, self.submissions):
            directory.mkdir(parents=True, exist_ok=True)
