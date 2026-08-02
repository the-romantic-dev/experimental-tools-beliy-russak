# План реализации: библиотека `aic`

> **Для агентов:** ОБЯЗАТЕЛЬНЫЙ САБ-СКИЛЛ — `superpowers:subagent-driven-development`
> (рекомендуется) либо `superpowers:executing-plans`. Шаги отмечены чекбоксами
> `- [ ]` для отслеживания.

**Цель:** вынуть инструменты из фреймворка `experimental_tools_beliy_russak` в
устанавливаемую библиотеку `aic`, которая не регламентирует способ обучения, и
перевести существующий пайплайн на неё, не потеряв ни одного из 30 прогонов.

**Архитектура:** девять плоских модулей без глобального состояния. Классов с
состоянием два: `AICAccumulator` и `Run`. Всё остальное — функции над numpy и
frozen dataclass'ы. Формат папки прогона не меняется; библиотека не заглядывает
внутрь `config.yaml`.

**Стек:** Python ≥3.10, numpy, pandas, pyarrow, scikit-learn, pyyaml,
opencv-python-headless, pillow. torch — только через extra, импортируется лениво.

Спека: [2026-08-02-aic-toolkit-design.md](2026-08-02-aic-toolkit-design.md).

## Global Constraints

- **Никаких прогонов обучения.** Запрет покрывает и smoke, и probe. Ни один шаг
  плана не запускает обучение; всё проверяется юнит-тестами на синтетике и на
  уже лежащих в `runs/` артефактах.
- **`import aic` обязан работать в среде без torch.** torch импортируется внутри
  функций `budget.*`, `Run.save_state`, `Run.load_state`, `submit.predict_folder`.
- **Переменные среды выставляются в `aic/__init__.py` до любого импорта torch,
  numpy и cv2:** `KMP_DUPLICATE_LIB_OK=TRUE`, `NO_ALBUMENTATIONS_UPDATE=1`,
  `OPENCV_LOG_LEVEL=ERROR`.
- **Никакого глобального состояния:** ни синглтона воркспейса, ни реестров, ни
  автозагрузки плагинов. Состояние передаётся аргументом.
- **Библиотека никогда не читает содержимое `config.yaml`** — только пишет и
  отдаёт как есть.
- **Формат папки прогона неизменен:** `ckpt/`, `config.yaml`, `metrics.jsonl`,
  `metrics.csv`, `oof/{val.npz, val_fullres.npz, val_rows.parquet}`, `preds/`,
  `summary.json`, `tb/`, `train.log`.
- **Строгие FLOPs только через `torch.utils.flop_counter.FlopCounterMode`.**
  fvcore, thop и ptflops считают MACs — вдвое меньше.
- **Язык комментариев и докстрингов — русский**, как во всём репозитории.
- **Питон запускается так:** `python -m pytest ...` из корня репозитория. Если
  среда не активирована, интерпретатор берётся из `$AIC_PYTHON`.

## Карта файлов

Создаются:

| файл | ответственность |
|---|---|
| `aic/__init__.py` | переменные среды, ленивый `__getattr__`, публичные имена |
| `aic/paths.py` | `Workspace` — где что лежит |
| `aic/metric.py` | AIC: аккумулятор гистограмм, свип порогов, прямой счёт |
| `aic/runs.py` | `Run` (папка прогона), `Eval` (аккумулятор + stem'ы) |
| `aic/stats.py` | бутстрап, вердикт, гейт, сопоставимость |
| `aic/analysis.py` | лидерборд, таблица различий, разбор валидации по корзинам |
| `aic/data.py` | индекс, фолды, чтение изображений, предкэш |
| `aic/submit.py` | раннер над `predict_fn`, запись, валидация, zip |
| `aic/budget.py` | строгие GFLOPs над `nn.Module` |
| `tests/aic/*` | тесты библиотеки, не импортируют пайплайн |

Переезжают в фазе 2: `experimental_tools_beliy_russak/` → `pipeline/aic_pipeline/`,
`tests/pipeline/` → `pipeline/tests/`.

**В библиотеку не идут, остаются пайплайном** (решено отдельно, в раскладку
спеки эти модули не попали):

* `compliance.py` — ограничения регламента помимо бюджета: источник весов
  энкодера и недостижимость путей теста из обучающего кода. Читает секцию
  `model` конфига, то есть завязан на форму конфига пайплайна;
* `provenance.py` — запись окружения в папку прогона. Завязана на конкретный
  набор пакетов, а не на формат папки.

---

## Фаза 1. Библиотека

### Task 1: Скелет пакета и разделение тестов

**Files:**
- Create: `aic/__init__.py`, `tests/aic/conftest.py`, `tests/aic/test_import.py`
- Create: `tests/pipeline/__init__.py`

**В `tests/aic/` НЕ должно быть `__init__.py`.** С ним каталог тестов
становится импортируемым пакетом по имени `aic` и затеняет библиотеку: pytest
кладёт `tests/` в `sys.path`, и внутрипроцессный `import aic` приводит в
каталог тестов. Подпроцессные проверки при этом проходят (у них свой
`PYTHONPATH`), так что расхождение выглядит необъяснимым. У `tests/pipeline/`
`__init__.py` нужен — там имена файлов пересекаются с будущими.
- Modify: `pyproject.toml`
- Move: `tests/*.py` → `tests/pipeline/*.py` (все 24 файла, включая `conftest.py`)

**Interfaces:**
- Produces: пакет `aic` импортируется; `aic.__version__` — строка;
  `aic.__getattr__(name)` поднимает подмодуль лениво.

- [ ] **Step 1: Перенести существующие тесты в `tests/pipeline/`**

```bash
mkdir -p tests/pipeline tests/aic
git mv tests/conftest.py tests/pipeline/conftest.py
for f in tests/test_*.py; do git mv "$f" "tests/pipeline/$(basename "$f")"; done
touch tests/pipeline/__init__.py tests/aic/__init__.py
git add tests/pipeline/__init__.py tests/aic/__init__.py
```

Причина переноса: `tests/conftest.py` на каждом тесте импортирует
`experimental_tools_beliy_russak` и тем самым торч. Тест «библиотека
импортируется без торча» под таким conftest бессмыслен — торч уже в процессе.

- [ ] **Step 2: Починить путь к корню в перенесённом conftest**

`tests/pipeline/conftest.py`, строка 19 — файл стал на уровень глубже:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
```

- [ ] **Step 3: Проверить, что перенос ничего не сломал**

Run: `python -m pytest tests/pipeline -q`
Expected: столько же пройденных тестов, сколько было до переноса; ни одного
`ERROR` про ненайденный воркспейс или конфиг.

- [ ] **Step 4: Написать падающий тест на импорт**

`tests/aic/test_import.py`:

```python
"""Библиотека обязана импортироваться без торча и без пайплайна."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_version_is_a_string():
    import aic

    assert isinstance(aic.__version__, str)
    assert aic.__version__


def test_import_sets_env_before_anything_else():
    """Переменные среды выставлены к моменту, когда импорт вернул управление."""
    code = textwrap.dedent(
        """
        import os, sys
        assert "torch" not in sys.modules
        import aic
        assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
        assert os.environ["NO_ALBUMENTATIONS_UPDATE"] == "1"
        assert os.environ["OPENCV_LOG_LEVEL"] == "ERROR"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_import_does_not_pull_torch():
    """`import aic` не тянет торч: половине задач он не нужен, а он тяжёлый."""
    code = textwrap.dedent(
        """
        import sys
        import aic
        assert "torch" not in sys.modules, "aic притащил torch на импорте"
        assert "timm" not in sys.modules, "aic притащил timm на импорте"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
```

Проверка отдельным процессом обязательна: в общем процессе pytest торч уже
импортирован тестами пайплайна, и проверка `"torch" not in sys.modules` прошла
бы или упала по причине, не имеющей отношения к делу.

- [ ] **Step 5: Убедиться, что тест падает**

Run: `python -m pytest tests/aic/test_import.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'aic'`

- [ ] **Step 6: Написать `aic/__init__.py`**

```python
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
    "Workspace": "paths",
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
```

`_EXPORTS` пополняется в каждой следующей задаче — там, где появляется новое
публичное имя.

- [ ] **Step 7: Добавить пустой conftest библиотеки**

`tests/aic/conftest.py`:

```python
"""Тесты библиотеки не знают ни про пайплайн, ни про воркспейс репозитория.

Пусто намеренно: любой импорт отсюда попадёт во ВСЕ тесты библиотеки, включая
тест «импортируется без торча». Всё, что нужно конкретному тесту, он создаёт
через tmp_path сам.
"""
```

- [ ] **Step 8: Прописать пакет в pyproject**

`pyproject.toml`, секция `[tool.setuptools.packages.find]`:

```toml
[tool.setuptools.packages.find]
include = ["aic*", "experimental_tools_beliy_russak*"]
```

Оба пакета на время фазы 1: пайплайн продолжает работать, свип по сидам не
встаёт. В фазе 2 останется один.

- [ ] **Step 9: Прогнать тесты**

Run: `python -m pytest tests/aic -q`
Expected: 3 passed

- [ ] **Step 10: Коммит**

```bash
git add aic tests pyproject.toml
git commit -m "aic: скелет пакета, тесты разделены по слоям"
```

---

### Task 2: `aic/paths.py` — воркспейс без синглтона

**Files:**
- Create: `aic/paths.py`, `tests/aic/test_paths.py`
- Modify: `aic/__init__.py` (уже экспортирует `Workspace`)

**Interfaces:**
- Produces:
  - `Workspace(root: str | Path)` — frozen dataclass, `root: Path` (resolved)
  - `Workspace.find(start: str | Path | None = None, *, marker: str = "data") -> Workspace` —
    поиск вверх **не поднимается до домашней папки и выше**: один `~/data`
    иначе делает воркспейсом весь домашний каталог, и `ws.train_csv` молча
    указывает в несуществующий путь
  - свойства → `Path`: `data`, `dataset_root`, `train_csv`, `src_dir`, `cache`,
    `test_root`, `test_csv`, `submission_template`, `test_img_dir`, `artifacts`,
    `index_path`, `split_path`, `runs`, `submissions`
  - `Workspace.resolve(rel_path: str, root: Path | None = None) -> Path`
  - `Workspace.ensure_dirs() -> None`

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_paths.py`:

```python
"""Воркспейс — обычный объект: создают и передают, глобального нет."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from aic.paths import Workspace


def test_derived_paths_hang_off_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.root == tmp_path.resolve()
    assert ws.data == tmp_path.resolve() / "data"
    assert ws.runs == tmp_path.resolve() / "runs"
    assert ws.train_csv == ws.dataset_root / "stage1" / "train.csv"
    assert ws.index_path == ws.artifacts / "index.parquet"
    assert ws.test_csv == ws.test_root / "test.csv"


def test_two_workspaces_do_not_interfere(tmp_path):
    """Ровно то, чего не умел синглтон: два воркспейса рядом в одном процессе."""
    a = Workspace(tmp_path / "a")
    b = Workspace(tmp_path / "b")
    assert a.runs != b.runs
    assert a.runs.name == b.runs.name == "runs"


def test_find_walks_up_to_the_data_marker(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    deep = tmp_path / "notebooks" / "нора"
    deep.mkdir(parents=True)
    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    assert Workspace.find(deep).root == tmp_path.resolve()


def test_find_prefers_the_env_variable(tmp_path, monkeypatch):
    (tmp_path / "here" / "data").mkdir(parents=True)
    (tmp_path / "there").mkdir()
    monkeypatch.setenv("AIC_WORKSPACE", str(tmp_path / "there"))
    assert Workspace.find(tmp_path / "here").root == (tmp_path / "there").resolve()


def test_find_falls_back_to_start_when_no_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    assert Workspace.find(tmp_path).root == tmp_path.resolve()


def test_resolve_normalises_windows_separators(tmp_path):
    ws = Workspace(tmp_path)
    got = ws.resolve("stage1\\train\\img\\a.jpg")
    assert got == ws.dataset_root / "stage1" / "train" / "img" / "a.jpg"


def test_ensure_dirs_creates_the_writable_ones(tmp_path):
    ws = Workspace(tmp_path)
    ws.ensure_dirs()
    for path in (ws.artifacts, ws.runs, ws.cache, ws.submissions):
        assert path.is_dir()


def test_frozen(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(Exception):
        ws.root = Path("/другое")
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `python -m pytest tests/aic/test_paths.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'aic.paths'`

- [ ] **Step 3: Написать `aic/paths.py`**

```python
"""Где что лежит. Обычный объект: создают и передают аргументом.

Раньше это был синглтон с двадцатью функциями-сокращениями (`runs_root()`,
`train_csv()`, ...) и `set_workspace()` для подмены. Работать с двумя папками
в одном процессе он не давал, а порядок импортов начинал влиять на результат.

    ws = aic.Workspace("D:/projects/aiijc")
    ws = aic.Workspace.find()      # AIC_WORKSPACE, иначе поиск вверх по data/
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
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
        """Переменная среды, иначе ближайшая вверх папка с `marker`, иначе `start`."""
        env_root = os.environ.get(WORKSPACE_ENV)
        if env_root:
            return cls(env_root)
        current = Path(start or Path.cwd()).resolve()
        for candidate in (current, *current.parents):
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
```

Свойств `configs` и `plans` здесь нет — их забирает `aic_pipeline` в фазе 2.

- [ ] **Step 4: Прогнать тесты**

Run: `python -m pytest tests/aic/test_paths.py -q`
Expected: 8 passed

- [ ] **Step 5: Коммит**

```bash
git add aic/paths.py tests/aic/test_paths.py
git commit -m "aic: Workspace без синглтона"
```

---

### Task 3: `aic/metric.py` — метрика AIC

Перенос `experimental_tools_beliy_russak/metrics.py` с одной содержательной
правкой: приватный `_tables()` становится публичным `tables()`. Сейчас разворот
гистограмм в `|P_t|` и `|P_t ∩ G|` продублирован в `analysis/oof_report.OofView`;
после публикации метода дубль убирается (Task 9).

**Files:**
- Create: `aic/metric.py`, `tests/aic/test_metric.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Produces:
  - `EPS = 1e-6`, `FP_AREA_THRESHOLD = 0.01`
  - `harmonic_aic(dice_pos: float, fpr_neg: float) -> float`
  - `dice_binary(pred: np.ndarray, gt: np.ndarray) -> float`
  - `AICResult` — dataclass с полями `aic, dice_pos, fpr_neg, n_pos, n_neg,`
    `mask_threshold, cls_threshold, min_area`; метод `as_dict() -> dict`
  - `score_masks(preds, gts, *, pred_positive_value=128, gt_positive_value=128) -> AICResult`
  - `AICAccumulator(n_bins: int = 256)` с `update`, `update_hist`, `tables`,
    `evaluate`, `sweep`, `best`, `save`, `load`, `thresholds`, `__len__`
  - `AICAccumulator.tables() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]` —
    `(pred_counts, inter_counts, gt_sum, n_pixels, cls_prob)`, формы `(N, n_bins)` и `(N,)`
  - `DEFAULT_MASK_GRID`, `DEFAULT_CLS_GRID`, `DEFAULT_AREA_GRID`

- [ ] **Step 1: Скопировать модуль**

```bash
git show HEAD:experimental_tools_beliy_russak/metrics.py > aic/metric.py
```

- [ ] **Step 2: Переименовать приватный метод**

В `aic/metric.py` заменить объявление и оба внутренних вызова:

```python
    def tables(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(pred_counts, inter_counts, gt_sum, n_pixels, cls_prob), формы (N, n_bins) и (N,).

        Публичный, а не приватный: разбор валидации по корзинам площади
        (`analysis.OofView`) строится ровно на этих таблицах. Пока метод был
        приватным, у разворота гистограмм было две реализации, и однажды они
        разошлись бы — а вердикты тогда считались бы не по той метрике, по
        которой отбираются модели.
        """
```

Вызов внутри `sweep`:

```python
        pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = self.tables()
```

- [ ] **Step 3: Написать тест на публичность и на инвариант**

`tests/aic/test_metric.py`:

```python
"""Метрика AIC. Перенесена дословно, поэтому тесты стерегут инварианты."""

from __future__ import annotations

import numpy as np
import pytest

from aic.metric import (
    DEFAULT_AREA_GRID,
    DEFAULT_CLS_GRID,
    DEFAULT_MASK_GRID,
    AICAccumulator,
    dice_binary,
    harmonic_aic,
    score_masks,
)


def _mask(h, w, filled):
    m = np.zeros((h, w), dtype=bool)
    m[:filled] = True
    return m


def test_harmonic_aic_is_zero_when_a_component_is_zero():
    assert harmonic_aic(0.0, 0.0) == 0.0
    assert harmonic_aic(0.9, 1.0) == 0.0


def test_dice_of_identical_masks_is_one():
    m = _mask(10, 10, 4)
    assert dice_binary(m, m) == pytest.approx(1.0, abs=1e-4)


def test_score_masks_counts_the_one_percent_rule():
    """Негатив с предсказанием >= 1% площади — ложная тревога, меньше — нет."""
    gt_neg = np.zeros((100, 100), dtype=bool)
    small = np.zeros((100, 100), dtype=bool)
    small[0, :50] = True                      # 0.5% площади
    big = np.zeros((100, 100), dtype=bool)
    big[:2] = True                            # 2% площади
    gt_pos = _mask(100, 100, 10)

    res = score_masks([gt_pos, small, big], [gt_pos, gt_neg, gt_neg])
    assert res.n_pos == 1 and res.n_neg == 2
    assert res.fpr_neg == pytest.approx(0.5)


def test_tables_is_public_and_matches_sweep():
    """Разворот гистограмм доступен снаружи и согласован со свипом."""
    rng = np.random.default_rng(0)
    probs = rng.random((4, 8, 8)).astype(np.float32)
    gts = (rng.random((4, 8, 8)) > 0.7).astype(np.float32)
    acc = AICAccumulator(n_bins=64)
    acc.update(probs, gts)

    pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = acc.tables()
    assert pred_counts.shape == (4, 64)
    assert inter_counts.shape == (4, 64)
    assert gt_sum.shape == n_pixels.shape == cls_prob.shape == (4,)
    assert not hasattr(acc, "_tables")

    # |P_t| при t=0 — все пиксели кадра
    assert pred_counts[:, 0].tolist() == n_pixels.tolist()


def test_sweep_is_sorted_by_aic_descending():
    rng = np.random.default_rng(1)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((6, 8, 8)), (rng.random((6, 8, 8)) > 0.5).astype(np.float32))
    results = acc.sweep([0.2, 0.5, 0.8], DEFAULT_CLS_GRID[:2], DEFAULT_AREA_GRID[:2])
    aics = [r.aic for r in results]
    assert aics == sorted(aics, reverse=True)
    assert acc.best([0.2, 0.5, 0.8]).aic == max(r.aic for r in acc.sweep([0.2, 0.5, 0.8]))


def test_evaluate_agrees_with_sweep_of_one_point():
    rng = np.random.default_rng(2)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((5, 6, 6)), (rng.random((5, 6, 6)) > 0.6).astype(np.float32))
    direct = acc.evaluate(0.5, 0.0, 0.0)
    swept = acc.sweep([0.5], [0.0], [0.0])[0]
    assert direct.aic == swept.aic


def test_save_and_load_roundtrip(tmp_path):
    rng = np.random.default_rng(3)
    acc = AICAccumulator(n_bins=16)
    acc.update(rng.random((3, 4, 4)), (rng.random((3, 4, 4)) > 0.5).astype(np.float32))
    path = tmp_path / "val.npz"
    acc.save(path)

    back = AICAccumulator.load(path)
    assert len(back) == len(acc)
    assert back.evaluate(0.5).aic == pytest.approx(acc.evaluate(0.5).aic)


def test_empty_accumulator_says_so():
    with pytest.raises(ValueError, match="пуст"):
        AICAccumulator().tables()


def test_default_grids_are_sane():
    assert min(DEFAULT_MASK_GRID) > 0.0 and max(DEFAULT_MASK_GRID) < 1.0
    assert DEFAULT_CLS_GRID[0] == 0.0
    assert DEFAULT_AREA_GRID[0] == 0.0
```

- [ ] **Step 4: Прогнать**

Run: `python -m pytest tests/aic/test_metric.py -q`
Expected: 9 passed

- [ ] **Step 5: Дописать экспорты**

`aic/__init__.py`, в `_EXPORTS`:

```python
    "Workspace": "paths",
    # метрика
    "AICAccumulator": "metric",
    "AICResult": "metric",
    "score_masks": "metric",
    "harmonic_aic": "metric",
    "dice_binary": "metric",
```

- [ ] **Step 6: Коммит**

```bash
git add aic/metric.py aic/__init__.py tests/aic/test_metric.py
git commit -m "aic: метрика AIC, tables() публичный"
```

---

### Task 4: `aic/runs.py` — папка прогона на запись

**Files:**
- Create: `aic/runs.py`, `tests/aic/test_runs_write.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: ничего из предыдущих задач
- Produces:
  - `Run.create(runs_root: str | Path, name: str, *, resume: bool = False, tensorboard: bool = True) -> Run`
  - `run.dir -> Path`
  - `run.info(message: str) -> None` — консоль + `train.log`
  - `run.log(step: int, metrics: Mapping[str, Any], prefix: str = "") -> None`
  - `run.save_snapshot(obj: Mapping[str, Any], name: str = "config.yaml") -> Path`
  - `run.save_summary(patch: Mapping[str, Any]) -> Path` — мерж по верхнему уровню
  - `run.flush_csv() -> None`, `run.close() -> None`

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_runs_write.py`:

```python
"""Папка прогона на запись. Формат не меняется — на нём стоят 30 прогонов."""

from __future__ import annotations

import json

import pytest
import yaml

from aic.runs import Run


def test_create_makes_the_standard_subdirs(tmp_path):
    run = Run.create(tmp_path, "проба", tensorboard=False)
    assert run.dir == tmp_path / "проба"
    for sub in ("ckpt", "oof", "tb", "preds"):
        assert (run.dir / sub).is_dir()
    run.close()


def test_create_does_not_overwrite_an_existing_run(tmp_path):
    first = Run.create(tmp_path, "занято", tensorboard=False)
    first.close()
    second = Run.create(tmp_path, "занято", tensorboard=False)
    assert second.dir != first.dir
    assert second.dir.name.startswith("занято__")
    second.close()


def test_resume_reuses_the_same_dir(tmp_path):
    first = Run.create(tmp_path, "продолжаем", tensorboard=False)
    first.close()
    second = Run.create(tmp_path, "продолжаем", resume=True, tensorboard=False)
    assert second.dir == first.dir
    second.close()


def test_resume_on_a_missing_dir_just_creates_it(tmp_path):
    run = Run.create(tmp_path, "нового-нет", resume=True, tensorboard=False)
    assert run.dir.is_dir()
    run.close()


def test_log_appends_jsonl_with_step_and_elapsed(tmp_path):
    run = Run.create(tmp_path, "лог", tensorboard=False)
    run.log(0, {"train/loss": 1.5})
    run.log(1, {"train/loss": 1.2, "samples": 8000})
    run.close()

    lines = (run.dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    assert [r["step"] for r in rows] == [0, 1]
    assert all("elapsed_s" in r for r in rows)
    assert rows[1]["samples"] == 8000


def test_log_prefix_applies_to_metric_names_only(tmp_path):
    run = Run.create(tmp_path, "префикс", tensorboard=False)
    run.log(0, {"loss": 1.0}, prefix="val/")
    run.close()
    row = json.loads((run.dir / "metrics.jsonl").read_text(encoding="utf-8").strip())
    assert row["val/loss"] == 1.0
    assert row["step"] == 0


def test_close_writes_the_csv_mirror(tmp_path):
    run = Run.create(tmp_path, "csv", tensorboard=False)
    run.log(0, {"a": 1})
    run.close()
    text = (run.dir / "metrics.csv").read_text(encoding="utf-8")
    assert "step" in text and "a" in text


def test_save_snapshot_writes_yaml_verbatim(tmp_path):
    run = Run.create(tmp_path, "снапшот", tensorboard=False)
    payload = {"модель": "unet", "lr": 3e-4, "вложенное": {"a": [1, 2]}}
    run.save_snapshot(payload)
    run.close()
    back = yaml.safe_load((run.dir / "config.yaml").read_text(encoding="utf-8"))
    assert back == payload


def test_save_summary_merges_top_level_keys(tmp_path):
    run = Run.create(tmp_path, "сводка", tensorboard=False)
    run.save_summary({"run": "сводка", "best_aic": 0.5})
    run.save_summary({"budget": {"gflops": 42.0}})
    run.save_summary({"best_aic": 0.8})
    run.close()

    summary = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert summary == {"run": "сводка", "best_aic": 0.8, "budget": {"gflops": 42.0}}


def test_save_summary_replaces_nested_dicts_whole(tmp_path):
    run = Run.create(tmp_path, "вложенное", tensorboard=False)
    run.save_summary({"best": {"aic": 0.5, "thr": 0.3}})
    run.save_summary({"best": {"aic": 0.9}})
    run.close()
    summary = json.loads((run.dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["best"] == {"aic": 0.9}


def test_info_goes_to_console_and_file(tmp_path, capsys):
    run = Run.create(tmp_path, "сообщения", tensorboard=False)
    run.info("эпоха 3 из 12")
    run.close()
    assert "эпоха 3 из 12" in capsys.readouterr().out
    assert "эпоха 3 из 12" in (run.dir / "train.log").read_text(encoding="utf-8")
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_runs_write.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'aic.runs'`

- [ ] **Step 3: Написать пишущую половину `aic/runs.py`**

```python
"""Папка прогона: ручка, которая пишет и читает документированный формат.

Формат тот же, что был у `train.run`, и меняться не будет — на нём стоят все
накопленные прогоны:

    <прогон>/
      config.yaml          непрозрачный снапшот: что вы захотели запомнить
      metrics.jsonl        источник правды, дописывается построчно
      metrics.csv          то же для Excel, перезаписывается на close()
      train.log            текстовый журнал
      summary.json         итоги, дописываются в несколько заходов
      ckpt/                чекпоинты
      oof/val.npz          гистограммы валидации
      oof/val_rows.parquet строки валидации в том же порядке
      tb/                  TensorBoard
      preds/               картинки для глаз

Библиотека не заглядывает в `config.yaml`: что там лежит — дело того, кто писал.

Писатели — глаголы (`log`, `save_*`), читатели — свойства (`history`,
`snapshot`, `summary`). Иначе `snapshot` пришлось бы делать и методом, и
свойством одновременно.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

#: подпапки, которые создаются сразу: пути на них ссылаются до первой записи
SUBDIRS = ("ckpt", "oof", "tb", "preds")


class Run:
    """Ручка на папку прогона."""

    def __init__(self, run_dir: str | Path, *, tensorboard: bool = False) -> None:
        self.dir = Path(run_dir)
        self.jsonl_path = self.dir / "metrics.jsonl"
        self.csv_path = self.dir / "metrics.csv"
        self.log_path = self.dir / "train.log"
        self.summary_path = self.dir / "summary.json"
        self.start = time.time()
        self.writer = None

        if tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(self.dir / "tb"))
            except Exception as exc:  # tensorboard не обязателен
                self.info(f"TensorBoard отключён: {exc}")

    # --- создание и открытие ---------------------------------------------

    @classmethod
    def create(
        cls,
        runs_root: str | Path,
        name: str,
        *,
        resume: bool = False,
        tensorboard: bool = True,
    ) -> "Run":
        """`<runs_root>/<name>`; занятое имя без `resume` получает суффикс времени.

        Затирать чужую папку нельзя ни при каких обстоятельствах: в ней лежат
        часы обучения, и восстановить их неоткуда.
        """
        runs_root = Path(runs_root)
        runs_root.mkdir(parents=True, exist_ok=True)
        run_dir = runs_root / name
        if run_dir.exists() and not resume:
            run_dir = runs_root / f"{name}__{time.strftime('%m%d-%H%M%S')}"
        for sub in SUBDIRS:
            (run_dir / sub).mkdir(parents=True, exist_ok=True)
        return cls(run_dir, tensorboard=tensorboard)

    @classmethod
    def open(cls, run_dir: str | Path) -> "Run":
        """Чужая папка на чтение. TensorBoard не поднимается."""
        run_dir = Path(run_dir)
        if not run_dir.is_dir():
            raise FileNotFoundError(f"нет папки прогона {run_dir}")
        return cls(run_dir, tensorboard=False)

    def __repr__(self) -> str:
        return f"Run({str(self.dir)!r})"

    # --- запись ------------------------------------------------------------

    def info(self, message: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {message}"
        print(line, flush=True)
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def log(self, step: int, metrics: Mapping[str, Any], prefix: str = "") -> None:
        """Строка в `metrics.jsonl` (+ TensorBoard). `step` и `elapsed_s` свои.

        Ось показов библиотека не считает: если сравнение прогонов пойдёт по
        числу показов, а не по номеру эпохи, логируйте `samples` обычной
        метрикой — см. `Run.curve`.
        """
        record: dict[str, Any] = {"step": int(step), "elapsed_s": round(time.time() - self.start, 1)}
        for key, value in metrics.items():
            record[f"{prefix}{key}" if prefix else key] = value

        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")

        if self.writer is not None:
            for key, value in record.items():
                if isinstance(value, (int, float)) and key not in {"step", "elapsed_s"}:
                    self.writer.add_scalar(key.replace("/", "_").replace("_", "/", 1), value, step)
            self.writer.flush()

    def save_snapshot(self, obj: Mapping[str, Any], name: str = "config.yaml") -> Path:
        """Записать снапшот как есть. Библиотека внутрь не смотрит."""
        path = self.dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(dict(obj), fh, allow_unicode=True, sort_keys=False)
        return path

    def save_summary(self, patch: Mapping[str, Any]) -> Path:
        """Дописать в `summary.json`, слив по ВЕРХНЕМУ уровню ключей.

        Мерж, а не перезапись: сводку заполняют в несколько заходов — обучение,
        потом калибровка, потом бюджет. Вложенные словари заменяются целиком:
        сливать `best` по частям некому и незачем, а частичный `best` был бы
        опаснее отсутствующего.
        """
        current = self.summary
        current.update(dict(patch))
        self.summary_path.write_text(
            json.dumps(current, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        return self.summary_path

    def flush_csv(self) -> None:
        if not self.jsonl_path.exists():
            return
        self._history_frame().to_csv(self.csv_path, index=False)

    def close(self) -> None:
        self.flush_csv()
        if self.writer is not None:
            self.writer.close()
            self.writer = None

    def __enter__(self) -> "Run":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # --- чтение (расширяется в следующей задаче) ---------------------------

    def _history_frame(self) -> pd.DataFrame:
        if not self.jsonl_path.exists():
            return pd.DataFrame()
        rows = [
            json.loads(line)
            for line in self.jsonl_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return pd.DataFrame(rows)

    @property
    def summary(self) -> dict:
        """`summary.json` или пустой словарь."""
        if not self.summary_path.exists():
            return {}
        return json.loads(self.summary_path.read_text(encoding="utf-8")) or {}
```

- [ ] **Step 4: Прогнать**

Run: `python -m pytest tests/aic/test_runs_write.py -q`
Expected: 11 passed

- [ ] **Step 5: Коммит**

```bash
git add aic/runs.py tests/aic/test_runs_write.py
git commit -m "aic: Run на запись"
```

---

### Task 5: `aic/runs.py` — чтение и ось показов

**Files:**
- Modify: `aic/runs.py`
- Create: `tests/aic/test_runs_read.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: `Run.create`, `Run.open`, `run.summary` из Task 4
- Produces:
  - `run.history -> pd.DataFrame` — все строки `metrics.jsonl`
  - `run.snapshot -> dict` — `config.yaml` как есть, `{}` если файла нет
  - `run.curve(y: str, x: str | tuple[str, float] = "samples") -> tuple[np.ndarray, np.ndarray]`

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_runs_read.py`:

```python
"""Чтение папки прогона, включая чужой и старый."""

from __future__ import annotations

import numpy as np
import pytest

from aic.runs import Run


def _make(tmp_path, rows, snapshot=None):
    run = Run.create(tmp_path, "прогон", tensorboard=False)
    for step, metrics in enumerate(rows):
        run.log(step, metrics)
    if snapshot is not None:
        run.save_snapshot(snapshot)
    run.close()
    return Run.open(run.dir)


def test_history_reads_every_row(tmp_path):
    run = _make(tmp_path, [{"val/aic_tuned": 0.3}, {"val/aic_tuned": 0.6}])
    assert list(run.history["val/aic_tuned"]) == [0.3, 0.6]
    assert list(run.history["step"]) == [0, 1]


def test_history_of_a_run_without_metrics_is_empty(tmp_path):
    run = Run.create(tmp_path, "пусто", tensorboard=False)
    run.close()
    assert Run.open(run.dir).history.empty


def test_snapshot_comes_back_unchanged(tmp_path):
    payload = {"модель": "unet", "epoch_size": 8000}
    run = _make(tmp_path, [{"a": 1}], snapshot=payload)
    assert run.snapshot == payload


def test_snapshot_of_a_run_without_config_is_empty(tmp_path):
    run = Run.create(tmp_path, "без-конфига", tensorboard=False)
    run.close()
    assert Run.open(run.dir).snapshot == {}


def test_open_on_a_missing_dir_says_so(tmp_path):
    with pytest.raises(FileNotFoundError, match="нет папки прогона"):
        Run.open(tmp_path / "такой-нет")


def test_curve_uses_a_logged_column(tmp_path):
    run = _make(
        tmp_path,
        [
            {"val/aic_tuned": 0.3, "samples": 8000},
            {"val/aic_tuned": 0.6, "samples": 16000},
        ],
    )
    xs, ys = run.curve("val/aic_tuned")
    assert xs.tolist() == [8000.0, 16000.0]
    assert ys.tolist() == [0.3, 0.6]


def test_curve_can_derive_the_axis_from_step(tmp_path):
    """Старые прогоны samples не логировали — множитель передаёт вызывающий."""
    run = _make(tmp_path, [{"val/aic_tuned": 0.3}, {"val/aic_tuned": 0.6}])
    xs, ys = run.curve("val/aic_tuned", x=("step", 8000))
    assert xs.tolist() == [8000.0, 16000.0]
    assert ys.tolist() == [0.3, 0.6]


def test_curve_skips_rows_without_the_metric(tmp_path):
    """Валидация может идти не на каждом шаге — такие строки в кривую не входят."""
    run = _make(
        tmp_path,
        [
            {"train/loss": 1.0, "samples": 8000},
            {"val/aic_tuned": 0.6, "samples": 16000},
        ],
    )
    xs, ys = run.curve("val/aic_tuned")
    assert xs.tolist() == [16000.0]
    assert ys.tolist() == [0.6]


def test_curve_without_the_axis_column_explains_how_to_fix_it(tmp_path):
    run = _make(tmp_path, [{"val/aic_tuned": 0.3}])
    with pytest.raises(KeyError) as err:
        run.curve("val/aic_tuned")
    message = str(err.value)
    assert "samples" in message
    assert "x=(\"step\"" in message or "x=('step'" in message


def test_curve_without_the_metric_column_says_which_one(tmp_path):
    run = _make(tmp_path, [{"samples": 8000}])
    with pytest.raises(KeyError, match="val/aic_tuned"):
        run.curve("val/aic_tuned")


def test_curve_of_an_empty_history_is_empty(tmp_path):
    run = Run.create(tmp_path, "пусто", tensorboard=False)
    run.close()
    xs, ys = Run.open(run.dir).curve("val/aic_tuned")
    assert xs.size == 0 and ys.size == 0
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_runs_read.py -q`
Expected: FAIL, `AttributeError: 'Run' object has no attribute 'history'`

- [ ] **Step 3: Дописать читающую часть в `aic/runs.py`**

В конец класса `Run`, заменив блок «чтение (расширяется в следующей задаче)»
на полноценный (метод `_history_frame` и свойство `summary` остаются как есть):

```python
    # --- чтение ------------------------------------------------------------

    @property
    def history(self) -> pd.DataFrame:
        """`metrics.jsonl` таблицей. Пустой DataFrame, если лога ещё нет."""
        return self._history_frame()

    @property
    def snapshot(self) -> dict:
        """`config.yaml` как есть. Библиотека внутрь не смотрит."""
        path = self.dir / "config.yaml"
        if not path.exists():
            return {}
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    def curve(
        self,
        y: str,
        x: str | tuple[str, float] = "samples",
    ) -> tuple["np.ndarray", "np.ndarray"]:
        """Две оси из лога — для `stats.gate_check` и для графиков.

        `x` — имя колонки либо пара `(колонка, множитель)`. Пара нужна старым
        прогонам: числа показов они не логировали, и ось строится из номера
        эпохи как `(step + 1) * epoch_size`. Множитель передаёт вызывающий,
        потому что только он знает, где у него лежит размер эпохи.

        Отсутствие колонки — ошибка, а не пустая кривая. Молчаливый ноль тут
        особенно дорог: `gate_check` на нулевой оси отвечает «вне кривой
        эталона» на каждой эпохе, то есть гейт выключается на весь прогон, и
        узнать об этом неоткуда.
        """
        import numpy as np

        frame = self._history_frame()
        if frame.empty:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)

        if y not in frame.columns:
            raise KeyError(
                f"в логе {self.jsonl_path} нет метрики {y!r}; "
                f"есть: {', '.join(sorted(frame.columns))}"
            )

        column, scale, shift = (x, 1.0, 0.0) if isinstance(x, str) else (x[0], float(x[1]), 1.0)
        if column not in frame.columns:
            raise KeyError(
                f"в логе {self.jsonl_path} нет колонки оси {column!r}. "
                f"Логируйте её обычной метрикой (`run.log(epoch, {{'samples': ...}})`) "
                f"или постройте ось из номера эпохи: x=(\"step\", размер_эпохи)"
            )

        rows = frame[frame[y].notna() & frame[column].notna()]
        xs = (rows[column].to_numpy(dtype=float) + shift) * scale
        return xs, rows[y].to_numpy(dtype=float)
```

`shift` равен единице только в форме с множителем: эпоха с номером 0 — это уже
один пройденный размер эпохи, а не ноль показов. Ровно так строил ось прежний
`load_reference`, и совместимость чисел с уже посчитанными вердиктами держится
на этом.

- [ ] **Step 4: Прогнать**

Run: `python -m pytest tests/aic/test_runs_read.py -q`
Expected: 11 passed

- [ ] **Step 5: Дописать экспорт `Run`**

`aic/__init__.py`, в `_EXPORTS`:

```python
    # прогон
    "Run": "runs",
```

- [ ] **Step 6: Коммит**

```bash
git add aic/runs.py aic/__init__.py tests/aic/test_runs_read.py
git commit -m "aic: чтение папки прогона и ось показов"
```

---

### Task 6: `Eval`, оценка и чекпоинты

**Files:**
- Modify: `aic/runs.py`
- Create: `tests/aic/test_runs_eval.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: `AICAccumulator`, `AICResult`, `DEFAULT_*_GRID` из `aic.metric`
- Produces:
  - `Eval(acc: AICAccumulator, stems: np.ndarray)` — frozen dataclass,
    `__post_init__` падает при несовпадении длин и при повторах stem'ов;
    `__len__`, `best(mask_thresholds=None, cls_thresholds=..., min_areas=...) -> AICResult`
  - `run.save_eval(acc: AICAccumulator, rows: pd.DataFrame, name: str = "val") -> Path`
  - `run.load_eval(name: str = "val") -> Eval`
  - `run.load_rows(name: str = "val") -> pd.DataFrame`
  - `run.operating_point() -> tuple[float, float, float]`
  - `run.save_state(state: Mapping[str, Any], name: str = "last.pt") -> Path`
  - `run.load_state(name: str = "last.pt", map_location: Any = "cpu") -> dict`

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_runs_eval.py`:

```python
"""Eval, операционная точка и чекпоинты."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aic.metric import AICAccumulator
from aic.runs import Eval, Run

torch = pytest.importorskip("torch", reason="чекпоинты требуют torch")


def _acc(n=4, seed=0):
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((n, 6, 6)), (rng.random((n, 6, 6)) > 0.6).astype(np.float32))
    return acc


def _rows(n=4):
    return pd.DataFrame({"stem": [f"кадр{i}" for i in range(n)], "fold": [0] * n})


def test_eval_rejects_a_length_mismatch():
    with pytest.raises(ValueError, match="кадров"):
        Eval(_acc(4), np.array(["a", "b"]))


def test_eval_rejects_duplicate_stems():
    with pytest.raises(ValueError, match="повторя"):
        Eval(_acc(2), np.array(["a", "a"]))


def test_eval_best_delegates_to_the_accumulator():
    acc = _acc(5)
    ev = Eval(acc, np.array([f"к{i}" for i in range(5)]))
    assert ev.best([0.3, 0.5]).aic == acc.best([0.3, 0.5]).aic
    assert len(ev) == 5


def test_save_and_load_eval_roundtrip(tmp_path):
    run = Run.create(tmp_path, "оценка", tensorboard=False)
    acc, rows = _acc(4), _rows(4)
    run.save_eval(acc, rows)
    run.close()

    assert (run.dir / "oof" / "val.npz").exists()
    assert (run.dir / "oof" / "val_rows.parquet").exists()

    ev = Run.open(run.dir).load_eval()
    assert len(ev) == 4
    assert ev.stems.tolist() == rows["stem"].tolist()
    assert ev.best([0.5]).aic == pytest.approx(acc.best([0.5]).aic)


def test_save_eval_under_another_name(tmp_path):
    run = Run.create(tmp_path, "полное", tensorboard=False)
    run.save_eval(_acc(3), _rows(3), name="val_fullres")
    run.close()
    assert (run.dir / "oof" / "val_fullres.npz").exists()
    assert len(Run.open(run.dir).load_eval("val_fullres")) == 3


def test_save_eval_refuses_mismatched_rows(tmp_path):
    """Иначе stem'ы молча разъехались бы с кадрами и сравнение мерило бы не то."""
    run = Run.create(tmp_path, "рассинхрон", tensorboard=False)
    with pytest.raises(ValueError, match="кадров"):
        run.save_eval(_acc(4), _rows(3))
    run.close()


def test_save_eval_requires_a_stem_column(tmp_path):
    run = Run.create(tmp_path, "без-stem", tensorboard=False)
    with pytest.raises(ValueError, match="stem"):
        run.save_eval(_acc(2), pd.DataFrame({"fold": [0, 0]}))
    run.close()


def test_operating_point_comes_from_the_summary(tmp_path):
    run = Run.create(tmp_path, "точка", tensorboard=False)
    run.save_summary(
        {"best": {"mask_threshold": 0.275, "cls_threshold": 0.5, "min_area": 0.03}}
    )
    run.save_eval(_acc(4), _rows(4))
    run.close()
    assert Run.open(run.dir).operating_point() == (0.275, 0.5, 0.03)


def test_operating_point_falls_back_to_a_sweep(tmp_path):
    run = Run.create(tmp_path, "без-сводки", tensorboard=False)
    run.save_eval(_acc(6), _rows(6))
    run.close()

    reopened = Run.open(run.dir)
    thr, cls_thr, area = reopened.operating_point()
    assert 0.0 < thr < 1.0
    best = reopened.load_eval().best()
    assert (thr, cls_thr, area) == (best.mask_threshold, best.cls_threshold, best.min_area)


def test_save_state_is_atomic_and_leaves_no_temp(tmp_path):
    run = Run.create(tmp_path, "чекпоинт", tensorboard=False)
    run.save_state({"model": {"w": torch.zeros(2)}, "epoch": 3})
    run.close()
    assert (run.dir / "ckpt" / "last.pt").exists()
    assert not list((run.dir / "ckpt").glob("*.tmp"))


def test_load_state_returns_whatever_was_put_in(tmp_path):
    run = Run.create(tmp_path, "состояние", tensorboard=False)
    run.save_state({"эпоха": 7, "своё": [1, 2, 3]}, name="best.pt")
    run.close()
    state = Run.open(run.dir).load_state("best.pt")
    assert state["эпоха"] == 7
    assert state["своё"] == [1, 2, 3]


def test_load_state_on_a_missing_file_says_so(tmp_path):
    run = Run.create(tmp_path, "нет-чекпоинта", tensorboard=False)
    run.close()
    with pytest.raises(FileNotFoundError, match="last.pt"):
        Run.open(run.dir).load_state()
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_runs_eval.py -q`
Expected: FAIL, `ImportError: cannot import name 'Eval' from 'aic.runs'`

- [ ] **Step 3: Дописать `Eval` в `aic/runs.py`**

Сразу после импортов, до класса `Run`:

```python
from dataclasses import dataclass

import numpy as np

from .metric import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID, AICAccumulator, AICResult


@dataclass(frozen=True)
class Eval:
    """Оценка на выборке: аккумулятор и имена кадров в том же порядке.

    Пара, а не два аргумента: величины берутся из аккумулятора по позиции, а
    совпадение кадров ищется по stem'ам. Разъехались длины — сравнение молча
    смешает разные кадры, и разница будет мерить не гипотезу, а рассинхрон.
    Проверка здесь делает такое состояние непредставимым.
    """

    acc: AICAccumulator
    stems: np.ndarray

    def __post_init__(self) -> None:
        stems = np.asarray(self.stems)
        object.__setattr__(self, "stems", stems)
        if stems.size != len(self.acc):
            raise ValueError(
                f"в аккумуляторе {len(self.acc)} кадров, а stem'ов {stems.size}"
            )
        if len(np.unique(stems)) != stems.size:
            raise ValueError("stem'ы повторяются — по ним нельзя сопоставить кадры")

    def __len__(self) -> int:
        return len(self.acc)

    def best(
        self,
        mask_thresholds=None,
        cls_thresholds=DEFAULT_CLS_GRID,
        min_areas=DEFAULT_AREA_GRID,
    ) -> AICResult:
        grid = list(DEFAULT_MASK_GRID) if mask_thresholds is None else list(mask_thresholds)
        return self.acc.best(grid, list(cls_thresholds), list(min_areas))
```

- [ ] **Step 4: Дописать методы `Run`**

В конец класса `Run`:

```python
    # --- оценка -------------------------------------------------------------

    def save_eval(self, acc: AICAccumulator, rows: pd.DataFrame, name: str = "val") -> Path:
        """`oof/<name>.npz` + `oof/<name>_rows.parquet`.

        Строки пишутся целиком, не только stem: разрезы по домену, генератору и
        площади GT в `analysis` берутся отсюда.
        """
        if "stem" not in rows.columns:
            raise ValueError(f"в таблице строк нет колонки stem (есть {list(rows.columns)})")
        if len(rows) != len(acc):
            raise ValueError(
                f"в аккумуляторе {len(acc)} кадров, а строк {len(rows)}. Сопоставление "
                "идёт по позиции, и на разной длине оно смешало бы разные кадры"
            )
        oof = self.dir / "oof"
        oof.mkdir(parents=True, exist_ok=True)
        acc.save(oof / f"{name}.npz")
        rows.to_parquet(oof / f"{name}_rows.parquet", index=False)
        return oof / f"{name}.npz"

    def load_rows(self, name: str = "val") -> pd.DataFrame:
        path = self.dir / "oof" / f"{name}_rows.parquet"
        if not path.exists():
            raise FileNotFoundError(f"нет {path}")
        return pd.read_parquet(path)

    def load_eval(self, name: str = "val") -> Eval:
        path = self.dir / "oof" / f"{name}.npz"
        if not path.exists():
            raise FileNotFoundError(f"нет {path}")
        return Eval(AICAccumulator.load(path), self.load_rows(name)["stem"].to_numpy())

    def operating_point(self, name: str = "val") -> tuple[float, float, float]:
        """`(mask_threshold, cls_threshold, min_area)` из сводки, иначе свипом.

        Остаток прежнего `load_reference`: остальная его работа разошлась по
        `Run.open`, `history` и `load_eval`.
        """
        best = self.summary.get("best") or {}
        if {"mask_threshold", "cls_threshold", "min_area"} <= set(best):
            return (
                float(best["mask_threshold"]),
                float(best["cls_threshold"]),
                float(best["min_area"]),
            )
        found = self.load_eval(name).best()
        return (found.mask_threshold, found.cls_threshold, found.min_area)

    # --- чекпоинты -----------------------------------------------------------

    def save_state(self, state: Mapping[str, Any], name: str = "last.pt") -> Path:
        """Записать состояние атомарно. Что в нём лежит — решает вызывающий.

        Библиотека не знает ни про EMA, ни про scaler, ни про шедулер. Запись
        идёт во временный файл рядом и `os.replace`: прогон, убитый посреди
        записи, не должен оставлять чекпоинт, с которого потом не поднимется
        resume.
        """
        import torch

        ckpt = self.dir / "ckpt"
        ckpt.mkdir(parents=True, exist_ok=True)
        target = ckpt / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        torch.save(dict(state), tmp)
        os.replace(tmp, target)
        return target

    def load_state(self, name: str = "last.pt", map_location: Any = "cpu") -> dict:
        import torch

        path = self.dir / "ckpt" / name
        if not path.exists():
            raise FileNotFoundError(f"нет чекпоинта {path}")
        return torch.load(str(path), map_location=map_location, weights_only=False)
```

- [ ] **Step 5: Прогнать**

Run: `python -m pytest tests/aic/test_runs_eval.py -q`
Expected: 12 passed

- [ ] **Step 6: Убедиться, что импорт всё ещё без торча**

Run: `python -m pytest tests/aic/test_import.py -q`
Expected: 3 passed — `import torch` живёт внутри `save_state`/`load_state`,
на импорте модуля его нет.

- [ ] **Step 7: Экспорт и коммит**

`aic/__init__.py`, в `_EXPORTS`: `"Eval": "runs",`

```bash
git add aic/runs.py aic/__init__.py tests/aic/test_runs_eval.py
git commit -m "aic: Eval, операционная точка, атомарные чекпоинты"
```

---

### Task 7: `aic/stats.py` — достоверность прироста

Арифметика переносится дословно: `per_image`, `align_by_stem`,
`paired_bootstrap`, `verdict`, `seeds_needed`, `gate_check`, `gate_metrics`,
`Boot`, `PerImage`, `Verdict`, `Gate`, `Comparison`. Убирается всё, что знало
про конфиг и про диск.

**Files:**
- Create: `aic/stats.py`, `tests/aic/test_stats.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: `Eval` из `aic.runs`; `AICAccumulator`, `harmonic_aic`, `EPS`,
  `FP_AREA_THRESHOLD` из `aic.metric`
- Produces:
  - `PerImage(dice, alarm, is_pos)` — frozen dataclass с массивами `np.ndarray`
  - `per_image(acc: AICAccumulator, op: tuple[float, float, float]) -> PerImage`
  - `align_by_stem(stems_a, stems_b) -> tuple[np.ndarray, np.ndarray]`
  - `Boot(delta, lo, hi, sd)`; `paired_bootstrap(a, b, *, n=2000, seed=0, block=250) -> Boot`
  - `seeds_needed(delta: float, train_sigma: float) -> int | None`
  - `Verdict(label, reason, seeds_needed)`;
    `verdict(delta, boot, train_sigma, *, diverged=()) -> Verdict`
  - `Gate(ref_aic, delta, fired, reason)`;
    `gate_check(curve, samples, aic, *, gate_delta=-0.05, after_samples=24000) -> Gate`
  - `gate_metrics(gate: Gate | None) -> dict`
  - `diverged_keys(a: Mapping, b: Mapping, keys: Sequence[str]) -> list[str]` — `keys` обязателен
  - `Comparison` с `as_dict()` и `report(ref_name, hours_per_run=1.35) -> list[str]`
  - `compare(own: Eval, ref: Eval, *, op, own_op=None, train_sigma=None,
    bootstrap_n=2000, bootstrap_seed=0, diverged=()) -> Comparison`

- [ ] **Step 1: Скопировать модуль и вырезать лишнее**

```bash
git show HEAD:experimental_tools_beliy_russak/stats.py > aic/stats.py
```

Удалить целиком: `BUDGET_KEYS`, `comparable`, `Reference`, `resolve_reference`,
`load_reference`, `GATE_ACTIONS`, `StatsSettings`, `stats_settings`,
`compare_to_reference`. Удалить импорты `json`, `yaml`, `Path`, `pandas` и
`from .config import get_path as _get_path`.

Заменить импорт метрики:

```python
from .metric import EPS, FP_AREA_THRESHOLD, AICAccumulator, harmonic_aic
```

`DEFAULT_*_GRID` здесь больше не нужны — сетки уехали в `Eval.best`.

- [ ] **Step 2: Заменить приватный вызов на публичный**

В `per_image`, строка с распаковкой таблиц:

```python
    pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = accumulator.tables()
```

- [ ] **Step 3: Дописать `diverged_keys` и `compare`**

В конец `aic/stats.py`:

```python
def diverged_keys(a: Mapping[str, Any], b: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    """Ключи, по которым два снапшота разошлись. `keys` обязателен.

    Значения по умолчанию тут нет намеренно. Какие параметры делают прогоны
    несопоставимыми, знает только тот, кто их запускал: у одного это
    `train.epochs`, у другого `n_steps`, у третьего вообще ничего — он сравнивает
    два прогона одного своего скрипта. Список по умолчанию был бы тихой
    регламентацией способа обучения.

    Пути читаются точечной нотацией: `diverged_keys(a, b, ["train.epochs"])`.
    """
    def get(node: Any, dotted: str) -> Any:
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        return node

    return [key for key in keys if get(a, key) != get(b, key)]


def compare(
    own: "Eval",
    ref: "Eval",
    *,
    op: tuple[float, float, float],
    own_op: tuple[float, float, float] | None = None,
    train_sigma: float | None = None,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 0,
    diverged: Sequence[str] = (),
) -> Comparison:
    """Дельта, интервал и вердикт против эталона на общих кадрах.

    `op` — операционная точка ЭТАЛОНА: оба прогона меряются в ней, иначе разница
    включала бы в себя разницу тюнинга порогов. `own_op` — своя тюненая точка,
    справочно; её разрыв с основной дельтой и есть накрутка от подбора порогов.
    """
    idx_self, idx_ref = align_by_stem(own.stems, ref.stems)

    def cut(sample: PerImage, idx: np.ndarray) -> PerImage:
        return PerImage(dice=sample.dice[idx], alarm=sample.alarm[idx], is_pos=sample.is_pos[idx])

    ref_at_op = cut(per_image(ref.acc, op), idx_ref)
    self_at_op = cut(per_image(own.acc, op), idx_self)
    boot = paired_bootstrap(ref_at_op, self_at_op, n=bootstrap_n, seed=bootstrap_seed)

    pos = ref_at_op.is_pos
    if own_op is None:
        delta_own = boot.delta
    else:
        self_at_own = cut(per_image(own.acc, own_op), idx_self)
        delta_own = harmonic_aic(
            self_at_own.dice[pos].mean(), self_at_own.alarm[~pos].mean()
        ) - harmonic_aic(ref_at_op.dice[pos].mean(), ref_at_op.alarm[~pos].mean())

    warning = None
    if idx_self.size < own.stems.size:
        warning = (
            f"val эталона совпадает с текущим только на {idx_self.size} из "
            f"{own.stems.size} кадров. Дельта считается на пересечении, CI будет шире."
        )

    return Comparison(
        delta_ref_op=boot.delta,
        delta_own_op=float(delta_own),
        boot=boot,
        verdict=verdict(boot.delta, boot, train_sigma, diverged=list(diverged)),
        n_common=int(idx_self.size),
        n_pos=int(pos.sum()),
        n_neg=int((~pos).sum()),
        ref_op=tuple(op),
        own_op=tuple(own_op) if own_op is not None else tuple(op),
        train_sigma=train_sigma,
        warning=warning,
    )
```

Добавить в шапку модуля недостающие импорты:

```python
from typing import TYPE_CHECKING, Any, Mapping, Sequence

if TYPE_CHECKING:
    from .runs import Eval
```

`Eval` только в аннотации: `runs` импортирует `metric`, а `stats` не должен
тянуть pandas ради подсказки типа.

Дефолты `gate_check` перенести в подпись (раньше они жили в `stats_settings`):

```python
def gate_check(
    curve: tuple[np.ndarray, np.ndarray],
    samples: int,
    aic: float,
    *,
    gate_delta: float = -0.05,
    after_samples: int = 24000,
) -> Gate:
```

- [ ] **Step 4: Написать тесты**

`tests/aic/test_stats.py`:

```python
"""Достоверность прироста. Арифметика перенесена — тесты стерегут её смысл."""

from __future__ import annotations

import numpy as np
import pytest

from aic.metric import AICAccumulator, harmonic_aic
from aic.runs import Eval
from aic.stats import (
    Boot,
    PerImage,
    align_by_stem,
    compare,
    diverged_keys,
    gate_check,
    gate_metrics,
    paired_bootstrap,
    per_image,
    seeds_needed,
    verdict,
)


def _eval(n_pos=30, n_neg=10, quality=0.8, seed=0, stems=None):
    """Аккумулятор с управляемым качеством: чем выше quality, тем точнее маски."""
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=64)
    probs, gts = [], []
    for i in range(n_pos + n_neg):
        gt = np.zeros((8, 8), dtype=np.float32)
        prob = rng.random((8, 8)).astype(np.float32) * 0.3
        if i < n_pos:
            gt[:4] = 1.0
            prob[:4] = quality
        probs.append(prob)
        gts.append(gt)
    acc.update(np.stack(probs), np.stack(gts))
    names = stems if stems is not None else [f"кадр{i}" for i in range(n_pos + n_neg)]
    return Eval(acc, np.array(names))


def test_per_image_means_match_the_accumulator():
    """Иначе вердикт считался бы не по той метрике, по которой отбирают модели."""
    ev = _eval()
    op = (0.5, 0.0, 0.0)
    sample = per_image(ev.acc, op)
    direct = ev.acc.evaluate(*op)
    assert sample.dice[sample.is_pos].mean() == pytest.approx(direct.dice_pos)
    assert sample.alarm[~sample.is_pos].mean() == pytest.approx(direct.fpr_neg)


def test_align_by_stem_finds_the_intersection():
    a = np.array(["a", "b", "c", "d"])
    b = np.array(["c", "a", "z"])
    idx_a, idx_b = align_by_stem(a, b)
    assert a[idx_a].tolist() == b[idx_b].tolist() == ["a", "c"]


def test_align_by_stem_rejects_duplicates():
    with pytest.raises(ValueError, match="повторяются"):
        align_by_stem(np.array(["a", "a"]), np.array(["a"]))


def test_paired_bootstrap_of_identical_samples_is_zero():
    ev = _eval()
    sample = per_image(ev.acc, (0.5, 0.0, 0.0))
    boot = paired_bootstrap(sample, sample, n=200, seed=0)
    assert boot.delta == pytest.approx(0.0, abs=1e-12)
    assert boot.lo == pytest.approx(0.0, abs=1e-12)
    assert boot.hi == pytest.approx(0.0, abs=1e-12)


def test_paired_bootstrap_is_reproducible():
    a = per_image(_eval(quality=0.6, seed=1).acc, (0.5, 0.0, 0.0))
    b = per_image(_eval(quality=0.9, seed=2).acc, (0.5, 0.0, 0.0))
    assert paired_bootstrap(a, b, n=300, seed=7) == paired_bootstrap(a, b, n=300, seed=7)


def test_paired_bootstrap_refuses_mismatched_samples():
    a = per_image(_eval(n_pos=30).acc, (0.5, 0.0, 0.0))
    b = per_image(_eval(n_pos=20).acc, (0.5, 0.0, 0.0))
    with pytest.raises(ValueError, match="разной длины"):
        paired_bootstrap(a, b, n=10)


def test_seeds_needed_grows_as_the_effect_shrinks():
    assert seeds_needed(0.02, 0.005) < seeds_needed(0.005, 0.005)
    assert seeds_needed(0.0, 0.005) is None


def test_seeds_needed_matches_the_two_sigma_rule():
    # k >= 8 * sigma^2 / delta^2
    assert seeds_needed(0.01, 0.005) == 2


def test_verdict_is_asymmetric_between_win_and_loss():
    """Заявка на выигрыш требует и пола шума, и чистого CI; провал — только пола."""
    sigma = 0.005
    floor = 2.0 * sigma * np.sqrt(2.0)

    dirty = Boot(delta=floor + 0.001, lo=-0.002, hi=0.02, sd=0.006)
    assert verdict(dirty.delta, dirty, sigma).label == "внутри шума обучения"

    clean = Boot(delta=floor + 0.001, lo=0.001, hi=0.02, sd=0.006)
    assert verdict(clean.delta, clean, sigma).label == "подтверждено"

    loss = Boot(delta=-floor - 0.001, lo=-0.03, hi=0.001, sd=0.006)
    assert verdict(loss.delta, loss, sigma).label == "хуже"


def test_verdict_without_sigma_refuses_to_judge():
    boot = Boot(delta=0.05, lo=0.01, hi=0.09, sd=0.02)
    assert verdict(0.05, boot, None).label == "пол шума не задан"


def test_verdict_on_diverged_keys_is_incomparable():
    boot = Boot(delta=0.05, lo=0.01, hi=0.09, sd=0.02)
    got = verdict(0.05, boot, 0.005, diverged=["train.epochs"])
    assert got.label == "несопоставимо"
    assert "train.epochs" in got.reason


def test_gate_holds_its_fire_before_the_threshold():
    curve = (np.array([8000.0, 16000.0]), np.array([0.4, 0.6]))
    gate = gate_check(curve, samples=8000, aic=0.01, after_samples=24000)
    assert gate.fired is False
    assert "рано судить" in gate.reason


def test_gate_fires_on_a_real_lag():
    curve = (np.array([8000.0, 16000.0, 24000.0]), np.array([0.4, 0.6, 0.7]))
    gate = gate_check(curve, samples=24000, aic=0.60, gate_delta=-0.05, after_samples=8000)
    assert gate.fired is True
    assert gate.ref_aic == pytest.approx(0.7)
    assert gate.delta == pytest.approx(-0.1)


def test_gate_interpolates_between_reference_points():
    curve = (np.array([8000.0, 24000.0]), np.array([0.4, 0.8]))
    gate = gate_check(curve, samples=16000, aic=0.6, after_samples=8000)
    assert gate.ref_aic == pytest.approx(0.6)
    assert gate.fired is False


def test_gate_outside_the_reference_curve_stays_silent():
    curve = (np.array([8000.0, 16000.0]), np.array([0.4, 0.6]))
    gate = gate_check(curve, samples=99999, aic=0.1, after_samples=1000)
    assert gate.fired is False
    assert "вне кривой" in gate.reason


def test_gate_metrics_of_none_is_empty():
    assert gate_metrics(None) == {}
    filled = gate_metrics(gate_check(
        (np.array([1000.0, 2000.0]), np.array([0.4, 0.5])), 2000, 0.45, after_samples=100
    ))
    assert set(filled) == {"ref/aic_at_samples", "ref/delta", "ref/gate"}


def test_diverged_keys_requires_an_explicit_list():
    a = {"train": {"epochs": 6}, "data": {"size": 768}}
    b = {"train": {"epochs": 12}, "data": {"size": 768}}
    assert diverged_keys(a, b, ["train.epochs", "data.size"]) == ["train.epochs"]
    assert diverged_keys(a, b, []) == []
    with pytest.raises(TypeError):
        diverged_keys(a, b)


def test_diverged_keys_treats_a_missing_path_as_none():
    assert diverged_keys({"a": 1}, {}, ["a"]) == ["a"]
    assert diverged_keys({}, {}, ["нет.такого"]) == []


def test_compare_against_itself_is_a_flat_zero():
    ev = _eval()
    cmp = compare(ev, ev, op=(0.5, 0.0, 0.0), train_sigma=0.005, bootstrap_n=200)
    assert cmp.delta_ref_op == pytest.approx(0.0, abs=1e-12)
    assert cmp.n_common == len(ev)
    assert cmp.verdict.label == "внутри шума обучения"


def test_compare_sees_a_better_arm():
    weak = _eval(quality=0.55, seed=1)
    strong = _eval(quality=0.95, seed=1)
    cmp = compare(strong, weak, op=(0.5, 0.0, 0.0), train_sigma=0.001, bootstrap_n=300)
    assert cmp.delta_ref_op > 0
    assert cmp.verdict.label in {"подтверждено", "внутри шума обучения"}


def test_compare_warns_about_a_partial_overlap():
    a = _eval(n_pos=20, n_neg=8, stems=[f"общий{i}" for i in range(28)])
    b_names = [f"общий{i}" for i in range(14)] + [f"чужой{i}" for i in range(14)]
    b = _eval(n_pos=20, n_neg=8, stems=b_names)
    cmp = compare(a, b, op=(0.5, 0.0, 0.0), bootstrap_n=100)
    assert cmp.n_common == 14
    assert "совпадает с текущим только" in cmp.warning


def test_compare_reports_both_operating_points():
    own = _eval(quality=0.9, seed=3)
    ref = _eval(quality=0.7, seed=3)
    cmp = compare(own, ref, op=(0.5, 0.0, 0.0), own_op=(0.3, 0.0, 0.0),
                  train_sigma=0.004, bootstrap_n=200)
    text = "\n".join(cmp.report("эталон"))
    assert "в точке эталона" in text
    assert "в своей тюненой точке" in text
    assert set(cmp.as_dict()) >= {"delta_ref_op", "delta_own_op", "ci_lo", "ci_hi", "label"}
```

- [ ] **Step 5: Прогнать**

Run: `python -m pytest tests/aic/test_stats.py -q`
Expected: 22 passed

- [ ] **Step 6: Экспорт и коммит**

`aic/__init__.py`, в `_EXPORTS`:

```python
    # статистика
    "compare": "stats",
    "verdict": "stats",
    "gate_check": "stats",
    "seeds_needed": "stats",
    "paired_bootstrap": "stats",
    "diverged_keys": "stats",
```

```bash
git add aic/stats.py aic/__init__.py tests/aic/test_stats.py
git commit -m "aic: статистика без конфига и без диска"
```

---

### Task 8: `aic/analysis.py` — лидерборд и сравнение прогонов

**Files:**
- Create: `aic/analysis.py`, `tests/aic/test_analysis_board.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: `Run` из `aic.runs`
- Produces:
  - `flatten(obj: Mapping, prefix: str = "") -> dict[str, Any]`
  - `leaderboard(runs_root: str | Path, *, sort_by: str = "best_aic", only_diff: bool = True) -> pd.DataFrame`
  - `compare_table(run_dirs: Sequence[str | Path]) -> pd.DataFrame`
  - `history(run_dirs: Sequence[str | Path], metric: str = "val/aic_tuned") -> pd.DataFrame`

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_analysis_board.py`:

```python
"""Лидерборд по папке runs/. Читает только то, что прогон после себя оставил."""

from __future__ import annotations

import pandas as pd
import pytest

from aic.analysis import compare_table, flatten, history, leaderboard
from aic.runs import Run


def _run(root, name, *, aic, snapshot, epochs=6):
    run = Run.create(root, name, tensorboard=False)
    run.save_snapshot(snapshot)
    for step in range(epochs):
        run.log(step, {"val/aic_tuned": aic * (step + 1) / epochs, "samples": 8000 * (step + 1)})
    run.save_summary({
        "run": name,
        "best_aic": aic,
        "best": {"dice_pos": 0.7, "fpr_neg": 0.07, "mask_threshold": 0.275,
                 "cls_threshold": 0.5, "min_area": 0.0},
        "epochs_done": epochs,
    })
    run.close()
    return run.dir


def test_flatten_uses_dotted_paths():
    assert flatten({"a": {"b": 1}, "c": 2}) == {"a.b": 1, "c": 2}
    assert flatten({}) == {}


def test_leaderboard_sorts_by_best_aic(tmp_path):
    _run(tmp_path, "слабый", aic=0.60, snapshot={"data": {"size": 512}})
    _run(tmp_path, "сильный", aic=0.80, snapshot={"data": {"size": 768}})
    board = leaderboard(tmp_path)
    assert list(board["run"]) == ["сильный", "слабый"]
    assert board["best_aic"].tolist() == [0.80, 0.60]


def test_leaderboard_shows_only_differing_keys(tmp_path):
    _run(tmp_path, "а", aic=0.6, snapshot={"data": {"size": 768}, "train": {"lr": 1e-4}})
    _run(tmp_path, "б", aic=0.7, snapshot={"data": {"size": 768}, "train": {"lr": 3e-4}})
    board = leaderboard(tmp_path)
    assert "lr" in board.columns
    assert "size" not in board.columns


def test_leaderboard_keeps_everything_when_asked(tmp_path):
    _run(tmp_path, "а", aic=0.6, snapshot={"data": {"size": 768}})
    _run(tmp_path, "б", aic=0.7, snapshot={"data": {"size": 768}})
    board = leaderboard(tmp_path, only_diff=False)
    assert "size" in board.columns


def test_leaderboard_of_an_empty_root_is_an_empty_frame(tmp_path):
    assert leaderboard(tmp_path).empty


def test_leaderboard_skips_dirs_without_a_snapshot(tmp_path):
    _run(tmp_path, "настоящий", aic=0.6, snapshot={"a": 1})
    (tmp_path / "мусор").mkdir()
    assert list(leaderboard(tmp_path)["run"]) == ["настоящий"]


def test_leaderboard_falls_back_to_the_log_without_a_summary(tmp_path):
    run = Run.create(tmp_path, "без-сводки", tensorboard=False)
    run.save_snapshot({"a": 1})
    run.log(0, {"val/aic_tuned": 0.42})
    run.close()
    board = leaderboard(tmp_path)
    assert board["best_aic"].iloc[0] == pytest.approx(0.42)


def test_leaderboard_disambiguates_colliding_leaf_names(tmp_path):
    """`loss.seg.bce` и `loss.area.small_seg.bce` не должны схлопнуться в bce."""
    _run(tmp_path, "а", aic=0.6, snapshot={
        "loss": {"seg": {"bce": 1.0}, "area": {"small_seg": {"bce": 0.0}}}})
    _run(tmp_path, "б", aic=0.7, snapshot={
        "loss": {"seg": {"bce": 2.0}, "area": {"small_seg": {"bce": 3.0}}}})
    board = leaderboard(tmp_path)
    assert "loss.seg.bce" in board.columns
    assert "loss.area.small_seg.bce" in board.columns


def test_compare_table_puts_runs_side_by_side(tmp_path):
    a = _run(tmp_path, "а", aic=0.6, snapshot={"train": {"lr": 1e-4, "bs": 8}})
    b = _run(tmp_path, "б", aic=0.7, snapshot={"train": {"lr": 3e-4, "bs": 8}})
    table = compare_table([a, b])
    assert list(table.columns) == ["а", "б"]
    assert "train.lr" in table.index
    assert "train.bs" not in table.index


def test_history_stacks_curves_of_several_runs(tmp_path):
    a = _run(tmp_path, "а", aic=0.6, epochs=3, snapshot={"x": 1})
    b = _run(tmp_path, "б", aic=0.8, epochs=3, snapshot={"x": 2})
    frame = history([a, b])
    assert set(frame["run"]) == {"а", "б"}
    assert len(frame) == 6
    assert {"step", "val/aic_tuned"} <= set(frame.columns)
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_analysis_board.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'aic.analysis'`

- [ ] **Step 3: Написать `aic/analysis.py`**

```python
"""Разбор накопленных прогонов: одна таблица по всей папке runs/.

Показывается не весь снапшот, а только те его ключи, которые между прогонами
РАЗЛИЧАЮТСЯ — иначе таблица становится нечитаемой уже на пятом эксперименте.

Модуль ничего не знает про способ обучения: он читает документированную папку
прогона и разворачивает снапшот точечными путями, чем бы тот ни был заполнен.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd

from .runs import Run

#: ключи снапшота, которые в таблицу не идут никогда: они различаются всегда и
#: ничего не объясняют
HIDE_PREFIXES = ("_source", "name", "seed")


def flatten(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Плоский вид словаря: {'train.lr': 0.0003}."""
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        full = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{full}."))
        else:
            flat[full] = value
    return flat


def _row(run_dir: Path) -> dict | None:
    """Одна строка лидерборда или None, если это не папка прогона."""
    if not (run_dir / "config.yaml").exists():
        return None

    run = Run.open(run_dir)
    row: dict[str, Any] = {"run": run_dir.name, "best_aic": None}

    summary = run.summary
    if summary:
        best = summary.get("best") or {}
        # прогоны, посчитанные до появления гейта по FLOPs, бюджета не знают —
        # у них тут остаётся пусто, и это честнее, чем подставить ноль
        cost = summary.get("budget") or {}
        row.update({
            "best_aic": summary.get("best_aic"),
            "dice_pos": best.get("dice_pos"),
            "fpr_neg": best.get("fpr_neg"),
            "thr": best.get("mask_threshold"),
            "cls_thr": best.get("cls_threshold"),
            "min_area": best.get("min_area"),
            "epochs": summary.get("epochs_done"),
            "gflops": cost.get("gflops"),
            # именно «вне бюджета», а не «не помечен»: прогон с exempt всё равно
            # вне лимита, и в таблице это должно быть видно
            "over_budget": (not cost["within_limit"]) if "within_limit" in cost else None,
        })

    frame = run.history
    if not frame.empty:
        last = frame.iloc[-1]
        row["last_epoch"] = last.get("step")
        row["elapsed"] = round(float(last.get("elapsed_s", 0) or 0) / 60, 1)
        if row.get("best_aic") is None and "val/aic_tuned" in frame.columns:
            row["best_aic"] = float(frame["val/aic_tuned"].max())

    row["_cfg"] = flatten(run.snapshot)
    return row


def _spread_configs(rows: list[dict], only_diff: bool) -> None:
    """Разложить снапшоты по колонкам, оставив (по умолчанию) только различия."""
    configs = [row.pop("_cfg") for row in rows]
    keys = sorted({k for cfg in configs for k in cfg})
    if only_diff:
        keys = [
            k for k in keys
            if not k.startswith(HIDE_PREFIXES)
            and len({str(cfg.get(k)) for cfg in configs}) > 1
        ]

    # колонку зовём коротко, только если короткое имя ни с чем не сталкивается:
    # у `loss.seg.bce` и `loss.area.small_seg.bce` последний кусок общий, и одна
    # колонка молча показывала бы значение только второго из них
    leaves = Counter(key.rsplit(".", 1)[-1] for key in keys)
    labels = {key: (leaf if leaves[leaf] == 1 else key)
              for key, leaf in ((k, k.rsplit(".", 1)[-1]) for k in keys)}

    for row, cfg in zip(rows, configs):
        for key in keys:
            row[labels[key]] = cfg.get(key)


def leaderboard(
    runs_root: str | Path,
    *,
    sort_by: str = "best_aic",
    only_diff: bool = True,
) -> pd.DataFrame:
    """Таблица по всем прогонам в папке, отсортированная по метрике."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return pd.DataFrame()

    rows = [r for r in (_row(d) for d in sorted(runs_root.iterdir()) if d.is_dir()) if r]
    if not rows:
        return pd.DataFrame()

    _spread_configs(rows, only_diff)
    table = pd.DataFrame(rows)
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False, na_position="last")
    return table.reset_index(drop=True)


def compare_table(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Различия снапшотов бок о бок: строки — ключи, колонки — прогоны."""
    dirs = [Path(d) for d in run_dirs]
    configs = [flatten(Run.open(d).snapshot) for d in dirs]
    keys = sorted({k for cfg in configs for k in cfg})
    keys = [
        k for k in keys
        if not k.startswith(HIDE_PREFIXES)
        and len({str(cfg.get(k)) for cfg in configs}) > 1
    ]
    return pd.DataFrame(
        {d.name: [cfg.get(k) for k in keys] for d, cfg in zip(dirs, configs)},
        index=keys,
    )


def history(
    run_dirs: Sequence[str | Path],
    metric: str = "val/aic_tuned",
) -> pd.DataFrame:
    """Кривые нескольких прогонов в одной длинной таблице — для графика."""
    frames = []
    for run_dir in run_dirs:
        run = Run.open(run_dir)
        frame = run.history
        if frame.empty or metric not in frame.columns:
            continue
        part = frame[["step", metric]].copy()
        if "samples" in frame.columns:
            part["samples"] = frame["samples"]
        part.insert(0, "run", run.dir.name)
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
```

- [ ] **Step 4: Прогнать**

Run: `python -m pytest tests/aic/test_analysis_board.py -q`
Expected: 11 passed

- [ ] **Step 5: Экспорт и коммит**

`aic/__init__.py`, в `_EXPORTS`:

```python
    # анализ
    "leaderboard": "analysis",
    "compare_table": "analysis",
```

```bash
git add aic/analysis.py aic/__init__.py tests/aic/test_analysis_board.py
git commit -m "aic: лидерборд и сравнение прогонов"
```

---

### Task 9: `OofView` — разбор валидации без второй копии арифметики

Сейчас `analysis/oof_report.OofView` разворачивает гистограммы своей копией
кода поверх сырого `npz`. Здесь дубль исчезает: таблицы приходят из
`AICAccumulator.tables()`.

**Files:**
- Modify: `aic/analysis.py`
- Create: `tests/aic/test_analysis_oof.py`

**Interfaces:**
- Consumes: `Eval`, `Run` из `aic.runs`; `AICAccumulator.tables()`,
  `FP_AREA_THRESHOLD`, `harmonic_aic`, `DEFAULT_AREA_GRID` из `aic.metric`
- Produces:
  - `AREA_BINS`, `MISS_DICE`
  - `OofView(ev: Eval, rows: pd.DataFrame | None = None)` со свойствами
    `pred`, `inter`, `area`, `dice`, `is_pos`, `gt_frac`, `n_bins`, `cls_prob`
  - `OofView.from_run(run_dir: str | Path, name: str = "val") -> OofView`
  - `OofView.by_area(op) -> pd.DataFrame` — корзины по площади GT
  - `OofView.by_column(column: str, op) -> pd.DataFrame` — разрез по домену/генератору
  - `OofView.ceilings(op) -> dict[str, float]` — потолки идеального классификатора
    кадра и идеального порога на кадр

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_analysis_oof.py`:

```python
"""Разбор валидации по корзинам. Арифметика — одна, из AICAccumulator.tables()."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aic.analysis import OofView
from aic.metric import AICAccumulator
from aic.runs import Eval, Run


def _view(n_pos=12, n_neg=6, seed=0):
    rng = np.random.default_rng(seed)
    acc = AICAccumulator(n_bins=64)
    probs, gts, gt_fracs = [], [], []
    for i in range(n_pos + n_neg):
        gt = np.zeros((10, 10), dtype=np.float32)
        prob = rng.random((10, 10)).astype(np.float32) * 0.2
        if i < n_pos:
            filled = 1 + i % 5          # от 1% до 5% площади
            gt[:filled] = 1.0
            prob[:filled] = 0.9
        probs.append(prob)
        gts.append(gt)
        gt_fracs.append(float(gt.mean()))
    acc.update(np.stack(probs), np.stack(gts))

    rows = pd.DataFrame({
        "stem": [f"кадр{i}" for i in range(n_pos + n_neg)],
        "domain": ["a" if i % 2 else "b" for i in range(n_pos + n_neg)],
        "generator": ["g1"] * (n_pos + n_neg),
    })
    return OofView(Eval(acc, rows["stem"].to_numpy()), rows)


def test_tables_come_from_the_accumulator_not_a_second_copy():
    """Гарантия, что дубль убран, а не переехал."""
    view = _view()
    pred, inter, gt_sum, n_pixels, cls_prob = view.ev.acc.tables()
    assert np.array_equal(view.pred, pred)
    assert np.array_equal(view.inter, inter)
    assert np.array_equal(view.area, pred / n_pixels[:, None])


def test_area_buckets_cover_every_positive_frame():
    view = _view()
    table = view.by_area((0.5, 0.0, 0.0))
    assert table["n"].sum() == int(view.is_pos.sum())
    assert {"n", "dice", "miss_share"} <= set(table.columns)


def test_area_buckets_show_small_masks_are_harder():
    """Ради этого разреза модуль и существует: где именно теряется Dice."""
    view = _view(n_pos=20)
    table = view.by_area((0.5, 0.0, 0.0))
    filled = table[table["n"] > 0]
    assert filled["dice"].is_monotonic_increasing or len(filled) == 1


def test_by_column_splits_on_a_row_field():
    view = _view()
    table = view.by_column("domain", (0.5, 0.0, 0.0))
    assert set(table.index) == {"a", "b"}
    assert table["n"].sum() == len(view.ev)


def test_by_column_without_rows_says_so():
    acc = AICAccumulator(n_bins=32)
    rng = np.random.default_rng(0)
    acc.update(rng.random((3, 4, 4)), (rng.random((3, 4, 4)) > 0.5).astype(np.float32))
    view = OofView(Eval(acc, np.array(["a", "b", "c"])))
    with pytest.raises(ValueError, match="строк"):
        view.by_column("domain", (0.5, 0.0, 0.0))


def test_ceilings_are_at_least_the_current_score():
    view = _view()
    op = (0.5, 0.0, 0.0)
    ceilings = view.ceilings(op)
    current = view.ev.acc.evaluate(*op).aic
    assert ceilings["current"] == pytest.approx(current)
    assert ceilings["perfect_cls"] >= current - 1e-9
    assert ceilings["perfect_thr"] >= current - 1e-9


def test_from_run_reads_the_folder(tmp_path):
    rng = np.random.default_rng(1)
    acc = AICAccumulator(n_bins=32)
    acc.update(rng.random((4, 6, 6)), (rng.random((4, 6, 6)) > 0.6).astype(np.float32))
    rows = pd.DataFrame({"stem": [f"к{i}" for i in range(4)], "domain": list("aabb")})

    run = Run.create(tmp_path, "разбор", tensorboard=False)
    run.save_eval(acc, rows)
    run.close()

    view = OofView.from_run(run.dir)
    assert len(view.ev) == 4
    assert set(view.by_column("domain", (0.5, 0.0, 0.0)).index) == {"a", "b"}
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_analysis_oof.py -q`
Expected: FAIL, `ImportError: cannot import name 'OofView' from 'aic.analysis'`

- [ ] **Step 3: Дописать `OofView` в `aic/analysis.py`**

В конец модуля; в шапку добавить импорты:

```python
import numpy as np

from .metric import FP_AREA_THRESHOLD, harmonic_aic
from .runs import Eval, Run
```

```python
#: ниже этого Dice предсказание считаем полным промахом, а не неточностью
MISS_DICE = 0.05
#: границы корзин по доле площади GT. Главный разрез: у baseline кадры с маской
#: меньше 1% дают Dice 0.135 и 71% полных промахов, а больше 12% — 0.86+ и 2%
AREA_BINS = (0.0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1.01)


class OofView:
    """Развёрнутая валидация прогона: где теряется Dice и каков потолок.

    Ни модели, ни GPU, ни повторного прогона не нужно — в аккумуляторе лежат
    гистограммы по каждому кадру, из которых восстанавливается |P_t| и |P_t ∩ G|
    для любого порога.

    Таблицы берутся из `AICAccumulator.tables()`, а не разворачиваются заново.
    Второй реализации той же арифметики здесь быть не должно: разойдясь, они
    заставили бы считать вердикты не по той метрике, по которой отбирают модели.
    """

    def __init__(self, ev: Eval, rows: pd.DataFrame | None = None) -> None:
        self.ev = ev
        self.rows = rows
        pred, inter, gt_sum, n_pixels, cls_prob = ev.acc.tables()
        self.n_bins = ev.acc.n_bins
        self.pred = pred.astype(np.float64)
        self.inter = inter.astype(np.float64)
        self.gt_sum = gt_sum
        self.n_pixels = n_pixels
        self.cls_prob = cls_prob
        self.area = self.pred / self.n_pixels[:, None]
        self.dice = 2.0 * self.inter / (self.pred + self.gt_sum[:, None] + 1e-6)
        self.is_pos = self.gt_sum > 0
        self.gt_frac = self.gt_sum / self.n_pixels

    @classmethod
    def from_run(cls, run_dir: str | Path, name: str = "val") -> "OofView":
        run = Run.open(run_dir)
        return cls(run.load_eval(name), run.load_rows(name))

    def _bin_index(self, mask_threshold: float) -> int:
        # ровно как в AICAccumulator.sweep, иначе разрез считался бы в другой точке
        return int(np.clip(int(mask_threshold * self.n_bins), 0, self.n_bins - 1))

    def at(self, op: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        """`(dice, area)` каждого кадра в операционной точке."""
        mask_threshold, cls_threshold, min_area = op
        k = self._bin_index(mask_threshold)
        keep = (self.cls_prob >= cls_threshold) & (self.area[:, k] >= min_area)
        return (
            np.where(keep, self.dice[:, k], 0.0),
            np.where(keep, self.area[:, k], 0.0),
        )

    def by_area(self, op: tuple[float, float, float]) -> pd.DataFrame:
        """Корзины по доле площади GT — главный разрез, только по позитивам."""
        dice, _ = self.at(op)
        pos = self.is_pos
        bucket = np.digitize(self.gt_frac[pos], AREA_BINS[1:-1])

        records = []
        for i in range(len(AREA_BINS) - 1):
            sel = bucket == i
            label = f"{AREA_BINS[i] * 100:g}-{AREA_BINS[i + 1] * 100:g}%"
            values = dice[pos][sel]
            records.append({
                "bucket": label,
                "n": int(sel.sum()),
                "dice": float(values.mean()) if values.size else 0.0,
                "miss_share": float((values < MISS_DICE).mean()) if values.size else 0.0,
            })
        return pd.DataFrame(records)

    def by_column(self, column: str, op: tuple[float, float, float]) -> pd.DataFrame:
        """Разрез по полю строк валидации: не выиграл ли прогон одним источником."""
        if self.rows is None or column not in self.rows.columns:
            available = [] if self.rows is None else list(self.rows.columns)
            raise ValueError(
                f"нет колонки {column!r} в таблице строк валидации (есть {available}). "
                "Передайте rows в OofView или сохраните их через Run.save_eval"
            )
        dice, area = self.at(op)
        frame = pd.DataFrame({
            column: self.rows[column].to_numpy(),
            "dice": dice,
            "alarm": (area >= FP_AREA_THRESHOLD).astype(float),
            "is_pos": self.is_pos,
        })

        records = {}
        for key, part in frame.groupby(column):
            pos, neg = part[part["is_pos"]], part[~part["is_pos"]]
            dice_pos = float(pos["dice"].mean()) if len(pos) else 0.0
            fpr_neg = float(neg["alarm"].mean()) if len(neg) else 0.0
            records[key] = {
                "n": len(part),
                "n_pos": len(pos),
                "dice_pos": dice_pos,
                "fpr_neg": fpr_neg,
                "aic": harmonic_aic(dice_pos, fpr_neg),
            }
        return pd.DataFrame(records).T

    def ceilings(self, op: tuple[float, float, float]) -> dict[str, float]:
        """Сколько ещё осталось у идеального классификатора и идеального порога.

        Если потолок близко, крутить эту ручку дальше бессмысленно — и это
        единственный способ узнать про неё, не потратив прогон.
        """
        mask_threshold, cls_threshold, min_area = op
        k = self._bin_index(mask_threshold)
        pos, neg = self.is_pos, ~self.is_pos

        dice_now, area_now = self.at(op)
        current = harmonic_aic(
            float(dice_now[pos].mean()) if pos.any() else 0.0,
            float((area_now[neg] >= FP_AREA_THRESHOLD).mean()) if neg.any() else 0.0,
        )

        # идеальный классификатор кадра: негативы обнулены, позитивы не тронуты
        perfect_cls = harmonic_aic(float(self.dice[pos, k].mean()) if pos.any() else 0.0, 0.0)

        # идеальный порог на кадр: у каждого позитива берётся его лучший Dice,
        # негативы остаются как есть в текущей точке
        best_per_frame = self.dice[pos].max(axis=1) if pos.any() else np.zeros(0)
        perfect_thr = harmonic_aic(
            float(best_per_frame.mean()) if best_per_frame.size else 0.0,
            float((area_now[neg] >= FP_AREA_THRESHOLD).mean()) if neg.any() else 0.0,
        )
        return {"current": current, "perfect_cls": perfect_cls, "perfect_thr": perfect_thr}
```

- [ ] **Step 4: Прогнать**

Run: `python -m pytest tests/aic/test_analysis_oof.py -q`
Expected: 7 passed

- [ ] **Step 5: Коммит**

```bash
git add aic/analysis.py tests/aic/test_analysis_oof.py
git commit -m "aic: разбор валидации без второй копии арифметики"
```

---

### Task 10: `aic/data.py` — разбор имён, чтение, индекс

**Files:**
- Create: `aic/data.py`, `tests/aic/test_data_index.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: `Workspace` из `aic.paths`
- Produces:
  - `GT_POSITIVE_VALUE = 128`
  - `imread(path, flags=cv2.IMREAD_COLOR) -> np.ndarray | None`
  - `imwrite(path, image, params=None) -> bool`
  - `strip_hash(name) -> str`, `stem_of(path) -> str`,
    `parse_domain(stem) -> str`, `parse_generator(stem) -> str`,
    `parse_group_id(chng_stem, orgl_path) -> str`
  - `build_index(ws: Workspace, *, csv_path=None, out_path=None, workers=12, limit=None) -> pd.DataFrame`
  - `load_index(path: str | Path) -> pd.DataFrame`
  - `summarize_index(df: pd.DataFrame) -> str`

- [ ] **Step 1: Собрать модуль из трёх существующих**

```bash
git show HEAD:experimental_tools_beliy_russak/imageio.py  > /tmp/aic_imageio.py
git show HEAD:experimental_tools_beliy_russak/indexing.py > /tmp/aic_indexing.py
```

`aic/data.py` = докстринг (ниже) + тело `imageio.py` без его шапки + тело
`indexing.py` без импорта `.workspace` и без импорта `.imageio`.

Докстринг модуля:

```python
"""Данные задачи: чтение, разбор имён, индекс, фолды, предкэш.

Специфично для датасета AI Challenge, но ничего не решает за вас в обучении:
здесь только то, из чего каждый строит свой Dataset.

Пути берутся из переданного `Workspace`, а не из глобального состояния — иначе
порядок импортов начинал бы влиять на то, какие файлы прочитаются.
"""
```

- [ ] **Step 2: Развязать `_probe` с глобальным воркспейсом**

`_probe` выполняется в отдельном процессе и звал модульный `resolve()`. Корень
датасета теперь едет в аргументах:

```python
def _probe(args: tuple[int, str, str, str]) -> tuple[int, int, int, float, int, int]:
    """Читает маску и заголовок изображения. Выполняется в отдельном процессе.

    Корень датасета приезжает аргументом, а не берётся из глобального состояния:
    дочерний процесс на Windows стартует заново и никакого `set_workspace` в нём
    не было бы.
    """
    import cv2
    from PIL import Image

    row_id, gt_rel, img_rel, dataset_root = args
    root = Path(dataset_root)
    mask = imread(root / str(gt_rel).replace("\\", "/"), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return row_id, 0, 0, -1.0, 0, 0
    height, width = mask.shape[:2]
    area = float(np.count_nonzero(mask >= GT_POSITIVE_VALUE)) / float(mask.size)

    try:
        with Image.open(root / str(img_rel).replace("\\", "/")) as im:
            img_w, img_h = im.size
    except Exception:
        img_w = img_h = 0
    return row_id, height, width, area, img_h, img_w
```

- [ ] **Step 3: Переписать подпись `build_index` под воркспейс**

Заменить первые строки функции и сборку задач:

```python
def build_index(
    ws: Workspace,
    *,
    csv_path: str | Path | None = None,
    out_path: str | Path | None = None,
    workers: int = 12,
    limit: int | None = None,
) -> pd.DataFrame:
    csv_path = Path(csv_path) if csv_path is not None else ws.train_csv
    out_path = Path(out_path) if out_path is not None else ws.index_path
    df = pd.read_csv(csv_path)
```

и список задач:

```python
    root = str(ws.dataset_root)
    tasks = [
        (i, gt, img, root)
        for i, (gt, img) in enumerate(zip(df["gt_path"], df["chng_path"]))
    ]
```

`load_index` теряет значение по умолчанию — путь обязателен:

```python
def load_index(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"нет индекса {path}. Собери его: aic.data.build_index(ws)")
    return pd.read_parquet(path)
```

`summarize` переименовать в `summarize_index`: в одном модуле с фолдами имя
`summarize` встретится дважды.

- [ ] **Step 4: Написать тесты**

`tests/aic/test_data_index.py`:

```python
"""Разбор имён датасета и сборка индекса."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from aic.data import (
    build_index,
    imread,
    imwrite,
    load_index,
    parse_domain,
    parse_generator,
    parse_group_id,
    stem_of,
    strip_hash,
    summarize_index,
)
from aic.paths import Workspace


def test_strip_hash_removes_only_a_real_hash_prefix():
    assert strip_hash("bd3895b63372e1f2_кадр.jpg") == "кадр.jpg"
    assert strip_hash("кадр.jpg") == "кадр.jpg"


def test_stem_of_drops_dir_hash_and_suffix():
    assert stem_of("stage1/train/img/bd3895b63372e1f2_coco_1234.jpg") == "coco_1234"


@pytest.mark.parametrize(
    "stem,domain",
    [
        ("coco_1234", "coco"),
        ("raise_1234", "raise"),
        ("openimages_abc", "openimages"),
        ("D01_something", "vision"),
        ("4242", "plain"),
        ("42_something", "numid"),
        ("что-то-своё", "other"),
    ],
)
def test_parse_domain(stem, domain):
    assert parse_domain(stem) == domain


def test_parse_generator_finds_the_method_token():
    assert parse_generator("coco_123_powerpaint-v2_large") == "powerpaint"
    assert parse_generator("coco_123") == "none"


def test_group_id_comes_from_the_original_when_it_is_known():
    """Префикс источника есть у изменённого файла и нет у оригинала."""
    assert parse_group_id("openimages_bd3895_removeanything", "bd3895.jpg") == "bd3895"


def test_group_id_is_derived_by_cutting_the_method_token():
    assert parse_group_id("coco_1234_powerpaint_large", None) == "1234"


def test_group_id_is_shared_by_manipulations_of_one_frame():
    """На этом стоит групповой сплит: разъедется — будет утечка."""
    a = parse_group_id("coco_1234_powerpaint_large", None)
    b = parse_group_id("coco_1234_lama_small", None)
    assert a == b


def test_imwrite_and_imread_roundtrip_a_cyrillic_path(tmp_path):
    """cv2.imwrite на путях с кириллицей молча возвращает False — обёртка нужна."""
    path = tmp_path / "папка с пробелом" / "маска.png"
    image = np.zeros((4, 6), dtype=np.uint8)
    image[0] = 255
    assert imwrite(path, image) is True
    back = imread(path, cv2.IMREAD_GRAYSCALE)
    assert back is not None
    assert np.array_equal(back, image)


def test_imread_of_a_missing_file_is_none(tmp_path):
    assert imread(tmp_path / "нет.png") is None


def _dataset(tmp_path):
    """Крошечный датасет: два позитива одного оригинала и один негатив."""
    ws = Workspace(tmp_path)
    stage = ws.dataset_root / "stage1"
    (stage / "train" / "img").mkdir(parents=True)
    (stage / "train" / "gt").mkdir(parents=True)

    rows = []
    for name, filled in [("coco_1_powerpaint", 3), ("coco_1_lama", 5), ("coco_2", 0)]:
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        gt = np.zeros((10, 10), dtype=np.uint8)
        gt[:filled] = 255
        imwrite(stage / "train" / "img" / f"{name}.png", img)
        imwrite(stage / "train" / "gt" / f"{name}.png", gt)
        rows.append({
            "chng_img_path": f"stage1/train/img/{name}.png",
            "gt_path": f"stage1/train/gt/{name}.png",
            "orgl_img_path": None,
        })
    ws.train_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(ws.train_csv, index=False)
    return ws


def test_build_index_collects_sizes_areas_and_groups(tmp_path):
    ws = _dataset(tmp_path)
    df = build_index(ws, workers=1)

    assert len(df) == 3
    assert set(df["stem"]) == {"coco_1_powerpaint", "coco_1_lama", "coco_2"}
    assert df["height"].tolist() == [10, 10, 10]
    assert int(df["is_negative"].sum()) == 1
    assert not df["broken"].any()
    assert df[df["stem"] != "coco_2"]["group_id"].nunique() == 1


def test_build_index_writes_and_reads_back(tmp_path):
    ws = _dataset(tmp_path)
    build_index(ws, workers=1)
    assert ws.index_path.exists()
    assert len(load_index(ws.index_path)) == 3


def test_load_index_of_a_missing_file_tells_how_to_build_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="build_index"):
        load_index(tmp_path / "нет.parquet")


def test_build_index_marks_a_broken_mask(tmp_path):
    ws = _dataset(tmp_path)
    (ws.dataset_root / "stage1" / "train" / "gt" / "coco_2.png").write_bytes(b"не картинка")
    df = build_index(ws, workers=1)
    assert int(df["broken"].sum()) == 1
    # битая строка не считается негативом: у неё площадь -1, а не 0
    assert int(df["is_negative"].sum()) == 0


def test_summarize_index_mentions_the_counts(tmp_path):
    ws = _dataset(tmp_path)
    text = summarize_index(build_index(ws, workers=1))
    assert "строк: 3" in text
```

- [ ] **Step 5: Прогнать**

Run: `python -m pytest tests/aic/test_data_index.py -q`
Expected: 20 passed

- [ ] **Step 6: Экспорт и коммит**

`aic/__init__.py`, в `_EXPORTS`: `"build_index": "data",` и `"load_index": "data",`

```bash
git add aic/data.py aic/__init__.py tests/aic/test_data_index.py
git commit -m "aic: индекс датасета и разбор имён без глобального воркспейса"
```

---

### Task 11: `aic/data.py` — фолды и предкэш

**Files:**
- Modify: `aic/data.py`
- Create: `tests/aic/test_data_folds.py`

**Interfaces:**
- Consumes: `load_index`, `imread`, `imwrite` из Task 10
- Produces:
  - `area_bucket(area: float) -> int`, `make_strata(df) -> pd.Series`
  - `make_folds(df: pd.DataFrame, *, n_folds=5, seed=42, out_path=None) -> pd.DataFrame`
  - `load_folds(path: str | Path) -> pd.DataFrame`
  - `check_leakage(df: pd.DataFrame) -> dict`
  - `summarize_folds(df: pd.DataFrame) -> str`
  - `cache_root(ws: Workspace, max_side: int) -> Path`
  - `cached_path(ws: Workspace, rel_path: str, max_side: int, is_mask: bool) -> Path`
  - `build_cache(ws, df, *, max_side=768, workers=12, quality=95, include_originals=True) -> dict[str, int]`
  - `cache_size_gb(ws: Workspace, max_side: int) -> float`

- [ ] **Step 1: Дописать содержимое `splits.py` и `precache.py`**

```bash
git show HEAD:experimental_tools_beliy_russak/splits.py   >> aic/data.py
git show HEAD:experimental_tools_beliy_russak/precache.py >> aic/data.py
```

Затем убрать из вставленного: повторные `from __future__`, шапочные докстринги,
дублирующиеся импорты, `from .workspace import ...`, `from .indexing import ...`.

- [ ] **Step 2: Развернуть `make_folds` на готовую таблицу**

Раньше функция сама читала индекс с диска. Теперь принимает его аргументом:
читать данные должен тот, кто их и так уже держит в руках.

```python
def make_folds(
    df: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    out_path: str | Path | None = None,
) -> pd.DataFrame:
    """Групповой стратифицированный сплит по `group_id`.

    Групповой обязательно: один оригинал порождает несколько манипуляций, и,
    разъехавшись по фолдам, они дают утечку — модель видит тот же кадр.
    """
    df = df[~df["broken"]].reset_index(drop=True)
    ...
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df
```

Середина функции (страты, схлопывание редких, `StratifiedGroupKFold`, проверка
`fold < 0`) не меняется.

`load_folds(path)` — путь обязателен, как у `load_index`.
`summarize` из `splits.py` переименовать в `summarize_folds`.

- [ ] **Step 3: Развязать предкэш с воркспейсом**

```python
def cache_root(ws: Workspace, max_side: int) -> Path:
    return ws.cache / f"s{max_side}"


def cached_path(ws: Workspace, rel_path: str, max_side: int, is_mask: bool) -> Path:
    rel = Path(str(rel_path).replace("\\", "/"))
    if is_mask:
        rel = rel.with_suffix(".png")
    return cache_root(ws, max_side) / rel
```

`_process_one` получает корни в кортеже задачи так же, как `_probe` в Task 10:
`(rel_path, max_side, is_mask, quality, dataset_root, cache_dir)`. `build_cache`
получает первым аргументом `ws` и складывает эти два пути в каждую задачу.

- [ ] **Step 4: Написать тесты**

`tests/aic/test_data_folds.py`:

```python
"""Фолды без утечки и предкэш."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from aic.data import (
    area_bucket,
    build_cache,
    cache_size_gb,
    cached_path,
    check_leakage,
    imread,
    imwrite,
    load_folds,
    make_folds,
    summarize_folds,
)
from aic.paths import Workspace


def _index(n_groups=20, per_group=3):
    rows = []
    for g in range(n_groups):
        for k in range(per_group):
            rows.append({
                "stem": f"кадр_{g}_{k}",
                "group_id": f"группа{g}",
                "domain": "coco" if g % 2 else "raise",
                "generator": "lama" if k else "none",
                "mask_area": 0.0 if k == 0 else 0.02 * (k + 1),
                "is_negative": k == 0,
                "broken": False,
                "chng_path": f"img/{g}_{k}.png",
                "gt_path": f"gt/{g}_{k}.png",
                "orgl_path": None,
            })
    return pd.DataFrame(rows)


def test_area_bucket_is_monotonic():
    assert area_bucket(0.0) <= area_bucket(0.02) <= area_bucket(0.5)


def test_every_row_lands_in_exactly_one_fold():
    folds = make_folds(_index(), n_folds=5, seed=0)
    assert set(folds["fold"]) == {0, 1, 2, 3, 4}
    assert (folds["fold"] >= 0).all()
    assert len(folds) == 60


def test_a_group_never_splits_across_folds():
    """Главное свойство: иначе модель видит тот же исходный кадр в валидации."""
    folds = make_folds(_index(), n_folds=5, seed=0)
    per_group = folds.groupby("group_id")["fold"].nunique()
    assert (per_group == 1).all()
    assert check_leakage(folds)["groups_in_many_folds"] == 0


def test_folds_are_reproducible_for_a_seed():
    a = make_folds(_index(), n_folds=5, seed=7)["fold"].tolist()
    b = make_folds(_index(), n_folds=5, seed=7)["fold"].tolist()
    assert a == b


def test_broken_rows_are_dropped():
    df = _index()
    df.loc[0, "broken"] = True
    assert len(make_folds(df, n_folds=5, seed=0)) == len(df) - 1


def test_make_folds_writes_only_when_asked(tmp_path):
    df = _index()
    make_folds(df, n_folds=5, seed=0)
    assert not list(tmp_path.iterdir())

    out = tmp_path / "folds.parquet"
    make_folds(df, n_folds=5, seed=0, out_path=out)
    assert len(load_folds(out)) == 60


def test_summarize_folds_mentions_the_split():
    text = summarize_folds(make_folds(_index(), n_folds=5, seed=0))
    assert "fold" in text.lower()


def test_cached_path_mirrors_the_relative_layout(tmp_path):
    ws = Workspace(tmp_path)
    got = cached_path(ws, "stage1/train/img/a.jpg", 768, is_mask=False)
    assert got == ws.cache / "s768" / "stage1" / "train" / "img" / "a.jpg"


def test_masks_are_cached_as_png(tmp_path):
    """JPEG на маске дал бы значения между 0 и 255 — GT перестал бы быть GT."""
    ws = Workspace(tmp_path)
    got = cached_path(ws, "stage1/train/gt/a.jpg", 768, is_mask=True)
    assert got.suffix == ".png"


def test_build_cache_shrinks_the_long_side(tmp_path):
    ws = Workspace(tmp_path)
    src = ws.dataset_root / "img"
    src.mkdir(parents=True)
    imwrite(src / "большая.png", np.zeros((400, 200, 3), dtype=np.uint8))
    gt = ws.dataset_root / "gt"
    gt.mkdir(parents=True)
    imwrite(gt / "большая.png", np.zeros((400, 200), dtype=np.uint8))

    df = pd.DataFrame([{"chng_path": "img/большая.png", "gt_path": "gt/большая.png",
                        "orgl_path": None}])
    build_cache(ws, df, max_side=100, workers=1)

    cached = imread(cached_path(ws, "img/большая.png", 100, is_mask=False))
    assert cached is not None
    assert max(cached.shape[:2]) == 100


def test_cache_size_of_an_absent_cache_is_zero(tmp_path):
    assert cache_size_gb(Workspace(tmp_path), 768) == 0.0
```

- [ ] **Step 5: Прогнать**

Run: `python -m pytest tests/aic/test_data_folds.py -q`
Expected: 11 passed

- [ ] **Step 6: Коммит**

```bash
git add aic/data.py tests/aic/test_data_folds.py
git commit -m "aic: фолды и предкэш"
```

---

### Task 12: `aic/budget.py` — строгие GFLOPs над модулем

**Files:**
- Create: `aic/budget.py`, `tests/aic/test_budget.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Produces:
  - `LIMIT_GFLOPS = 100.0`
  - `count_gflops(model: nn.Module, size: int, *, channels: int = 3) -> float`
  - `Verdict(gflops, limit, size, n_views=1, n_models=1)` со свойствами
    `within_limit`, `ok`, `text` и методом `as_dict()`
  - `check(model, size, *, n_views=1, n_models=1, limit=LIMIT_GFLOPS) -> Verdict`
  - `largest_fitting_size(build, *, start=512, n_views=1, n_models=1, limit=LIMIT_GFLOPS, step=32) -> int | None`
  - `rejection_text(build, verdict: Verdict) -> str`

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_budget.py`:

```python
"""Бюджет вычислений: строгие FLOPs, а не MACs."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", reason="бюджет требует torch")
import torch.nn as nn  # noqa: E402

from aic.budget import (  # noqa: E402
    LIMIT_GFLOPS,
    check,
    count_gflops,
    largest_fitting_size,
    rejection_text,
)


class Conv(nn.Module):
    """Одна свёртка: её FLOPs считаются на бумаге и проверяются точно."""

    def __init__(self, cin=3, cout=16, k=3):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)

    def forward(self, x):
        return self.conv(x)


def test_count_gflops_counts_strict_flops_not_macs():
    """MAC = две операции. Счётчики, пишущие MACs, дали бы вдвое меньше."""
    model = Conv(3, 16, 3).eval()
    size = 64
    macs = 3 * 16 * 3 * 3 * size * size
    assert count_gflops(model, size) == pytest.approx(2 * macs / 1e9, rel=1e-6)


def test_count_gflops_scales_quadratically_with_side():
    model = Conv().eval()
    assert count_gflops(model, 128) == pytest.approx(4 * count_gflops(model, 64), rel=1e-6)


def test_count_gflops_works_on_the_meta_device():
    """FlopCounterMode работает на уровне диспетчера: настоящих тензоров не надо."""
    model = Conv().eval().to("meta")
    assert count_gflops(model, 64) > 0


def test_check_says_within_limit_for_a_tiny_model():
    verdict = check(Conv().eval(), 64)
    assert verdict.within_limit is True
    assert verdict.ok is True
    assert "в бюджете" in verdict.text
    assert verdict.limit == LIMIT_GFLOPS


def test_tta_and_ensemble_multiply_the_cost():
    model = Conv().eval()
    one = check(model, 64)
    four = check(model, 64, n_views=2, n_models=2)
    assert four.gflops == pytest.approx(4 * one.gflops)
    assert "x2 видов TTA" in four.text
    assert "x2 моделей" in four.text


def test_check_rejects_what_does_not_fit():
    verdict = check(Conv(3, 512, 7).eval(), 512, limit=1.0)
    assert verdict.within_limit is False
    assert verdict.ok is False
    assert "превышение" in verdict.text


def test_as_dict_is_json_ready():
    payload = check(Conv().eval(), 64).as_dict()
    assert set(payload) == {"gflops", "limit_gflops", "within_limit"}


def test_largest_fitting_size_finds_a_multiple_of_step():
    got = largest_fitting_size(lambda size: Conv(3, 64, 3).eval(), start=512, limit=5.0)
    assert got is not None and got % 32 == 0
    build = lambda size: Conv(3, 64, 3).eval()  # noqa: E731
    assert count_gflops(build(got), got) <= 5.0
    assert count_gflops(build(got + 32), got + 32) > 5.0


def test_largest_fitting_size_is_none_when_nothing_fits():
    assert largest_fitting_size(lambda size: Conv(3, 512, 7).eval(), limit=1e-9) is None


def test_the_builder_receives_the_size():
    """Энкодеры с оконным вниманием собираются только под свой img_size."""
    seen = []

    def build(size):
        seen.append(size)
        return Conv().eval()

    largest_fitting_size(build, start=256, limit=5.0)
    assert seen and all(isinstance(s, int) for s in seen)


def test_rejection_text_answers_the_next_question():
    verdict = check(Conv(3, 256, 5).eval(), 512, limit=2.0)
    text = rejection_text(lambda size: Conv(3, 256, 5).eval(), verdict)
    assert "превышение" in text
    assert "укладывается вход" in text or "ни одно разрешение" in text
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_budget.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'aic.budget'`

- [ ] **Step 3: Написать `aic/budget.py`**

```python
"""Бюджет вычислений: не больше 100 строгих GFLOPs на одно изображение.

Регламент задаёт лимит в классических FLOPs, где умножение-сложение (MAC)
считается за две операции. Популярные счётчики (fvcore, thop, ptflops) пишут в
выводе «FLOPs», а считают MACs, то есть вдвое меньше — и решение, собранное по
их числу, укладывается в лимит только на бумаге. Источником правды в спорных
случаях регламент называет `torch.utils.flop_counter.FlopCounterMode`, поэтому
здесь используется только он.

Считается ГОТОВАЯ модель, а не конфиг: библиотека не знает, как вы её собираете.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

#: лимит регламента, строгие FLOPs на одно изображение
LIMIT_GFLOPS = 100.0


def count_gflops(model, size: int, *, channels: int = 3) -> float:
    """Строгие GFLOPs одного forward на входе (1, channels, size, size).

    Модель считается там, где лежит: перекладывать её здесь нельзя, иначе
    вызывающий получил бы обратно испорченный объект. На `meta`-устройстве
    работает и считает то же самое, но без арифметики и без памяти — 0.1 с
    против нескольких секунд на 768px.
    """
    import torch
    from torch.utils.flop_counter import FlopCounterMode

    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    counter = FlopCounterMode(display=False)
    with torch.no_grad(), counter:
        model(torch.zeros(1, channels, size, size, device=device))
    return counter.get_total_flops() / 1e9


@dataclass(frozen=True)
class Verdict:
    """Влезает ли одно изображение в лимит."""

    gflops: float
    limit: float
    size: int
    n_views: int = 1
    n_models: int = 1

    @property
    def within_limit(self) -> bool:
        return self.gflops <= self.limit

    #: пометок и послаблений здесь нет: это уровень пайплайна, а не библиотеки
    ok = within_limit

    @property
    def text(self) -> str:
        recipe = f"{self.size}px"
        if self.n_views > 1:
            recipe += f" x{self.n_views} видов TTA"
        if self.n_models > 1:
            recipe += f" x{self.n_models} моделей"
        head = f"{self.gflops:.1f} из {self.limit:.0f} GFLOPs на изображение ({recipe})"
        if self.within_limit:
            return f"{head} — в бюджете"
        return f"{head} — превышение в {self.gflops / self.limit:.2f} раза"

    def as_dict(self) -> dict:
        return {
            "gflops": round(self.gflops, 1),
            "limit_gflops": self.limit,
            "within_limit": self.within_limit,
        }


def check(
    model,
    size: int,
    *,
    n_views: int = 1,
    n_models: int = 1,
    limit: float = LIMIT_GFLOPS,
) -> Verdict:
    """Вердикт с учётом фактического рецепта инференса.

    Лимит регламента задан на ИЗОБРАЖЕНИЕ, а не на forward: TTA и ансамбль
    входят множителями. `--tta hflip,vflip` это два вида, сабмит по двум
    прогонам — две модели, вместе — четырёхкратная стоимость.
    """
    return Verdict(
        gflops=count_gflops(model, int(size)) * int(n_views) * int(n_models),
        limit=float(limit),
        size=int(size),
        n_views=int(n_views),
        n_models=int(n_models),
    )


def largest_fitting_size(
    build: Callable[[int], object],
    *,
    start: int = 512,
    n_views: int = 1,
    n_models: int = 1,
    limit: float = LIMIT_GFLOPS,
    step: int = 32,
) -> int | None:
    """Самая большая сторона входа, кратная `step`, которая ещё влезает.

    Отвечает на вопрос, который возникает сразу после отказа: «а на чём тогда
    учить». `None` значит, что не влезает даже минимальный вход — тогда дело не
    в разрешении, а в самой сети или в числе видов TTA.

    Фабрика, а не готовая модель: у энкодеров с оконным вниманием размер входа
    зашит в маски внимания, и посчитать такую сеть на другой стороне нельзя —
    она там просто не собирается.

    Полного двоичного поиска не нужно: FLOPs почти квадратичны по стороне, и из
    одного замера получается близкая оценка, которую остаётся подвинуть на
    шаг-другой.
    """
    per_image = int(n_views) * int(n_models)

    def cost(side: int) -> float:
        return count_gflops(build(int(side)), int(side)) * per_image

    guess = int(start * math.sqrt(limit / cost(start)))
    size = max(step, guess - guess % step)

    while size >= step and cost(size) > limit:
        size -= step
    if size < step or cost(size) > limit:
        return None
    while cost(size + step) <= limit:
        size += step
    return size


def rejection_text(build: Callable[[int], object], verdict: Verdict) -> str:
    """Отказ вместе с ответом на вопрос, который возникает сразу следом.

    «Не влезает» без «а на чём тогда» заставляет подбирать размер вручную,
    перезапуская проверку на каждой попытке.
    """
    fits = largest_fitting_size(
        build,
        start=verdict.size,
        n_views=verdict.n_views,
        n_models=verdict.n_models,
        limit=verdict.limit,
    )
    tail = (f"в бюджет укладывается вход {fits}px" if fits
            else "в бюджет не укладывается ни одно разрешение — дело в самой сети")
    return f"{verdict.text}; {tail}"
```

`ok = within_limit` присваивается на уровне класса и остаётся свойством —
синоним оставлен ради совместимости с вызывающим кодом пайплайна, который
проверяет именно `verdict.ok`.

- [ ] **Step 4: Прогнать**

Run: `python -m pytest tests/aic/test_budget.py -q`
Expected: 11 passed

- [ ] **Step 5: Проверить, что импорт пакета по-прежнему без торча**

Run: `python -m pytest tests/aic/test_import.py -q`
Expected: 3 passed

- [ ] **Step 6: Экспорт и коммит**

`aic/__init__.py`, в `_EXPORTS`: `"count_gflops": "budget",`

```bash
git add aic/budget.py aic/__init__.py tests/aic/test_budget.py
git commit -m "aic: бюджет над nn.Module, без конфига"
```

---

### Task 13: `aic/submit.py` — сабмит над чужой функцией

**Files:**
- Create: `aic/submit.py`, `tests/aic/test_submit.py`
- Modify: `aic/__init__.py`

**Interfaces:**
- Consumes: `imread`, `imwrite` из `aic.data`; `FP_AREA_THRESHOLD` из `aic.metric`
- Produces:
  - `REQUIRED_COLUMNS = ["img_path", "prediction_path"]`
  - `TTA_OPS: dict[str, tuple[Callable, Callable]]` — `none`, `hflip`, `vflip`, `hvflip`
  - `load_test_table(test_csv, template=None) -> tuple[pd.DataFrame, Path]`
  - `postprocess(prob, cls_prob=1.0, *, mask_threshold=0.5, cls_threshold=0.0, min_area=0.0) -> np.ndarray`
  - `to_original(prob, orig_h, orig_w, val_mode="resize") -> np.ndarray`
  - `predict_folder(paths, predict_fn, *, size, val_mode="resize", tta=("none",),
    batch_size=8, device=None, progress=None) -> Iterator[tuple[str, np.ndarray, float]]`
  - `write(out_dir, items, *, table=None, mask_threshold=0.5, cls_threshold=0.0, min_area=0.0) -> dict`
  - `validate(path, test_csv, *, max_problems=20, check_sizes=True, partial=False) -> dict`
  - `pack_zip(out_dir, zip_path=None) -> Path`

`predict_fn: (B, 3, H, W) float tensor -> (B, 1, H, W) вероятности`. Опционально
возвращает пару `(probs, cls_probs)`; вторая — вероятность «кадр изменён» формы
`(B,)`. Больше библиотека о вашей модели не предполагает ничего.

- [ ] **Step 1: Написать падающие тесты**

`tests/aic/test_submit.py`:

```python
"""Сабмит: механика регламента плюс раннер над своей функцией."""

from __future__ import annotations

import zipfile

import numpy as np
import pandas as pd
import pytest

from aic.data import imread, imwrite
from aic.submit import (
    load_test_table,
    pack_zip,
    postprocess,
    predict_folder,
    to_original,
    validate,
    write,
)

torch = pytest.importorskip("torch", reason="раннер требует torch")


def _test_set(tmp_path, n=4, size=(10, 12)):
    """Папка теста с test.csv и картинками."""
    root = tmp_path / "test_stage1"
    (root / "img").mkdir(parents=True)
    rows = []
    for i in range(n):
        rel = f"img/кадр{i}.png"
        imwrite(root / rel, np.zeros((*size, 3), dtype=np.uint8))
        rows.append({"img_path": rel})
    pd.DataFrame(rows).to_csv(root / "test.csv", index=False)
    return root


def test_load_test_table_derives_prediction_paths(tmp_path):
    root = _test_set(tmp_path)
    table, got_root = load_test_table(root / "test.csv")
    assert got_root == root
    assert list(table.columns) == ["img_path", "prediction_path"]
    assert table["prediction_path"].iloc[0] == "predictions/кадр0_pred.png"


def test_load_test_table_prefers_the_organisers_template(tmp_path):
    """Правило именования задано организаторами — угадывать его не надо."""
    root = _test_set(tmp_path, n=2)
    pd.DataFrame({
        "img_path": ["img/кадр0.png", "img/кадр1.png"],
        "prediction_path": ["preds/нулевой.png", "preds/первый.png"],
    }).to_csv(root / "submission.csv", index=False)

    table, _ = load_test_table(root / "test.csv", root / "submission.csv")
    assert table["prediction_path"].tolist() == ["preds/нулевой.png", "preds/первый.png"]


def test_load_test_table_rejects_an_incomplete_template(tmp_path):
    root = _test_set(tmp_path, n=2)
    pd.DataFrame({
        "img_path": ["img/кадр0.png"], "prediction_path": ["preds/a.png"],
    }).to_csv(root / "submission.csv", index=False)
    with pytest.raises(ValueError, match="не покрывает"):
        load_test_table(root / "test.csv", root / "submission.csv")


def test_postprocess_makes_a_zero_or_255_mask():
    prob = np.array([[0.1, 0.9], [0.6, 0.2]])
    mask = postprocess(prob, 1.0, mask_threshold=0.5)
    assert set(np.unique(mask).tolist()) <= {0, 255}
    assert mask.dtype == np.uint8
    assert mask[0, 1] == 255 and mask[0, 0] == 0


def test_postprocess_blanks_the_frame_below_the_cls_threshold():
    prob = np.ones((4, 4))
    assert postprocess(prob, 0.1, mask_threshold=0.5, cls_threshold=0.5).max() == 0


def test_postprocess_blanks_a_mask_under_min_area():
    prob = np.zeros((10, 10))
    prob[0, 0] = 1.0                      # 1% площади
    assert postprocess(prob, 1.0, mask_threshold=0.5, min_area=0.05).max() == 0
    assert postprocess(prob, 1.0, mask_threshold=0.5, min_area=0.005).max() == 255


def test_to_original_returns_the_input_resolution():
    prob = torch.zeros(1, 8, 8)
    assert to_original(prob, 13, 21).shape == (13, 21)


def test_to_original_crops_the_padding_first():
    """В режиме pad правая и нижняя части сетки — паддинг, а не картинка."""
    prob = torch.zeros(1, 8, 8)
    prob[:, :4, :] = 1.0
    out = to_original(prob, 40, 80, val_mode="pad")
    assert out.shape == (40, 80)
    assert out.mean() == pytest.approx(1.0, abs=1e-6)


def test_predict_folder_calls_your_function_and_returns_stems(tmp_path):
    root = _test_set(tmp_path, n=3, size=(10, 12))
    paths = sorted((root / "img").glob("*.png"))
    seen = []

    def predict_fn(batch):
        seen.append(tuple(batch.shape))
        return torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 0.8)

    items = list(predict_folder(paths, predict_fn, size=16, batch_size=2))
    assert [stem for stem, _, _ in items] == ["кадр0", "кадр1", "кадр2"]
    assert all(prob.shape == (10, 12) for _, prob, _ in items)
    assert seen and all(shape[1:] == (3, 16, 16) for shape in seen)


def test_predict_folder_averages_tta_views(tmp_path):
    root = _test_set(tmp_path, n=1)
    paths = sorted((root / "img").glob("*.png"))
    calls = []

    def predict_fn(batch):
        calls.append(1)
        return torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 0.5)

    list(predict_folder(paths, predict_fn, size=16, tta=("none", "hflip")))
    assert len(calls) == 2


def test_predict_folder_accepts_a_cls_probability(tmp_path):
    root = _test_set(tmp_path, n=1)
    paths = sorted((root / "img").glob("*.png"))

    def predict_fn(batch):
        probs = torch.full((batch.shape[0], 1, batch.shape[2], batch.shape[3]), 0.9)
        return probs, torch.full((batch.shape[0],), 0.25)

    (_, _, cls_prob), = list(predict_folder(paths, predict_fn, size=16))
    assert cls_prob == pytest.approx(0.25)


def test_write_and_validate_a_clean_submission(tmp_path):
    root = _test_set(tmp_path, n=3, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    items = [(f"кадр{i}", np.full((10, 12), 0.9), 1.0) for i in range(3)]

    out_dir = tmp_path / "сабмит"
    stats = write(out_dir, items, table=table, mask_threshold=0.5)
    assert stats["n_images"] == 3
    assert (out_dir / "submission.csv").exists()

    report = validate(out_dir, root / "test.csv")
    assert report["ok"] is True, report["problems"]
    assert report["n_checked"] == 3


def test_validate_catches_a_wrong_mask_size(tmp_path):
    root = _test_set(tmp_path, n=1, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "кривой"
    write(out_dir, [("кадр0", np.zeros((5, 5)), 1.0)], table=table)

    report = validate(out_dir, root / "test.csv")
    assert report["ok"] is False
    assert any("размер маски" in p for p in report["problems"])


def test_validate_catches_a_missing_frame(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "неполный"
    write(out_dir, [("кадр0", np.zeros((10, 12)), 1.0)], table=table.head(1))

    report = validate(out_dir, root / "test.csv")
    assert report["ok"] is False
    assert any("нет предсказаний" in p for p in report["problems"])


def test_validate_accepts_a_partial_submission_when_asked(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "частичный"
    write(out_dir, [("кадр0", np.zeros((10, 12)), 1.0)], table=table.head(1))
    assert validate(out_dir, root / "test.csv", partial=True)["ok"] is True


def test_validate_catches_values_other_than_0_and_255(tmp_path):
    root = _test_set(tmp_path, n=1, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "серый"
    write(out_dir, [("кадр0", np.zeros((10, 12)), 1.0)], table=table)
    imwrite(out_dir / table["prediction_path"].iloc[0],
            np.full((10, 12), 128, dtype=np.uint8))

    report = validate(out_dir, root / "test.csv")
    assert any("не только 0/255" in p for p in report["problems"])


def test_pack_zip_puts_the_csv_in_the_archive_root(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "архив"
    write(out_dir, [(f"кадр{i}", np.zeros((10, 12)), 1.0) for i in range(2)], table=table)

    zip_path = pack_zip(out_dir)
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert "submission.csv" in names
    assert sum(n.endswith(".png") for n in names) == 2


def test_validate_reads_a_zip_the_same_way(tmp_path):
    root = _test_set(tmp_path, n=2, size=(10, 12))
    table, _ = load_test_table(root / "test.csv")
    out_dir = tmp_path / "зип"
    write(out_dir, [(f"кадр{i}", np.zeros((10, 12)), 1.0) for i in range(2)], table=table)
    assert validate(pack_zip(out_dir), root / "test.csv")["ok"] is True
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest tests/aic/test_submit.py -q`
Expected: FAIL, `ModuleNotFoundError: No module named 'aic.submit'`

- [ ] **Step 3: Собрать `aic/submit.py`**

Переносятся без изменений: `REQUIRED_COLUMNS`, `pack_zip`, `_Source`,
`TTA_OPS`, тело `validate_submission` (переименовать в `validate`), тело
`postprocess`, тело `_to_original` (переименовать в публичный `to_original`).

`load_test_table` теряет обращения к воркспейсу — `test_csv` становится
обязательным, маркер `DEFAULT` исчезает:

```python
def load_test_table(
    test_csv: str | Path,
    template: str | Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    """Возвращает (таблицу с img_path и prediction_path, корень тестовых путей).

    Шаблон организаторов, если он есть, переиспользуется как есть: правило
    именования уже задано ими, угадывать его не нужно.
    """
```

Тело — прежнее, за вычетом веток `workspace()` и сравнения с `DEFAULT`.

`validate` получает `test_csv` вторым позиционным и передаёт его в
`load_test_table(test_csv, template=None)`.

- [ ] **Step 4: Написать раннер**

Новый код, взамен `predict_stream` + `predict_folder`:

```python
def _load_batch(paths: Sequence[Path], size: int, val_mode: str):
    """Читает картинки и приводит к модельной сетке. Возвращает батч и размеры."""
    import cv2
    import torch

    images, shapes = [], []
    for path in paths:
        image = imread(path)
        if image is None:
            raise FileNotFoundError(f"не читается: {path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        shapes.append(image.shape[:2])
        if val_mode == "pad":
            scale = size / float(max(image.shape[:2]))
            h = max(1, int(round(image.shape[0] * scale)))
            w = max(1, int(round(image.shape[1] * scale)))
            resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
            canvas = np.zeros((size, size, 3), dtype=resized.dtype)
            canvas[:h, :w] = resized
            image = canvas
        else:
            image = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
        images.append(image)

    batch = np.stack(images).astype(np.float32) / 255.0
    return torch.from_numpy(batch).permute(0, 3, 1, 2).contiguous(), shapes


def predict_folder(
    paths: Sequence[str | Path],
    predict_fn: Callable,
    *,
    size: int,
    val_mode: str = "resize",
    tta: Sequence[str] = ("none",),
    batch_size: int = 8,
    device: "str | object | None" = None,
    progress: Callable[[str], None] | None = None,
) -> Iterator[tuple[str, np.ndarray, float]]:
    """Обход теста над вашей функцией. Отдаёт `(stem, вероятности, cls_prob)`.

    `predict_fn(batch) -> (B, 1, H, W)` вероятностей либо пара
    `(вероятности, (B,) вероятность «кадр изменён»)`. Ни про модель, ни про
    чекпоинты библиотека не знает.

    Итератор, а не список: полный тест не обязан помещаться в память, а между
    предсказанием и записью кто-то захочет вставить своё — ансамбль, свою
    постобработку, свои пороги.

    Нормализация входа НЕ делается: чем нормировать, знает только ваша модель.
    На вход `predict_fn` приходит RGB в [0, 1].
    """
    import torch

    paths = [Path(p) for p in paths]
    if not paths:
        raise ValueError("список изображений пуст")

    for start in range(0, len(paths), batch_size):
        chunk = paths[start:start + batch_size]
        batch, shapes = _load_batch(chunk, int(size), val_mode)
        if device is not None:
            batch = batch.to(device)

        prob_sum, cls_sum = None, None
        with torch.no_grad():
            for op_name in tta:
                forward_op, inverse_op = TTA_OPS[op_name]
                output = predict_fn(forward_op(batch))
                probs, cls = output if isinstance(output, tuple) else (output, None)
                probs = inverse_op(probs).float()
                if cls is None:
                    cls = torch.ones(probs.shape[0], device=probs.device)
                prob_sum = probs if prob_sum is None else prob_sum + probs
                cls_sum = cls.reshape(-1).float() if cls_sum is None else cls_sum + cls.reshape(-1)
        probs, cls_probs = prob_sum / len(tta), cls_sum / len(tta)

        for i, path in enumerate(chunk):
            orig_h, orig_w = shapes[i]
            yield (
                path.stem,
                to_original(probs[i], orig_h, orig_w, val_mode),
                float(cls_probs[i]),
            )
        if progress is not None:
            progress(f"  {min(start + batch_size, len(paths))}/{len(paths)}")


def write(
    out_dir: str | Path,
    items: Iterable[tuple[str, np.ndarray, float]],
    *,
    table: pd.DataFrame | None = None,
    mask_threshold: float = 0.5,
    cls_threshold: float = 0.0,
    min_area: float = 0.0,
) -> dict:
    """Записать маски и `submission.csv`. `items` — то, что отдал раннер.

    `table` — результат `load_test_table`: из неё берутся пути назначения. Без
    неё маски кладутся в `predictions/<stem>_pred.png`, а csv не пишется — так
    можно собрать половину сабмита и дописать вторую другим прогоном.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets: dict[str, str] = {}
    if table is not None:
        targets = {Path(img).stem: rel
                   for img, rel in zip(table["img_path"], table["prediction_path"])}

    areas: list[float] = []
    for stem, prob, cls_prob in items:
        mask = postprocess(
            prob, cls_prob,
            mask_threshold=mask_threshold, cls_threshold=cls_threshold, min_area=min_area,
        )
        rel = targets.get(stem, f"predictions/{stem}_pred.png")
        target = out_dir / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        imwrite(target, mask)
        areas.append(float(np.count_nonzero(mask)) / mask.size)

    if table is not None:
        table.to_csv(out_dir / "submission.csv", index=False)

    values = np.asarray(areas) if areas else np.zeros(0)
    non_empty = values > 0
    return {
        "n_images": len(areas),
        "out_dir": str(out_dir),
        "empty_masks": int((~non_empty).sum()),
        "flagged_share": round(float((values >= FP_AREA_THRESHOLD).mean()), 4) if areas else 0.0,
        "area_median_nonempty": round(float(np.median(values[non_empty])), 4)
        if non_empty.any() else 0.0,
    }
```

Шапка модуля:

```python
"""Сабмит: механика регламента и раннер поверх вашей функции предсказания.

Формат по условию задачи:

    submission.zip
      submission.csv          колонки img_path, prediction_path
      predictions/*.png       одноканальные PNG, значения 0/255,
                              размер совпадает с входным изображением

Отдельная проверка существует потому, что сабмит легко испортить незаметно:
маска не того размера, PNG с тремя каналами, значения 0/1 вместо 0/255,
пропущенная строка. Всё это выясняется уже после загрузки, когда попытка
потрачена.

О моделях и чекпоинтах модуль не знает ничего: `predict_fn` ваша.
"""
```

- [ ] **Step 5: Прогнать**

Run: `python -m pytest tests/aic/test_submit.py -q`
Expected: 18 passed

- [ ] **Step 6: Экспорт и коммит**

`aic/__init__.py`, в `_EXPORTS`:

```python
    "predict_folder": "submit",
    "validate_submission": "submit:validate",
```

```bash
git add aic/submit.py aic/__init__.py tests/aic/test_submit.py
git commit -m "aic: сабмит над своей predict_fn"
```

---

### Task 14: Проверка совместимости на настоящей папке

Три обещания спеки проверяются здесь: библиотека читает все существующие
прогоны, воспроизводит уже посчитанные числа и не тянет торч.

**Files:**
- Create: `tests/aic/test_real_runs.py`

**Interfaces:**
- Consumes: всё публичное API библиотеки

- [ ] **Step 1: Написать тест**

`tests/aic/test_real_runs.py`:

```python
"""Совместимость с накопленными прогонами. Без этих тестов переезд бессмыслен.

Тесты пропускаются, если папки runs/ нет: библиотека ставится и без воркспейса.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from aic.analysis import OofView, leaderboard
from aic.runs import Run
from aic.stats import compare, per_image

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUNS = REPO_ROOT / "runs"


def _runs_with_eval() -> list[Path]:
    if not RUNS.is_dir():
        return []
    return sorted(d for d in RUNS.iterdir() if (d / "oof" / "val.npz").exists())


pytestmark = pytest.mark.skipif(not _runs_with_eval(), reason="в репозитории нет прогонов")


def test_every_run_dir_opens():
    """Ни одна из накопленных папок не должна отвалиться на чтении."""
    for run_dir in _runs_with_eval():
        run = Run.open(run_dir)
        assert isinstance(run.snapshot, dict)
        assert isinstance(run.summary, dict)
        assert not run.history.empty, run_dir.name


def test_eval_loads_and_lengths_agree():
    for run_dir in _runs_with_eval():
        ev = Run.open(run_dir).load_eval()
        assert len(ev) == ev.stems.size
        assert len(np.unique(ev.stems)) == ev.stems.size, run_dir.name


def test_best_aic_reproduces_the_stored_summary():
    """Главная проверка: числа не поехали."""
    checked = 0
    for run_dir in _runs_with_eval():
        run = Run.open(run_dir)
        stored = (run.summary or {}).get("best")
        if not stored:
            continue
        op = run.operating_point()
        got = run.load_eval().acc.evaluate(*op)
        assert got.aic == pytest.approx(stored["aic"], abs=1e-9), run_dir.name
        assert got.dice_pos == pytest.approx(stored["dice_pos"], abs=1e-9)
        assert got.fpr_neg == pytest.approx(stored["fpr_neg"], abs=1e-9)
        checked += 1
    assert checked > 0, "ни у одного прогона нет summary.best — проверять нечего"


def test_per_image_means_match_the_summary():
    """Вердикт обязан считаться по той же метрике, по которой отбирают модели."""
    for run_dir in _runs_with_eval()[:3]:
        run = Run.open(run_dir)
        op = run.operating_point()
        sample = per_image(run.load_eval().acc, op)
        direct = run.load_eval().acc.evaluate(*op)
        assert sample.dice[sample.is_pos].mean() == pytest.approx(direct.dice_pos, abs=1e-9)
        assert sample.alarm[~sample.is_pos].mean() == pytest.approx(direct.fpr_neg, abs=1e-9)


def test_comparing_a_run_with_itself_is_zero():
    run_dir = _runs_with_eval()[0]
    run = Run.open(run_dir)
    ev = run.load_eval()
    cmp = compare(ev, ev, op=run.operating_point(), bootstrap_n=200)
    assert cmp.delta_ref_op == pytest.approx(0.0, abs=1e-12)
    assert cmp.n_common == len(ev)


def test_curve_from_step_matches_the_old_axis():
    """Ось показов из номера эпохи: (step + 1) * epoch_size, как считал load_reference."""
    for run_dir in _runs_with_eval():
        run = Run.open(run_dir)
        epoch_size = (run.snapshot.get("data") or {}).get("epoch_size")
        if not epoch_size or "val/aic_tuned" not in run.history.columns:
            continue
        xs, ys = run.curve("val/aic_tuned", x=("step", int(epoch_size)))
        assert xs[0] == pytest.approx(float(epoch_size))
        assert xs.size == ys.size > 0
        return
    pytest.skip("ни у одного прогона нет data.epoch_size в снапшоте")


def test_oof_view_opens_a_real_run():
    for run_dir in _runs_with_eval():
        if not (run_dir / "oof" / "val_rows.parquet").exists():
            continue
        view = OofView.from_run(run_dir)
        op = Run.open(run_dir).operating_point()
        buckets = view.by_area(op)
        assert buckets["n"].sum() == int(view.is_pos.sum())
        ceilings = view.ceilings(op)
        assert ceilings["perfect_cls"] >= ceilings["current"] - 1e-9
        return
    pytest.skip("ни у одного прогона нет val_rows.parquet")


def test_leaderboard_covers_the_whole_runs_folder():
    board = leaderboard(RUNS)
    assert not board.empty
    assert "run" in board.columns
    names = set(board["run"])
    for run_dir in _runs_with_eval():
        assert run_dir.name in names
```

- [ ] **Step 2: Прогнать**

Run: `python -m pytest tests/aic/test_real_runs.py -q -v`
Expected: 8 passed. Если `test_best_aic_reproduces_the_stored_summary` падает —
это не «поправить допуск», а расхождение арифметики; разбираться до конца.

- [ ] **Step 3: Прогнать всю библиотеку целиком**

Run: `python -m pytest tests/aic -q`
Expected: ~110 passed, 0 failed

- [ ] **Step 4: Убедиться, что пайплайн всё ещё цел**

Run: `python -m pytest tests/pipeline -q`
Expected: столько же пройденных, сколько до начала фазы 1.

- [ ] **Step 5: Коммит**

```bash
git add tests/aic/test_real_runs.py
git commit -m "aic: совместимость с накопленными прогонами под тестом"
```

---

## Фаза 2. Пайплайн переезжает на библиотеку

Фаза 1 самодостаточна: после неё библиотекой уже можно пользоваться из
ноутбука. Здесь пайплайн перестаёт держать свои копии инструментов.

Порядок внутри фазы важен: сначала механическое переименование (Task 15),
потом содержательная замена (Task 16–17). Смешивать нельзя — в одном коммите
переименования и правки логики невозможно отличить одно от другого при разборе.

### Task 15: `experimental_tools_beliy_russak` → `aic_pipeline`

Чисто механический шаг: ни строчки логики не меняется.

**Files:**
- Move: `experimental_tools_beliy_russak/` → `pipeline/aic_pipeline/`
- Move: `tests/pipeline/` → `pipeline/tests/`
- Modify: `pyproject.toml`, `cli.py`

- [ ] **Step 1: Перенести пакет и тесты**

```bash
mkdir -p pipeline
git mv experimental_tools_beliy_russak pipeline/aic_pipeline
git mv tests/pipeline pipeline/tests
```

- [ ] **Step 2: Переписать имя пакета во всех импортах**

```bash
grep -rl "experimental_tools_beliy_russak" --include=*.py --include=*.toml \
     --include=*.md --include=*.cfg --include=*.yaml . \
  | xargs sed -i 's/experimental_tools_beliy_russak/aic_pipeline/g'
```

- [ ] **Step 3: Починить корень репозитория в conftest пайплайна**

`pipeline/tests/conftest.py` — файл снова сменил глубину:

```python
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
```

- [ ] **Step 4: Прописать оба пакета и оба каталога тестов**

`pyproject.toml`:

```toml
[tool.setuptools.packages.find]
where = [".", "pipeline"]
include = ["aic*", "aic_pipeline*"]

[project.scripts]
aic = "aic_pipeline.cli:app"

[tool.pytest.ini_options]
testpaths = ["tests", "pipeline/tests"]
```

- [ ] **Step 5: Переустановить пакет и прогнать всё**

```bash
python -m pip install -e ".[dev]" --no-deps
python -m pytest -q
```

Expected: столько же пройденных тестов, сколько было; ни одного
`ModuleNotFoundError: experimental_tools_beliy_russak`.

- [ ] **Step 6: Проверить, что CLI жив**

Run: `python cli.py registry`
Expected: список зарегистрированных компонент, без трассировки.

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "Переименование: experimental_tools_beliy_russak -> aic_pipeline"
```

---

### Task 16: Пайплайн берёт инструменты из библиотеки

Восемь модулей перестают существовать в пайплайне. Там, где от модуля остаётся
только знание о конфиге, он превращается в тонкий адаптер; где не остаётся
ничего — удаляется, а импорты переписываются на `aic`.

**Files:**
- Delete: `pipeline/aic_pipeline/metrics.py`, `logging_utils.py`, `imageio.py`,
  `indexing.py`, `splits.py`, `precache.py`, `analysis/leaderboard.py`,
  `analysis/compare_runs.py`, `analysis/oof_report.py`
- Move: `pipeline/aic_pipeline/analysis/viz.py` → `pipeline/aic_pipeline/viz.py`
  (наложение масок на кадры в библиотеку не входит — в раскладке спеки его нет)
- Rewrite as adapters: `pipeline/aic_pipeline/workspace.py`, `stats.py`, `budget.py`
- Modify: `pipeline/aic_pipeline/__init__.py`, `datasets.py`, `inference.py`,
  `submission.py`, `train.py`, `plans.py`, `utils.py`, `cli.py`

- [ ] **Step 1: Переписать импорты удаляемых модулей**

```bash
cd pipeline/aic_pipeline
grep -rl "from \.metrics import\|from \.\.metrics import" --include=*.py . \
  | xargs sed -i 's/from \.\.metrics import/from aic.metric import/; s/from \.metrics import/from aic.metric import/'
grep -rl "from \.imageio import" --include=*.py . \
  | xargs sed -i 's/from \.imageio import/from aic.data import/'
grep -rl "from \.indexing import\|from \.splits import" --include=*.py . \
  | xargs sed -i 's/from \.indexing import/from aic.data import/; s/from \.splits import/from aic.data import/'
grep -rl "from \.logging_utils import RunLogger" --include=*.py . \
  | xargs sed -i 's/from \.logging_utils import RunLogger/from aic.runs import Run/'
cd ../..
git rm pipeline/aic_pipeline/{metrics.py,logging_utils.py,imageio.py,indexing.py,splits.py,precache.py}
git mv pipeline/aic_pipeline/analysis/viz.py pipeline/aic_pipeline/viz.py
git rm -r pipeline/aic_pipeline/analysis
```

`cli.py` звал `analysis.leaderboard`, `analysis.compare_runs` и
`analysis.oof_report` — переписать на `aic.analysis`:

```python
from aic.analysis import OofView, compare_table, history, leaderboard
```

Команда `aic leaderboard` теперь передаёт путь явно: `leaderboard(runs_root())`.
Отчёт по корзинам собирается из `OofView.from_run(run_dir)` — `by_area`,
`by_column("domain", op)`, `by_column("generator", op)`, `ceilings(op)`;
форматирование строк остаётся в `cli.py`, где ему и место.

`load_index`, `load_folds`, `build_index`, `make_folds` в `aic.data` больше не
берут пути из глобалей — на каждом вызове дописать аргумент:

- `splits.load_folds()` → `aic.data.load_folds(workspace().split_path)`
- `indexing.load_index()` → `aic.data.load_index(workspace().index_path)`
- `make_folds(n_folds=..., seed=...)` → `aic.data.make_folds(load_index(ws.index_path),
  n_folds=..., seed=..., out_path=ws.split_path)`
- `build_index(...)` → `aic.data.build_index(workspace(), ...)`
- `build_cache(df, ...)` → `aic.data.build_cache(workspace(), df, ...)`
- `cached_path(rel, side, is_mask)` → `aic.data.cached_path(workspace(), rel, side, is_mask)`

- [ ] **Step 2: Превратить `workspace.py` в адаптер**

Синглтон и сокращения остаются — на них стоит весь CLI. Пути считает `aic`.

```python
"""Воркспейс пайплайна: синглтон поверх `aic.Workspace` плюс конфиги и планы.

Библиотека намеренно не держит глобального состояния, а CLI без него неудобен:
`aic train -c baseline` не должен требовать пути в каждой команде. Синглтон —
свойство ЭТОГО слоя, и живёт он здесь.
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

from aic.paths import WORKSPACE_ENV, Workspace as _Workspace

#: пайплайн опознаёт свой воркспейс по конфигам, а не по данным: без configs/
#: команда `aic train -c ...` всё равно ничего не сделает
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
    return Workspace.find(start, marker=MARKER).root


_current: Workspace | None = None


def workspace() -> Workspace:
    global _current
    if _current is None:
        env_root = os.environ.get(WORKSPACE_ENV)
        _current = Workspace(env_root) if env_root else Workspace(find_workspace_root())
    return _current


def set_workspace(root: str | Path | None) -> Workspace:
    global _current
    _current = None if root is None else Workspace(root)
    return workspace()


@contextlib.contextmanager
def use_workspace(root: str | Path):
    global _current
    previous = _current
    try:
        yield set_workspace(root)
    finally:
        _current = previous


# --- сокращения, чтобы вызывающий код не писал workspace() каждый раз ------

def project_root() -> Path: return workspace().root
def data_root() -> Path: return workspace().data
def dataset_root() -> Path: return workspace().dataset_root
def train_csv() -> Path: return workspace().train_csv
def src_dir() -> Path: return workspace().src_dir
def cache_root() -> Path: return workspace().cache
def test_root() -> Path: return workspace().test_root
def test_csv() -> Path: return workspace().test_csv
def submission_template() -> Path: return workspace().submission_template
def test_img_dir() -> Path: return workspace().test_img_dir
def artifacts_root() -> Path: return workspace().artifacts
def index_path() -> Path: return workspace().index_path
def split_path() -> Path: return workspace().split_path
def runs_root() -> Path: return workspace().runs
def submissions_root() -> Path: return workspace().submissions
def configs_root() -> Path: return workspace().configs
def plans_root() -> Path: return workspace().plans
def resolve(rel_path: str, root: Path | None = None) -> Path:
    return workspace().resolve(rel_path, root)
def ensure_dirs() -> None: workspace().ensure_dirs()
```

- [ ] **Step 3: Превратить `stats.py` в адаптер**

Остаётся ровно то, что знает про конфиг и про формат прогона.

```python
"""Статистика прогонов: арифметика из `aic.stats`, конфиг — здесь.

Что именно делает два прогона несопоставимыми, библиотека не решает: у каждого
своя форма конфига. Список ключей бюджета и разбор блока `stats` — свойство
этого слоя.
"""

from __future__ import annotations

from pathlib import Path

from aic.runs import Eval, Run
from aic.stats import (  # noqa: F401 — переэкспорт для вызывающего кода
    Boot, Comparison, Gate, PerImage, Verdict,
    align_by_stem, compare, diverged_keys, gate_check, gate_metrics,
    paired_bootstrap, per_image, seeds_needed, verdict,
)

from .config import get_path as _get_path
from .workspace import runs_root

#: ключи, определяющие бюджет прогона. Сравнивать между собой можно только
#: прогоны, у которых они совпадают: иначе разница мерит бюджет, а не гипотезу
BUDGET_KEYS = (
    "data.size",
    "data.epoch_size",
    "data.val_frac",
    # val_limit и val_keep_negatives задают состав валидации наравне с val_frac:
    # разойдясь по ним, два прогона меряются на разных наборах кадров
    "data.val_limit",
    "data.val_keep_negatives",
    "data.val_seed",
    "train.epochs",
    "train.bs",
    "train.accum_steps",
    "train.fold",
)


def comparable(cfg: dict, cfg_ref: dict, keys=BUDGET_KEYS) -> tuple[bool, list[str]]:
    """Совпадают ли бюджеты двух прогонов; вторым — список разошедшихся ключей."""
    diverged = diverged_keys(cfg, cfg_ref, list(keys))
    return not diverged, diverged


def resolve_reference(name: str) -> Path:
    """Имя прогона или путь к его папке."""
    candidate = Path(name)
    return candidate if candidate.exists() else runs_root() / name


class Reference:
    """Опорный прогон, поднятый с диска: то, что раньше делал `load_reference`."""

    def __init__(self, run_dir) -> None:
        self.run = Run.open(run_dir)
        self.name = self.run.dir.name
        self.eval: Eval = self.run.load_eval()
        self.op = self.run.operating_point()
        self.cfg = self.run.snapshot
        self.curve = self._curve()

    def _curve(self):
        """Ось показов. У эталона её даёт только `data.epoch_size` из конфига.

        Пустая кривая означает, что гейт молча выключается на весь прогон, —
        поэтому про неё говорится вслух один раз на старте, а не замалчивается.
        """
        import numpy as np

        epoch_size = int(_get_path(self.cfg, "data.epoch_size") or 0)
        if epoch_size <= 0 or "val/aic_tuned" not in self.run.history.columns:
            return np.zeros(0, dtype=float), np.zeros(0, dtype=float)
        return self.run.curve("val/aic_tuned", x=("step", epoch_size))


def load_reference(run_dir) -> Reference:
    return Reference(run_dir)
```

`StatsSettings` и `stats_settings` переносятся из старого `stats.py` в этот
файл без изменений — они целиком про блок `stats` в конфиге.

- [ ] **Step 4: Превратить `budget.py` в адаптер**

```python
"""Бюджет по конфигу: счёт — из `aic.budget`, сборка модели и пометки — здесь."""

from __future__ import annotations

import json
from typing import Any, Mapping

from aic.budget import LIMIT_GFLOPS, count_gflops, largest_fitting_size  # noqa: F401
from aic.budget import Verdict as _Verdict

from .config import get_path

#: (нормализованная секция model, размер) -> GFLOPs. Сеть не зависит от секций
#: data.* и loss.*, поэтому свип по ним считается один раз, а не на каждой точке
_CACHE: dict[tuple[str, int], float] = {}


def config_gflops(cfg_model: Mapping[str, Any], size: int) -> float:
    """Собрать сеть по секции `model` и посчитать её forward.

    Считается на `meta`-устройстве: FlopCounterMode работает на уровне
    диспетчера, настоящие тензоры ему не нужны. Предобученные веса не
    запрашиваются — на число операций они не влияют.
    """
    from .models import build_model

    spec = dict(cfg_model)
    spec["encoder_weights"] = None
    # у энкодеров с оконным вниманием размер входа зашит в маски внимания:
    # ведём `img_size` за размером, а не наоборот
    encoder_kwargs = dict(spec.get("encoder_kwargs") or {})
    if "img_size" in encoder_kwargs:
        encoder_kwargs["img_size"] = int(size)
        spec["encoder_kwargs"] = encoder_kwargs

    key = (json.dumps(spec, sort_keys=True, default=str), int(size))
    if key not in _CACHE:
        _CACHE[key] = count_gflops(build_model(spec).eval().to("meta"), int(size))
    return _CACHE[key]
```

Дальше в том же файле — послабление для исследовательского прогона. Оно
свойство конфига, а не регламента, поэтому в библиотеке его нет:

```python
class Verdict(_Verdict):
    """Библиотечный вердикт плюс пометка `budget.exempt` из конфига."""

    exempt: bool = False
    exempt_reason: str | None = None

    def __init__(self, *args, exempt: bool = False, exempt_reason: str | None = None, **kw):
        super().__init__(*args, **kw)
        object.__setattr__(self, "exempt", bool(exempt))
        object.__setattr__(self, "exempt_reason", exempt_reason)

    @property
    def ok(self) -> bool:
        """Настоящий ответ регламента — `within_limit`; это послабление поверх."""
        return self.within_limit or self.exempt

    @property
    def text(self) -> str:
        base = super().text
        if self.within_limit or not self.exempt:
            return base
        return f"{base}, но помечен budget.exempt: {self.exempt_reason or 'причина не указана'}"

    def as_dict(self) -> dict:
        return {**super().as_dict(), "exempt": self.exempt}


def inference_gflops(cfg: Mapping[str, Any], *, n_views: int = 1, n_models: int = 1) -> float:
    """Во что обходится ОДНО изображение целиком, а не один forward.

    Лимит регламента задан на изображение, поэтому TTA и ансамбль входят
    множителями: `--tta hflip,vflip` это два вида, сабмит по двум прогонам —
    две модели, вместе — четырёхкратная стоимость.
    """
    size = int(get_path(cfg, "data.size", 512))
    return config_gflops(cfg.get("model", {}), size) * int(n_views) * int(n_models)


def check(
    cfg: Mapping[str, Any],
    *,
    n_views: int = 1,
    n_models: int = 1,
    allow_exempt: bool = True,
    limit: float = LIMIT_GFLOPS,
) -> Verdict:
    """Вердикт по конфигу. `allow_exempt=False` — для сабмита, там послаблений нет."""
    exempt = bool(get_path(cfg, "budget.exempt", False)) if allow_exempt else False
    return Verdict(
        gflops=inference_gflops(cfg, n_views=n_views, n_models=n_models),
        limit=float(limit),
        size=int(get_path(cfg, "data.size", 512)),
        n_views=int(n_views),
        n_models=int(n_models),
        exempt=exempt,
        exempt_reason=get_path(cfg, "budget.exempt_reason") if exempt else None,
    )
```

`check_submission` и `rejection_text` переносятся из старого `budget.py` без
изменений, кроме одного: `rejection_text` зовёт библиотечный
`largest_fitting_size` с фабрикой `lambda size: build_model({**cfg_model,
"encoder_weights": None}).eval().to("meta")`.

- [ ] **Step 4b: Разделить `inference.py`**

Из `aic_pipeline/inference.py` уходят `TTA_OPS`, `_forward_tta`, `_to_original`,
`predict_stream` и `postprocess` — всё это теперь в `aic.submit`. Остаются
`load_checkpoint` и `evaluate_full_res`: они знают про формат чекпоинта и про
сборку модели по конфигу, то есть про пайплайн.

`evaluate_full_res` переписывается на библиотечный раннер:

```python
from aic.data import imread
from aic.metric import AICAccumulator
from aic.submit import predict_folder
```

и внутри вместо `predict_stream` по `DataLoader` — обход путей из `df` через
`predict_folder` с `predict_fn`, замыкающей загруженную модель. Маска GT
читается тем же `imread`, что и раньше; логика бинаризации по
`cfg.data.gt_binarize` не меняется.

- [ ] **Step 5: Пересобрать `submission.build_submission` поверх `aic.submit`**

`aic_pipeline/submission.py` оставляет себе только то, что знает про чекпоинты
и конфиг: загрузку моделей, сверку `size`/`val_mode` ансамбля, гейт по бюджету,
разрешение порогов из `calib.json`. Всё остальное переезжает на библиотеку:

```python
from aic.submit import load_test_table, pack_zip, predict_folder  # noqa: F401
from aic.submit import validate as validate_submission            # noqa: F401
from aic.submit import write as write_submission
```

Место, где строился `DataLoader` и звался `predict_stream`, заменяется на:

```python
    def predict_fn(batch):
        outputs = [net(batch) for net in models]
        probs = sum(torch.sigmoid(o["logits"].float()) for o in outputs) / len(outputs)
        cls = sum(torch.sigmoid(o["cls_logits"].float()).reshape(-1) for o in outputs) / len(outputs)
        return probs, cls

    items = predict_folder(
        paths, predict_fn, size=size, val_mode=val_mode, tta=tuple(tta),
        batch_size=batch_size, device=device, progress=progress,
    )
    stats = write_submission(out_dir, items, table=table, **thresholds)
```

`load_test_table` теперь требует путь: `load_test_table(test_csv or workspace().test_csv,
template if template is not DEFAULT else workspace().submission_template)`.

- [ ] **Step 6: Прогнать тесты пайплайна**

Run: `python -m pytest pipeline/tests -q`
Expected: все проходят. Тесты, дублирующие библиотечные (`test_metrics.py`,
`test_workspace.py` в части путей), удалить — они переехали в `tests/aic/`.

- [ ] **Step 7: Прогнать всё**

Run: `python -m pytest -q`
Expected: 0 failed.

- [ ] **Step 8: Коммит**

```bash
git add -A
git commit -m "Пайплайн берёт метрику, статистику, данные и бюджет из aic"
```

---

### Task 17: `train.py` пишет через `Run` и логирует `samples`

**Files:**
- Modify: `pipeline/aic_pipeline/train.py`
- Create: `pipeline/tests/test_train_run_record.py`

- [ ] **Step 1: Написать падающий тест**

Обучение не запускается: проверяется, что нужное попадает в папку. Тест зовёт
только те части `train.py`, которые пишут артефакты.

`pipeline/tests/test_train_run_record.py`:

```python
"""Прогон обязан оставлять ось показов — иначе гейт по эталону нечем питать."""

from __future__ import annotations

from aic.runs import Run

from aic_pipeline.train import epoch_samples, open_run_record


def test_epoch_samples_prefers_the_explicit_epoch_size():
    assert epoch_samples({"data": {"epoch_size": 8000}}, n_train_rows=99999) == 8000


def test_epoch_samples_falls_back_to_the_dataset_size():
    assert epoch_samples({"data": {}}, n_train_rows=1234) == 1234


def test_run_record_logs_samples_on_every_epoch(tmp_path):
    run = open_run_record(tmp_path, "проба", cfg={"data": {"epoch_size": 100}})
    for epoch in range(3):
        run.log(epoch, {"val/aic_tuned": 0.1 * epoch, "samples": 100 * (epoch + 1)})
    run.close()

    xs, ys = Run.open(run.dir).curve("val/aic_tuned")
    assert xs.tolist() == [100.0, 200.0, 300.0]
    assert ys.tolist() == [0.0, 0.1, 0.2]
```

- [ ] **Step 2: Убедиться, что падает**

Run: `python -m pytest pipeline/tests/test_train_run_record.py -q`
Expected: FAIL, `ImportError: cannot import name 'epoch_samples'`

- [ ] **Step 3: Заменить `RunLogger` на `Run` в `train.py`**

Точечные замены по всему файлу:

| было | стало |
|---|---|
| `run_dir = make_run_dir(name, resume=...)` | `run = Run.create(runs_root(), name, resume=..., tensorboard=cfg.get("tensorboard", True))`, далее `run_dir = run.dir` |
| `logger = RunLogger(run_dir, use_tensorboard=...)` | удаляется, `logger` → `run` |
| `logger.info(...)` | `run.info(...)` |
| `logger.log_metrics(epoch, metrics)` | `run.log(epoch, {**metrics, "samples": (epoch + 1) * samples_per_epoch})` |
| `logger.flush_csv()` / `logger.close()` | `run.flush_csv()` / `run.close()` |
| `save_config(cfg, run_dir / "config.yaml")` | `run.save_snapshot(cfg)` |
| ручная запись `summary.json` | `run.save_summary({...})` |
| `torch.save(state, run_dir / "ckpt" / f"{name}.pt")` | `run.save_state(state, f"{name}.pt")` |
| `accumulator.save(run_dir / "oof" / "val.npz")` + запись parquet | `run.save_eval(accumulator, val_rows)` |

`logger.log_hparams(flat_cfg, metrics)` вызова не имеет замены в библиотеке —
строчку удалить: hparams в TensorBoard пайплайн и так дублирует снапшотом, а
тащить ради неё метод в библиотеку смысла нет.

- [ ] **Step 4: Добавить две функции в `train.py`**

```python
def epoch_samples(cfg, n_train_rows: int) -> int:
    """Сколько кадров показывается за эпоху.

    Ось сравнения прогонов — ЧИСЛО ПОКАЗОВ, а не номер эпохи: иначе плечо с
    `epoch_size: 4000, epochs: 12` нельзя сопоставить с эталоном 8000x6.
    """
    return int(get_path(cfg, "data.epoch_size") or n_train_rows)


def open_run_record(runs_root, name: str, cfg, *, resume: bool = False) -> Run:
    """Папка прогона со снапшотом конфига. Отдельной функцией — ради теста."""
    run = Run.create(runs_root, name, resume=resume,
                     tensorboard=bool(get_path(cfg, "tensorboard", True)))
    run.save_snapshot(dict(cfg))
    return run
```

- [ ] **Step 5: Прогнать**

Run: `python -m pytest pipeline/tests -q`
Expected: 0 failed.

- [ ] **Step 6: Проверить, что старый прогон дочитывается по-новому**

Run:

```bash
python -c "from aic_pipeline.stats import load_reference; r = load_reference('runs/f0-control-768'); print(r.name, r.op, len(r.curve[0]))"
```

Expected: `f0-control-768 (0.275, 0.5, 0.0) 6` — та же операционная точка, что
лежит в `summary.json`, и шесть точек кривой по числу эпох.

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "train: папка прогона через aic.Run, ось показов в логе"
```

---

### Task 18: Упаковка и документация

**Files:**
- Modify: `pyproject.toml`, `requirements.txt`, `README.md`, `GUIDE.md`,
  `docs/EXTENDING.md`
- Create: `docs/aic-quickstart.md`

- [ ] **Step 1: Разнести зависимости по extras**

`pyproject.toml`:

```toml
[project]
name = "aic"
dynamic = ["version"]
description = "Инструменты для AI Challenge: метрика AIC, статистика прироста, бюджет, прогоны"
requires-python = ">=3.10"

dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "pyarrow>=14.0",          # parquet для индекса, фолдов и строк валидации
    "scikit-learn>=1.3",      # StratifiedGroupKFold в data.make_folds
    "pyyaml>=6.0",
    "pillow>=10.0",
    "opencv-python-headless>=4.10",
]

[project.optional-dependencies]
# torch ставится отдельно и под свою CUDA. Если вписать его в основные
# зависимости, `pip install -e .` в conda-среде `challenges` попытается заменить
# conda-сборку колесом с PyPI — с другой версией CUDA
torch    = ["torch>=2.1"]
tb       = ["tensorboard>=2.16"]
viz      = ["matplotlib>=3.7"]
pipeline = [
    "typer>=0.12",
    "timm>=1.0.20",
    "segmentation-models-pytorch>=0.5.0",
    "albumentations>=2.0.0",
    "tensorboard>=2.16",
    "matplotlib>=3.7",
]
dev = ["pytest>=8.0", "jupyterlab>=4.0", "nbformat>=5.9", "ipykernel>=6.29"]

[tool.setuptools.dynamic]
version = { attr = "aic.__version__" }
```

- [ ] **Step 2: Проверить, что лёгкая установка действительно лёгкая**

```bash
python -m pip install --dry-run . 2>&1 | grep -iE "timm|segmentation-models|albumentations|typer"
```

Expected: пусто. Если что-то нашлось — зависимость осталась в основных.

- [ ] **Step 3: Обновить `requirements.txt`**

```
# Список зависимостей живёт в pyproject.toml — здесь только способ его поставить.
#
#   D:/Apps/anaconda3/envs/challenges/python.exe -m pip install -r requirements.txt
#
# Ставится поверх conda-среды `challenges` (torch, numpy, pandas, sklearn уже
# там). torch в зависимости пакета намеренно не включён, чтобы pip не подменил
# conda-сборку колесом с PyPI и другой версией CUDA.

-e .[pipeline,dev]
```

- [ ] **Step 4: Написать `docs/aic-quickstart.md`**

Одна страница на весь путь из ноутбука: своя модель, свой цикл, инструменты
библиотеки. Обязательно показать всё, что перечислено ниже, — это же и есть
приёмка API целиком:

```python
import aic
import numpy as np

ws = aic.Workspace.find()

# данные
df = aic.data.load_index(ws.index_path)
folds = aic.data.load_folds(ws.split_path)

# бюджет — до первой эпохи
model = build_my_model()
print(aic.budget.check(model, 768).text)

# прогон
run = aic.Run.create(ws.runs, "мой-эксперимент")
run.save_snapshot({"модель": "своя", "lr": 3e-4, "epoch_size": 8000})

acc = aic.AICAccumulator()
for epoch in range(12):
    ...                                        # свой цикл обучения
    acc = aic.AICAccumulator()
    for probs, gts, cls in my_validation():    # своя валидация
        acc.update(probs, gts, cls)
    best = acc.best()
    run.log(epoch, {"val/aic_tuned": best.aic, "samples": (epoch + 1) * 8000})
    run.save_state({"model": model.state_dict(), "epoch": epoch})

run.save_eval(acc, val_rows)                   # val_rows со столбцом stem
run.save_summary({"best_aic": best.aic, "best": best.as_dict()})
run.close()

# вердикт против эталона
ref = aic.Run.open(ws.runs / "f0-control-768")
cmp = aic.stats.compare(run.load_eval(), ref.load_eval(),
                        op=ref.operating_point(), train_sigma=0.0035)
print("\n".join(cmp.report("f0-control-768")))

# где именно теряется Dice
view = aic.analysis.OofView.from_run(run.dir)
print(view.by_area(run.operating_point()))
print(view.ceilings(run.operating_point()))

# сабмит
items = aic.submit.predict_folder(paths, my_predict_fn, size=768, tta=("none", "hflip"))
table, _ = aic.submit.load_test_table(ws.test_csv, ws.submission_template)
aic.submit.write("submissions/мой", items, table=table, mask_threshold=0.275)
print(aic.submit.validate("submissions/мой", ws.test_csv))
aic.submit.pack_zip("submissions/мой")
```

- [ ] **Step 5: Поправить README и GUIDE**

В `README.md`: заменить имя пакета и команду установки, добавить абзац про два
слоя (`import aic` — инструменты, `aic train` — эталонный пайплайн), сослаться
на `docs/aic-quickstart.md`. В `GUIDE.md` — найти все упоминания
`experimental_tools_beliy_russak`, `etbr`, `set_workspace`, `RunLogger`,
`make_run_dir` и привести к новому API. В `docs/EXTENDING.md` явно написать,
что реестры и `aic_plugins.py` — механика ПАЙПЛАЙНА, а библиотека расширяется
обычным способом: своими функциями вокруг её объектов.

- [ ] **Step 6: Прогнать всё в последний раз**

Run: `python -m pytest -q`
Expected: 0 failed.

Run: `python cli.py registry` и `python cli.py leaderboard`
Expected: обе команды отрабатывают.

- [ ] **Step 7: Коммит**

```bash
git add -A
git commit -m "Упаковка: дистрибутив aic с extras, документация под новое API"
```

---

## Приёмка

План считается выполненным, когда все три обещания спеки проверены командой,
а не рассуждением:

1. `python -m pytest -q` — ноль падений на обоих слоях.
2. `python -m pytest tests/aic/test_real_runs.py -q` — накопленные прогоны
   читаются, и `summary.json` каждого воспроизводится из его же `oof/val.npz`.
3. `python -m pytest tests/aic/test_import.py -q` — `import aic` не тянет ни
   torch, ни timm.
