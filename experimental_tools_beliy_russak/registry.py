"""Точки расширения: свои лоссы, аугментации, оптимизаторы — без правки библиотеки.

Раньше каждый билдер разбирал имя из конфига через `if/elif` и падал на всём
незнакомом. Пока конфиги писал один человек, это работало; когда у каждого в
команде свой воркспейс и свои гипотезы, любая новая компонента означала правку
библиотеки и новую версию у всех.

Теперь так: в корне воркспейса лежит `aic_plugins.py`, библиотека подхватывает
его сама.

    # ~/моя-папка/aic_plugins.py
    from experimental_tools_beliy_russak import register_loss

    @register_loss("lovasz")
    def lovasz(logits, targets, *, per_image=True):
        ...

    # ~/моя-папка/configs/мой.yaml
    # loss:
    #   seg: {bce: 1.0, lovasz: 1.0}
    #   lovasz: {per_image: true}

Именованные параметры компоненты приезжают из секции `loss.<имя>` — тем же
способом, каким уже настраиваются встроенные `tversky` и `focal`. Новой
грамматики в конфигах не появляется.

Загрузка ленивая и одноразовая: её запускает первое же обращение к любому
реестру. Поэтому неважно, зашёл ты через `aic train`, через `build_loss`
напрямую или из ноутбука — плагины будут на месте.

Что зарегистрировано и откуда — показывает `aic registry`.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from .workspace import workspace

#: файл в корне воркспейса, который подхватывается автоматически
PLUGIN_FILE = "aic_plugins.py"

_PACKAGE = __name__.split(".")[0]


class Registry:
    """Именованные компоненты одного вида плюс внятная ошибка на опечатку."""

    def __init__(self, key: str, what: str, contract: str, builtins: str) -> None:
        self.key = key            # как называется ключ в конфиге: loss, aug, ...
        self.what = what          # «компонента лосса» — для текста ошибки
        self.contract = contract  # сигнатура, которую ждём от функции
        self.builtins = builtins  # модуль, в котором регистрируются встроенные
        self._items: dict[str, Callable[..., Any]] = {}
        self._builtins_loaded = False

    def _ensure_ready(self) -> None:
        """Сначала встроенные, потом плагины — чтобы override=True мог перебить.

        Импортируется ровно один модуль, свой для каждого реестра: иначе запрос
        пресета аугментаций тянул бы за собой и модели, и оптимизаторы.
        """
        if not self._builtins_loaded:
            # флаг ставится ДО импорта: импортируемый модуль сам дёргает реестр
            self._builtins_loaded = True
            importlib.import_module(self.builtins, __package__)
        ensure_plugins_loaded()

    # --- регистрация -----------------------------------------------------

    def register(
        self,
        name: str,
        fn: Callable[..., Any] | None = None,
        *,
        override: bool = False,
    ):
        """Работает и декоратором, и обычным вызовом `register("имя", функция)`."""
        def decorate(function: Callable[..., Any]) -> Callable[..., Any]:
            key = str(name)
            existing = self._items.get(key)
            if existing is not None and not override:
                raise ValueError(
                    f"{self.what} '{key}' уже зарегистрирована в {existing.__module__}. "
                    f"Возьми другое имя или, если подменяешь осознанно, "
                    f"передай override=True"
                )
            self._items[key] = function
            return function

        return decorate if fn is None else decorate(fn)

    # --- чтение -----------------------------------------------------------

    def get(self, name: str) -> Callable[..., Any]:
        self._ensure_ready()
        try:
            return self._items[str(name)]
        except KeyError:
            raise ValueError(self._unknown_message(str(name))) from None

    def names(self) -> list[str]:
        self._ensure_ready()
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        self._ensure_ready()
        return str(name) in self._items

    def __len__(self) -> int:
        self._ensure_ready()
        return len(self._items)

    def by_source(self) -> dict[str, list[str]]:
        """{'встроенные': [...], 'aic_plugins.py': [...]} — для `aic registry`."""
        groups: dict[str, list[str]] = {}
        for name, function in sorted(self._items.items()):
            module = getattr(function, "__module__", "?")
            label = "встроенные" if module.split(".")[0] == _PACKAGE else module
            groups.setdefault(label, []).append(name)
        return groups

    def _unknown_message(self, name: str) -> str:
        lines = [f"{self.what}: нет варианта {name!r}"]
        for label, items in self.by_source().items():
            lines.append(f"  {label + ':':<20} {', '.join(items)}")
        close = difflib.get_close_matches(name, list(self._items), n=1, cutoff=0.6)
        if close:
            lines.append(f"  похоже на {close[0]!r}?")
        lines.append(
            f"  как добавить свою: @register_{self.key}({name!r}) в {PLUGIN_FILE}, "
            f"сигнатура {self.contract}"
        )
        return "\n".join(lines)


# --------------------------------------------------------------------------
# сами реестры
# --------------------------------------------------------------------------

LOSSES = Registry(
    "loss", "компонента лосса",
    "fn(logits, targets, **params) -> тензор (B,), значение на каждый кадр",
    builtins=".losses",
)
AUGS = Registry(
    "aug", "пресет аугментаций",
    "fn() -> список пиксельных операций albumentations",
    builtins=".transforms",
)
VAL_MODES = Registry(
    "val_mode", "режим валидации",
    "fn(size) -> список операций геометрии albumentations",
    builtins=".transforms",
)
OPTIMIZERS = Registry(
    "optimizer", "оптимизатор",
    "fn(param_groups, cfg_train) -> torch.optim.Optimizer",
    builtins=".engine",
)
SCHEDULERS = Registry(
    "scheduler", "планировщик",
    "fn(optimizer, cfg_train, total_steps, warmup_steps) -> scheduler | None",
    builtins=".engine",
)
BACKENDS = Registry(
    "backend", "backend модели",
    'fn(cfg_model) -> nn.Module, forward которого даёт {"logits", "cls_logits"}',
    builtins=".models",
)

ALL_REGISTRIES: tuple[Registry, ...] = (
    LOSSES, AUGS, VAL_MODES, OPTIMIZERS, SCHEDULERS, BACKENDS,
)


def register_loss(name: str, fn=None, *, override: bool = False):
    """`fn(logits, targets, **params) -> тензор (B,)`; параметры из `loss.<имя>`.

    Значение возвращается по каждому кадру, а не усреднённое: библиотека сама
    взвешивает кадры при включённом профиле по площади (`loss.area`).
    """
    return LOSSES.register(name, fn, override=override)


def register_aug(name: str, fn=None, *, override: bool = False):
    """`fn() -> список операций albumentations`; выбирается через `data.aug`.

    Это только пиксельная часть. Геометрия (кроп, флипы) и нормализация остаются
    за библиотекой: они завязаны на `data.size` и на возврат маски в исходное
    разрешение.
    """
    return AUGS.register(name, fn, override=override)


def register_val_mode(name: str, fn=None, *, override: bool = False):
    """`fn(size) -> список операций геометрии`; выбирается через `data.val_mode`."""
    return VAL_MODES.register(name, fn, override=override)


def register_optimizer(name: str, fn=None, *, override: bool = False):
    """`fn(param_groups, cfg_train) -> Optimizer`; выбирается через `train.optimizer`.

    Группы параметров (энкодер/декодер × decay/no-decay) библиотека собирает
    сама и отдаёт готовыми.
    """
    return OPTIMIZERS.register(name, fn, override=override)


def register_scheduler(name: str, fn=None, *, override: bool = False):
    """`fn(optimizer, cfg_train, total_steps, warmup_steps)`; через `train.scheduler`.

    Полное число шагов и длину прогрева библиотека считает сама — она знает и
    размер эпохи, и накопление градиента.
    """
    return SCHEDULERS.register(name, fn, override=override)


def register_backend(name: str, fn=None, *, override: bool = False):
    """`fn(cfg_model) -> nn.Module`; выбирается через `model.backend`.

    Forward обязан вернуть `{"logits": (B, 1, H, W), "cls_logits": (B, 1)}` —
    на этом контракте держатся и метрика, и сборка сабмита.
    """
    return BACKENDS.register(name, fn, override=override)


# --------------------------------------------------------------------------
# загрузка плагинов
# --------------------------------------------------------------------------

_loaded_paths: set[Path] = set()
_loaded_modules: set[str] = set()
_loading = False


def ensure_plugins_loaded() -> None:
    """Подтянуть `aic_plugins.py` из корня воркспейса, если он есть.

    Идемпотентно и по каждому пути отдельно: после `set_workspace()` на другую
    папку её плагины тоже загрузятся, а уже загруженные повторно не пойдут.
    """
    global _loading
    if _loading:
        # плагин, который при импорте сам что-то строит, иначе ушёл бы в рекурсию
        return

    path = workspace().root / PLUGIN_FILE
    if path in _loaded_paths or not path.is_file():
        return

    _loading = True
    try:
        _load_file(path)
        _loaded_paths.add(path)
    finally:
        _loading = False


def load_plugins(modules: list[str] | tuple[str, ...] | str | None = None) -> list[str]:
    """Загрузить плагины воркспейса и дополнительно перечисленные модули.

    Список модулей берётся из ключа `plugins` в конфиге — так команда может
    держать общие компоненты в отдельном устанавливаемом пакете, а не в файле
    у каждого.
    """
    ensure_plugins_loaded()
    if isinstance(modules, str):
        modules = [modules]

    loaded: list[str] = []
    for name in modules or []:
        name = str(name)
        if name in _loaded_modules:
            continue
        try:
            importlib.import_module(name)
        except Exception as error:
            raise ImportError(
                f"не смог загрузить плагин {name!r} из ключа `plugins`: "
                f"{type(error).__name__}: {error}"
            ) from error
        _loaded_modules.add(name)
        loaded.append(name)
    return loaded


def _load_file(path: Path) -> None:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"не смог прочитать плагин {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    try:
        spec.loader.exec_module(module)
    except Exception as error:
        del sys.modules[path.stem]
        raise ImportError(
            f"плагин {path} упал при импорте: {type(error).__name__}: {error}"
        ) from error


def describe() -> dict[str, dict[str, list[str]]]:
    """{ключ реестра: {источник: [имена]}} — данные для `aic registry`."""
    for registry in ALL_REGISTRIES:
        registry._ensure_ready()
    return {registry.key: registry.by_source() for registry in ALL_REGISTRIES}


@contextlib.contextmanager
def sandbox() -> Iterator[None]:
    """Снимок всех реестров; на выходе состояние возвращается.

    Нужен тестам: тест, зарегистрировавший свою компоненту, не должен утащить её
    в следующий.
    """
    global _loaded_paths, _loaded_modules
    saved = [(dict(r._items), r._builtins_loaded) for r in ALL_REGISTRIES]
    saved_paths, saved_modules = set(_loaded_paths), set(_loaded_modules)
    try:
        yield
    finally:
        for registry, (items, builtins_loaded) in zip(ALL_REGISTRIES, saved):
            registry._items = items
            registry._builtins_loaded = builtins_loaded
        _loaded_paths, _loaded_modules = saved_paths, saved_modules
