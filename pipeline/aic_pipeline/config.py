"""Конфиги: YAML c наследованием через `_base_` + переопределения из CLI.

    cfg = load_config("configs/unet_convnext_512.yaml", overrides=["train.lr=3e-4", "train.bs=8"])
    cfg.train.lr  # 0.0003  (и cfg["train"]["lr"] тоже работает)

Значения из CLI парсятся как YAML-скаляры, поэтому `null`, `true`, `[1,2]`
и `3e-4` приезжают уже нужным типом, а не строкой.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import yaml

from .schema import check_config
from .workspace import configs_root, project_root


def get_path(node: Any, dotted: str, default: Any = None) -> Any:
    """Значение по пути `a.b.c` в любом вложенном словаре; нет ключа — `default`.

    Отдельной функцией, а не только методом `Cfg`: снапшот чужого прогона
    приезжает из YAML обычным dict'ом, и оборачивать его ради одного чтения
    (`stats.py` так и делает) незачем.
    """
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


class Cfg(dict):
    """dict с доступом через точку. Вложенные dict оборачиваются лениво."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(
                f"нет ключа '{name}' в конфиге; доступны: {sorted(self.keys())}"
            ) from exc
        return Cfg(value) if isinstance(value, dict) else value

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        return get_path(self, dotted, default)


def _deep_merge(base: dict, override: dict) -> dict:
    """override поверх base; вложенные словари сливаются, остальное заменяется."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _resolve_config_path(spec: str | Path) -> Path:
    """Принимает 'smoke', 'smoke.yaml' или полный путь."""
    path = Path(spec)
    root = configs_root()
    candidates = [path, path.with_suffix(".yaml"), root / path,
                  root / path.with_suffix(".yaml")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"конфиг не найден: {spec} (искал в {root})")


def _load_raw(spec: str | Path, _seen: set[Path] | None = None) -> dict:
    path = _resolve_config_path(spec)
    _seen = _seen or set()
    if path in _seen:
        raise ValueError(f"циклическое наследование конфигов на {path}")
    _seen.add(path)

    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался словарь на верхнем уровне")

    bases = data.pop("_base_", None)
    if bases is None:
        return data
    if isinstance(bases, str):
        bases = [bases]

    merged: dict = {}
    for base in bases:
        base_path = base if Path(base).is_absolute() else path.parent / base
        merged = _deep_merge(merged, _load_raw(base_path, set(_seen)))
    return _deep_merge(merged, data)


def apply_override(cfg: dict, dotted_assignment: str) -> None:
    """'train.lr=3e-4' -> cfg['train']['lr'] = 0.0003 (in-place)."""
    if "=" not in dotted_assignment:
        raise ValueError(f"ожидался вид key.sub=value, получено: {dotted_assignment!r}")
    key, raw_value = dotted_assignment.split("=", 1)
    value = yaml.safe_load(raw_value)
    if isinstance(value, str):
        # YAML 1.1 не считает числом запись вида 3e-4 (нужно 3.0e-4), а из
        # командной строки её пишут именно так — доводим руками
        try:
            value = float(value)
        except ValueError:
            pass

    node = cfg
    parts = key.strip().split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def load_config(
    spec: str | Path,
    overrides: Iterable[str] | None = None,
    *,
    check: bool = True,
) -> Cfg:
    """Собрать конфиг из файла, его баз и переопределений из командной строки.

    `check=False` отключает проверку имён ключей — нужно там, где конфиг
    заведомо частичный (например, снапшот старого прогона).
    """
    cfg = _load_raw(spec)
    for override in overrides or []:
        apply_override(cfg, override)
    cfg.setdefault("_source", _describe_source(_resolve_config_path(spec)))

    if check:
        check_config(cfg, source=str(cfg["_source"]))

    # ключ `plugins` подтягивает чужие компоненты до того, как конфиг пойдёт в
    # билдеры; `aic_plugins.py` воркспейса грузится и без этого ключа
    from .registry import load_plugins

    load_plugins(cfg.get("plugins"))
    return Cfg(cfg)


def _describe_source(path: Path) -> str:
    """Путь конфига для снапшота: короткий относительный, если он внутри воркспейса.

    Конфиг вполне может лежать и снаружи — например, у соседа по команде свои
    рецепты в отдельной папке. Тогда пишем абсолютный путь, а не падаем.
    """
    try:
        return str(path.relative_to(project_root()))
    except ValueError:
        return str(path)


def save_config(cfg: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, allow_unicode=True, sort_keys=False)


def config_hash(cfg: dict, length: int = 8) -> str:
    """Стабильный хэш конфига — для имён прогонов и сравнения экспериментов."""
    payload = json.dumps(
        {k: v for k, v in cfg.items() if not k.startswith("_")},
        sort_keys=True, default=str,
    )
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:length]


def flatten(cfg: dict, prefix: str = "") -> dict[str, Any]:
    """Плоский вид для таблицы сравнения прогонов и для TensorBoard hparams."""
    flat: dict[str, Any] = {}
    for key, value in cfg.items():
        full = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, prefix=f"{full}."))
        else:
            flat[full] = value
    return flat
