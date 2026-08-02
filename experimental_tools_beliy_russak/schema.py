"""Проверка конфига на опечатки в именах ключей.

Зачем. Неизвестный ключ раньше проходил молча: `-s train.epohs=20` создавал
новый ключ, а `epochs` оставался прежним — прогон шёл на двенадцати эпохах, и
никто этого не замечал. Пока конфиги писал один человек по образцу, это было
терпимо; когда их пишут вчетвером с нуля, ловится регулярно.

Что делает проверка:

* ключ, похожий на известный, — **ошибка** с подсказкой. Настоящие опечатки
  почти всегда такие: `epohs`, `weigth_decay`, `enocder`;
* ключ, ни на что не похожий, — предупреждение. Молча игнорировать его тоже
  неправильно (он ничего не делает), но и падать нельзя: это может быть
  параметр чужой компоненты из `aic_plugins.py`.

Открытые ветки конфиг задаёт сам, своими же значениями. Если написано
`train.scheduler: step`, то `train.step.*` — это параметры выбранного
планировщика, а не опечатка. То же для оптимизатора, аугментаций, режима
валидации, бэкенда и компонент лосса. Поэтому проверке не нужен реестр, а
`load_config` остаётся лёгким — без импорта torch.
"""

from __future__ import annotations

import difflib
import warnings
from typing import Any, Iterator, NamedTuple

#: обычный ключ: значение любое, вложенности не ждём
LEAF = object()
#: ветка целиком свободна — внутрь не смотрим
OPEN = object()

#: насколько имя должно быть похоже на известное, чтобы счесть это опечаткой.
#: 0.8 разделяет `epohs` -> `epochs` (0.92) и `step_gamma` -> `grad_clip` (0.3)
SIMILARITY = 0.8


class UnknownConfigKey(UserWarning):
    """Ключ, которого библиотека не читает. Не опечатка, но и не работает."""


SCHEMA: dict[str, Any] = {
    "name": LEAF,
    "seed": LEAF,
    "device": LEAF,
    "deterministic": LEAF,
    "tensorboard": LEAF,
    "plugins": LEAF,
    "_source": LEAF,

    "data": {
        "source": LEAF,
        "cache_size": LEAF,
        "size": LEAF,
        "aug": LEAF,
        "crop_scale": LEAF,
        "val_mode": LEAF,
        "gt_binarize": LEAF,
        "negative_fraction": LEAF,
        "extra_negatives": LEAF,
        "small_area_fraction": LEAF,
        "small_area_threshold": LEAF,
        "synth": {
            "fraction": LEAF,
            "area_range": LEAF,
            "ops": LEAF,
            "feather": LEAF,
            "post_jpeg": LEAF,
        },
        "val_keep_negatives": LEAF,
        "val_seed": LEAF,
        "epoch_size": LEAF,
        "train_frac": LEAF,
        "val_frac": LEAF,
        "train_limit": LEAF,
        "val_limit": LEAF,
    },

    "model": {
        "backend": LEAF,
        "arch": LEAF,
        "encoder": LEAF,
        "encoder_weights": LEAF,
        "cls_head": LEAF,
        "cls_dropout": LEAF,
        "stream": LEAF,
        "fuse": LEAF,
        "stream_width": LEAF,
        "skip_norm": LEAF,
        "fp32_decoder_stem": LEAF,
        "hf_name": LEAF,
        # имена голов проверяет реестр AUX_HEADS при сборке модели, а не схема
        "aux_heads": OPEN,
        # параметры конструктора энкодера: их набор задаёт timm, не мы
        "encoder_kwargs": OPEN,
    },

    "loss": {
        # имена компонент проверяет реестр при сборке лосса, а не схема
        "seg": OPEN,
        "cls_weight": LEAF,
        "dice_smooth": LEAF,
        "pos_weight": LEAF,
        "tversky": OPEN,
        "focal": OPEN,
        "area": {
            "threshold": LEAF,
            "small_seg": OPEN,
            "small_weight": LEAF,
        },
    },

    "train": {
        "fold": LEAF,
        "epochs": LEAF,
        "bs": LEAF,
        "val_bs": LEAF,
        "accum_steps": LEAF,
        "lr": LEAF,
        "encoder_lr": LEAF,
        "weight_decay": LEAF,
        "optimizer": LEAF,
        "scheduler": LEAF,
        "warmup_frac": LEAF,
        "min_lr_factor": LEAF,
        "amp": LEAF,
        "grad_clip": LEAF,
        "ema": LEAF,
        "ema_decay": LEAF,
        "num_workers": LEAF,
        "log_every": LEAF,
        "early_stop": LEAF,
        "max_train_steps": LEAF,
        "max_val_steps": LEAF,
    },

    "calib": {
        "n_bins": LEAF,
        "mask_grid": LEAF,
        "cls_grid": LEAF,
        "area_grid": LEAF,
    },

    "budget": {
        "exempt": LEAF,
        "exempt_reason": LEAF,
    },

    "stats": {
        "reference": LEAF,
        "train_sigma": LEAF,
        "bootstrap_n": LEAF,
        "bootstrap_seed": LEAF,
        "gate_delta": LEAF,
        "gate_after_samples": LEAF,
        "gate_action": LEAF,
    },
}


class Unknown(NamedTuple):
    path: str
    #: похожий известный ключ, если нашёлся — тогда это опечатка
    suggestion: str | None


def _selected_names(section: dict, keys: tuple[str, ...]) -> set[str]:
    """Значения ключей выбора — имена компонент, чьи параметры лежат рядом."""
    chosen = set()
    for key in keys:
        value = section.get(key)
        if isinstance(value, str):
            chosen.add(value)
    return chosen


def _open_branches(path: str, section: dict) -> set[str]:
    """Какие ключи этой секции — параметры выбранной компоненты, а не опечатки."""
    if path == "train":
        return _selected_names(section, ("optimizer", "scheduler"))
    if path == "data":
        return _selected_names(section, ("aug", "val_mode"))
    if path == "model":
        return _selected_names(section, ("backend",))
    if path == "loss":
        # компоненты, упомянутые в loss.seg и loss.area.small_seg, могут иметь
        # секцию параметров рядом: loss.tversky, loss.<своя компонента>
        used = set()
        for holder in (section.get("seg"), (section.get("area") or {}).get("small_seg")):
            if isinstance(holder, dict):
                used |= set(holder)
        return used
    return set()


def _walk(node: dict, schema: dict, path: str = "") -> Iterator[Unknown]:
    known = set(schema) | _open_branches(path, node)

    for key, value in node.items():
        full = f"{path}.{key}" if path else str(key)
        if key not in known:
            match = difflib.get_close_matches(str(key), sorted(known), n=1, cutoff=SIMILARITY)
            yield Unknown(full, f"{path}.{match[0]}" if (match and path) else
                          (match[0] if match else None))
            continue

        expected = schema.get(key, OPEN)
        if expected is OPEN or expected is LEAF:
            continue
        if isinstance(expected, dict) and isinstance(value, dict):
            yield from _walk(value, expected, full)


def find_unknown_keys(cfg: dict) -> list[Unknown]:
    """Все ключи, которых библиотека не читает, с подсказкой по каждому."""
    return list(_walk(dict(cfg), SCHEMA))


def check_config(cfg: dict, *, source: str = "конфиг") -> None:
    """Падает на опечатках, предупреждает про остальное незнакомое."""
    unknown = find_unknown_keys(cfg)
    if not unknown:
        return

    typos = [item for item in unknown if item.suggestion]
    if typos:
        lines = [f"{source}: ключи, которых библиотека не знает:"]
        lines += [f"  {item.path}  ->  может быть, {item.suggestion}?" for item in typos]
        lines.append(
            "Такой ключ молча ничего не делает: настоящий параметр остаётся "
            "прежним. Все ключи с комментариями — в configs/_base.yaml."
        )
        raise ValueError("\n".join(lines))

    warnings.warn(
        f"{source}: библиотека не читает "
        + ", ".join(item.path for item in unknown)
        + ". Если это параметры своей компоненты, положи их в секцию с её именем "
        "(например, train.scheduler=step -> train.step.*) — тогда предупреждения не будет.",
        UnknownConfigKey,
        stacklevel=3,
    )
