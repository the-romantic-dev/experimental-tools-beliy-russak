"""Воспроизводимость: одинаковый сид обязан давать одинаковые аугментации и выборку.

Главная ловушка здесь — albumentations 2.x. У `A.Compose` собственный генератор
случайных чисел, и `random.seed` / `np.random.seed` / `torch.manual_seed` на него
не действуют. Без явного `seed=` два прогона с одним конфигом видят разные
аугментации, и сравнение экспериментов превращается в сравнение шума.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import albumentations as A
import numpy as np
import pytest
import torch

from aic_pipeline.datasets import build_sampler
from aic_pipeline.transforms import build_train_transform, build_transform, build_val_transform

IMAGE = np.random.RandomState(0).randint(0, 255, (64, 96, 3), dtype=np.uint8)
MASK = np.zeros((64, 96), dtype=np.float32)


def apply(transform) -> np.ndarray:
    out = transform(image=IMAGE.copy(), mask=MASK.copy())["image"]
    # наши пресеты заканчиваются ToTensorV2, голый Compose в тестах — нет
    return out.numpy() if hasattr(out, "numpy") else np.asarray(out)


def test_global_seeds_do_not_control_albumentations():
    """Документируем причину, по которой нужен явный seed у Compose.

    Если этот тест однажды упадёт, значит albumentations начал слушать
    глобальные сиды — и костыль с worker_init_fn можно упрощать."""
    import random

    def once():
        random.seed(777)
        np.random.seed(777)
        torch.manual_seed(777)
        return apply(A.Compose([A.RandomBrightnessContrast(p=1.0)]))

    assert not np.array_equal(once(), once())


def test_train_transform_is_reproducible_with_seed():
    a = apply(build_train_transform(size=32, preset="medium", seed=123))
    b = apply(build_train_transform(size=32, preset="medium", seed=123))
    assert np.array_equal(a, b)


def test_train_transform_differs_across_seeds():
    a = apply(build_train_transform(size=32, preset="medium", seed=1))
    b = apply(build_train_transform(size=32, preset="medium", seed=2))
    assert not np.array_equal(a, b)


def test_train_transform_without_seed_is_not_reproducible():
    """Прямое доказательство исходного бага — на случай регресса."""
    a = apply(build_train_transform(size=32, preset="medium", seed=None))
    b = apply(build_train_transform(size=32, preset="medium", seed=None))
    assert not np.array_equal(a, b)


def test_build_transform_passes_seed_through():
    cfg = {"size": 32, "aug": "medium", "crop_scale": (0.3, 1.0)}
    a = apply(build_transform(cfg, train=True, seed=5))
    b = apply(build_transform(cfg, train=True, seed=5))
    c = apply(build_transform(cfg, train=True, seed=6))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


@pytest.mark.parametrize("mode", ["resize", "pad"])
def test_val_transform_is_deterministic(mode):
    """В валидации случайных операций нет, и результат обязан совпадать всегда."""
    a = apply(build_val_transform(size=32, mode=mode))
    b = apply(build_val_transform(size=32, mode=mode))
    assert np.array_equal(a, b)


def test_set_random_seed_resets_the_stream():
    """На этом стоит worker_init_fn: воркер переустанавливает поток своим сидом."""
    transform = build_train_transform(size=32, preset="medium", seed=7)
    first = apply(transform)
    transform.set_random_seed(7)
    assert np.array_equal(apply(transform), first)
    transform.set_random_seed(8)
    assert not np.array_equal(apply(transform), first)


# --- сэмплер ---------------------------------------------------------------

class FakeDataset:
    is_negative = np.array([True] * 20 + [False] * 80)
    area = np.full(100, 0.2, dtype=np.float32)

    def __len__(self):
        return 100


def draw(seed, torch_seed=0):
    torch.manual_seed(torch_seed)  # состояние глобального RNG не должно влиять
    sampler = build_sampler(
        FakeDataset(), {"negative_fraction": 0.25, "epoch_size": 64}, seed=seed
    )
    return list(sampler)


def test_sampler_is_reproducible_with_seed():
    assert draw(42) == draw(42)


def test_sampler_ignores_global_rng_state():
    """Иначе любая правка выше по коду незаметно сдвигала бы состав эпох."""
    assert draw(42, torch_seed=0) == draw(42, torch_seed=99999)


def test_sampler_differs_across_seeds():
    assert draw(1) != draw(2)


def test_sampler_respects_epoch_size():
    assert len(draw(42)) == 64


def test_sampler_without_seed_still_works():
    sampler = build_sampler(FakeDataset(), {"negative_fraction": 0.25, "epoch_size": 32})
    assert len(list(sampler)) == 32
