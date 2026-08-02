"""Аугментации (albumentations 2.x).

Пресеты выбираются строкой в конфиге: `data.aug: light|medium|heavy|none`.

Особенность forensics-задачи: модель ловит подделку по низкоуровневым следам —
шуму сенсора, JPEG-сетке, границам блендинга. Поэтому:
* цветовые/яркостные аугментации держим слабыми — они смывают полезный сигнал;
* JPEG-рекомпрессия и downscale, наоборот, полезны: тестовые кадры почти
  наверняка пережаты, и модель не должна разваливаться от смены качества;
* геометрия ограничена флипами и поворотами на 90° — аффинные искажения
  ресемплят изображение и создают артефакты, которых в тесте нет.
"""

from __future__ import annotations

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

from .registry import AUGS, VAL_MODES, register_aug, register_val_mode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _geometry_train(size: int, scale: tuple[float, float]) -> list:
    return [
        A.RandomResizedCrop(size=(size, size), scale=scale, ratio=(0.75, 1.333),
                            interpolation=cv2.INTER_LINEAR,
                            mask_interpolation=cv2.INTER_LINEAR, p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
    ]


# Пресеты — обычные записи в реестре, свой добавляется из aic_plugins.py
# декоратором @register_aug. Геометрия и нормализация не настраиваются: они
# завязаны на data.size и на возврат маски в исходное разрешение.

@register_aug("none")
def _pixel_none() -> list:
    return []


@register_aug("light")
def _pixel_light() -> list:
    return [
        A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.3),
        A.ImageCompression(compression_type="jpeg", quality_range=(60, 100), p=0.3),
    ]


@register_aug("medium")
def _pixel_medium() -> list:
    return _pixel_light() + [
        A.HueSaturationValue(hue_shift_limit=6, sat_shift_limit=12, val_shift_limit=8, p=0.2),
        A.OneOf([
            A.GaussNoise(std_range=(0.02, 0.08), p=1.0),
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        ], p=0.2),
        A.Downscale(scale_range=(0.6, 0.9), p=0.1),
    ]


@register_aug("heavy")
def _pixel_heavy() -> list:
    return _pixel_medium() + [
        A.ImageCompression(compression_type="jpeg", quality_range=(35, 75), p=0.25),
        A.Sharpen(alpha=(0.1, 0.3), p=0.1),
        A.CLAHE(clip_limit=2.0, p=0.1),
    ]


def _finalize() -> list:
    return [
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD, max_pixel_value=255.0),
        ToTensorV2(transpose_mask=True),
    ]


def build_train_transform(
    size: int = 512,
    preset: str = "medium",
    scale: tuple[float, float] = (0.35, 1.0),
    seed: int | None = None,
) -> A.Compose:
    """`seed` обязателен для воспроизводимости.

    У albumentations 2.x свой генератор случайных чисел внутри Compose, и
    `random.seed`/`np.random.seed`/`torch.manual_seed` на него НЕ влияют: без
    явного `seed` каждый Compose берёт энтропию у ОС, и два прогона с одним
    конфигом видят разные аугментации.

    В воркерах DataLoader сид переустанавливается индивидуально (см.
    `worker_init_fn` в train.py) — иначе все воркеры, получив объект
    пиклом вместе с состоянием генератора, выдавали бы один и тот же поток
    аугментаций.
    """
    pixel = AUGS.get(preset)()
    return A.Compose(_geometry_train(size, tuple(scale)) + pixel + _finalize(), seed=seed)


@register_val_mode("resize")
def _val_resize(size: int) -> list:
    return [A.Resize(size, size, interpolation=cv2.INTER_LINEAR,
                     mask_interpolation=cv2.INTER_LINEAR)]


@register_val_mode("pad")
def _val_pad(size: int) -> list:
    return [
        A.LongestMaxSize(max_size=size, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(min_height=size, min_width=size, position="top_left",
                      border_mode=cv2.BORDER_CONSTANT, fill=0, fill_mask=0),
    ]


def build_val_transform(size: int = 512, mode: str = "resize", seed: int | None = 0) -> A.Compose:
    """`resize` — растянуть в квадрат; `pad` — вписать по длинной стороне и добить паддингом.

    `pad` бережёт пропорции (и мелкие правки на краях), но даёт «пустые» поля.
    Что лучше — проверяется экспериментом; по умолчанию resize, как быстрее.

    Свой режим — `@register_val_mode`. Но помни: `inference.py` умеет возвращать
    маску в исходное разрешение только для этих двух геометрий, третью придётся
    научить и его.
    """
    geometry = VAL_MODES.get(mode)(size)
    # валидация детерминирована по построению (случайных операций нет), но сид
    # задаём явно, чтобы Compose не тянул энтропию из ОС на каждом создании
    return A.Compose(geometry + _finalize(), seed=seed)


def build_transform(cfg_data: dict, train: bool, seed: int | None = None) -> A.Compose:
    size = int(cfg_data.get("size", 512))
    if train:
        return build_train_transform(
            size=size,
            preset=str(cfg_data.get("aug", "medium")),
            scale=tuple(cfg_data.get("crop_scale", (0.35, 1.0))),
            seed=seed,
        )
    return build_val_transform(
        size=size, mode=str(cfg_data.get("val_mode", "resize")), seed=0
    )
