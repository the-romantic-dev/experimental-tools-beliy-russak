"""Датасеты и сэмплеры.

`SegDataset` отдаёт словарь:
    image     float32 (3, S, S), нормализован под ImageNet
    mask      float32 (1, S, S), 0/1
    label     float32 (1,)  — 1, если кадр изменён (для aux-головы)
    area      float32 (1,)  — доля кадра под маской ПОСЛЕ аугментаций
    index     int           — строка в исходном DataFrame
    orig_h/w  int           — исходный размер, нужен для инференса
    valid_h/w int           — размер полезной части внутри модельной сетки;
                              в режиме resize это вся сетка, в режиме pad —
                              область без паддинга

Негативы бывают двух сортов:
1. «родные» — строки, где GT полностью пустой (их в данных ~5-6k);
2. синтетические — чистые оригиналы из `src`, для них маска = нули.
Второй сорт добавляется только в train и только из групп этого же фолда,
поэтому валидация не протекает.

Позитивы бывают тех же двух сортов: `data.synth` подмешивает в пул кадры,
собранные из чистых оригиналов на лету (см. `synth.py`). Маска у них известна
точно, а площадь вставки задаётся конфигом — это и есть способ дать модели
столько мелких правок, сколько нужно, вместо перевзвешивания дефицитных
настоящих. Синтетика тоже живёт только в train.

`area` нужна лоссу: 22% позитивов имеют маску меньше 6% кадра, и именно на них
приходится почти весь недобор Dice (у корзины <1% Dice 0.135 против 0.86+ у
крупных). Профиль лосса и веса сэмплера по этой величине разводятся отдельно.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, RandomSampler, WeightedRandomSampler

from .geometry import mask_geometry
from .imageio import imread
from .synth import SynthSettings, needs_donor, pick_op, synthesize
from .workspace import resolve
from .precache import cached_path

cv2.setNumThreads(0)  # иначе воркеры DataLoader дерутся за ядра


@dataclass
class Record:
    img_path: str
    gt_path: str | None
    is_negative: bool
    row_index: int
    area: float = 0.0  # доля кадра под GT-маской до аугментаций (из индекса)
    #: кадр-основа для синтеза: `img_path` указывает на чистый оригинал, а
    #: подделка и маска к нему делаются на лету в `__getitem__`
    synth: bool = False


def _read_image(path: Path) -> np.ndarray:
    image = imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"не читается изображение: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"не читается маска: {path}")
    if mask.shape[:2] != shape:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
    return mask


def valid_region(grid_h: int, grid_w: int, orig_h: int, orig_w: int, pad_mode: bool) -> tuple[int, int]:
    """Размер полезной (не-паддинговой) части внутри модельной сетки.

    Повторяет геометрию `LongestMaxSize + PadIfNeeded(position=top_left)` из
    `transforms.build_val_transform` и обратное преобразование в
    `inference._to_original`. Нужен и метрике: правило FPR смотрит на долю
    площади кадра, и считать её от холста вместе с паддингом — значит завышать
    допустимый размер предсказания в полтора раза на широких снимках.
    """
    if not pad_mode:
        return grid_h, grid_w
    scale = max(grid_h, grid_w) / float(max(orig_h, orig_w))
    return (
        min(grid_h, max(1, int(round(orig_h * scale)))),
        min(grid_w, max(1, int(round(orig_w * scale)))),
    )


class SegDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform,
        *,
        source: str = "raw",
        cache_size: int = 768,
        gt_binarize: float | None = 0.5,
        extra_negatives: float = 0.0,
        seed: int = 0,
        pad_mode: bool = False,
        aux_targets: bool = False,
        synth: dict | SynthSettings | None = None,
    ) -> None:
        self.transform = transform
        self.source = source
        self.cache_size = cache_size
        self.gt_binarize = gt_binarize
        self.pad_mode = pad_mode
        self.synth = synth if isinstance(synth, SynthSettings) else SynthSettings.from_config(synth)
        self.synth_bases: list[str] = []
        self._synth_rng: np.random.Generator | None = None
        # геометрия считается только когда её кто-то просит: connectedComponents
        # на маске 768x768 стоит ~1.5 мс, и платить их зря на каждом сэмпле
        # незачем
        self.aux_targets = aux_targets
        self.df = df.reset_index(drop=True)

        areas = (
            self.df["mask_area"].to_numpy(dtype=np.float32)
            if "mask_area" in self.df.columns
            else np.zeros(len(self.df), dtype=np.float32)
        )
        self.records: list[Record] = [
            Record(row.chng_path, row.gt_path, bool(row.is_negative), i, float(areas[i]))
            for i, row in enumerate(self.df.itertuples())
        ]

        if extra_negatives > 0 and "orgl_path" in self.df.columns:
            self.records += self._sample_originals(extra_negatives, seed)

        if self.synth is not None:
            self.records += self._sample_synthetic(seed)

        self.is_negative = np.array([r.is_negative for r in self.records], dtype=bool)
        self.area = np.array([r.area for r in self.records], dtype=np.float32)

    def _originals(self) -> list[str]:
        if "orgl_path" not in self.df.columns:
            return []
        return [str(p) for p in self.df["orgl_path"].dropna().unique().tolist()]

    def _sample_originals(self, fraction: float, seed: int) -> list[Record]:
        originals = self._originals()
        if not originals:
            return []
        n = int(round(len(self.df) * fraction))
        rng = np.random.default_rng(seed)
        chosen = rng.choice(originals, size=min(n, len(originals)), replace=False)
        return [Record(str(p), None, True, -1, 0.0) for p in chosen]

    def _sample_synthetic(self, seed: int) -> list[Record]:
        """Кадры-основы под синтез: столько, чтобы их доля среди позитивов была
        ровно `synth.fraction`.

        Основы берутся с возвращением: один и тот же оригинал даёт каждый раз
        другую подделку, поэтому повтор пути — не повтор примера. Доля считается
        от НАСТОЯЩИХ позитивов, уже лежащих в пуле.
        """
        originals = self._originals()
        if not originals:
            raise ValueError(
                "data.synth включён, но в индексе нет ни одного `orgl_path` — "
                "синтезировать подделки не из чего"
            )
        self.synth_bases = originals

        n_positive = sum(1 for record in self.records if not record.is_negative)
        fraction = self.synth.fraction
        n_synth = int(round(n_positive * fraction / (1.0 - fraction)))
        if n_synth <= 0:
            return []

        rng = np.random.default_rng(seed + 1)
        chosen = rng.choice(originals, size=n_synth, replace=True)
        # площадь для сэмплера — геометрическая середина заказанного диапазона:
        # настоящей она станет только после розыгрыша в `__getitem__`
        area = float(np.sqrt(self.synth.area_range[0] * self.synth.area_range[1]))
        return [Record(str(p), None, False, -1, area, synth=True) for p in chosen]

    def _rng(self) -> np.random.Generator:
        """Генератор синтеза — ленивый и свой у каждого воркера DataLoader.

        Сид берётся из глобального numpy-RNG, а его torch засевает в каждом
        воркере от base_seed прогона. Поэтому поток воспроизводим по `seed`
        конфига, но разный между воркерами: иначе все они лепили бы на одном
        шаге одинаковые подделки.
        """
        if self._synth_rng is None:
            self._synth_rng = np.random.default_rng(int(np.random.randint(0, 2 ** 31 - 1)))
        return self._synth_rng

    def _synthesize(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        rng = self._rng()
        op = pick_op(rng, self.synth)
        donor = None
        if needs_donor(op):
            donor = _read_image(self._resolve(str(rng.choice(self.synth_bases)), False))
        return synthesize(image, rng, self.synth, op=op, donor=donor)

    def _resolve(self, rel_path: str, is_mask: bool) -> Path:
        if self.source == "cache":
            path = cached_path(rel_path, self.cache_size, is_mask)
            if path.exists():
                return path
        return resolve(rel_path)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        record = self.records[idx]
        image = _read_image(self._resolve(record.img_path, False))
        h, w = image.shape[:2]

        if record.synth:
            image, mask = self._synthesize(image)
        elif record.gt_path is None:
            mask = np.zeros((h, w), dtype=np.float32)
        else:
            mask = _read_mask(self._resolve(record.gt_path, True), (h, w)).astype(np.float32) / 255.0

        augmented = self.transform(image=image, mask=mask)
        image_t = augmented["image"]
        mask_t = augmented["mask"].float()
        if mask_t.ndim == 2:
            mask_t = mask_t.unsqueeze(0)

        if self.gt_binarize is not None:
            mask_t = (mask_t >= self.gt_binarize).float()

        valid_h, valid_w = valid_region(mask_t.shape[-2], mask_t.shape[-1], h, w, self.pad_mode)
        # площадь считается по полезной области: в pad-режиме паддинг занимает
        # до трети холста и иначе systematically занижал бы долю маски
        area = float(mask_t[..., :valid_h, :valid_w].mean())

        geometry = mask_geometry(mask_t, valid_h, valid_w) if self.aux_targets else {}

        return {
            "image": image_t,
            "mask": mask_t,
            "label": torch.tensor([float(mask_t.max() > 0)], dtype=torch.float32),
            "area": torch.tensor([area], dtype=torch.float32),
            # площадь уже лежит выше в сыром виде — дублировать её под aux_
            # не надо, цель для головы считается из неё нормировкой в лоссе
            **{f"aux_{k}": v for k, v in geometry.items()
               if k not in {"geom_valid", "area"}},
            **({"geom_valid": geometry["geom_valid"]} if geometry else {}),
            "index": record.row_index,
            "orig_h": h,
            "orig_w": w,
            "valid_h": valid_h,
            "valid_w": valid_w,
        }


class PredictDataset(Dataset):
    """Инференс по списку файлов: без масок, с возвратом исходного размера."""

    def __init__(self, paths: Sequence[Path | str], transform) -> None:
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        path = self.paths[idx]
        image = _read_image(path)
        h, w = image.shape[:2]
        augmented = self.transform(image=image, mask=np.zeros((h, w), dtype=np.float32))
        return {
            "image": augmented["image"],
            "orig_h": h,
            "orig_w": w,
            "index": idx,
            "name": path.stem,
        }


def build_balanced_sampler(
    dataset: SegDataset,
    negative_fraction: float,
    num_samples: int | None = None,
    *,
    small_area_fraction: float | None = None,
    small_area_threshold: float = 0.06,
    generator: torch.Generator | None = None,
) -> WeightedRandomSampler:
    """Держит заданную долю негативов (и, опционально, мелких масок) в батче.

    Половина метрики — FPR_neg, а негативов в трейне ~3%. Без перевзвешивания
    модель почти не видит чистых кадров и щедро галлюцинирует на них маски.

    `small_area_fraction` добавляет второй перекос — внутри позитивов. Кадры с
    маской меньше `small_area_threshold` составляют 22% позитивов, но держат
    почти весь запас по Dice: модель промахивает 71% масок мельче 1% кадра и
    36% масок 1-3%, тогда как на крупных Dice уже 0.86+. При естественной доле
    таких кадров градиента на них приходится слишком мало.
    """
    is_neg = dataset.is_negative
    n_neg, n_pos = int(is_neg.sum()), int((~is_neg).sum())
    if n_neg == 0 or n_pos == 0:
        weights = np.ones(len(dataset), dtype=np.float64)
    elif not small_area_fraction:
        weights = np.where(is_neg, negative_fraction / n_neg, (1.0 - negative_fraction) / n_pos)
    else:
        is_small = (~is_neg) & (dataset.area < small_area_threshold)
        is_large = (~is_neg) & ~is_small
        n_small, n_large = int(is_small.sum()), int(is_large.sum())
        positive_budget = 1.0 - negative_fraction
        weights = np.where(is_neg, negative_fraction / n_neg, 0.0)
        if n_small:
            weights[is_small] = positive_budget * small_area_fraction / n_small
        if n_large:
            weights[is_large] = positive_budget * (1.0 - small_area_fraction) / n_large
        if not n_small or not n_large:  # вырожденный случай — не оставлять нули
            weights[~is_neg] = positive_budget / n_pos

    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=num_samples or len(dataset),
        replacement=True,
        generator=generator,
    )


def build_sampler(dataset, cfg_data: dict, seed: int | None = None):
    """Сэмплер по конфигу: `negative_fraction` задаёт баланс, `epoch_size` — длину эпохи.

    Оба параметра независимы. Раньше `epoch_size` работал только вместе с
    `negative_fraction` и молча игнорировался без него — теперь для этого случая
    берётся обычный RandomSampler с нужным числом отсчётов.

    `seed` даёт сэмплеру собственный генератор. Без него порядок выборки зависит
    от состояния глобального RNG на момент начала итерации, то есть от того,
    сколько раз его дёрнули выше по коду — и любая правка в `run()` незаметно
    меняла бы состав эпох.
    """
    neg_frac = cfg_data.get("negative_fraction")
    epoch_size = cfg_data.get("epoch_size")
    num_samples = int(epoch_size) if epoch_size else None

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(seed))

    if neg_frac:
        return build_balanced_sampler(
            dataset, float(neg_frac), num_samples,
            small_area_fraction=cfg_data.get("small_area_fraction"),
            small_area_threshold=float(cfg_data.get("small_area_threshold", 0.06)),
            generator=generator,
        )
    if num_samples:
        return RandomSampler(
            dataset, replacement=True, num_samples=num_samples, generator=generator
        )
    return None
