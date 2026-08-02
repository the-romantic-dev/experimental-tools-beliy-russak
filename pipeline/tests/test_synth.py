"""Синтез подделок из чистых оригиналов: контракт операций и встройка в датасет.

Главное, что здесь проверяется, — два инварианта, без которых эксперимент
теряет смысл:

1. **Маска точна.** Кадр меняется ровно там, где маска ненулевая. Если правка
   протекает за маску, обучение получает ложные негативы в тех самых мелких
   корзинах, ради которых всё затевается.
2. **`post_jpeg` трогает весь кадр.** Если сжимать до правки, граница вставки
   не попадает в JPEG-сетку, и модель выучивает артефакт синтеза за одну эпоху:
   train красивый, валидация не двигается.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aic_pipeline.datasets import SegDataset
from aic_pipeline.imageio import imwrite
from aic_pipeline.schema import find_unknown_keys
from aic_pipeline.synth import (
    OPS,
    SynthSettings,
    needs_donor,
    pick_op,
    synthesize,
)
from aic_pipeline.transforms import build_val_transform
from aic_pipeline.workspace import resolve, set_workspace

SIZE = 512


def noise_image(seed: int = 0) -> np.ndarray:
    """Шумный кадр: на нём любая правка заведомо меняет пиксели."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def settings(**over) -> SynthSettings:
    base = dict(fraction=0.3, area_range=(0.005, 0.02), feather=(1, 3), post_jpeg=None)
    return SynthSettings(**{**base, **over})


# --- настройки -----------------------------------------------------------


def test_settings_are_off_when_not_configured():
    assert SynthSettings.from_config(None) is None
    assert SynthSettings.from_config({}) is None
    assert SynthSettings.from_config({"fraction": 0.0}) is None


def test_settings_read_the_config_block():
    parsed = SynthSettings.from_config({
        "fraction": 0.25, "area_range": [0.002, 0.05],
        "ops": ["copy_move"], "feather": [2, 4], "post_jpeg": [60, 90],
    })
    assert parsed.fraction == 0.25
    assert parsed.area_range == (0.002, 0.05)
    assert parsed.ops == ("copy_move",)
    assert parsed.post_jpeg == (60, 90)


def test_settings_reject_an_unknown_op():
    with pytest.raises(ValueError, match="copy_move"):
        SynthSettings.from_config({"fraction": 0.3, "ops": ["copy_paste"]})


def test_settings_reject_an_inverted_area_range():
    with pytest.raises(ValueError, match="area_range"):
        SynthSettings.from_config({"fraction": 0.3, "area_range": [0.05, 0.001]})


def test_schema_catches_a_typo_inside_the_synth_block():
    (unknown,) = find_unknown_keys({"data": {"synth": {"fracton": 0.3}}})
    assert unknown.path == "data.synth.fracton"
    assert unknown.suggestion == "data.synth.fraction"


# --- контракт операций ---------------------------------------------------


@pytest.mark.parametrize("op", OPS)
def test_every_op_keeps_pixels_outside_the_mask_intact(op):
    image = noise_image()
    donor = noise_image(1) if needs_donor(op) else None
    out, mask = synthesize(image, np.random.default_rng(7), settings(), op=op, donor=donor)

    outside = mask <= 0.0
    assert np.array_equal(out[outside], image[outside])


@pytest.mark.parametrize("op", OPS)
def test_every_op_actually_changes_the_frame(op):
    image = noise_image()
    donor = noise_image(1) if needs_donor(op) else None
    out, mask = synthesize(image, np.random.default_rng(7), settings(), op=op, donor=donor)

    core = mask >= 0.9
    assert core.any(), "маска пустая — правки не было"
    assert not np.array_equal(out[core], image[core])


@pytest.mark.parametrize("op", OPS)
def test_output_contract(op):
    image = noise_image()
    donor = noise_image(1) if needs_donor(op) else None
    out, mask = synthesize(image, np.random.default_rng(3), settings(), op=op, donor=donor)

    assert out.shape == image.shape and out.dtype == np.uint8
    assert mask.shape == image.shape[:2] and mask.dtype == np.float32
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0


def test_post_jpeg_touches_the_whole_frame():
    """Сжатие ПОСЛЕ правки: иначе граница вставки не в JPEG-сетке и модель
    выучивает артефакт синтеза, а не подделку."""
    image = noise_image()
    cfg = settings(post_jpeg=(60, 60), ops=("copy_move",))
    out, mask = synthesize(image, np.random.default_rng(5), cfg, op="copy_move")

    outside = mask <= 0.0
    assert not np.array_equal(out[outside], image[outside])


def test_mask_area_stays_inside_the_requested_range():
    cfg = settings(area_range=(0.005, 0.02))
    areas = []
    for seed in range(30):
        _, mask = synthesize(noise_image(), np.random.default_rng(seed), cfg, op="copy_move")
        areas.append(float((mask >= 0.5).mean()))

    # растушёвка размывает границу, поэтому бинаризованная площадь совпадает с
    # заказанной лишь приблизительно — допуск на это, но не на порядок
    assert min(areas) >= 0.005 * 0.75
    assert max(areas) <= 0.02 * 1.25
    assert len(set(np.round(areas, 4))) > 1, "площадь должна разыгрываться, а не быть константой"


def test_splice_pastes_donor_content():
    image = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    donor = np.full((SIZE, SIZE, 3), 255, dtype=np.uint8)
    out, mask = synthesize(
        image, np.random.default_rng(11), settings(), op="splice", donor=donor
    )
    assert out[mask >= 0.9].min() > 200


def test_splice_without_a_donor_is_an_error():
    with pytest.raises(ValueError, match="donor"):
        synthesize(noise_image(), np.random.default_rng(0), settings(), op="splice")


def test_pick_op_only_returns_configured_ops():
    cfg = settings(ops=("recompress", "local_blur"))
    rng = np.random.default_rng(0)
    drawn = {pick_op(rng, cfg) for _ in range(50)}
    assert drawn == {"recompress", "local_blur"}


def test_same_seed_gives_the_same_forgery():
    image = noise_image()
    first = synthesize(image, np.random.default_rng(42), settings(), op="copy_move")
    second = synthesize(image, np.random.default_rng(42), settings(), op="copy_move")
    assert np.array_equal(first[0], second[0])
    assert np.array_equal(first[1], second[1])


# --- встройка в датасет --------------------------------------------------


def frame(index: int, negative: bool = False) -> dict:
    return {
        "chng_path": f"chng/{index}.png",
        "gt_path": None if negative else f"gt/{index}.png",
        "orgl_path": f"src/{index}.png",
        "is_negative": negative,
        "mask_area": 0.0 if negative else 0.1,
    }


def test_synthetic_positives_take_the_requested_share_of_positives():
    df = pd.DataFrame([frame(i) for i in range(8)] + [frame(100, negative=True)])
    dataset = SegDataset(df, build_val_transform(size=64), synth={"fraction": 0.5})

    is_synth = np.array([record.synth for record in dataset.records])
    positives = ~dataset.is_negative
    assert is_synth[positives].mean() == pytest.approx(0.5)
    assert not is_synth[dataset.is_negative].any(), "синтетика не может быть негативом"


def test_synthesis_is_off_by_default():
    df = pd.DataFrame([frame(i) for i in range(4)])
    dataset = SegDataset(df, build_val_transform(size=64))
    assert not any(record.synth for record in dataset.records)


def test_only_the_training_half_gets_synthesis(monkeypatch):
    """Синтетика в валидации сделала бы метрику несравнимой с остальными прогонами."""
    import logging

    from aic_pipeline import train as train_module
    from aic_pipeline.config import load_config

    seen: list = []

    class Spy:
        def __init__(self, df, transform, **kwargs):
            seen.append(kwargs.get("synth"))
            self.df = df
            self.is_negative = np.zeros(len(df), dtype=bool)

        def __len__(self) -> int:
            return len(self.df)

    folds = pd.DataFrame([{**frame(i), "fold": i % 2, "stem": f"s{i}"} for i in range(20)])
    monkeypatch.setattr(train_module, "SegDataset", Spy)
    monkeypatch.setattr(train_module, "load_folds", lambda: folds)

    cfg = load_config("h1_synth", ["train.num_workers=0"])
    train_module.build_dataloaders(cfg, logging.getLogger("test-synth"))

    train_synth, val_synth = seen
    assert train_synth, "у train синтез обязан быть включён"
    assert not val_synth, "в валидации синтетики быть не должно"


def test_synthetic_item_carries_a_nonempty_mask(tmp_path):
    """Сквозной проход: синтетический кадр приходит из датасета позитивом."""
    set_workspace(tmp_path)

    for index in range(4):
        path = resolve(f"src/{index}.png")
        path.parent.mkdir(parents=True, exist_ok=True)
        # не cv2.imwrite: на путях с кириллицей он молча ничего не пишет,
        # ради этого в пакете и живёт обёртка imageio
        imwrite(path, noise_image(index))

    df = pd.DataFrame([frame(i) for i in range(4)])
    dataset = SegDataset(
        df, build_val_transform(size=128),
        synth={"fraction": 0.5, "area_range": [0.02, 0.05], "ops": ["copy_move"]},
    )
    synthetic = next(i for i, record in enumerate(dataset.records) if record.synth)
    item = dataset[synthetic]

    assert float(item["label"]) == 1.0
    assert float(item["area"]) > 0.0
    assert 0.0 < float(item["mask"].mean()) < 1.0
