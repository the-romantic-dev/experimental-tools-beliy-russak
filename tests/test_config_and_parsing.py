"""Конфиги, разбор имён файлов и группировка — то, на чём ломается сплит."""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import pytest

from experimental_tools_beliy_russak.config import apply_override, config_hash, flatten, load_config
from experimental_tools_beliy_russak.indexing import parse_domain, parse_generator, parse_group_id, stem_of


def test_base_config_inheritance_and_overrides():
    cfg = load_config("smoke")
    assert cfg.model.backend == "smp"          # приехало из _base.yaml
    assert cfg.data.size == 256                # переопределено в smoke.yaml
    assert cfg.train.epochs == 1


def test_cli_overrides_are_parsed_as_yaml_scalars():
    cfg = load_config("smoke", ["train.lr=3e-4", "train.ema=false", "data.crop_scale=[0.5,1.0]"])
    assert cfg.train.lr == pytest.approx(3e-4)
    assert cfg.train.ema is False
    assert cfg.data.crop_scale == [0.5, 1.0]


def test_override_creates_missing_nested_keys():
    cfg = {}
    apply_override(cfg, "a.b.c=7")
    assert cfg == {"a": {"b": {"c": 7}}}


def test_missing_key_raises_readable_error():
    cfg = load_config("smoke")
    with pytest.raises(AttributeError, match="нет ключа"):
        _ = cfg.train.nonexistent_param


def test_config_hash_is_stable_and_sensitive():
    a = load_config("smoke")
    b = load_config("smoke")
    c = load_config("smoke", ["train.lr=1e-5"])
    assert config_hash(a) == config_hash(b)
    assert config_hash(a) != config_hash(c)


def test_flatten_produces_dotted_keys():
    flat = flatten({"train": {"lr": 1, "sched": {"kind": "cosine"}}})
    assert flat == {"train.lr": 1, "train.sched.kind": "cosine"}


@pytest.mark.parametrize(
    "name,domain",
    [
        ("coco_000000059319_powerpaint_realisticvision_1_Blended", "coco"),
        ("raise_rbda4cda5t_Q72_removeanything_lama_1_None", "raise"),
        ("D21_L1S2C1_small_1_inpainted-0", "vision"),
        ("000196093", "plain"),
        ("2761119252_small_2_inpainted-0", "numid"),
        ("openimages_bd3895b63372e1f2_removeanything_lama_1_None", "openimages"),
    ],
)
def test_domain_parsing(name, domain):
    assert parse_domain(name) == domain


@pytest.mark.parametrize(
    "name,generator",
    [
        ("coco_000000059319_powerpaint_realisticvision_1_Blended", "powerpaint"),
        ("raise_rbda4cda5t_Q72_removeanything_lama_1_None", "removeanything"),
        ("D21_L1S2C1_small_1_inpainted-0", "inpainted"),
        ("000196093", "none"),
    ],
)
def test_generator_parsing(name, generator):
    assert parse_generator(name) == generator


def test_stem_strips_twelve_hex_prefix():
    assert stem_of("stage1/train/img/8d13de81e6c8_coco_000000059319_x.jpg") == "coco_000000059319_x"
    assert stem_of("stage1/train/mask/000196093.png") == "000196093"


def test_group_id_is_shared_between_manipulations_of_one_source():
    """Разные манипуляции одного кадра обязаны попасть в одну группу,
    иначе они разъедутся по фолдам и валидация будет с утечкой."""
    a = parse_group_id("coco_000000059319_powerpaint_realisticvision_1_Blended", None)
    b = parse_group_id("coco_000000059319_removeanything_lama_1_None", None)
    assert a == b == "000000059319"


def test_group_id_from_original_matches_group_id_from_name():
    """Часть строк даёт оригинал, часть — нет. Идентификатор должен совпасть."""
    from_name = parse_group_id("coco_000000059319_powerpaint_realisticvision_1_Blended", None)
    from_orig = parse_group_id("whatever", "stage1/train/src/41a28965566b_000000059319.jpg")
    assert from_name == from_orig

    raise_name = parse_group_id("raise_rbda4cda5t_Q72_removeanything_lama_1_None", None)
    raise_orig = parse_group_id("x", "stage1/train/src/0f186f51db0f_rbda4cda5t_Q72.jpg")
    assert raise_name == raise_orig

    # openimages: в имени изменённого файла префикс есть, в имени оригинала — нет
    oi_name = parse_group_id("openimages_bd3895b63372e1f2_removeanything_lama_1_None", None)
    oi_orig = parse_group_id("x", "stage1/train/src/c87eb2d17d81_bd3895b63372e1f2.jpg")
    assert oi_name == oi_orig == "bd3895b63372e1f2"


def test_group_id_cuts_at_size_token():
    assert parse_group_id("D21_L1S2C1_small_1_inpainted-0", None) == "D21_L1S2C1"
    assert parse_group_id("2761119252_small_2_inpainted-0", None) == "2761119252"
