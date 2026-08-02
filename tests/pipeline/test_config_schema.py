"""Опечатка в имени ключа должна падать, а не тихо ничего не делать.

`-s train.epohs=20` раньше создавал новый ключ, а `epochs` оставался прежним:
прогон шёл на двенадцати эпохах вместо двадцати, и заметить это было нечем.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import pytest
import yaml

from experimental_tools_beliy_russak.config import load_config
from experimental_tools_beliy_russak.schema import (
    SCHEMA,
    UnknownConfigKey,
    check_config,
    find_unknown_keys,
)
from experimental_tools_beliy_russak.workspace import configs_root, use_workspace


@pytest.fixture
def workspace_with_config(tmp_path):
    """Воркспейс с одним конфигом, содержимое задаёт тест."""
    def make(body: dict, name: str = "проба"):
        (tmp_path / "configs").mkdir(exist_ok=True)
        (tmp_path / "configs" / f"{name}.yaml").write_text(
            yaml.safe_dump(body, allow_unicode=True), encoding="utf-8"
        )
        return tmp_path, name
    return make


# --- опечатки ---------------------------------------------------------------


@pytest.mark.parametrize(
    "typo, expected",
    [
        ("train.epohs", "train.epochs"),
        ("train.weigth_decay", "train.weight_decay"),
        ("model.enocder", "model.encoder"),
        ("data.siz", "data.size"),
    ],
)
def test_near_miss_is_reported_with_the_intended_key(typo, expected):
    section, key = typo.split(".")
    unknown = find_unknown_keys({section: {key: 1}})

    assert [(item.path, item.suggestion) for item in unknown] == [(typo, expected)]


def test_typo_in_override_stops_the_run(workspace_with_config):
    root, name = workspace_with_config({"train": {"epochs": 12}})

    with use_workspace(root), pytest.raises(ValueError) as error:
        load_config(name, ["train.epohs=20"])

    message = str(error.value)
    assert "train.epohs" in message and "train.epochs" in message


def test_misspelled_section_is_caught_too():
    unknown = find_unknown_keys({"trian": {"epochs": 12}})

    assert [(item.path, item.suggestion) for item in unknown] == [("trian", "train")]


def test_misspelled_section_is_not_descended_into():
    """Иначе на одну описку сыпалась бы гора сообщений про её содержимое."""
    unknown = find_unknown_keys({"trian": {"epochs": 12, "bs": 8, "lr": 1e-4}})

    assert len(unknown) == 1


# --- незнакомое, но не похожее ----------------------------------------------


def test_unrelated_unknown_key_warns_but_does_not_stop():
    with pytest.warns(UnknownConfigKey, match="train.вообще_мимо"):
        check_config({"train": {"вообще_мимо": 1}})


def test_warning_says_where_component_parameters_belong():
    with pytest.warns(UnknownConfigKey, match=r"train\.step\.\*"):
        check_config({"train": {"step_gamma": 0.3}})


# --- открытые ветки ---------------------------------------------------------


def test_selected_component_gets_its_own_section():
    """train.scheduler=step объявляет train.step.* параметрами планировщика."""
    assert find_unknown_keys({"train": {"scheduler": "step", "step": {"gamma": 0.3}}}) == []


def test_the_same_holds_for_optimizer_aug_and_backend():
    cfg = {
        "train": {"optimizer": "lion", "lion": {"betas": [0.9, 0.99]}},
        "data": {"aug": "forensic", "forensic": {"quality": 40}},
        "model": {"backend": "my_unet", "my_unet": {"width": 32}},
    }

    assert find_unknown_keys(cfg) == []


def test_loss_component_parameters_are_open():
    cfg = {"loss": {"seg": {"bce": 1.0, "soft_iou": 1.0}, "soft_iou": {"smooth": 2.0}}}

    assert find_unknown_keys(cfg) == []


def test_parameters_of_a_component_that_is_not_used_are_still_flagged():
    """`loss.soft_iou` без `soft_iou` в `loss.seg` — скорее всего забытая правка."""
    unknown = find_unknown_keys({"loss": {"seg": {"bce": 1.0}, "soft_iou": {"smooth": 2.0}}})

    assert [item.path for item in unknown] == ["loss.soft_iou"]


def test_check_can_be_switched_off(workspace_with_config):
    root, name = workspace_with_config({"train": {"epohs": 20}})

    with use_workspace(root):
        cfg = load_config(name, check=False)

    assert cfg.train["epohs"] == 20


# --- схема не должна отставать от базового конфига --------------------------


def test_every_key_of_the_base_config_is_in_the_schema():
    """Схема ведётся руками, и это единственное, что мешает ей отстать.

    Добавили ключ в `_base.yaml`, забыли в `SCHEMA` — падает здесь, а не у
    тиммейта в виде «библиотека не знает такой ключ».
    """
    base = yaml.safe_load((configs_root() / "_base.yaml").read_text(encoding="utf-8"))
    base.pop("_base_", None)

    assert find_unknown_keys(base) == []


def test_all_configs_in_the_workspace_pass_the_check():
    for path in sorted(configs_root().glob("*.yaml")):
        body = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        body.pop("_base_", None)
        unknown = [item.path for item in find_unknown_keys(body)]
        assert unknown == [], f"{path.name}: {unknown}"


def test_schema_covers_every_top_level_section_the_code_reads():
    assert {"data", "model", "loss", "train", "calib"} <= set(SCHEMA)
