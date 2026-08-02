"""Точки расширения: свои компоненты подключаются без правки библиотеки.

Ради этого реестр и заводился, поэтому проверяется весь путь целиком — от файла
`aic_plugins.py` в воркспейсе до собранного лосса и оптимизатора.

Изоляция между тестами обеспечена автофикстурой `_isolate_registries` в
conftest.py: всё зарегистрированное здесь исчезает после каждого теста.
"""

from __future__ import annotations

import aic_pipeline  # noqa: F401

import pytest
import torch

from aic_pipeline.config import load_config
from aic_pipeline.engine import build_optimizer, build_scheduler
from aic_pipeline.losses import build_loss
from aic_pipeline.registry import (
    LOSSES,
    OPTIMIZERS,
    PLUGIN_FILE,
    Registry,
    describe,
    register_loss,
    register_optimizer,
)
from aic_pipeline.transforms import build_train_transform
from aic_pipeline.workspace import use_workspace


@pytest.fixture
def workspace_with_plugin(tmp_path):
    """Воркспейс, у которого есть свой aic_plugins.py."""
    def make(body: str):
        (tmp_path / "configs").mkdir(exist_ok=True)
        (tmp_path / PLUGIN_FILE).write_text(body, encoding="utf-8")
        return tmp_path
    return make


# --- базовая механика -------------------------------------------------------


def test_register_works_as_decorator_and_as_plain_call():
    shelf = Registry("проба", "проба", "fn()", builtins=".metrics")

    @shelf.register("через-декоратор")
    def first():
        return 1

    shelf.register("через-вызов", lambda: 2)

    assert shelf.get("через-декоратор")() == 1
    assert shelf.get("через-вызов")() == 2
    assert first() == 1, "декоратор обязан вернуть саму функцию, а не обёртку"


def test_duplicate_name_is_refused_unless_override_asked():
    """Чужой плагин не должен молча подменить компоненту — это ищется часами."""
    @register_loss("столкновение")
    def first(logits, targets):
        return logits.flatten(1).mean(dim=1)

    with pytest.raises(ValueError, match="уже зарегистрирована"):
        @register_loss("столкновение")
        def second(logits, targets):
            return logits.flatten(1).mean(dim=1)

    @register_loss("столкновение", override=True)
    def third(logits, targets):
        return logits.flatten(1).mean(dim=1) * 0

    assert LOSSES.get("столкновение") is third


def test_unknown_name_explains_what_is_available_and_how_to_add():
    with pytest.raises(ValueError) as error:
        OPTIMIZERS.get("adamwww")

    message = str(error.value)
    assert "adamw" in message and "sgd" in message      # что есть
    assert "похоже на 'adamw'?" in message              # подсказка по опечатке
    assert "@register_optimizer" in message             # как добавить своё
    assert "param_groups" in message                    # какая нужна сигнатура


def test_builtins_are_visible_without_importing_their_modules():
    """`aic registry` не должен зависеть от того, что успели импортировать."""
    table = describe()

    assert set(table) == {"loss", "aug", "val_mode", "optimizer", "scheduler", "backend"}
    assert table["loss"]["встроенные"] == ["bce", "dice", "focal", "tversky"]
    assert table["val_mode"]["встроенные"] == ["pad", "resize"]


# --- свои компоненты в деле -------------------------------------------------


def test_own_loss_component_reaches_the_combined_loss():
    @register_loss("постоянный")
    def constant(logits, targets, *, value=0.25):
        return torch.full((logits.shape[0],), float(value))

    criterion = build_loss({"seg": {"постоянный": 1.0}, "постоянный": {"value": 0.5}})
    logits = torch.randn(3, 1, 8, 8)
    total, stats = criterion({"logits": logits}, {"mask": torch.zeros(3, 1, 8, 8)})

    assert float(total) == pytest.approx(0.5)
    assert stats["постоянный"] == pytest.approx(0.5)


def test_own_loss_component_participates_in_the_area_profile():
    """Контракт «вернуть (B,)» существует ради этого: кадры взвешиваются по площади."""
    @register_loss("по-кадру")
    def per_frame(logits, targets):
        return torch.arange(logits.shape[0], dtype=torch.float32)

    criterion = build_loss({
        "seg": {"по-кадру": 1.0},
        "area": {"threshold": 0.05, "small_seg": {"по-кадру": 1.0}, "small_weight": 3.0},
    })
    logits = torch.randn(2, 1, 8, 8)
    batch = {
        "mask": torch.zeros(2, 1, 8, 8),
        "area": torch.tensor([[0.01], [0.5]]),   # первый кадр мелкий, второй нет
    }
    total, stats = criterion({"logits": logits}, batch)

    # значения компоненты 0 и 1, веса кадров 3 и 1 -> (3*0 + 1*1) / (3 + 1)
    assert float(total) == pytest.approx(0.25)
    assert stats["small_frac"] == pytest.approx(0.5)


def test_own_optimizer_gets_ready_made_parameter_groups():
    seen: dict = {}

    @register_optimizer("шпион")
    def spy(param_groups, cfg_train):
        seen["groups"] = param_groups
        return torch.optim.SGD(param_groups, lr=float(cfg_train["lr"]))

    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(model, {"optimizer": "шпион", "lr": 0.1, "weight_decay": 0.01})

    assert isinstance(optimizer, torch.optim.SGD)
    # группировка на decay/no-decay остаётся за библиотекой
    assert {g["weight_decay"] for g in seen["groups"]} == {0.01, 0.0}


def test_own_scheduler_receives_computed_step_counts():
    from aic_pipeline.registry import register_scheduler

    seen: dict = {}

    @register_scheduler("шпион")
    def spy(optimizer, cfg_train, total_steps, warmup_steps):
        seen.update(total=total_steps, warmup=warmup_steps)
        return None

    model = torch.nn.Linear(4, 4)
    optimizer = build_optimizer(model, {"lr": 0.1})
    build_scheduler(
        optimizer,
        {"scheduler": "шпион", "epochs": 4, "accum_steps": 2, "warmup_frac": 0.25},
        steps_per_epoch=10,
    )

    assert seen == {"total": 20, "warmup": 5}   # (10 // 2) * 4 шагов, четверть на прогрев


def test_own_aug_preset_is_inserted_between_geometry_and_normalisation():
    import albumentations as A

    from aic_pipeline.registry import register_aug

    @register_aug("мой")
    def mine():
        return [A.ToGray(p=1.0)]

    names = [type(t).__name__ for t in build_train_transform(size=64, preset="мой", seed=0)]

    assert "ToGray" in names
    assert names.index("RandomResizedCrop") < names.index("ToGray") < names.index("Normalize")


# --- загрузка из воркспейса -------------------------------------------------


PLUGIN_BODY = """
import torch
from aic_pipeline import register_loss

@register_loss("из-файла")
def from_file(logits, targets, *, value=0.75):
    return torch.full((logits.shape[0],), float(value))
"""


def test_plugin_file_is_picked_up_from_the_workspace(workspace_with_plugin):
    root = workspace_with_plugin(PLUGIN_BODY)

    with use_workspace(root):
        criterion = build_loss({"seg": {"из-файла": 1.0}})
        total, _ = criterion(
            {"logits": torch.randn(2, 1, 4, 4)}, {"mask": torch.zeros(2, 1, 4, 4)}
        )

    assert float(total) == pytest.approx(0.75)


def test_plugin_is_attributed_to_its_file_in_the_listing(workspace_with_plugin):
    root = workspace_with_plugin(PLUGIN_BODY)

    with use_workspace(root):
        groups = describe()["loss"]

    assert "из-файла" in groups["aic_plugins"]
    assert "из-файла" not in groups["встроенные"]


def test_broken_plugin_names_the_file_it_came_from(workspace_with_plugin):
    root = workspace_with_plugin("raise RuntimeError('я сломан')")

    with use_workspace(root), pytest.raises(ImportError) as error:
        LOSSES.get("bce")

    assert PLUGIN_FILE in str(error.value)
    assert "я сломан" in str(error.value)


def test_workspace_without_plugin_file_works_as_before(tmp_path):
    (tmp_path / "configs").mkdir()

    with use_workspace(tmp_path):
        assert "bce" in LOSSES
        assert "из-файла" not in LOSSES


def test_plugins_key_in_config_loads_a_module(tmp_path, monkeypatch):
    """Общие компоненты команды можно держать отдельным пакетом, а не файлом."""
    (tmp_path / "configs").mkdir()
    (tmp_path / "общий_плагин.py").write_text(
        "import torch\n"
        "from aic_pipeline import register_loss\n"
        "@register_loss('из-модуля')\n"
        "def f(logits, targets):\n"
        "    return torch.zeros(logits.shape[0])\n",
        encoding="utf-8",
    )
    (tmp_path / "configs" / "мой.yaml").write_text(
        "plugins: [общий_плагин]\nloss: {seg: {'из-модуля': 1.0}}\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    with use_workspace(tmp_path):
        cfg = load_config("мой")
        assert "из-модуля" in LOSSES
        assert build_loss(cfg["loss"]) is not None
