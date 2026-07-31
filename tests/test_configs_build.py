"""Каждый конфиг в репозитории должен собираться в рабочую модель.

Без этого теста опечатка в имени энкодера живёт до первого запуска и съедает
время в самый неудачный момент. Именно так `tu-mit_b2` (такой модели в timm нет,
MiT — нативный энкодер smp и пишется без префикса) доехал до репозитория.

Веса всегда None: тест проверяет форму архитектуры, а не качество, и не должен
ходить в сеть.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import pytest
import torch

from experimental_tools_beliy_russak.config import load_config
from experimental_tools_beliy_russak.models import build_model
from experimental_tools_beliy_russak.workspace import configs_root

# conftest.py прибивает воркспейс к корню репозитория ещё до сбора тестов,
# поэтому здесь configs_root() уже указывает куда надо
CONFIGS = configs_root()
CONFIG_FILES = sorted(p for p in CONFIGS.glob("*.yaml") if not p.name.startswith("_"))


def test_configs_are_discovered():
    assert CONFIG_FILES, f"в {CONFIGS} не нашлось ни одного конфига"


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_builds_model_with_expected_output_contract(config_path):
    cfg = load_config(config_path, ["model.encoder_weights=null"])
    model = build_model(cfg.model).eval()

    # у энкодеров с оконным вниманием (Swin) размер входа зашит в маски
    # внимания, и прогнать их можно только на том размере, под который они
    # собраны — поэтому пробный тензор берём из самого конфига
    size = int(dict(cfg.model.get("encoder_kwargs") or {}).get("img_size", 256))

    with torch.no_grad():
        out = model(torch.randn(1, 3, size, size))

    # контракт — «эти ключи обязаны быть», а не «только эти»: на них держатся
    # метрика и сборка сабмита. Всё лишнее допустимо, но обязано быть
    # вспомогательной головой, а не случайно протёкшим тензором
    assert {"logits", "cls_logits"} <= set(out)
    assert all(key.startswith("aux_") for key in set(out) - {"logits", "cls_logits"})
    assert out["logits"].shape == (1, 1, size, size)
    assert out["cls_logits"].shape == (1, 1)
    assert torch.isfinite(out["logits"]).all()
    assert all(torch.isfinite(value).all() for value in out.values())


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_encoder_img_size_matches_the_input_size(config_path):
    """`encoder_kwargs.img_size` обязан совпадать с `data.size`.

    Swin падает прямо на forward, если они разъехались:
    `Input height (768) doesn't match model (224)`. Ловить это тестом дешевле,
    чем на первом батче четырёхчасового прогона.
    """
    cfg = load_config(config_path)
    img_size = dict(cfg.model.get("encoder_kwargs") or {}).get("img_size")
    if img_size is None:
        pytest.skip("энкодеру не нужен фиксированный размер входа")
    assert int(img_size) == int(cfg.data.size), (
        f"encoder_kwargs.img_size={img_size} против data.size={cfg.data.size}"
    )


@pytest.mark.parametrize("config_path", CONFIG_FILES, ids=lambda p: p.stem)
def test_config_has_sane_training_section(config_path):
    cfg = load_config(config_path)
    assert cfg.train.bs >= 1
    assert cfg.train.epochs >= 1
    assert cfg.train.accum_steps >= 1
    assert cfg.data.size % 32 == 0, "сторона входа должна делиться на 32 под шаги энкодера"
    assert cfg.train.amp in {"fp16", "bf16", "off", "none"}
    assert cfg.data.source in {"raw", "cache"}
    assert cfg.data.val_mode in {"resize", "pad"}
