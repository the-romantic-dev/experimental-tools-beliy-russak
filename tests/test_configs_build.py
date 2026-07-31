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

    with torch.no_grad():
        out = model(torch.randn(1, 3, 256, 256))

    assert set(out) == {"logits", "cls_logits"}
    assert out["logits"].shape == (1, 1, 256, 256)
    assert out["cls_logits"].shape == (1, 1)
    assert torch.isfinite(out["logits"]).all()


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
