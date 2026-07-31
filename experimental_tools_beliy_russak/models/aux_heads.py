"""Вспомогательные головы поверх признаков энкодера.

Все головы сидят на самой глубокой карте энкодера (глобальный пулинг ->
dropout -> линейный слой): представление шевелится там же, откуда его берёт
декодер, а не на выходе сегментации.

Реестр `AUX_HEADS` задаёт для каждой головы три вещи: размерность выхода, вид
лосса и нужно ли считать её только на позитивах. Геометрия у чистого кадра не
определена (нет ни центра, ни компонент), поэтому такие головы учатся только
там, где маска непустая. Площадь — исключение: у чистого кадра она честно равна
нулю, и это ровно тот сигнал, который метрика проверяет правилом FPR.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class HeadSpec(NamedTuple):
    out_dim: int
    loss: str            # mse | bce
    positives_only: bool
    target_key: str      # какой ключ батча служит целью
    description: str


AUX_HEADS: dict[str, HeadSpec] = {
    "area": HeadSpec(
        1, "mse", False, "aux_area",
        "нормированный логарифм доли кадра; у чистых кадров ровно 0",
    ),
    "border": HeadSpec(
        1, "bce", True, "aux_border",
        "касается ли маска границы полезной области",
    ),
    "centroid": HeadSpec(
        2, "mse", True, "aux_centroid",
        "центр масс маски в долях кадра (y, x)",
    ),
    "components": HeadSpec(
        1, "mse", True, "aux_components",
        "нормированный log1p числа связных компонент",
    ),
}


def parse_aux_spec(spec) -> dict[str, float]:
    """`{area: 0.1, border: 0.0}` -> только головы с ненулевым весом.

    Принимает и список имён — тогда всем даётся вес по умолчанию 0.1.
    """
    if not spec:
        return {}
    if isinstance(spec, (list, tuple, set)):
        spec = {str(name): 0.1 for name in spec}

    weights: dict[str, float] = {}
    for name, weight in dict(spec).items():
        if name not in AUX_HEADS:
            raise ValueError(
                f"неизвестная aux-голова: {name}; есть {sorted(AUX_HEADS)}"
            )
        if float(weight) != 0.0:
            weights[str(name)] = float(weight)
    return weights


class AuxHead(nn.Module):
    """Глобальный пулинг признаков энкодера -> предсказание одной характеристики."""

    def __init__(self, in_channels: int, out_dim: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_channels, out_dim)
        # старт из нуля: на первом шаге голова не тянет представление никуда,
        # градиент по ней при этом ненулевой — тот же приём, что у проекции
        # в fuse: gate и у восстановленного стема
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(features).flatten(1)
        return self.fc(self.drop(pooled))


def build_aux_heads(in_channels: int, weights: dict[str, float], dropout: float = 0.2):
    """ModuleDict из голов, перечисленных в конфиге."""
    return nn.ModuleDict({
        name: AuxHead(in_channels, AUX_HEADS[name].out_dim, dropout)
        for name in weights
    })
