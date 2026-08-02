"""Геометрические характеристики маски — цели для вспомогательных голов.

Важное отличие от метки генератора (inpainted/powerpaint/...): всё здесь
выводится из САМОЙ GT-маски, а не из имени файла. Шортката «узнать, из какого
датасета картинка» нет по построению.

Обратная сторона: новой информации эти цели тоже не несут. Если модель
предскажет маску идеально, площадь, центр и компоненты выводятся из неё
автоматически. Выигрыш может прийти только от переформовки градиента —
это регуляризатор, а не дополнительный источник разметки. Исключение —
площадь: правило `FPR_neg` в метрике это буквально порог по площади
(`area >= 0.01`), поэтому голова на площадь предсказывает собственную
решающую переменную метрики.

Замеры по 2000 масок обучающей выборки:

    площадь        0.0004 .. 0.765 (диапазон в 2141 раз), медиана 0.159
    касание границы  61% в среднем, но 20% у масок <1% и 85% у масок >20%
    компонент        70.5% масок — ровно одна связная область
    центр масс       std 0.175 по x и 0.198 по y (равномерное дало бы 0.29)

Из них по-настоящему выделяется только площадь; остальное — слабые цели,
и поодиночке проверять их не стоит.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

#: сдвиг под логарифм площади: 1e-4 это ~1/4 самой мелкой маски в данных
AREA_EPS = 1e-4
#: верхняя граница для нормировки числа компонент
MAX_COMPONENTS = 16.0


def normalize_area(area: float | torch.Tensor) -> float | torch.Tensor:
    """Площадь -> [0, 1] в логарифмической шкале.

    Линейная шкала здесь не годится: диапазон площадей 2141x, и MSE по сырому
    значению определялся бы крупными масками, полностью игнорируя мелкие —
    ровно ту популяцию, где модель промахивается в 71% случаев.

    Ноль (чистый кадр) отображается в ноль, площадь во весь кадр — в единицу.
    """
    log_eps = float(np.log(AREA_EPS))
    span = float(np.log(1.0 + AREA_EPS)) - log_eps
    if isinstance(area, torch.Tensor):
        return (torch.log(area.clamp(min=0.0) + AREA_EPS) - log_eps) / span
    return (float(np.log(max(float(area), 0.0) + AREA_EPS)) - log_eps) / span


def denormalize_area(value: float | torch.Tensor) -> float | torch.Tensor:
    """Обратное к `normalize_area` — чтобы предсказание головы читалось как площадь."""
    log_eps = float(np.log(AREA_EPS))
    span = float(np.log(1.0 + AREA_EPS)) - log_eps
    if isinstance(value, torch.Tensor):
        return torch.exp(value * span + log_eps) - AREA_EPS
    return float(np.exp(float(value) * span + log_eps)) - AREA_EPS


def normalize_components(count: float) -> float:
    """Число связных компонент -> [0, 1]. log1p, потому что хвост длинный."""
    return float(np.log1p(max(float(count), 0.0)) / np.log1p(MAX_COMPONENTS))


def mask_geometry(mask: torch.Tensor, valid_h: int, valid_w: int) -> dict[str, torch.Tensor]:
    """Характеристики бинарной маски (1, H, W) в пределах полезной области.

    `valid_h`/`valid_w` вырезают паддинг: в режиме `val_mode: pad` он занимает
    до трети холста, и без вырезания и площадь, и центр масс, и касание границы
    считались бы по чужой области.

    Для пустой маски (чистый кадр) геометрия не определена: возвращаются
    нейтральные значения и `geom_valid = 0`, чтобы лосс их не учитывал.
    """
    region = mask[0, :valid_h, :valid_w]
    binary = region > 0.5
    total = int(binary.sum())

    if total == 0:
        return {
            "area": torch.tensor([0.0], dtype=torch.float32),
            "border": torch.tensor([0.0], dtype=torch.float32),
            "centroid": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "components": torch.tensor([0.0], dtype=torch.float32),
            "geom_valid": torch.tensor([0.0], dtype=torch.float32),
        }

    touches = bool(
        binary[0].any() or binary[-1].any() or binary[:, 0].any() or binary[:, -1].any()
    )

    ys, xs = binary.nonzero(as_tuple=True)
    centre_y = float(ys.float().mean()) / max(valid_h - 1, 1)
    centre_x = float(xs.float().mean()) / max(valid_w - 1, 1)

    plane = binary.to(torch.uint8).cpu().numpy()
    n_components = int(cv2.connectedComponents(plane, connectivity=8)[0]) - 1

    return {
        "area": torch.tensor([total / float(binary.numel())], dtype=torch.float32),
        "border": torch.tensor([float(touches)], dtype=torch.float32),
        "centroid": torch.tensor([centre_y, centre_x], dtype=torch.float32),
        "components": torch.tensor([normalize_components(n_components)], dtype=torch.float32),
        "geom_valid": torch.tensor([1.0], dtype=torch.float32),
    }
