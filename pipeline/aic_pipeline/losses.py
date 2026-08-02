"""Лоссы. Собираются из конфига как взвешенная сумма компонент.

    loss:
      seg: {bce: 1.0, dice: 1.0, focal: 0.0, tversky: 0.0}
      cls_weight: 0.3
      pos_weight: null
      tversky: {alpha: 0.3, beta: 0.7}

Про негативы: на кадре без манипуляций числитель Dice всегда ноль, поэтому
сглаживание в Dice берётся такое, чтобы пара «пусто-пусто» давала лосс 0,
иначе чистые кадры превращаются в постоянный штраф и модель учится рисовать
хоть что-нибудь.

Профили по площади (`loss.area`) — отдельный режим:

    loss:
      seg: {bce: 1.0, dice: 1.0}              # крупные маски и негативы
      area:
        threshold: 0.06                       # граница «мелкой» маски
        small_seg: {bce: 0.5, focal: 1.0, tversky: 1.5}
        small_weight: 2.0                     # множитель вклада мелких кадров

Смысл. На кадре с маской в 1% площади BCE почти целиком состоит из фона:
предсказать везде ноль — уже отличная BCE, и градиент по редкому классу тонет.
Dice на такой маске, наоборот, скачет от единичных пикселей. Поэтому мелким
кадрам даётся другой состав: focal (гасит лёгкий фон) плюс Tversky с beta>alpha
(штраф за пропуск дороже штрафа за лишнее). Негативы (площадь ровно 0) всегда
идут по профилю по умолчанию — у них «мелкой маски» нет вовсе.

Веса нормируются на среднее по батчу, поэтому `small_weight` меняет баланс
между корзинами, но не масштаб лосса — расписание lr остаётся сравнимым
с прогонами без профилей.

Компоненты берутся из реестра, так что своя добавляется из воркспейса без
правки библиотеки — см. `registry.py` и `@register_loss`.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import normalize_area
from .registry import LOSSES, register_loss


# Всё, что суммирует ПО ВСЕМУ КАДРУ, считается в fp32 явным `.float()`, а не в
# dtype логитов. Под autocast(fp16) логиты приходят половинками, `sigmoid` их
# такими и оставляет, а у fp16 потолок 65504 — при 768x768 = 589824 пикселях
# сумма переполняется в inf, стоит средней вероятности (или площади маски)
# превысить 11.1% кадра. Дальше знаменатель становится inf, дробь — нулём, и
# лосс молча залипает ровно на 1.0 с НУЛЕВЫМ градиентом; а когда модель
# научится и числитель тоже перевалит 65504, получается inf/inf = nan.
# В индексе 61% позитивов имеют маску крупнее 11%, то есть без этого каста
# dice на 768 не учил больше половины позитивных кадров, а на хорошей модели
# ронял прогон в nan. BCE каста не требует: autocast сам считает
# binary_cross_entropy_with_logits в fp32.
def soft_dice_loss(
    logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0, reduce: bool = True
) -> torch.Tensor:
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)
    intersection = (probs * targets).sum(dim=1)
    denominator = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    loss = 1.0 - dice
    return loss.mean() if reduce else loss


def tversky_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0, reduce: bool = True,
) -> torch.Tensor:
    """alpha штрафует FP, beta — FN. alpha<beta тянет к полноте, alpha>beta — к точности.

    Для AIC полезно уметь двигать этот баланс: FP на негативах бьют по метрике
    сильнее, чем недобор площади на позитивах.
    """
    probs = torch.sigmoid(logits.float()).flatten(1)
    targets = targets.float().flatten(1)
    true_pos = (probs * targets).sum(dim=1)
    false_pos = (probs * (1 - targets)).sum(dim=1)
    false_neg = ((1 - probs) * targets).sum(dim=1)
    index = (true_pos + smooth) / (true_pos + alpha * false_pos + beta * false_neg + smooth)
    loss = 1.0 - index
    return loss.mean() if reduce else loss


def focal_loss(
    logits: torch.Tensor, targets: torch.Tensor,
    alpha: float = 0.25, gamma: float = 2.0, reduce: bool = True,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1 - probs) * (1 - targets)
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * (1 - p_t).pow(gamma) * bce
    return loss.mean() if reduce else loss.flatten(1).mean(dim=1)


# Встроенные компоненты регистрируются тем же способом, что и чужие: один путь
# кода, и пример прямо перед глазами у того, кто пишет свою.

@register_loss("bce")
def _bce_component(logits, targets, *, pos_weight=None):
    per_pixel = F.binary_cross_entropy_with_logits(
        logits, targets, pos_weight=pos_weight, reduction="none"
    )
    return per_pixel.flatten(1).mean(dim=1)


@register_loss("dice")
def _dice_component(logits, targets, *, smooth=1.0):
    return soft_dice_loss(logits, targets, smooth, reduce=False)


@register_loss("focal")
def _focal_component(logits, targets, *, alpha=0.25, gamma=2.0):
    return focal_loss(logits, targets, alpha, gamma, reduce=False)


@register_loss("tversky")
def _tversky_component(logits, targets, *, alpha=0.3, beta=0.7, smooth=1.0):
    return tversky_loss(logits, targets, alpha, beta, smooth, reduce=False)


def _params_for(cfg_loss: dict, name: str) -> dict:
    """Именованные параметры компоненты: секция `loss.<имя>` конфига.

    Плюс две исторические формы записи, которые остаются рабочими: `bce` берёт
    `loss.pos_weight`, а `dice` и `tversky` — `loss.dice_smooth`. Ломать ими
    существующие конфиги и чекпоинты незачем.
    """
    section = cfg_loss.get(name)
    params = dict(section) if isinstance(section, dict) else {}
    if name in {"dice", "tversky"}:
        params.setdefault("smooth", float(cfg_loss.get("dice_smooth", 1.0)))
    return params


class CombinedLoss(nn.Module):
    def __init__(self, cfg_loss: dict, aux_weights: dict[str, float] | None = None) -> None:
        super().__init__()
        # веса вспомогательных голов приходят из model.aux_heads, а не из loss:
        # там же перечислено, какие головы вообще собираются, и держать список
        # в двух местах — верный способ их рассинхронизировать
        self.aux_weights = dict(aux_weights or {})
        seg = dict(cfg_loss.get("seg", {"bce": 1.0, "dice": 1.0}))
        self.weights = {k: float(v) for k, v in seg.items() if float(v) != 0.0}
        if not self.weights:
            raise ValueError("в loss.seg не задано ни одной компоненты с ненулевым весом")

        area = dict(cfg_loss.get("area", {}) or {})
        small_seg = dict(area.get("small_seg", {}) or {})
        self.small_weights = {k: float(v) for k, v in small_seg.items() if float(v) != 0.0}
        self.area_threshold = float(area.get("threshold", 0.06))
        self.small_weight = float(area.get("small_weight", 1.0))
        self.use_area = bool(self.small_weights) or self.small_weight != 1.0
        if self.use_area and not self.small_weights:
            self.small_weights = dict(self.weights)

        # разрешение имён здесь, а не в forward: опечатка в конфиге должна
        # вылезти при сборке лосса, а не на первом шаге обучения
        self._fns = {
            name: LOSSES.get(name)
            for name in sorted(set(self.weights) | set(self.small_weights))
        }
        self._params = {name: _params_for(cfg_loss, name) for name in self._fns}

        self.cls_weight = float(cfg_loss.get("cls_weight", 0.0))

        # pos_weight остаётся буфером, а не обычным параметром в _params:
        # только так он переезжает на device вместе с модулем. Пишется он и
        # по-старому (`loss.pos_weight`), и по общему правилу (`loss.bce.pos_weight`);
        # из _params вынимается в любом случае, чтобы не уехать в компоненту дважды
        nested = self._params.get("bce", {}).pop("pos_weight", None)
        pos_weight = cfg_loss.get("pos_weight") or nested
        self.register_buffer(
            "pos_weight",
            torch.tensor([float(pos_weight)]) if pos_weight else None,
            persistent=False,
        )

    def _component(self, name: str, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Значение компоненты для каждого кадра батча, форма (B,)."""
        params = self._params[name]
        if name == "bce":
            params = {**params, "pos_weight": self.pos_weight}
        return self._fns[name](logits, targets, **params)

    def forward(self, outputs: dict, batch: dict) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["logits"]
        targets = batch["mask"].to(logits.device, dtype=logits.dtype)

        # _fns уже собран по объединению обычного и «мелкого» профилей
        components = {name: self._component(name, logits, targets) for name in self._fns}

        if not self.use_area:
            total = sum(self.weights[name] * components[name].mean() for name in self.weights)
            stats = {name: float(value.mean().detach()) for name, value in components.items()}
        else:
            area = batch.get("area")
            if area is None:  # старые чекпоинты / внешние вызовы без поля area
                area = (targets.flatten(1).mean(dim=1)).unsqueeze(1)
            area = area.to(logits.device, dtype=logits.dtype).reshape(-1)
            # негативы (площадь ровно 0) остаются на профиле по умолчанию
            is_small = (area > 0) & (area < self.area_threshold)

            per_sample = torch.zeros_like(area)
            for name, value in components.items():
                weight = torch.where(
                    is_small,
                    torch.full_like(area, self.small_weights.get(name, 0.0)),
                    torch.full_like(area, self.weights.get(name, 0.0)),
                )
                per_sample = per_sample + weight * value

            sample_weight = torch.where(
                is_small, torch.full_like(area, self.small_weight), torch.ones_like(area)
            )
            total = (per_sample * sample_weight).sum() / sample_weight.sum().clamp(min=1e-6)
            stats = {name: float(value.mean().detach()) for name, value in components.items()}
            stats["small_frac"] = float(is_small.float().mean().detach())

        if self.cls_weight > 0 and "cls_logits" in outputs:
            labels = batch["label"].to(logits.device, dtype=logits.dtype)
            cls = F.binary_cross_entropy_with_logits(outputs["cls_logits"], labels)
            stats["cls"] = float(cls.detach())
            total = total + self.cls_weight * cls

        for name, weight in self.aux_weights.items():
            value = self._aux_component(name, outputs, batch, logits)
            if value is None:
                continue
            stats[f"aux_{name}"] = float(value.detach())
            total = total + weight * value

        return total, stats

    def _aux_component(self, name, outputs, batch, logits):
        """Лосс одной вспомогательной головы, или None если её нет в выходе."""
        from .models.aux_heads import AUX_HEADS

        prediction = outputs.get(f"aux_{name}")
        if prediction is None:
            return None

        spec = AUX_HEADS[name]
        if name == "area":
            # цель считается из сырой площади нормировкой, а не хранится
            # отдельным полем: иначе в батче было бы два похожих числа
            # в разных шкалах, и перепутать их — вопрос времени
            raw = batch.get("area")
            if raw is None:
                raw = batch["mask"].flatten(1).mean(dim=1, keepdim=True)
            target = normalize_area(raw.to(logits.device, dtype=logits.dtype))
        else:
            target = batch.get(f"aux_{name}")
            if target is None:
                return None
            target = target.to(logits.device, dtype=logits.dtype)

        prediction = prediction.reshape(target.shape)
        if spec.loss == "bce":
            per_sample = F.binary_cross_entropy_with_logits(
                prediction, target, reduction="none"
            ).mean(dim=1)
        else:
            per_sample = F.mse_loss(prediction, target, reduction="none").mean(dim=1)

        if not spec.positives_only:
            return per_sample.mean()

        # у чистого кадра геометрии нет: ни центра, ни компонент. Учить на них
        # нейтральную заглушку значило бы заставлять модель предсказывать
        # «середина кадра» там, где маски вообще не существует
        valid = batch.get("geom_valid")
        if valid is None:
            valid = batch["label"]
        valid = valid.to(logits.device, dtype=logits.dtype).reshape(-1)
        return (per_sample * valid).sum() / valid.sum().clamp(min=1.0)


def build_loss(cfg_loss: dict, aux_weights: dict[str, float] | None = None) -> CombinedLoss:
    return CombinedLoss(cfg_loss, aux_weights)
