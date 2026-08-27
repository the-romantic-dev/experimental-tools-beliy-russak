"""Слагаемые лосса серии L, вынесенные из ноутбука 16.

В ячейке им не место по той же причине, что и датасету: ячейку нельзя покрыть
тестом, а эти четыре функции — ровно тот код, где ошибка не падает, а тихо
портит обучение. Терм, который всегда отдаёт ноль, неотличим от терма, которому
нечего лечить, пока на него не посмотришь отдельно.

Проверки: `tests/test_loss_terms.py`.

Каждый терм включается своим весом и по умолчанию выключен. Ни один не заменяет
`bce + dice + 0.3 cls` — все добавляются, поэтому плечо отличается от якоря
ровно одним числом в конфиге.

Почему именно эти четыре — по замерам разбора ошибок прогона
`s3-T-mitb2-native640_long` (AIC 0.9070, 5025 позитивов валидации):

* `gated_seg_loss` — гейт зануляет 157 верных позитивов, и ни один градиент за
  это не платится: `loss_seg` и `loss_cls` сейчас независимы;
* `component_recall_loss` — у 30.1% кадров GT состоит из нескольких кусков, и
  модель покрывает примерно половину, теряя 19% площади GT;
* `far_false_positive_loss` — 96% ложных пикселей прилипают к границе, но в 232
  провальных кадрах больше половины ложной массы лежит далеко от GT;
* `lovasz_hinge` — на `plain_l9` порог не сломан, сломано ранжирование
  (AUC внутри кадра 0.892 при Dice 0.65).
"""

from __future__ import annotations

# `aic` импортируется раньше torch намеренно: он выставляет KMP_DUPLICATE_LIB_OK,
# без которого две копии OpenMP в conda-среде роняют интерпретатор на
# OMP: Error #15. Тот же порядок, что в `datasets_s3.py`.
import aic  # noqa: F401

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from .datasets_s3 import SegDataset  # noqa: E402


def soft_dice_from_probs(probs, targets, smooth=1.0):
    probs = probs.flatten(1)
    targets = targets.flatten(1)
    inter = (probs * targets).sum(1)
    return (1.0 - (2.0 * inter + smooth) / (probs.sum(1) + targets.sum(1) + smooth)).mean()


def soft_dice(logits, targets, smooth=1.0):
    return soft_dice_from_probs(torch.sigmoid(logits), targets, smooth)


# ── серия L: четыре терма под измеренный профиль ошибок ─────────────────────
#
# Каждый включается своим весом и по умолчанию выключен. Ни один не заменяет
# `bce + dice + 0.3 cls` — все добавляются, поэтому плечо отличается от якоря
# ровно одним числом в конфиге.


def gated_seg_loss(logits, cls_logits, targets):
    """Учить на том, что реально уходит в метрику: маска УМНОЖЕННАЯ на гейт.

    Сейчас `loss_seg` и `loss_cls` независимы, а на инференсе маска зануляется
    по гейту целиком. Модель не получает ни одного градиента за то, что гейт
    убил верную маску, — а это 157 позитивов на валидации.

    Внимание, чего этот терм НЕ делает: он не взвешивает лосс на `cls_prob`.
    Замер говорит, почему так нельзя — на позитивах гейт это фактически детектор
    мелкой правки (корреляция с Dice 0.716, у зарубленных медианная площадь GT
    1.1% против 16%). Вес на нём душил бы градиент ровно там, где модель слепа.
    Множитель в ПРЕДСКАЗАНИИ такой обратной связи не создаёт: чем сильнее гейт
    ошибся, тем больше штраф.
    """
    probs = torch.sigmoid(logits.float()) * torch.sigmoid(cls_logits.float())[..., None, None]
    probs = probs.clamp(1e-6, 1.0 - 1e-6)
    bce = -(targets * probs.log() + (1.0 - targets) * (-probs).log1p()).mean()
    return bce + soft_dice_from_probs(probs, targets)


def component_recall_loss(probs, comp, tau: float = 0.25):
    """Штраф за куски правки, на которые модель не откликнулась ВООБЩЕ.

    Dice взвешен по площади: потерять маленький второй фрагмент почти ничего не
    стоит, поэтому градиента на него нет. Замер по валидации: у 30.1% кадров GT
    состоит из нескольких кусков, среди них компонент в среднем 6.27, а модель
    покрывает 3.09 — половина теряется целиком, унося 19% площади GT.

    Штрафуется только недобор НИЖЕ `tau`: на уже покрытые куски терм не давит,
    иначе он превратился бы в ещё один Dice и начал спорить с основным.

    Считается без единой синхронизации с GPU: корзин фиксированное число, а
    какие из них реальные — видно по счётчику пикселей.
    """
    b = probs.shape[0]
    labels = comp.reshape(b, -1).long()
    flat = probs.reshape(b, -1).float()
    width = SegDataset.MAX_COMPONENTS
    sums = torch.zeros(b, width, device=flat.device, dtype=flat.dtype)
    counts = torch.zeros(b, width, device=flat.device, dtype=flat.dtype)
    sums.scatter_add_(1, labels, flat)
    counts.scatter_add_(1, labels, torch.ones_like(flat))

    recall = sums[:, 1:] / counts[:, 1:].clamp(min=1.0)
    valid = counts[:, 1:] > 0
    # хиндж ЛИНЕЙНЫЙ, а не квадратичный: у квадрата градиент затухает по мере
    # приближения к tau, то есть слабее всего толкает там, где кусок ещё почти
    # не найден. Здесь нужно ровно обратное — постоянный толчок до порога.
    penalty = torch.relu(tau - recall) * valid
    per_frame = penalty.sum(1) / valid.sum(1).clamp(min=1)
    has_gt = valid.any(1)
    return (per_frame * has_gt).sum() / has_gt.sum().clamp(min=1)


def far_false_positive_loss(probs, dist, targets, margin: float = 0.01):
    """Уверенность ВДАЛИ от GT дороже, чем размазанная граница.

    Замер, ради которого терм и стоит того: 96% ложных пикселей прилипают к
    границе GT и этим членом почти не трогаются, зато в 232 провальных кадрах
    больше половины ложной массы лежит дальше 5% диагонали.

    Чего терм не сделает: Dice не отличает далёкий FP от близкого, они стоят
    одинаково. То есть это правка индуктивного смещения, а не цели. И на
    кадрах с пустым GT расстояние не определено — там он молчит, а не выдаёт
    бесконечность.
    """
    has_gt = targets.flatten(1).amax(1) > 0.5
    flat = probs.float().flatten(1)
    beyond = (dist - margin).clamp(min=0.0).flatten(1)
    # Нормировка на ПРЕДСКАЗАННУЮ МАССУ, а не на число пикселей кадра. Иначе
    # величина терма зависела бы от площади маски, и один вес не годился бы
    # сразу для правки в 1% кадра и в 50%. В такой форме это просто «средняя
    # дальность предсказанной массы от GT», число в [0, ~0.7], и вес читается.
    penalty = (flat * beyond).sum(1) / flat.sum(1).clamp(min=1e-6)
    return (penalty * has_gt).sum() / has_gt.sum().clamp(min=1)


def lovasz_hinge(logits, targets):
    """Суррогат IoU по ПОРЯДКУ пикселей, а не по их калибровке.

    Взят потому, что диагностика `plain_l9` говорит прямо: порог не сломан,
    сломано ранжирование — AUC внутри кадра 0.892 при Dice 0.65. В свипе из
    семнадцати рецептов этого лосса не было ни в одном.

    Батчевая версия: сортировка сразу по всем кадрам, без цикла и без
    синхронизаций. Кадры целиком фоновые обрабатываются сами собой — у них
    градиент Ловаша вырождается в штраф за максимальный логит.
    """
    b = logits.shape[0]
    y = targets.reshape(b, -1).float()
    signs = 2.0 * y - 1.0
    errors = 1.0 - logits.reshape(b, -1).float() * signs
    errors_sorted, perm = torch.sort(errors, dim=1, descending=True)
    gt_sorted = torch.gather(y, 1, perm)

    total = gt_sorted.sum(1, keepdim=True)
    intersection = total - gt_sorted.cumsum(1)
    union = total + (1.0 - gt_sorted).cumsum(1)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    jaccard = torch.cat([jaccard[:, :1], jaccard[:, 1:] - jaccard[:, :-1]], dim=1)
    return (F.relu(errors_sorted) * jaccard).sum(1).mean()


def compute_loss(out, batch, cfg, progress: float = 1.0):
    """`(total, слагаемые)`. Слагаемые возвращаются, чтобы их можно было СМОТРЕТЬ.

    Без этого эксперимент с лоссами читается только по итоговому AIC, и терм с
    неудачно подобранным весом неотличим от терма, который не работает: первый
    даёт ноль, потому что его задавили, второй — потому что лечить нечего.

    `progress` — доля пройденного обучения, от неё идёт разогрев гейта.
    """
    mask = batch["mask"]
    parts = {}

    bce = F.binary_cross_entropy_with_logits(out["logits"], mask)
    dice = soft_dice(out["logits"], mask)
    cls = F.binary_cross_entropy_with_logits(out["cls_logits"], batch["label"])
    total = bce + dice + 0.3 * cls
    parts["bce"], parts["dice"], parts["cls"] = bce.detach(), dice.detach(), cls.detach()

    aux_weight = float(cfg.get("aux_weight", 0.0))
    if aux_weight > 0 and "aux_logits" in out:
        aux = (F.binary_cross_entropy_with_logits(out["aux_logits"], mask)
               + soft_dice(out["aux_logits"], mask))
        total = total + aux_weight * aux
        parts["aux"] = aux.detach()

    deep_weight = float(cfg.get("deep_weight", 0.0))
    if deep_weight > 0 and "states" in out:
        states = out["states"]
        last = len(states) - 1
        deep = sum(((t + 1) / (last + 1)) ** 2
                   * (F.binary_cross_entropy_with_logits(state, mask) + soft_dice(state, mask))
                   for t, state in enumerate(states))
        total = total + deep_weight * deep
        parts["deep"] = deep.detach()

    evidence_weight = float(cfg.get("evidence_weight", 0.0))
    if evidence_weight > 0 and "evidence_logits" in out:
        evidence = F.binary_cross_entropy_with_logits(out["evidence_logits"], mask)
        total = total + evidence_weight * evidence
        parts["evidence"] = evidence.detach()

    # --- серия L -----------------------------------------------------------
    w_gate = float(cfg.get("w_gate", 0.0))
    if w_gate > 0:
        # разогрев: несозревший гейт в первые шаги душил бы сегментацию
        ramp = min(1.0, progress / max(float(cfg.get("gate_warmup", 0.1)), 1e-9))
        gated = gated_seg_loss(out["logits"], out["cls_logits"], mask)
        total = total + w_gate * ramp * gated
        parts["gate"] = gated.detach()

    w_comp = float(cfg.get("w_comp", 0.0))
    if w_comp > 0 and "comp" in batch:
        comp_loss = component_recall_loss(
            torch.sigmoid(out["logits"]), batch["comp"], float(cfg.get("comp_tau", 0.25)))
        total = total + w_comp * comp_loss
        parts["comp"] = comp_loss.detach()

    w_far = float(cfg.get("w_far", 0.0))
    if w_far > 0 and "dist" in batch:
        far = far_false_positive_loss(
            torch.sigmoid(out["logits"]), batch["dist"], mask,
            float(cfg.get("far_margin", 0.01)))
        total = total + w_far * far
        parts["far"] = far.detach()

    w_lovasz = float(cfg.get("w_lovasz", 0.0))
    if w_lovasz > 0:
        lov = lovasz_hinge(out["logits"], mask)
        total = total + w_lovasz * lov
        parts["lovasz"] = lov.detach()

    return total, parts
