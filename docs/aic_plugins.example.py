"""Пример файла со своими компонентами. Скопируй в КОРЕНЬ своего воркспейса
под именем `aic_plugins.py` — рядом с папкой `configs/`, и всё.

    cp docs/aic_plugins.example.py ~/моя-папка/aic_plugins.py

Библиотека подхватывает этот файл сама, регистрировать его нигде не нужно.
Проверить, что подхватила: `aic registry`.

Здесь по одному примеру на каждый вид расширения. Всё, что не нужно, смело
удаляй — файл целиком необязателен.

Подробности и полный справочник: docs/EXTENDING.md
"""

import albumentations as A
import torch
import torch.nn.functional as F

from experimental_tools_beliy_russak import (
    register_aug,
    register_loss,
    register_optimizer,
    register_scheduler,
)


# --------------------------------------------------------------------------
# лосс: soft IoU как альтернатива Dice
# --------------------------------------------------------------------------
# В конфиге:
#     loss:
#       seg: {bce: 1.0, soft_iou: 1.0}
#       soft_iou: {smooth: 1.0}

@register_loss("soft_iou")
def soft_iou(logits, targets, *, smooth=1.0):
    """ВАЖНО: вернуть значение НА КАЖДЫЙ КАДР — тензор (B,), не скаляр.

    Иначе сломается профиль по площади (`loss.area`): библиотеке нужно значение
    по каждому кадру, чтобы взвесить мелкие маски отдельно.
    """
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1) - intersection
    return 1.0 - (intersection + smooth) / (union + smooth)


# --------------------------------------------------------------------------
# аугментации: пресет под пережатые кадры
# --------------------------------------------------------------------------
# В конфиге:  data: {aug: forensic}

@register_aug("forensic")
def forensic_pixel():
    """Только пиксельная часть. Геометрия (кроп, флипы) и нормализация — за
    библиотекой: они завязаны на `data.size` и на возврат маски в исходное
    разрешение."""
    return [
        A.ImageCompression(compression_type="jpeg", quality_range=(30, 95), p=0.6),
        A.Downscale(scale_range=(0.5, 0.9), p=0.2),
    ]


# --------------------------------------------------------------------------
# оптимизатор
# --------------------------------------------------------------------------
# В конфиге:  train: {optimizer: adam}

@register_optimizer("adam")
def adam(param_groups, cfg_train):
    """`param_groups` приезжают готовыми: библиотека уже разложила параметры на
    энкодер/декодер и decay/no-decay, включая свои lr. Твоё дело — сам оптимизатор."""
    return torch.optim.Adam(param_groups, lr=float(cfg_train.get("lr", 3e-4)))


# --------------------------------------------------------------------------
# планировщик
# --------------------------------------------------------------------------
# В конфиге:
#     train:
#       scheduler: step
#       step: {gamma: 0.3}      # параметры — в секции с именем компоненты

@register_scheduler("step")
def step_decay(optimizer, cfg_train, total_steps, warmup_steps):
    """`total_steps` и `warmup_steps` библиотека считает сама — она знает и длину
    эпохи, и накопление градиента. Планировщик шагает по СТУПЕНЯМ ОПТИМИЗАТОРА,
    а не по эпохам.

    Свои параметры клади в `train.<имя компоненты>`: проверка конфига считает
    такую секцию своей и не ругается. Ключ россыпью в `train` вызовет
    предупреждение — он неотличим от опечатки.
    """
    gamma = float((cfg_train.get("step") or {}).get("gamma", 0.3))
    milestones = [int(total_steps * part) for part in (0.5, 0.8)]
    return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=gamma)


# --------------------------------------------------------------------------
# своя модель целиком
# --------------------------------------------------------------------------
# В конфиге:  model: {backend: my_unet}
#
# Раскомментируй, если нужен свой бэкенд. Единственное жёсткое требование —
# форма выхода: на ней держатся и метрика, и сборка сабмита.
#
# from experimental_tools_beliy_russak import register_backend
#
# @register_backend("my_unet")
# def my_unet(cfg_model):
#     class Net(torch.nn.Module):
#         def forward(self, images):
#             logits = ...              # (B, 1, H, W), тех же H и W, что на входе
#             cls_logits = ...          # (B, 1) — «в кадре есть манипуляция»
#             return {"logits": logits, "cls_logits": cls_logits}
#     return Net()
