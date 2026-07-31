"""Реестр моделей: одна строка в конфиге -> собранная сеть.

Любая модель отдаёт словарь одинаковой формы:
    {"logits": (B, 1, H, W), "cls_logits": (B, 1)}

`cls_logits` — вероятность «в кадре есть манипуляция» до сигмоиды. На инференсе
по ней целиком обнуляются маски на чистых кадрах, и это главный рычаг против
FPR_neg в метрике AIC.

Конфиг:
    model:
      backend: smp            # smp | hf
      arch: unet              # smp: unet|unetplusplus|fpn|deeplabv3plus|upernet|segformer|...
      encoder: tu-convnext_tiny
      encoder_weights: imagenet
      cls_head: aux           # aux | from_logits | none
      stream: none            # none | srm | bayar | srm_bayar — низкоуровневый поток
      fuse: input             # input | gate — как этот поток подмешивается
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKENDS, register_backend
from ..streams import GatedDualEncoder, InputFusion, NoiseBranch, encoder_strides


def _stream_kinds(name: str) -> tuple[str, ...]:
    """'srm_bayar' -> ('srm', 'bayar'); 'none'/'' -> ()."""
    name = str(name or "none").lower()
    if name in {"none", "off", "false"}:
        return ()
    kinds = tuple(part for part in name.split("_") if part)
    unknown = set(kinds) - {"srm", "bayar"}
    if unknown:
        raise ValueError(f"неизвестный поток: {sorted(unknown)}; есть srm|bayar|srm_bayar")
    return kinds


class LogitsClsHead(nn.Module):
    """Классификация кадра из карты сегментации: [avg, max] -> линейный слой."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(2, 1)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[0.5, 0.5]]))
            self.fc.bias.zero_()

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        flat = logits.flatten(1)
        features = torch.stack([flat.mean(dim=1), flat.amax(dim=1)], dim=1)
        return self.fc(features)


class SegModel(nn.Module):
    """Общая обёртка: приводит любой бэкенд к единому контракту выхода."""

    def __init__(
        self,
        core: nn.Module,
        backend: str,
        cls_head: str = "aux",
        fusion: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.core = core
        self.backend = backend
        self.cls_mode = cls_head
        self.logits_head = LogitsClsHead() if cls_head == "from_logits" else None
        # при fuse=input остаток приклеивается к RGB здесь, а core уже собран
        # под расширенное число входных каналов
        self.fusion = fusion

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        size = images.shape[-2:]
        if self.fusion is not None:
            images = self.fusion(images)

        if self.backend == "smp":
            out = self.core(images)
            if isinstance(out, (tuple, list)):
                logits, cls_logits = out[0], out[1]
            else:
                logits, cls_logits = out, None
        else:  # hf
            logits = self.core(pixel_values=images).logits
            cls_logits = None

        if logits.shape[-2:] != size:
            logits = F.interpolate(logits, size=size, mode="bilinear", align_corners=False)

        if self.cls_mode == "from_logits":
            cls_logits = self.logits_head(logits)
        elif cls_logits is None:
            cls_logits = logits.flatten(1).amax(dim=1, keepdim=True)

        if cls_logits.ndim == 1:
            cls_logits = cls_logits.unsqueeze(1)
        return {"logits": logits, "cls_logits": cls_logits}


def find_stem_conv(encoder: nn.Module, in_channels: int) -> nn.Conv2d | None:
    """Первая свёртка, которая получает сам вход. Ищется прогоном, а не по имени.

    У ConvNeXt это patchify-свёртка, у ResNet — conv1, у EfficientNet — conv_stem;
    угадывать по имени ненадёжно, а порядок в `modules()` не гарантирует порядок
    вызова. Хук на forward даёт настоящий первый слой.
    """
    found: list[nn.Conv2d] = []
    handles = []

    def hook(module, inputs, output):
        if not found and inputs and inputs[0].shape[1] == in_channels:
            found.append(module)

    for module in encoder.modules():
        if isinstance(module, nn.Conv2d):
            handles.append(module.register_forward_hook(hook))
    was_training = encoder.training
    try:
        encoder.eval()
        with torch.no_grad():
            encoder(torch.zeros(1, in_channels, 64, 64))
    except Exception:
        # прогон мог упасть дальше по сети — неважно, хук на нужном слое
        # уже сработал; если не сработал, вернём None и вызывающий решит сам
        pass
    finally:
        encoder.train(was_training)
        for handle in handles:
            handle.remove()
    return found[0] if found else None


def restore_pretrained_stem(encoder: nn.Module, in_channels: int, rgb_channels: int = 3) -> bool:
    """Вернуть стему исходные RGB-веса и обнулить каналы низкоуровневого потока.

    Зачем. timm адаптирует предобученный стем под `in_chans != 3` так: копирует
    блок RGB-весов нужное число раз и делит всё на `3 / in_chans`. Для 9 каналов
    это значит, что отклик на RGB ослаблен втрое, а две трети веса стема отданы
    шумовому остатку, пропущенному через ЦВЕТОВЫЕ фильтры ImageNet. Модель
    стартует с испорченного представления: замер показывает расхождение признаков
    энкодера с RGB-baseline на 37% уже на нулевом шаге, до всякого обучения.

    Для многоспектральных данных, под которые эта адаптация писалась, такое
    поведение разумно — там лишние каналы похожи на RGB. Для forensic-остатка
    оно вредно.

    После восстановления модель на шаге 0 побитово равна RGB-baseline, а каналы
    остатка учатся с нуля: градиент по ним ненулевой, потому что вход ненулевой.
    Ровно тот же приём, что и зануление проекции в `fuse: gate`.
    """
    conv = find_stem_conv(encoder, in_channels)
    if conv is None or conv.weight.shape[1] != in_channels:
        return False

    with torch.no_grad():
        weight = conv.weight.data
        # timm поделил на 3 / in_channels — возвращаем исходный масштаб
        rgb = weight[:, :rgb_channels] * (in_channels / rgb_channels)
        weight.zero_()
        weight[:, :rgb_channels] = rgb
    return True


def _build_smp(cfg: dict) -> tuple[nn.Module, str, nn.Module | None]:
    import segmentation_models_pytorch as smp

    cls_head = cfg.get("cls_head", "aux")
    aux_params = dict(classes=1, dropout=float(cfg.get("cls_dropout", 0.2))) \
        if cls_head == "aux" else None

    kinds = _stream_kinds(cfg.get("stream", "none"))
    fuse = str(cfg.get("fuse", "input")).lower()
    # при input-фьюзе энкодер строится сразу под 3 + K каналов: timm корректно
    # переносит предобученные веса стема на новое число входов
    fusion = InputFusion(kinds) if (kinds and fuse == "input") else None

    kwargs = dict(
        arch=cfg.get("arch", "unet"),
        encoder_name=cfg.get("encoder", "tu-convnext_tiny"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=fusion.out_channels if fusion is not None else 3,
        classes=1,
    )
    if aux_params is None:
        core = smp.create_model(**kwargs)
    else:
        try:
            core = smp.create_model(aux_params=aux_params, **kwargs)
        except TypeError:
            # UPerNet, Segformer и часть архитектур в smp не принимают aux_params —
            # тогда классификацию считаем из карты логитов, контракт выхода не меняется
            core, cls_head = smp.create_model(**kwargs), "from_logits"

    if fusion is not None and cfg.get("encoder_weights"):
        # без этого модель стартует с испорченного предобученного стема —
        # см. restore_pretrained_stem
        restore_pretrained_stem(core.encoder, fusion.out_channels)

    if kinds and fuse == "gate":
        core.encoder = _wrap_gated(core.encoder, kinds, int(cfg.get("stream_width", 32)))
    elif kinds and fuse != "input":
        raise ValueError(f"неизвестный способ фьюза: {fuse}; есть input|gate")

    return core, cls_head, fusion


def _wrap_gated(encoder: nn.Module, kinds: tuple[str, ...], width: int) -> GatedDualEncoder:
    """Подменяет энкодер smp на пару «энкодер + ветка по остатку» с гейтом."""
    strides = encoder_strides(encoder)
    fuse_index = [i for i, ch in enumerate(encoder.out_channels) if ch > 0 and i >= 2]
    if not fuse_index:
        raise ValueError("у энкодера нет стадий, пригодных для фьюза")
    branch = NoiseBranch(tuple(strides[i] for i in fuse_index), kinds, width=width)
    return GatedDualEncoder(encoder, branch)


def _build_hf(cfg: dict) -> tuple[nn.Module, str]:
    from transformers import AutoModelForSemanticSegmentation

    if _stream_kinds(cfg.get("stream", "none")):
        raise ValueError("model.stream поддержан только для backend=smp")

    core = AutoModelForSemanticSegmentation.from_pretrained(
        cfg.get("hf_name", "nvidia/mit-b0"),
        num_labels=1,
        ignore_mismatched_sizes=True,
    )
    # у HF-моделей своей aux-головы нет — считаем классификацию из карты логитов
    cls_head = cfg.get("cls_head", "from_logits")
    return core, ("from_logits" if cls_head == "aux" else cls_head)


@register_backend("smp")
def _backend_smp(cfg_model: dict) -> SegModel:
    core, cls_head, fusion = _build_smp(cfg_model)
    return SegModel(core, "smp", cls_head, fusion)


@register_backend("hf")
def _backend_hf(cfg_model: dict) -> SegModel:
    core, cls_head = _build_hf(cfg_model)
    return SegModel(core, "hf", cls_head, None)


def build_model(cfg_model: dict) -> nn.Module:
    """Собрать модель по `model.backend`.

    Свой бэкенд регистрируется через `@register_backend` и обязан вернуть модуль,
    forward которого даёт `{"logits": (B, 1, H, W), "cls_logits": (B, 1)}` —
    на этом контракте держатся и метрика, и сборка сабмита. Наследовать
    `SegModel` не обязательно.
    """
    return BACKENDS.get(str(cfg_model.get("backend", "smp")).lower())(cfg_model)


def describe_encoder(encoder_name: str, arch: str = "unet") -> dict[str, Any]:
    """Пирамида признаков энкодера и то, сколько каналов реально приходит в скипы.

    Нужно, чтобы отвечать на вопрос «а есть ли у этого энкодера карта признаков
    на нужном шаге разрешения». У ConvNeXt, например, стем сразу режет вход в 4
    раза, карты на stride 2 не существует, и два последних блока U-Net-декодера
    остаются вообще без skip-connection. У ResNet там 64 канала.
    """
    import segmentation_models_pytorch as smp

    model = smp.create_model(
        arch, encoder_name=encoder_name, encoder_weights=None, in_channels=3, classes=1
    )
    out_channels = list(model.encoder.out_channels)
    blocks = list(getattr(model.decoder, "blocks", []))

    decoder_in = [b.conv1[0].in_channels for b in blocks]
    decoder_out = [b.conv2[0].out_channels for b in blocks]
    previous = [out_channels[-1]] + decoder_out[:-1]
    skips = [din - prev for din, prev in zip(decoder_in, previous)]

    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    return {
        "encoder": encoder_name,
        "arch": arch,
        "out_channels": out_channels,
        "decoder_in": decoder_in,
        "decoder_out": decoder_out,
        "skip_channels": skips,
        "blocks_without_skip": int(sum(1 for s in skips if s == 0)),
        "encoder_m": round(encoder_params / 1e6, 2),
        "total_m": round(total_params / 1e6, 2),
    }


def count_parameters(model: nn.Module) -> dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_m": round(total / 1e6, 2), "trainable_m": round(trainable / 1e6, 2)}
