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

import functools
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import BACKENDS, register_backend
from ..streams import GatedDualEncoder, InputFusion, NoiseBranch, encoder_strides
from .aux_heads import build_aux_heads, parse_aux_spec


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


class _FeatureCatcher:
    """Складывает выход энкодера, чтобы его достали aux-головы.

    Намеренно НЕ метод SegModel: `ModelEma` копирует модель через `deepcopy`,
    и хук, замкнутый на связанный метод, утащил бы за собой всю модель. Обычный
    объект копируется вместе с ссылкой на себя же в хуке, и копия остаётся
    согласованной.
    """

    def __init__(self) -> None:
        self.features = None

    def __call__(self, module, inputs, output):
        self.features = output


class SegModel(nn.Module):
    """Общая обёртка: приводит любой бэкенд к единому контракту выхода."""

    def __init__(
        self,
        core: nn.Module,
        backend: str,
        cls_head: str = "aux",
        fusion: nn.Module | None = None,
        aux_weights: dict[str, float] | None = None,
        aux_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.core = core
        self.backend = backend
        self.cls_mode = cls_head
        self.logits_head = LogitsClsHead() if cls_head == "from_logits" else None
        # при fuse=input остаток приклеивается к RGB здесь, а core уже собран
        # под расширенное число входных каналов
        self.fusion = fusion

        self.aux_weights = dict(aux_weights or {})
        self.aux_heads = None
        self.catcher = None
        if self.aux_weights:
            if backend != "smp":
                raise ValueError("aux-головы поддержаны только для backend=smp")
            channels = int(core.encoder.out_channels[-1])
            self.aux_heads = build_aux_heads(channels, self.aux_weights, aux_dropout)
            self.catcher = _FeatureCatcher()
            core.encoder.register_forward_hook(self.catcher)

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

        outputs = {"logits": logits, "cls_logits": cls_logits}

        if self.aux_heads is not None:
            features = self.catcher.features
            if features is None:
                raise RuntimeError("энкодер не отдал признаки — хук aux-голов не сработал")
            deepest = features[-1] if isinstance(features, (list, tuple)) else features
            for name, head in self.aux_heads.items():
                outputs[f"aux_{name}"] = head(deepest)
            self.catcher.features = None  # не держим карту признаков после шага

        return outputs


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


@functools.lru_cache(maxsize=None)
def _fp32_class(cls: type) -> type:
    """Подкласс, который считает forward с выключенным autocast.

    Подмена `__class__` вместо обёртки-модуля — не эстетство, а требование двух
    мест. Во-первых, обёртка добавила бы уровень в имена параметров, и все
    существующие чекпоинты перестали бы грузиться. Во-вторых, `ModelEma` копирует
    модель через `deepcopy`: подмена метода `forward` на замыкание утащила бы за
    собой ссылку на ИСХОДНЫЙ модуль, и EMA-копия молча считала бы чужие веса (та
    же грабля, из-за которой `_FeatureCatcher` сделан отдельным объектом).
    Классы deepcopy копирует по ссылке, поэтому подмена переживает и его.
    """
    def forward(self, x, *args, **kwargs):
        with torch.amp.autocast("cuda", enabled=False):
            return cls.forward(self, x.float(), *args, **kwargs)

    return type(f"Fp32{cls.__name__}", (cls,), {"forward": forward, "_is_fp32": True})


def force_fp32(module: nn.Module) -> None:
    """Считать этот модуль в fp32 даже под autocast(fp16)."""
    if not getattr(module, "_is_fp32", False):
        module.__class__ = _fp32_class(module.__class__)


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
    # Проброс параметров конструктора энкодера. Нужен, например, Swin: у него
    # размер входа зашит в предвычисленные маски внимания, и без img_size он
    # падает на 768 с `Input height (768) doesn't match model (224)`.
    kwargs.update(dict(cfg.get("encoder_kwargs") or {}))
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

    # нормировка скипов навешивается ПОСЛЕ гейт-фьюза: она должна видеть уже
    # собранную пирамиду, а не только основной энкодер
    if cfg.get("skip_norm", False):
        core.encoder = SkipNormEncoder(core.encoder)

    if cfg.get("fp32_decoder_stem", True):
        _protect_decoder_stem(core)

    return core, cls_head, fusion


def _protect_decoder_stem(core: nn.Module) -> None:
    """Первую свёртку декодера считать в fp32: она упирается в потолок fp16.

    Замер на long_baseline_768 (unet + convnext_tiny, 768 px, 13 эпох): у
    `decoder.blocks.0.conv1` BatchNorm накопил running_var 1.16e8 (sigma ~10 800)
    при running_mean 19 200, тогда как у соседних блоков это 4579 и 1629. То есть
    выход ЭТОЙ свёртки живёт на одном порядке с пределом fp16 (65504), и хвост
    распределения за него изредка вылезает.

    Дальше цепочка необратимая: свёртка отдаёт inf, следующий BatchNorm делает из
    него (inf-inf)/inf = nan и НАВСЕГДА пишет inf в running_mean/running_var —
    градиентный скалер такое не откатывает, буферы обновляются в forward. В
    train-режиме BN считает по батчу и лосс «выздоравливает», а eval берёт
    испорченные буферы и валидация падает.

    Почему именно этот блок: он один принимает самую глубокую фичу энкодера
    (у ConvNeXt в последней стадии активации известно крупные и растут по ходу
    обучения — за 84k шагов running_var вырос с 1209 до 1.16e8). Остальная сеть
    остаётся в fp16, так что цена — одна свёртка 1152->256 на 48x48.

    Функцию сети это не меняет, только точность её вычисления, и имена
    параметров остаются прежними — старые чекпоинты грузятся как есть.
    """
    blocks = getattr(getattr(core, "decoder", None), "blocks", None)
    if blocks:
        force_fp32(blocks[0].conv1)


class SkipNormEncoder(nn.Module):
    """Энкодер, у которого каждая фича пирамиды нормируется перед декодером.

    Зачем. timm отдаёт `features_only` без финальной нормы на всех стадиях,
    кроме последней: у tu-convnext_tiny на реальных кадрах глубокая фича идёт
    с RMS 1.03, а skip в первый блок декодера — с RMS 22.95 и выбросами за 1000
    (замер на f0-control-768 после 6 эпох; до обучения было 12.35 и 700, то есть
    поток ещё и дрейфует вверх по ходу обучения). Остаточный поток ConvNeXt
    между стадиями не нормируется вообще — LayerNorm живёт ВНУТРИ блока, — а
    weight decay штрафует веса, а не магнитуду активаций.

    Что из этого следует. В конкатенации на входе `decoder.blocks.0.conv1` скип
    даёт 79.5% энергии выхода против 20.5% у глубокой фичи: блок, который по
    замыслу вливает семантику, на четыре пятых состоит из скипа. Плюс ровно этот
    дрейф уводил свёртку в потолок fp16 на 13-й эпохе long_baseline_768
    (running_var 1.16e8 против 1209 у f0).

    Почему GroupNorm(1, C), а не LayerNorm2d из timm. LayerNorm2d нормирует
    только по каналам, отдельно в каждой точке — это стирает разницу масштабов
    между участками кадра, а для forensic-задачи «здесь отклик сильнее» само по
    себе признак. GroupNorm с одной группой нормирует по (C, H, W) целиком:
    убирает общий сдвиг и масштаб карты, а пространственный контраст оставляет.

    Ранние стадии (индексы 0 и 1) пропускаются по той же причине, что и в
    `GatedDualEncoder`: нулевой индекс — это сам вход, а у части энкодеров
    ранние уровни пустые.
    """

    def __init__(self, encoder: nn.Module, from_stage: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.out_channels = list(encoder.out_channels)
        self.output_stride = getattr(encoder, "output_stride", 32)

        self.norm_index = [
            i for i, ch in enumerate(self.out_channels) if ch > 0 and i >= from_stage
        ]
        if not self.norm_index:
            raise ValueError("у энкодера нет стадий, пригодных для нормировки скипов")
        # имя `skip_norms` перечислено в SCRATCH_MARKERS: модуль лежит внутри
        # энкодера, но учится с нуля, и пониженный encoder_lr ему не нужен
        self.skip_norms = nn.ModuleList(
            [nn.GroupNorm(1, self.out_channels[i]) for i in self.norm_index]
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        features = list(self.encoder(x))
        for slot, index in enumerate(self.norm_index):
            features[index] = self.skip_norms[slot](features[index])
        return features


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
    return SegModel(
        core, "smp", cls_head, fusion,
        aux_weights=parse_aux_spec(cfg_model.get("aux_heads")),
        aux_dropout=float(cfg_model.get("cls_dropout", 0.2)),
    )


@register_backend("hf")
def _backend_hf(cfg_model: dict) -> SegModel:
    core, cls_head = _build_hf(cfg_model)
    return SegModel(
        core, "hf", cls_head, None,
        aux_weights=parse_aux_spec(cfg_model.get("aux_heads")),
    )


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
