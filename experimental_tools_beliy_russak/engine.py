"""Циклы обучения и валидации.

Всё, что нужно на 8 ГБ VRAM: AMP, накопление градиента, channels_last,
клиппинг, EMA. Валидация не считает метрику «в лоб», а копит гистограммы
для `AICAccumulator` — один проход даёт возможность потом перебрать
любые пороги без повторного прогона модели.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import AICAccumulator
from .registry import OPTIMIZERS, SCHEDULERS, register_optimizer, register_scheduler
from .utils import AverageMeter, format_seconds


#: не-целевые поля батча: индексы и размеры нужны на CPU, а тензорами не являются
_NON_TARGET = {"image", "index", "orig_h", "orig_w", "valid_h", "valid_w", "name"}


def _target_keys(batch: dict) -> list[str]:
    """Поля батча, которые надо перенести на device.

    Список раньше был захардкожен ("mask", "label", "area"), и любая новая цель
    молча оставалась на CPU до первой ошибки про несовпадение устройств.
    """
    return [
        key for key, value in batch.items()
        if key not in _NON_TARGET and isinstance(value, torch.Tensor)
    ]


def _amp_dtype(name: str) -> torch.dtype | None:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "off": None, "none": None}[name]


@torch.no_grad()
def _histograms(
    probs: torch.Tensor,
    masks: torch.Tensor,
    n_bins: int,
    valid_h: torch.Tensor | None = None,
    valid_w: torch.Tensor | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Гистограммы вероятностей по каждому кадру, считается на GPU.

    `valid_h`/`valid_w` задают полезную часть сетки. В режиме `val_mode: pad`
    паддинг занимает до трети холста, и без вырезания он попадал бы и в |P_t|,
    и в знаменатель площади — правило FPR («площадь >= 1% кадра») срабатывало бы
    в полтора раза позже, чем на настоящем сабмите.
    """
    batch, height, width = probs.shape[0], probs.shape[-2], probs.shape[-1]
    probs = probs.clamp(0, 1).float().reshape(batch, height, width)
    masks = masks.reshape(batch, height, width)

    if valid_h is not None and valid_w is not None:
        rows = torch.arange(height, device=probs.device).view(1, height, 1)
        cols = torch.arange(width, device=probs.device).view(1, 1, width)
        keep = (rows < valid_h.to(probs.device).view(-1, 1, 1)) & (
            cols < valid_w.to(probs.device).view(-1, 1, 1)
        )
        probs = probs * keep
        masks = masks * keep
        n_pixels = (valid_h.to(torch.int64) * valid_w.to(torch.int64)).cpu().numpy()
    else:
        n_pixels = np.full(batch, height * width, dtype=np.int64)

    flat = probs.reshape(batch, -1)
    if not torch.isfinite(flat).all():
        # без этой проверки nan уезжал в `.long()` (это INT64_MIN, а `clamp_`
        # ниже держит только верхнюю границу) и вылезал через двести строк
        # стека как «bincount only supports non-negative inputs» — сообщение,
        # по которому настоящую причину не найти
        raise RuntimeError(
            "модель выдала не-конечные вероятности на валидации. Чаще всего это "
            "значит, что в train переполнился forward: BatchNorm навсегда записал "
            "inf/nan в running_mean/running_var, и в train-режиме лосс выглядит "
            "здоровым (там BN считает по батчу), а eval берёт испорченные буферы. "
            "Смотри nan-шаги в логе эпохи."
        )
    idx = (flat * n_bins).long().clamp_(min=0, max=n_bins - 1)
    offsets = torch.arange(batch, device=idx.device).unsqueeze(1) * n_bins
    shifted = idx + offsets

    hist_all = torch.bincount(shifted.reshape(-1), minlength=batch * n_bins)
    gt_flat = masks.reshape(batch, -1) > 0.5
    hist_gt = torch.bincount(shifted[gt_flat], minlength=batch * n_bins)

    n_pixels = np.asarray(n_pixels, dtype=np.int64)
    counts = hist_all.reshape(batch, n_bins).cpu().numpy()
    # Обнулённый выше паддинг осел в нулевом бине, а в знаменателе площади его
    # нет. Без этой поправки рушится инвариант «сумма гистограммы = n_pixels»:
    # на порогах ниже 1/n_bins доля предсказанной площади выходила больше
    # единицы, и правило FPR («площадь >= 1% кадра») срабатывало на пустоте.
    counts[:, 0] -= height * width - n_pixels

    return (
        counts,
        hist_gt.reshape(batch, n_bins).cpu().numpy(),
        gt_flat.sum(dim=1).cpu().numpy(),
        n_pixels,
    )


@torch.no_grad()
def _nan_report(model: nn.Module, images: torch.Tensor, batch: dict, outputs: dict,
                dtype: torch.dtype | None) -> str:
    """Где именно родился nan: во входе, в цели, в выходе или в конкретном слое.

    Вызывается только на аварийном шаге, поэтому может позволить себе второй
    forward с хуками на каждом листовом модуле. Дешёвая проверка «весь ли
    тензор конечен» на каждом слое КАЖДОГО шага стоила бы сотни синхронизаций
    с GPU, а нужна она раз в несколько тысяч шагов.

    Первый модуль в цепочке и есть виновник: дальше по сети inf сам собой
    превращается в nan (например, любой BatchNorm даёт (inf-inf)/inf), и по
    последнему звену причину не найти.
    """
    lines: list[str] = []

    def flag(name: str, tensor) -> None:
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            return
        finite = torch.isfinite(tensor)
        if bool(finite.all()):
            lines.append(f"    {name}: ок, |max|={float(tensor.abs().max()):.4g}")
        else:
            lines.append(f"    {name}: НЕ КОНЕЧЕН ({int((~finite).sum())} из {tensor.numel()})")

    lines.append("  вход и цели:")
    flag("image", images)
    for key in ("mask", "label", "area"):
        if key in batch:
            flag(key, batch[key])
    lines.append("  выход модели:")
    outputs_bad = False
    for key, value in outputs.items():
        flag(key, value)
        outputs_bad |= (isinstance(value, torch.Tensor) and value.is_floating_point()
                        and not bool(torch.isfinite(value).all()))

    chain: list[str] = []

    def make_hook(name: str):
        def hook(module, inputs, output):
            tensors = output if isinstance(output, (list, tuple)) else [output]
            for tensor in tensors:
                if (isinstance(tensor, torch.Tensor) and tensor.is_floating_point()
                        and not bool(torch.isfinite(tensor).all())):
                    chain.append(name)
                    return
        return hook

    handles = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules() if not list(module.children())
    ]
    # momentum=0 замораживает running-статистику BatchNorm на время пробы:
    # сам forward её обновляет, и диагностика не должна добавлять модели
    # второй порции порчи поверх той, что уже случилась на боевом проходе
    norms = [m for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)]
    momenta = [m.momentum for m in norms]
    try:
        for norm in norms:
            norm.momentum = 0.0
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            model(images)
    finally:
        for norm, momentum in zip(norms, momenta):
            norm.momentum = momentum
        for handle in handles:
            handle.remove()

    if chain:
        lines.append(f"  первым переполнился: {chain[0]}")
        lines.append(f"  дальше по цепочке ({len(chain)} модулей): {' -> '.join(chain[1:6])}")
    elif outputs_bad:
        # повтор на тех же весах и том же батче обязан воспроизвести настоящее
        # переполнение свёртки; если не воспроизвёл — виноват недетерминизм
        # (дропаут в cls-голове, выбор алгоритма cudnn), и это тоже улика
        lines.append(
            "  выход модели не конечен, но повторный forward на том же батче чист — "
            "переполнение плавающее, а не детерминированное"
        )
    else:
        lines.append("  forward чист и выход конечен — значит nan родился в лоссе, а не в сети")
    if "index" in batch:
        lines.append(f"  строки датасета в батче: {batch['index'].tolist()}")
    return "\n".join(lines)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    scaler: torch.amp.GradScaler | None = None,
    scheduler=None,
    amp: str = "fp16",
    accum_steps: int = 1,
    grad_clip: float = 0.0,
    ema=None,
    epoch: int = 0,
    log_every: int = 50,
    logger=None,
    max_steps: int | None = None,
) -> dict[str, float]:
    model.train()
    dtype = _amp_dtype(amp)
    meters: dict[str, AverageMeter] = {"loss": AverageMeter()}
    start = time.time()
    n_steps = min(len(loader), max_steps) if max_steps else len(loader)
    nan_steps = 0

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        for key in _target_keys(batch):
            batch[key] = batch[key].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            outputs = model(images)
            loss, parts = criterion(outputs, batch)

        # Синхронизация с GPU тут не лишняя: `float(loss)` всё равно нужен
        # метру ниже, так что значение берётся один раз и переиспользуется.
        loss_value = float(loss.detach())
        if not np.isfinite(loss_value):
            # Раньше nan-шаг молча уходил в backward. При fp16 его гасил
            # GradScaler (шаг пропускался по found_inf), но при bf16 и amp=off
            # скалера нет вовсе, и nan-градиент убивал веса насмерть. Плюс в
            # любом режиме forward уже успевал записать inf в running-статистику
            # BatchNorm — это не откатывается и ломает eval до конца прогона.
            nan_steps += 1
            if logger and nan_steps <= 3:
                logger.info(
                    f"  ЭПОХА {epoch} ШАГ {step + 1}: лосс не конечен "
                    f"({', '.join(f'{k}={v}' for k, v in parts.items())}). Разбор:\n"
                    + _nan_report(model, images, batch, outputs, dtype)
                )
            optimizer.zero_grad(set_to_none=True)  # накопленное тоже под подозрением
            continue

        scaled = loss / accum_steps
        if scaler is not None and scaler.is_enabled():
            scaler.scale(scaled).backward()
        else:
            scaled.backward()

        if (step + 1) % accum_steps == 0:
            if grad_clip > 0:
                if scaler is not None and scaler.is_enabled():
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if scaler is not None and scaler.is_enabled():
                # при inf/NaN в градиентах scaler молча пропускает шаг и делит
                # масштаб пополам; вверх масштаб тоже двигается (раз в 2000
                # удачных шагов), поэтому пропуск ловится сравнением, а не равенством
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                stepped = scaler.get_scale() >= scale_before
            else:
                optimizer.step()
                stepped = True
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            # расписание двигается только вслед за реальным шагом весов. На
            # первой итерации fp16 стартовый масштаб 65536 почти всегда даёт
            # переполнение, и безусловный step() уводил LR вперёд пропущенного
            # апдейта — плюс ровно на нём срабатывало предупреждение torch
            # «lr_scheduler.step() before optimizer.step()».
            if scheduler is not None and stepped:
                scheduler.step()

        meters["loss"].update(loss_value, images.size(0))
        for name, value in parts.items():
            meters.setdefault(name, AverageMeter()).update(value, images.size(0))

        if logger and log_every and (step + 1) % log_every == 0:
            done = (step + 1) / n_steps
            eta = (time.time() - start) * (1 - done) / max(done, 1e-9)
            lr = optimizer.param_groups[0]["lr"]
            detail = " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
            skipped = f" nan-шагов={nan_steps}" if nan_steps else ""
            logger.info(
                f"  эпоха {epoch} [{step + 1}/{n_steps}] {detail} lr={lr:.2e} "
                f"ETA {format_seconds(eta)}{skipped}"
            )

    return {name: meter.avg for name, meter in meters.items()} | {
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time_s": round(time.time() - start, 1),
        "nan_steps": nan_steps,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module | None,
    device: torch.device,
    *,
    amp: str = "fp16",
    n_bins: int = 256,
    max_steps: int | None = None,
) -> tuple[AICAccumulator, dict[str, float]]:
    model.eval()
    dtype = _amp_dtype(amp)
    accumulator = AICAccumulator(n_bins=n_bins)
    loss_meter = AverageMeter()

    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break
        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        masks = batch["mask"].to(device, non_blocking=True)
        batch["mask"] = masks
        for key in _target_keys(batch):
            if key != "mask":  # маску уже перенесли выше
                batch[key] = batch[key].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            outputs = model(images)
            if criterion is not None:
                loss, _ = criterion(outputs, batch)
                loss_meter.update(float(loss.detach()), images.size(0))

        probs = torch.sigmoid(outputs["logits"].float())
        cls_prob = torch.sigmoid(outputs["cls_logits"].float()).reshape(-1)

        hist_all, hist_gt, gt_sum, n_pixels = _histograms(
            probs, masks, n_bins, batch.get("valid_h"), batch.get("valid_w")
        )
        accumulator.update_hist(hist_all, hist_gt, gt_sum, n_pixels, cls_prob.cpu().numpy())

    return accumulator, {"val_loss": loss_meter.avg}


# Модули, которые лежат ВНУТРИ энкодера, но обучаются с нуля: пониженный
# encoder_lr для них — это не бережное отношение к предобученным весам,
# а просто медленное обучение случайной инициализации.
SCRATCH_MARKERS = ("aux_stream", "projections", "gates", "fusion", "skip_norms")


@register_optimizer("adamw")
def _adamw(param_groups: list[dict], cfg_train: dict) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        param_groups, lr=float(cfg_train.get("lr", 3e-4)), betas=(0.9, 0.999)
    )


@register_optimizer("sgd")
def _sgd(param_groups: list[dict], cfg_train: dict) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        param_groups, lr=float(cfg_train.get("lr", 3e-4)), momentum=0.9, nesterov=True
    )


def build_optimizer(model: nn.Module, cfg_train: dict) -> torch.optim.Optimizer:
    """Разные lr для энкодера и декодера + no-decay для норм и биасов.

    Разбиение на группы — самая неочевидная часть, и она общая для всех
    оптимизаторов, поэтому остаётся здесь. Зарегистрированная через
    `@register_optimizer` функция получает готовые `param_groups`.
    """
    encoder_lr = float(cfg_train.get("encoder_lr", cfg_train.get("lr", 3e-4)))
    lr = float(cfg_train.get("lr", 3e-4))
    weight_decay = float(cfg_train.get("weight_decay", 1e-4))

    groups: dict[str, dict] = {
        "enc_decay": {"params": [], "lr": encoder_lr, "weight_decay": weight_decay},
        "enc_nodecay": {"params": [], "lr": encoder_lr, "weight_decay": 0.0},
        "dec_decay": {"params": [], "lr": lr, "weight_decay": weight_decay},
        "dec_nodecay": {"params": [], "lr": lr, "weight_decay": 0.0},
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        pretrained = "encoder" in name and not any(m in name for m in SCRATCH_MARKERS)
        part = "enc" if pretrained else "dec"
        decay = "nodecay" if (param.ndim <= 1 or name.endswith(".bias")) else "decay"
        groups[f"{part}_{decay}"]["params"].append(param)

    param_groups = [g for g in groups.values() if g["params"]]
    return OPTIMIZERS.get(str(cfg_train.get("optimizer", "adamw")).lower())(
        param_groups, cfg_train
    )


@register_scheduler("none")
def _no_scheduler(optimizer, cfg_train: dict, total_steps: int, warmup_steps: int):
    return None


@register_scheduler("cosine")
def _cosine(optimizer, cfg_train: dict, total_steps: int, warmup_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    min_factor = float(cfg_train.get("min_lr_factor", 0.02))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_factor + (1 - min_factor) * 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    return LambdaLR(optimizer, lr_lambda)


@register_scheduler("onecycle")
def _onecycle(optimizer, cfg_train: dict, total_steps: int, warmup_steps: int):
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[g["lr"] for g in optimizer.param_groups],
        total_steps=total_steps,
        pct_start=float(cfg_train.get("warmup_frac", 0.05)),
    )


def build_scheduler(optimizer, cfg_train: dict, steps_per_epoch: int):
    """Полное число шагов и длину прогрева считаем здесь: только тут известны
    и размер эпохи, и накопление градиента. Зарегистрированный планировщик
    получает их готовыми."""
    epochs = int(cfg_train.get("epochs", 10))
    accum = max(1, int(cfg_train.get("accum_steps", 1)))
    total = max(1, (steps_per_epoch // accum) * epochs)
    warmup = int(float(cfg_train.get("warmup_frac", 0.05)) * total)

    name = str(cfg_train.get("scheduler", "cosine")).lower()
    return SCHEDULERS.get(name)(optimizer, cfg_train, total, warmup)
