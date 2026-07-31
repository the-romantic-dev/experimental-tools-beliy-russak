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
    idx = (flat * n_bins).long().clamp_(max=n_bins - 1)
    offsets = torch.arange(batch, device=idx.device).unsqueeze(1) * n_bins
    shifted = (idx + offsets).reshape(-1)

    hist_all = torch.bincount(shifted, minlength=batch * n_bins).reshape(batch, n_bins)
    gt_flat = masks.reshape(batch, -1) > 0.5
    hist_gt = torch.bincount(
        (idx + offsets)[gt_flat], minlength=batch * n_bins
    ).reshape(batch, n_bins)

    return (
        hist_all.cpu().numpy(),
        hist_gt.cpu().numpy(),
        gt_flat.sum(dim=1).cpu().numpy(),
        np.asarray(n_pixels, dtype=np.int64),
    )


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

    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        if max_steps and step >= max_steps:
            break

        images = batch["image"].to(device, non_blocking=True, memory_format=torch.channels_last)
        for key in ("mask", "label", "area"):
            if key in batch:
                batch[key] = batch[key].to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            outputs = model(images)
            loss, parts = criterion(outputs, batch)

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
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            if ema is not None:
                ema.update(model)
            if scheduler is not None:
                scheduler.step()

        meters["loss"].update(float(loss.detach()), images.size(0))
        for name, value in parts.items():
            meters.setdefault(name, AverageMeter()).update(value, images.size(0))

        if logger and log_every and (step + 1) % log_every == 0:
            done = (step + 1) / n_steps
            eta = (time.time() - start) * (1 - done) / max(done, 1e-9)
            lr = optimizer.param_groups[0]["lr"]
            detail = " ".join(f"{k}={m.avg:.4f}" for k, m in meters.items())
            logger.info(
                f"  эпоха {epoch} [{step + 1}/{n_steps}] {detail} lr={lr:.2e} "
                f"ETA {format_seconds(eta)}"
            )

    return {name: meter.avg for name, meter in meters.items()} | {
        "lr": optimizer.param_groups[0]["lr"],
        "epoch_time_s": round(time.time() - start, 1),
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
        for key in ("label", "area"):
            if key in batch:
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
SCRATCH_MARKERS = ("aux_stream", "projections", "gates", "fusion")


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
