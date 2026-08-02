"""Замер пиковой VRAM для конфига — до того, как ставить прогон на часы.

Оценивать память по памяти («на 512 влезало, значит и на 768 влезет») бесполезно:
активации растут по пикселям, EMA удваивает веса, оптимизатор держит два момента
на параметр, а дополнительные ветки модели вообще нигде не учтены.

Прогоняется настоящая модель с настоящим лоссом и оптимизатором на случайном
батче нужной формы, несколько шагов — чтобы аллокатор вышел на устойчивый пик.
"""

from __future__ import annotations

import torch

from .config import Cfg
from .engine import build_optimizer
from .losses import build_loss
from .models import build_model, count_parameters
from .models.aux_heads import parse_aux_spec
from .utils import ModelEma, pick_device


def _fake_batch(batch_size: int, size: int, device: torch.device) -> dict:
    """Батч той же формы, что отдаёт SegDataset, но из шума."""
    mask = (torch.rand(batch_size, 1, size, size, device=device) > 0.85).float()
    label = (mask.flatten(1).amax(dim=1, keepdim=True) > 0).float()
    return {
        "image": torch.randn(batch_size, 3, size, size, device=device),
        "mask": mask,
        "label": label,
        "area": mask.flatten(1).mean(dim=1, keepdim=True),
        # цели aux-голов кладём всегда: лосс возьмёт только включённые конфигом,
        # а без них головы выпадали бы из графа и замер не увидел бы их части
        "aux_border": (torch.rand(batch_size, 1, device=device) > 0.5).float(),
        "aux_centroid": torch.rand(batch_size, 2, device=device),
        "aux_components": torch.rand(batch_size, 1, device=device),
        "geom_valid": label,
    }


def probe_memory(cfg: Cfg, steps: int = 3, include_ema: bool | None = None) -> dict:
    device = pick_device(str(cfg.get("device", "auto")))
    if device.type != "cuda":
        raise RuntimeError("замер памяти имеет смысл только на GPU")

    free_before, total = torch.cuda.mem_get_info(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    size = int(cfg.data.size)
    batch_size = int(cfg.train.bs)
    amp = str(cfg.train.get("amp", "fp16"))
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp)

    model = build_model(cfg.model).to(device, memory_format=torch.channels_last)
    params = count_parameters(model)
    # aux-веса те же, что в train: иначе лосс не трогал бы выходы голов, и замер
    # молчал бы про их часть графа — а модель их всё равно считает
    criterion = build_loss(cfg.loss, parse_aux_spec(cfg.model.get("aux_heads"))).to(device)
    optimizer = build_optimizer(model, cfg.train)
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16"))

    if include_ema is None:
        include_ema = bool(cfg.train.get("ema"))
    ema = ModelEma(model, float(cfg.train.get("ema_decay", 0.999))) if include_ema else None

    batch = _fake_batch(batch_size, size, device)
    images = batch["image"].to(memory_format=torch.channels_last)

    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            outputs = model(images)
            loss, _ = criterion(outputs, batch)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if ema is not None:
            ema.update(model)

    peak_alloc = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    # валидация идёт без градиентов, но с большим батчем
    val_bs = int(cfg.train.get("val_bs", batch_size))
    torch.cuda.reset_peak_memory_stats(device)
    eval_model = ema.module if ema is not None else model
    eval_model.eval()
    with torch.no_grad():
        val_images = torch.randn(val_bs, 3, size, size, device=device).to(
            memory_format=torch.channels_last
        )
        with torch.amp.autocast("cuda", dtype=dtype, enabled=dtype is not None):
            eval_model(val_images)
    peak_val = torch.cuda.max_memory_allocated(device)

    result = {
        "size": size,
        "bs": batch_size,
        "val_bs": val_bs,
        "amp": amp,
        "ema": bool(ema),
        "params_m": params["total_m"],
        "peak_train_gb": round(peak_alloc / 1e9, 2),
        "peak_train_reserved_gb": round(peak_reserved / 1e9, 2),
        "peak_val_gb": round(peak_val / 1e9, 2),
        "gpu_total_gb": round(total / 1e9, 2),
        "gpu_free_at_start_gb": round(free_before / 1e9, 2),
    }

    del model, criterion, optimizer, ema, batch, images
    torch.cuda.empty_cache()
    return result


def verdict(result: dict, headroom_gb: float = 2.6) -> tuple[bool, str]:
    """Влезет ли на полностью свободную карту.

    `headroom_gb` закрывает то, чего счётчик torch не видит: контекст CUDA и
    воркспейсы cuDNN (~1.5 ГБ, замерено как расхождение nvidia-smi 6036 МиБ
    против 4.26 ГБ reserved на одном и том же прогоне), плюс ~1 ГБ, который
    на Windows держит рабочий стол. Без этой поправки вердикт врал бы
    «влезает» на пограничных конфигах.
    """
    need = max(result["peak_train_reserved_gb"], result["peak_val_gb"])
    total = result["gpu_total_gb"]
    budget = total - headroom_gb
    fits = need <= budget
    if fits:
        text = (f"влезает: пик {need:.2f} ГБ из {budget:.2f} ГБ полезных "
                f"(запас {budget - need:.2f} ГБ)")
    else:
        text = (f"НЕ влезает: нужно {need:.2f} ГБ, полезных {budget:.2f} ГБ. "
                f"Снижай train.bs или data.size")
    return fits, text
