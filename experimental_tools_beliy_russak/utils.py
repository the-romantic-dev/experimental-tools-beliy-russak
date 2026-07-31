"""Мелкие утилиты: сиды, EMA, замеры, работа с run-папками."""

from __future__ import annotations

import contextlib
import os
import random
import subprocess
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from .workspace import runs_root


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def pick_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


class AverageMeter:
    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0


class ModelEma:
    """Экспоненциальное скользящее среднее весов. Обычно +0.3-1.0 к Dice бесплатно."""

    def __init__(self, model: torch.nn.Module, decay: float = 0.999) -> None:
        self.module = deepcopy(model).eval()
        for param in self.module.parameters():
            param.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
            if ema_v.dtype.is_floating_point:
                ema_v.mul_(self.decay).add_(model_v.detach(), alpha=1.0 - self.decay)
            else:
                ema_v.copy_(model_v)


@contextlib.contextmanager
def timed(label: str, sink=print):
    start = time.perf_counter()
    yield
    sink(f"{label}: {time.perf_counter() - start:.1f} c")


def gpu_memory_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / 1e9


def git_revision() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or "no-git"
    except Exception:
        return "no-git"


def make_run_dir(name: str, resume: bool = False) -> Path:
    """runs/<name>; при коллизии добавляется суффикс, чтобы ничего не затирать."""
    runs = runs_root()
    runs.mkdir(parents=True, exist_ok=True)
    run_dir = runs / name
    if run_dir.exists() and not resume:
        stamp = time.strftime("%m%d-%H%M%S")
        run_dir = runs / f"{name}__{stamp}"
    for sub in ("ckpt", "oof", "tb", "preds"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:d}:{minutes:02d}:{secs:02d}" if hours else f"{minutes:d}:{secs:02d}"
