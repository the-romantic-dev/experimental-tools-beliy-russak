"""Продолженный прогон обязан идти по той же кривой LR, что и непрерывный.

Шедулер строится заново на каждом запуске и в `__init__` выставляет LR начала
warmup. Если состояние не восстановить, resume с середины косинуса возвращает
модель к прогревочному LR и доучивает её по второму кругу расписания — прогон
перестаёт быть продолжением того, что записано в `last.pt`, и сравнивать его с
непрерывным прогоном уже нельзя.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import pytest
import torch
import torch.nn as nn

from experimental_tools_beliy_russak.engine import build_optimizer, build_scheduler
from experimental_tools_beliy_russak.train import _checkpoint_state, _restore_state

CFG_TRAIN = {
    "lr": 1e-3,
    "encoder_lr": 1e-4,
    "epochs": 4,
    "scheduler": "cosine",
    "warmup_frac": 0.25,
}
STEPS_PER_EPOCH = 10


def make_run(scheduler_name: str = "cosine"):
    """Модель, оптимизатор и шедулер — ровно так, как их строит `train()`."""
    cfg_train = CFG_TRAIN | {"scheduler": scheduler_name}
    model = nn.Linear(4, 1)
    optimizer = build_optimizer(model, cfg_train)
    return model, optimizer, build_scheduler(optimizer, cfg_train, STEPS_PER_EPOCH)


def lrs(optimizer) -> list[float]:
    return [group["lr"] for group in optimizer.param_groups]


def advance(optimizer, scheduler, n_steps: int) -> None:
    for _ in range(n_steps):
        optimizer.step()  # градиентов нет, но счётчик шагов растёт
        scheduler.step()


def save(model, optimizer, scheduler, epoch: int = 1, best: float = 0.5) -> dict:
    return _checkpoint_state(
        model, None, optimizer, scheduler, {}, epoch=epoch, best=best, calib={}
    )


def test_checkpoint_carries_scheduler_state():
    model, optimizer, scheduler = make_run()
    advance(optimizer, scheduler, 12)

    assert save(model, optimizer, scheduler)["scheduler"], (
        "без состояния шедулера resume начинает расписание LR заново"
    )


def test_resumed_run_continues_the_lr_curve():
    """Главное свойство: LR после resume совпадает с непрерывным прогоном."""
    model, optimizer, scheduler = make_run()
    advance(optimizer, scheduler, 20)  # две эпохи из четырёх
    state = save(model, optimizer, scheduler)

    advance(optimizer, scheduler, 5)  # как если бы прогон не прерывался
    expected = lrs(optimizer)

    model2, optimizer2, scheduler2 = make_run()
    _restore_state(state, model2, optimizer2, scheduler2, None)
    advance(optimizer2, scheduler2, 5)

    assert lrs(optimizer2) == pytest.approx(expected)


def test_resume_lands_on_the_saved_lr_before_any_step():
    """Сразу после восстановления LR — тот же, что был в момент сохранения."""
    model, optimizer, scheduler = make_run()
    advance(optimizer, scheduler, 20)
    saved_lrs = lrs(optimizer)
    state = save(model, optimizer, scheduler)

    model2, optimizer2, scheduler2 = make_run()
    warmup_lrs = lrs(optimizer2)
    _restore_state(state, model2, optimizer2, scheduler2, None)

    assert lrs(optimizer2) == pytest.approx(saved_lrs)
    assert lrs(optimizer2) != pytest.approx(warmup_lrs), (
        "тест бесполезен, если середина косинуса совпала с началом warmup"
    )


def test_restore_returns_epoch_and_best():
    model, optimizer, scheduler = make_run()
    state = save(model, optimizer, scheduler, epoch=3, best=0.71)

    model2, optimizer2, scheduler2 = make_run()
    assert _restore_state(state, model2, optimizer2, scheduler2, None) == (4, 0.71)


def test_old_checkpoint_without_scheduler_key_still_resumes():
    """Чекпоинты, записанные до появления ключа, должны читаться как раньше."""
    model, optimizer, scheduler = make_run()
    advance(optimizer, scheduler, 20)
    state = save(model, optimizer, scheduler)
    del state["scheduler"]

    model2, optimizer2, scheduler2 = make_run()
    assert _restore_state(state, model2, optimizer2, scheduler2, None) == (2, 0.5)


def test_run_without_scheduler_round_trips():
    """`scheduler: none` — шедулера нет вообще, сохранять и грузить нечего."""
    model, optimizer, scheduler = make_run("none")
    assert scheduler is None

    state = save(model, optimizer, scheduler)
    assert state["scheduler"] is None

    model2, optimizer2, _ = make_run("none")
    assert _restore_state(state, model2, optimizer2, None, None) == (2, 0.5)


def test_checkpoint_carries_the_gradient_scaler():
    """Новый GradScaler стартует с масштаба 65536 — на порядки выше устоявшегося,
    поэтому первые шаги после resume гарантированно переполнялись и пропускались."""
    model, optimizer, scheduler = make_run()
    scaler = torch.amp.GradScaler("cuda", init_scale=128.0, enabled=True)

    state = _checkpoint_state(
        model, None, optimizer, scheduler, {}, epoch=1, best=0.5, calib={}, scaler=scaler
    )

    restored = torch.amp.GradScaler("cuda", enabled=True)
    _restore_state(state, *make_run(), None, restored)
    assert restored.get_scale() == pytest.approx(128.0)


def test_checkpoint_without_scaler_stays_readable():
    """`amp: off` и старые чекпоинты — скалера в них нет, resume это переживает."""
    model, optimizer, scheduler = make_run()
    state = save(model, optimizer, scheduler)
    assert state["scaler"] is None

    disabled = torch.amp.GradScaler("cuda", enabled=False)
    assert _restore_state(state, *make_run(), None, disabled) == (2, 0.5)
