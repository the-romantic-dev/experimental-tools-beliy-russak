"""Оркестрация обучения одного эксперимента.

Результат прогона целиком лежит в runs/<name>/:
    config.yaml      снапшот конфига со всеми переопределениями
    env.json         коммит, версии пакетов, командная строка — чем посчитано
    train.log        человекочитаемый лог
    metrics.jsonl    построчные метрики (источник правды)
    metrics.csv      то же для Excel
    tb/              TensorBoard
    ckpt/best.pt     лучший чекпоинт по AIC на валидации
    ckpt/last.pt     последний (для resume)
    oof/val.npz      сжатая статистика валидации -> свип порогов без модели
    summary.json     итог: лучший AIC, подобранные пороги и бюджет GFLOPs
"""

from __future__ import annotations

import json

import pandas as pd
import torch
from torch.utils.data import DataLoader

from . import budget
from .config import Cfg, config_hash, flatten, save_config
from .datasets import SegDataset, build_sampler
from .engine import build_optimizer, build_scheduler, train_one_epoch, validate
from .logging_utils import RunLogger
from .losses import build_loss
from aic.metric import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID, AICAccumulator
from .models import build_model, count_parameters
from .models.aux_heads import parse_aux_spec
from .provenance import write_environment
from aic.data import load_folds
from .stats import (
    compare_to_reference,
    gate_check,
    gate_metrics,
    load_reference,
    resolve_reference,
    stats_settings,
)
from .transforms import build_transform
from .utils import ModelEma, gpu_memory_gb, make_run_dir, pick_device, seed_everything
from .workspace import split_path


def _subset(
    df: pd.DataFrame,
    frac: float | None,
    limit: int | None,
    seed: int,
    keep_negatives: bool = False,
) -> pd.DataFrame:
    """Подвыборка для укороченных прогонов.

    `keep_negatives` прореживает только позитивы. Негативов в данных ~3%, и на
    них держится вся оценка FPR_neg: при val_frac=0.25 их остаётся полторы
    сотни, то есть одна ложная тревога стоит 0.007 FPR и порог классификатора
    подбирается по десятку событий. Позитивов же хватает и в четверти выборки.
    """
    if keep_negatives and "is_negative" in df.columns and (limit or (frac and frac < 1.0)):
        negatives = df[df["is_negative"]]
        positives = df[~df["is_negative"]]
        if limit:
            limit = min(limit, len(df))
            # жёсткий лимит важнее сохранения негативов: иначе на маленьком
            # val_limit (как в smoke) выборка вырождалась в одни негативы и
            # Dice_pos становился нулём при живой модели
            if len(negatives) * 2 > limit:
                negatives = negatives.sample(n=limit // 2, random_state=seed)
            take = max(0, limit - len(negatives))
            positives = positives.sample(n=min(take, len(positives)), random_state=seed)
        else:
            positives = positives.sample(frac=frac, random_state=seed)
        return (
            pd.concat([positives, negatives])
            .sample(frac=1.0, random_state=seed)
            .reset_index(drop=True)
        )
    if limit:
        return df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)
    if frac and frac < 1.0:
        return df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    return df


def _checkpoint_state(
    model: torch.nn.Module,
    ema,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg,
    *,
    epoch: int,
    best: float,
    calib: dict,
    scaler=None,
) -> dict:
    """Всё, что нужно, чтобы продолжить прогон ровно с этой точки.

    Состояние шедулера лежит здесь рядом с оптимизатором не для симметрии:
    шедулер строится заново на каждом запуске и в `__init__` выставляет LR
    начала warmup. Без этого ключа resume с середины косинуса откатывал LR к
    прогревочному и проходил расписание по второму кругу.

    По той же причине сохраняется и масштаб GradScaler: заново созданный скалер
    стартует с 65536, а это на порядки больше устоявшегося значения, и первые
    шаги после resume гарантированно уходят в переполнение и пропускаются.
    """
    return {
        "model": model.state_dict(),
        "ema": ema.module.state_dict() if ema is not None else None,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
        "epoch": epoch,
        "best": best,
        "cfg": dict(cfg),
        "calib": calib,
    }


def _restore_state(state: dict, model, optimizer, scheduler, ema, scaler=None) -> tuple[int, float]:
    """Разворачивает чекпоинт обратно; возвращает эпоху продолжения и лучший AIC.

    Порядок важен: `optimizer.load_state_dict` возвращает param_groups (вместе
    с их lr) на момент сохранения, поэтому шедулер восстанавливаем после него.
    `state.get` вместо `state[...]` — чекпоинты, записанные до появления ключа,
    должны читаться по-прежнему, просто без расписания и без масштаба скалера.
    """
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and state.get("scheduler"):
        scheduler.load_state_dict(state["scheduler"])
    if scaler is not None and state.get("scaler"):
        scaler.load_state_dict(state["scaler"])
    if ema is not None and state.get("ema"):
        ema.module.load_state_dict(state["ema"])
    return state["epoch"] + 1, state.get("best", -1.0)


def worker_init_fn(worker_id: int) -> None:
    """Индивидуальный, но детерминированный сид каждому воркеру DataLoader.

    torch сам сеет `random`, `torch` и `np.random` в воркерах от base_seed, но
    про генератор внутри albumentations.Compose он ничего не знает. А Compose
    приезжает в воркер пиклом ВМЕСТЕ с состоянием генератора, так что без этой
    функции все воркеры выдавали бы один и тот же поток аугментаций.

    `info.seed` torch выводит из base_seed и номера воркера, а base_seed — из
    глобального сида, выставленного `seed_everything`. Поэтому поток остаётся
    воспроизводимым от прогона к прогону и при этом разным между воркерами.
    """
    info = torch.utils.data.get_worker_info()
    if info is None:
        return
    seed = int(info.seed) % (2 ** 31)
    transform = getattr(info.dataset, "transform", None)
    setter = getattr(transform, "set_random_seed", None)
    if setter is not None:
        setter(seed)


def build_dataloaders(cfg: Cfg, logger: RunLogger) -> tuple[DataLoader, DataLoader, pd.DataFrame]:
    folds = load_folds(split_path())
    fold = int(cfg.train.get("fold", 0))
    seed = int(cfg.get("seed", 42))

    train_df = folds[folds["fold"] != fold].reset_index(drop=True)
    val_df = folds[folds["fold"] == fold].reset_index(drop=True)

    # Состав валидации намеренно НЕ зависит от `seed`: иначе прогон с другим
    # сидом мерился бы на другой подвыборке, и разница между ним и опорным
    # прогоном смешивала бы шум обучения с шумом выборки валидации. Ровно эту
    # смесь и получил бы замер пола шума (f1_seed7), ради которого всё затевалось.
    val_seed = int(cfg.data.get("val_seed", 42))

    train_df = _subset(train_df, cfg.data.get("train_frac"), cfg.data.get("train_limit"), seed)
    val_df = _subset(
        val_df, cfg.data.get("val_frac"), cfg.data.get("val_limit"), val_seed,
        keep_negatives=bool(cfg.data.get("val_keep_negatives", True)),
    )

    # геометрию считаем, только если её реально просит хоть одна aux-голова
    aux_targets = bool(parse_aux_spec(cfg.model.get("aux_heads")))
    common = dict(
        source=cfg.data.get("source", "raw"),
        cache_size=int(cfg.data.get("cache_size", 768)),
        gt_binarize=cfg.data.get("gt_binarize", 0.5),
        aux_targets=aux_targets,
    )
    # синтез — только в train: подмешать его в валидацию значит мерить метрику
    # на других кадрах, чем все остальные прогоны, и потерять сравнимость
    train_ds = SegDataset(
        train_df, build_transform(cfg.data, train=True, seed=seed),
        extra_negatives=float(cfg.data.get("extra_negatives", 0.0)), seed=seed,
        synth=cfg.data.get("synth"), **common,
    )
    val_ds = SegDataset(
        val_df, build_transform(cfg.data, train=False), extra_negatives=0.0,
        pad_mode=str(cfg.data.get("val_mode", "resize")) == "pad", **common,
    )

    sampler = build_sampler(train_ds, cfg.data, seed=seed)

    workers = int(cfg.train.get("num_workers", 6))
    loader_kwargs = dict(
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
        prefetch_factor=4 if workers > 0 else None,
        worker_init_fn=worker_init_fn,
    )
    # явный генератор: иначе порядок сэмплера зависит от того, сколько раз
    # глобальный RNG успели дёрнуть до начала итерации, и любая вставка кода
    # выше по течению незаметно сдвигает выборку
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg.train.bs), sampler=sampler,
        shuffle=sampler is None, drop_last=True, generator=loader_generator,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg.train.get("val_bs", cfg.train.bs)),
        shuffle=False, drop_last=False, **loader_kwargs,
    )

    logger.info(
        f"данные: train {len(train_ds)} (негативов {int(train_ds.is_negative.sum())}), "
        f"val {len(val_ds)} (негативов {int(val_ds.is_negative.sum())}), фолд {fold}, "
        f"источник={common['source']}"
    )
    return train_loader, val_loader, val_df


def run(cfg: Cfg, resume: str | None = None) -> dict:
    # Бюджет вычислений — самое первое, что проверяется: прогон вне лимита
    # получил бы 0 за этап, и тратить на него ни данные, ни карту, ни папку в
    # runs/ незачем. Пометка `budget.exempt` пропускает исследовательские
    # прогоны — но она же уедет в summary.json, и сабмит её не признает.
    verdict = budget.check(cfg)
    if not verdict.ok:
        raise ValueError(budget.rejection_text(cfg, verdict))

    seed_everything(int(cfg.get("seed", 42)), bool(cfg.get("deterministic", False)))
    name = cfg.get("name") or f"{cfg.model.get('arch')}-{config_hash(cfg)}"
    run_dir = make_run_dir(str(name), resume=bool(resume))
    logger = RunLogger(run_dir, use_tensorboard=bool(cfg.get("tensorboard", True)))
    save_config(cfg, run_dir / "config.yaml")
    # снапшота конфига мало: тот же конфиг на другой версии timm даёт другую сеть
    write_environment(run_dir)
    logger.info(f"прогон: {run_dir.name}  ->  {run_dir}")
    logger.info(f"бюджет: {verdict.text}")

    device = pick_device(str(cfg.get("device", "auto")))
    train_loader, val_loader, val_df = build_dataloaders(cfg, logger)

    model = build_model(cfg.model).to(device, memory_format=torch.channels_last)
    logger.info(f"модель: {cfg.model.get('arch')}/{cfg.model.get('encoder')} {count_parameters(model)}")

    criterion = build_loss(cfg.loss, parse_aux_spec(cfg.model.get("aux_heads"))).to(device)
    optimizer = build_optimizer(model, cfg.train)
    scheduler = build_scheduler(optimizer, cfg.train, len(train_loader))
    amp = str(cfg.train.get("amp", "fp16"))
    scaler = torch.amp.GradScaler("cuda", enabled=(amp == "fp16" and device.type == "cuda"))
    ema = ModelEma(model, float(cfg.train.get("ema_decay", 0.999))) if cfg.train.get("ema") else None

    start_epoch, best = 0, -1.0
    if resume:
        state = torch.load(resume, map_location=device, weights_only=False)
        start_epoch, best = _restore_state(state, model, optimizer, scheduler, ema, scaler)
        logger.info(f"продолжаю с эпохи {start_epoch}, лучший AIC пока {best:.4f}")

    settings = stats_settings(cfg)
    reference = None
    if settings is not None:
        reference = load_reference(resolve_reference(settings.reference))
        logger.info(
            f"эталон: {reference.name}, операционная точка {reference.op}, "
            f"train_sigma={settings.train_sigma}"
        )
        if len(reference.curve[0]) == 0:
            logger.info(
                "  у эталона не задан data.epoch_size — кривой по числу показов нет, "
                "онлайн-гейт выключен (итоговое сравнение считается как обычно)"
            )
    gate_stop = False

    epochs = int(cfg.train.get("epochs", 10))
    patience = int(cfg.train.get("early_stop", 0))
    since_improved = 0
    best_result = None
    epoch = start_epoch - 1  # если цикл не выполнится ни разу (resume на последней эпохе)

    calib = dict(cfg.get("calib", {}) or {})
    n_bins = int(calib.get("n_bins", 256))
    mask_grid = list(calib.get("mask_grid", DEFAULT_MASK_GRID))
    cls_grid = list(calib.get("cls_grid", DEFAULT_CLS_GRID))
    area_grid = list(calib.get("area_grid", DEFAULT_AREA_GRID))

    for epoch in range(start_epoch, epochs):
        train_stats = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            scaler=scaler, scheduler=scheduler, amp=amp,
            accum_steps=int(cfg.train.get("accum_steps", 1)),
            grad_clip=float(cfg.train.get("grad_clip", 0.0)),
            ema=ema, epoch=epoch, logger=logger,
            log_every=int(cfg.train.get("log_every", 50)),
            max_steps=cfg.train.get("max_train_steps"),
        )

        eval_model = ema.module if ema is not None else model
        accumulator, val_stats = validate(
            eval_model, val_loader, criterion, device, amp=amp,
            n_bins=n_bins, max_steps=cfg.train.get("max_val_steps"),
        )
        at_half = accumulator.evaluate(mask_threshold=0.5)
        tuned = accumulator.best(mask_grid, cls_grid, area_grid)

        per_epoch = int(cfg.data.get("epoch_size") or len(train_loader.dataset))
        gate = None
        if reference is not None:
            gate = gate_check(
                reference.curve,
                samples=(epoch + 1) * per_epoch,
                aic=tuned.aic,
                gate_delta=settings.gate_delta,
                after_samples=settings.gate_after_samples,
            )

        metrics = {
            **{f"train/{k}": v for k, v in train_stats.items()},
            "val/loss": val_stats["val_loss"],
            "val/aic@0.5": at_half.aic,
            "val/dice@0.5": at_half.dice_pos,
            "val/fpr@0.5": at_half.fpr_neg,
            "val/aic_tuned": tuned.aic,
            "val/dice_tuned": tuned.dice_pos,
            "val/fpr_tuned": tuned.fpr_neg,
            "val/best_thr": tuned.mask_threshold,
            "val/best_cls_thr": tuned.cls_threshold,
            "val/best_min_area": tuned.min_area,
            "gpu_gb": round(gpu_memory_gb(), 2),
            **gate_metrics(gate),
        }
        logger.log_metrics(epoch, metrics)
        nan_note = (
            f" | ПРОПУЩЕНО nan-шагов: {train_stats['nan_steps']}"
            if train_stats.get("nan_steps") else ""
        )
        logger.info(
            f"эпоха {epoch}: train_loss={train_stats['loss']:.4f} "
            f"val_loss={val_stats['val_loss']:.4f} | {tuned}{nan_note}"
        )
        if gate is not None and gate.ref_aic is not None:
            logger.info(
                f"  эталон {reference.name} @{(epoch + 1) * per_epoch} показов = "
                f"{gate.ref_aic:.4f}   {gate.reason}"
                + ("   ГЕЙТ СРАБОТАЛ" if gate.fired else "")
            )

        improved = tuned.aic > best
        if improved:
            best, best_result, since_improved = tuned.aic, tuned, 0
        else:
            since_improved += 1

        # `best` в чекпоинте учитывает и ТЕКУЩУЮ эпоху. Раньше last.pt писался до
        # сравнения и хранил рекорд предыдущих эпох: resume с такого файла начинал
        # с заниженной планки и на первом же улучшении относительно неё затирал
        # best.pt моделью хуже той, что там уже лежала.
        state = _checkpoint_state(
            model, ema, optimizer, scheduler, cfg,
            epoch=epoch, best=best, calib=tuned.as_dict(), scaler=scaler,
        )
        torch.save(state, run_dir / "ckpt" / "last.pt")

        if improved:
            torch.save(state, run_dir / "ckpt" / "best.pt")
            accumulator.save(run_dir / "oof" / "val.npz")
            val_df.to_parquet(run_dir / "oof" / "val_rows.parquet", index=False)
            logger.info(f"  новый лучший AIC {best:.4f} -> ckpt/best.pt")
        elif patience and since_improved >= patience:
            logger.info(f"ранняя остановка: {patience} эпох без улучшения")
            break

        # снимаем ПОСЛЕ сохранения чекпоинта: у снятого плеча всё равно должны
        # остаться его артефакты, иначе разбираться в причине будет не по чему
        if gate is not None and gate.fired and settings.gate_action == "stop":
            logger.info("снимаю прогон по гейту (stats.gate_action=stop)")
            gate_stop = True
            break

    summary = {
        "run": run_dir.name,
        "best_aic": best,
        "best": best_result.as_dict() if best_result else None,
        "config": str(cfg.get("_source", "")),
        "epochs_done": epoch + 1,
        "status": "killed" if gate_stop else "ok",
        # чтобы в `board` цена прогона стояла рядом с его AIC: прирост, купленный
        # выходом за лимит, виден сразу, а не после отдельного пересчёта
        "budget": verdict.as_dict(),
    }
    if gate_stop:
        summary["killed_reason"] = f"гейт по эталону {reference.name} на эпохе {epoch}"

    if reference is not None and best_result is not None and not gate_stop:
        saved = AICAccumulator.load(run_dir / "oof" / "val.npz")
        # Сопоставление идёт по stem'ам, а в аккумуляторе кадры лежат в порядке
        # val_df. Усечённая валидация (train.max_val_steps) даёт кадров меньше,
        # чем строк, и тогда индексы разъезжаются — считать вердикт по огрызку
        # выборки всё равно нельзя, поэтому честнее сказать это вслух.
        if len(saved) != len(val_df):
            logger.info(
                f"сравнение с эталоном пропущено: провалидировано {len(saved)} кадров "
                f"из {len(val_df)} (train.max_val_steps)"
            )
        else:
            comparison = compare_to_reference(
                saved,
                val_df["stem"].to_numpy(),
                dict(cfg),
                reference,
                own_op=(best_result.mask_threshold, best_result.cls_threshold,
                        best_result.min_area),
                train_sigma=settings.train_sigma,
                bootstrap_n=settings.bootstrap_n,
                bootstrap_seed=settings.bootstrap_seed,
            )
            summary["verdict"] = comparison.as_dict()
            for line in comparison.report(reference.name):
                logger.info(line)

    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.log_hparams(flatten(dict(cfg)), {"best_aic": best})
    logger.info(f"готово. лучший AIC={best:.4f}. Всё в {run_dir}")
    logger.close()
    return summary
