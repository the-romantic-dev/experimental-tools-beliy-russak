"""Единая точка входа во все операции проекта.

Две равноправные формы запуска: `aic <команда>` после `pip install -e .` и
`python cli.py <команда>` из корня воркспейса — второе просто зовёт первое.

    aic env                      проверка среды и железа
    aic index                    train.csv -> artifacts/index.parquet
    aic split                    групповые фолды без утечки
    aic precache --max-side 768  ресайз-кэш для быстрых эпох
    aic profile                  что вообще лежит в данных
    aic smoke                    весь пайплайн на 200 картинках
    aic budget --fit             GFLOPs на кадр против лимита регламента
    aic train -c unet_convnext_512 -s train.lr=3e-4
    aic calibrate runs/<name>    подобрать пороги по OOF
    aic eval runs/<name>         честная метрика в полном разрешении
    aic predict runs/<name> --images <dir> --out <dir>
    aic board                    таблица всех прогонов
    aic viz runs/<name>          картинки: вход / GT / предсказание

Где искать данные и куда писать прогоны, решает `workspace.py`: команда
работает с той папкой, из которой запущена (точнее — с ближайшей вверх, где
есть `configs/`), или с той, что задана в `AIC_WORKSPACE`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import typer

from .workspace import (
    configs_root, ensure_dirs, index_path, runs_root, split_path, submission_template,
    submissions_root, test_csv as workspace_test_csv, workspace,
)

app = typer.Typer(add_completion=False, help="Тулкит экспериментов AIC / детекция манипуляций")


# --------------------------------------------------------------------------
# среда и данные
# --------------------------------------------------------------------------

@app.command()
def env() -> None:
    """Проверить интерпретатор, GPU и наличие ключевых пакетов."""
    import importlib
    import os
    import sys

    import torch

    typer.echo(f"python      {sys.version.split()[0]}  ({sys.executable})")
    typer.echo(f"torch       {torch.__version__}  cuda={torch.cuda.is_available()}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        typer.echo(f"gpu         {props.name}  {props.total_memory / 1e9:.1f} GB")
    typer.echo(f"cpu         {os.cpu_count()} ядер")
    for module in ("timm", "segmentation_models_pytorch", "albumentations", "cv2", "tensorboard"):
        try:
            mod = importlib.import_module(module)
            typer.echo(f"  {module:32s} {getattr(mod, '__version__', 'ok')}")
        except Exception as exc:
            typer.secho(f"  {module:32s} НЕТ ({exc})", fg="red")

    typer.echo(f"воркспейс   {workspace().root}")

    from .registry import PLUGIN_FILE

    plugins = workspace().root / PLUGIN_FILE
    typer.echo(f"плагины     {plugins}  [{'подключены' if plugins.exists() else 'нет файла'}]")

    for path, label in ((index_path(), "индекс"), (split_path(), "фолды")):
        mark = "есть" if path.exists() else "НЕТ — собери его"
        typer.echo(f"{label:12s}{path}  [{mark}]")


@app.command()
def registry(
    kind: Optional[str] = typer.Argument(None, help="показать только один реестр: loss, aug, …"),
) -> None:
    """Что можно писать в конфиге и откуда оно взялось.

    Первое, что стоит спросить, когда `loss.seg` или `train.optimizer` не
    принимает нужное имя, или когда у соседа конфиг работает, а у тебя нет.
    """
    from .registry import ALL_REGISTRIES, PLUGIN_FILE, describe

    contracts = {reg.key: reg.contract for reg in ALL_REGISTRIES}
    table = describe()
    if kind:
        if kind not in table:
            typer.secho(f"нет такого реестра: {kind}; есть {', '.join(table)}", fg="red")
            raise typer.Exit(code=1)
        table = {kind: table[kind]}

    for key, groups in table.items():
        typer.secho(f"\n{key}", fg="cyan", bold=True)
        typer.echo(f"  сигнатура: {contracts[key]}")
        for source, names in groups.items():
            typer.echo(f"  {source + ':':<20} {', '.join(names)}")

    plugins = workspace().root / PLUGIN_FILE
    if not plugins.exists():
        typer.secho(
            f"\nСвои компоненты добавляются файлом {plugins} — см. docs/EXTENDING.md",
            fg="yellow",
        )


@app.command()
def index(
    workers: int = typer.Option(12, help="процессов на чтение масок"),
    limit: Optional[int] = typer.Option(None, help="только первые N строк (для отладки)"),
) -> None:
    """Собрать индекс датасета: домены, генераторы, группы, площади масок, негативы."""
    from .indexing import build_index, summarize

    ensure_dirs()
    typer.echo("сканирую маски (это единственный долгий шаг, делается один раз)...")
    df = build_index(workers=workers, limit=limit)
    typer.echo(summarize(df))
    typer.secho(f"\nсохранено: {index_path()}", fg="green")


@app.command()
def split(
    folds: int = typer.Option(5, help="число фолдов"),
    seed: int = typer.Option(42),
) -> None:
    """Нарезать групповые стратифицированные фолды (без утечки исходных кадров)."""
    from .splits import make_folds, summarize

    df = make_folds(n_folds=folds, seed=seed)
    typer.echo(summarize(df))
    typer.secho(f"\nсохранено: {split_path()}", fg="green")


@app.command()
def precache(
    max_side: int = typer.Option(768, help="длинная сторона после ресайза"),
    workers: int = typer.Option(12),
    quality: int = typer.Option(95, help="качество JPEG; ниже 90 не стоит — стираются артефакты"),
    limit: Optional[int] = typer.Option(None),
) -> None:
    """Собрать ресайз-кэш. После этого ставь в конфиге data.source=cache."""
    from .indexing import load_index
    from .precache import build_cache, cache_size_gb

    df = load_index()
    if limit:
        df = df.head(limit)
    stats = build_cache(df, max_side=max_side, workers=workers, quality=quality)
    typer.echo(f"{stats}")
    typer.secho(f"кэш s{max_side}: {cache_size_gb(max_side):.1f} ГБ", fg="green")


@app.command()
def profile(
    out: Optional[Path] = typer.Option(None, help="куда сохранить отчёт (.md)"),
) -> None:
    """Профиль датасета: домены, генераторы, площади, разрешения, негативы."""
    from .analysis.profile_data import profile_dataset

    report = profile_dataset()
    typer.echo(report)
    if out:
        Path(out).write_text(report, encoding="utf-8")
        typer.secho(f"отчёт: {out}", fg="green")


# --------------------------------------------------------------------------
# обучение
# --------------------------------------------------------------------------

@app.command()
def train(
    config: str = typer.Option("baseline", "--config", "-c", help="имя или путь конфига"),
    set_: List[str] = typer.Option([], "--set", "-s", help="переопределение: train.lr=3e-4"),
    name: Optional[str] = typer.Option(None, help="имя прогона (папка в runs/)"),
    fold: Optional[int] = typer.Option(None, help="какой фолд валидировать"),
    resume: Optional[Path] = typer.Option(None, help="путь к ckpt/last.pt"),
) -> None:
    """Запустить эксперимент."""
    from .config import load_config
    from .train import run

    overrides = list(set_)
    if name:
        overrides.append(f"name={name}")
    if fold is not None:
        overrides.append(f"train.fold={fold}")

    cfg = load_config(config, overrides)
    summary = run(cfg, resume=str(resume) if resume else None)
    typer.secho(json.dumps(summary, indent=2, ensure_ascii=False), fg="green")


@app.command()
def plan(
    plan_file: str = typer.Argument("series", help="имя плана из plans/ или путь к YAML"),
    dry_run: bool = typer.Option(False, "--dry-run", help="показать очередь и выйти"),
    check: bool = typer.Option(False, "--check", help="только префлайт: собрать все конфиги и модели"),
    force: bool = typer.Option(False, "--force", help="пересчитать даже готовые прогоны"),
    stop_on_error: bool = typer.Option(False, "--stop-on-error", help="оборвать очередь на первом падении"),
    only: List[str] = typer.Option([], "--only", help="взять только эти прогоны (по имени или конфигу)"),
) -> None:
    """Запустить очередь экспериментов, описанную одним YAML-файлом.

    Очередь идёт до конца: упавший прогон помечается и не роняет остальные.
    Прогоны с готовым summary.json пропускаются, поэтому после сбоя план
    можно просто перезапустить — он доделает остаток.

    Перед стартом всегда выполняется префлайт: каждый конфиг загружается и
    каждая модель собирается на CPU. Опечатка в пятом прогоне вылезет сразу,
    а не через четыре часа.
    """
    from .plans import describe, load_plan, preflight, run_plan

    plan_name, queue = load_plan(plan_file)
    if only:
        wanted = set(only)
        queue = [r for r in queue if r.name in wanted or r.config in wanted]

    typer.echo(describe(plan_name, queue, skip_done=not force))

    if dry_run:
        raise typer.Exit()

    problems = preflight(queue)
    if problems:
        typer.secho("\nпрефлайт не прошёл:", fg="red")
        for problem in problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(code=1)
    typer.secho(f"\nпрефлайт пройден: {len(queue)} конфигов собираются", fg="green")

    if check:
        raise typer.Exit()

    report = run_plan(
        plan_file, skip_done=not force, stop_on_error=stop_on_error,
        only=list(only) or None, check_first=False, logger=typer.echo,
    )

    typer.echo("\nитог:")
    for item in report["runs"]:
        aic_value = item.get("best_aic")
        line = f"  {item['name']:<28} {item['status']:<8}"
        line += f" AIC={aic_value:.4f}" if isinstance(aic_value, float) else ""
        if item.get("error"):
            line += f"  {item['error']}"
        typer.echo(line)

    failed = [item for item in report["runs"] if item["status"] == "failed"]
    colour = "red" if failed else "green"
    typer.secho(f"\nплан `{report['plan']}` завершён за {report['total_seconds']:.0f} с, "
                f"падений: {len(failed)}", fg=colour)
    if failed:
        raise typer.Exit(code=1)


@app.command()
def probe(
    configs: List[str] = typer.Argument(..., help="имена конфигов, например e0_control e1_res768"),
    set_: List[str] = typer.Option([], "--set", "-s", help="переопределения, как в train"),
    steps: int = typer.Option(3, help="шагов обучения для выхода на устойчивый пик"),
) -> None:
    """Измерить пиковую VRAM для конфигов, не запуская обучение.

    Отвечает на вопрос «влезет ли в 8 ГБ» до того, как ставить прогон на часы.
    Считать надо на свободной карте: команда печатает, сколько было свободно
    на старте, и если там занято посторонним, вердикт будет заниженным.
    """
    import torch

    from .config import load_config
    from .probe import probe_memory, verdict

    if not torch.cuda.is_available():
        typer.secho("CUDA недоступна — мерить нечего", fg="red")
        raise typer.Exit(code=1)

    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    total_gb = torch.cuda.mem_get_info()[1] / 1e9
    typer.echo(f"на карте свободно {free_gb:.2f} из {total_gb:.2f} ГБ\n")
    # на Windows рабочий стол и драйвер сами держат ~1 ГБ — это норма,
    # ругаться имеет смысл только на заметно большее постороннее потребление
    if free_gb < total_gb - 1.6:
        typer.secho(
            f"посторонним занято {total_gb - free_gb:.1f} ГБ — освободи карту, "
            "иначе замер упрётся в чужую память, а не в реальный пик\n",
            fg="yellow",
        )

    rows = []
    for name in configs:
        cfg = load_config(name, list(set_))
        try:
            result = probe_memory(cfg, steps=steps)
        except torch.cuda.OutOfMemoryError:
            typer.secho(f"{name:18s} OOM уже на замере — точно не влезет", fg="red")
            torch.cuda.empty_cache()
            continue
        fits, text = verdict(result)
        rows.append({"config": name, **result})
        colour = "green" if fits else "red"
        typer.echo(
            f"{name:18s} {result['size']}px bs={result['bs']}/{result['val_bs']} "
            f"{result['params_m']:.1f}M  train {result['peak_train_reserved_gb']:.2f} ГБ  "
            f"val {result['peak_val_gb']:.2f} ГБ"
        )
        typer.secho(f"{'':18s} {text}", fg=colour)

    if len(rows) > 1:
        import pandas as pd

        typer.echo("\n" + pd.DataFrame(rows).to_string(index=False))


@app.command()
def budget(
    configs: List[str] = typer.Argument(None, help="имена конфигов; без аргументов — все из configs/"),
    tta: str = typer.Option("none", help="рецепт TTA, как в eval/submit: none | hflip,vflip"),
    models: int = typer.Option(1, help="сколько моделей в ансамбле"),
    fit: bool = typer.Option(False, "--fit", help="дописать, какой вход ещё влезает"),
    weights: bool = typer.Option(False, "--weights", help="показать источники предобученных весов"),
) -> None:
    """Сколько строгих GFLOPs стоит одно изображение и влезает ли это в лимит.

    Лимит регламента — 100 GFLOPs, причём на ИЗОБРАЖЕНИЕ, а не на forward:
    `--tta hflip,vflip` удваивает счёт, ансамбль из двух прогонов удваивает ещё
    раз. Считает `torch.utils.flop_counter.FlopCounterMode` — тот самый счётчик,
    который регламент называет источником правды в спорных случаях. Карта не
    нужна: forward идёт на meta-устройстве.
    """
    from .budget import LIMIT_GFLOPS, check, largest_fitting_size
    from .compliance import pretrained_sources
    from .config import load_config

    names = list(configs) if configs else sorted(
        p.stem for p in configs_root().glob("*.yaml") if not p.name.startswith("_")
    )
    views = len([part for part in tta.split(",") if part.strip()]) or 1
    recipe = f"{views} вид(ов) TTA x {models} модель(ей)" if views > 1 or models > 1 else "один forward"
    typer.echo(f"лимит {LIMIT_GFLOPS:.0f} строгих GFLOPs на изображение; рецепт: {recipe}\n")

    over = 0
    for name in names:
        try:
            cfg = load_config(name)
            verdict = check(cfg, n_views=views, n_models=models)
        except Exception as error:  # noqa: BLE001 — конфиг соседа не должен ронять таблицу
            typer.secho(f"{name:24s} не собрался: {type(error).__name__}: {error}", fg="red")
            continue

        note = "в бюджете" if verdict.within_limit else f"ВНЕ, x{verdict.gflops / verdict.limit:.2f}"
        if verdict.exempt:
            note += " (помечен exempt)"
        if fit and not verdict.within_limit:
            side = largest_fitting_size(cfg, n_views=views, n_models=models)
            note += f"; влезает {side}px" if side else "; не влезает ни на каком входе"

        over += not verdict.within_limit
        typer.secho(
            f"{name:24s} {verdict.size:>4d}px {verdict.gflops:>8.1f}  {note}",
            fg="green" if verdict.within_limit else "red",
        )
        if weights:
            sources = pretrained_sources(cfg.model) or ["с нуля, внешних весов нет"]
            for source in sources:
                typer.echo(f"{'':24s}      веса: {source}")

    if over:
        typer.secho(
            f"\nвне бюджета: {over} из {len(names)}. Такой прогон получит 0 за этап; "
            "если он исследовательский — пометь его `budget.exempt: true`",
            fg="yellow",
        )


@app.command()
def smoke(
    config: str = typer.Option("smoke", "--config", "-c"),
    set_: List[str] = typer.Option([], "--set", "-s"),
) -> None:
    """Прогнать весь пайплайн на крошечной подвыборке. Занимает ~1 минуту.

    Это проверка «ничего не сломано», а не обучение. Запускай после любой
    правки кода и перед тем, как ставить длинный прогон.
    """
    from .config import load_config
    from .train import run

    cfg = load_config(config, list(set_))
    summary = run(cfg)
    typer.secho("smoke пройден: " + json.dumps(summary, ensure_ascii=False), fg="green")


# --------------------------------------------------------------------------
# оценка и сабмит
# --------------------------------------------------------------------------

@app.command()
def calibrate(
    run_dir: Path = typer.Argument(..., help="папка прогона в runs/"),
    top: int = typer.Option(10, help="сколько лучших комбинаций показать"),
) -> None:
    """Подобрать пороги по сохранённой статистике валидации (без GPU, за секунды)."""
    from .calibrate import calibrate_run

    result = calibrate_run(run_dir, top_k=top)
    typer.echo(result["table"].to_string(index=False))
    typer.secho(f"\nлучшее: {json.dumps(result['best'], ensure_ascii=False)}", fg="green")
    typer.echo(f"записано в {Path(run_dir) / 'calib.json'}")


@app.command()
def eval(
    run_dir: Path = typer.Argument(..., help="папка прогона в runs/"),
    checkpoint: str = typer.Option("best", help="best|last или путь к .pt"),
    tta: str = typer.Option("none", help="none | hflip,vflip | none,hflip,vflip,hvflip"),
    limit: Optional[int] = typer.Option(None, help="ограничить число кадров валидации"),
    batch_size: int = typer.Option(8),
    workers: int = typer.Option(6),
    ema: bool = typer.Option(True, help="брать веса EMA; --no-ema возьмёт сырые"),
) -> None:
    """Честная метрика в ИСХОДНОМ разрешении (в train.py она считается на сетке модели)."""
    import pandas as pd

    from .inference import evaluate_full_res
    from .metrics import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID

    run_dir = Path(run_dir)
    ckpt = Path(checkpoint) if checkpoint.endswith(".pt") else run_dir / "ckpt" / f"{checkpoint}.pt"
    rows_path = run_dir / "oof" / "val_rows.parquet"
    if not rows_path.exists():
        raise typer.BadParameter(f"нет {rows_path}: прогон не доучился до первой валидации")

    df = pd.read_parquet(rows_path)
    if limit:
        df = df.head(limit)

    accumulator = evaluate_full_res(
        ckpt, df, tta=tuple(t.strip() for t in tta.split(",")),
        batch_size=batch_size, num_workers=workers, use_ema=ema,
    )
    accumulator.save(run_dir / "oof" / "val_fullres.npz")
    best = accumulator.best(list(DEFAULT_MASK_GRID), list(DEFAULT_CLS_GRID), list(DEFAULT_AREA_GRID))
    typer.echo(f"при пороге 0.5:  {accumulator.evaluate(0.5)}")
    typer.secho(f"после подбора:   {best}", fg="green")


@app.command()
def predict(
    run_dir: Path = typer.Argument(..., help="папка прогона в runs/"),
    images: Path = typer.Option(..., help="папка с входными изображениями"),
    out: Path = typer.Option(..., help="куда писать PNG-маски"),
    checkpoint: str = typer.Option("best"),
    thr: Optional[float] = typer.Option(None, help="порог маски; по умолчанию из calib.json"),
    cls_thr: Optional[float] = typer.Option(None, help="порог классификатора"),
    min_area: Optional[float] = typer.Option(None, help="минимальная доля площади"),
    tta: str = typer.Option("none"),
    batch_size: int = typer.Option(8),
    workers: int = typer.Option(6),
    limit: Optional[int] = typer.Option(None),
    ema: bool = typer.Option(True, help="брать веса EMA; --no-ema возьмёт сырые"),
) -> None:
    """Сгенерировать бинарные маски 0/255 в исходном разрешении."""
    from .inference import predict_folder

    run_dir = Path(run_dir)
    calib_path = run_dir / "calib.json"
    calib = json.loads(calib_path.read_text(encoding="utf-8")) if calib_path.exists() else {}
    if not calib:
        typer.secho(
            "calib.json не найден — беру пороги по умолчанию. "
            "Сначала лучше запустить: python cli.py calibrate " + str(run_dir),
            fg="yellow",
        )

    ckpt = Path(checkpoint) if checkpoint.endswith(".pt") else run_dir / "ckpt" / f"{checkpoint}.pt"
    stats = predict_folder(
        ckpt, images, out,
        mask_threshold=thr if thr is not None else calib.get("mask_threshold", 0.5),
        cls_threshold=cls_thr if cls_thr is not None else calib.get("cls_threshold", 0.0),
        min_area=min_area if min_area is not None else calib.get("min_area", 0.0),
        tta=tuple(t.strip() for t in tta.split(",")),
        batch_size=batch_size, num_workers=workers, limit=limit, use_ema=ema,
    )
    typer.secho(json.dumps(stats, indent=2, ensure_ascii=False), fg="green")


@app.command()
def submit(
    run_dirs: List[Path] = typer.Argument(..., help="одна или несколько папок прогонов (ансамбль)"),
    name: Optional[str] = typer.Option(None, help="имя сабмита; по умолчанию по имени прогона"),
    test_csv: Optional[Path] = typer.Option(None, help="таблица тестовой выборки; по умолчанию из воркспейса"),
    template: Optional[Path] = typer.Option(None, help="шаблон submission.csv; по умолчанию из воркспейса"),
    checkpoint: str = typer.Option("best"),
    thr: Optional[float] = typer.Option(None, help="порог маски; по умолчанию из calib.json"),
    cls_thr: Optional[float] = typer.Option(None, help="порог классификатора"),
    min_area: Optional[float] = typer.Option(None, help="минимальная доля площади"),
    tta: str = typer.Option("none"),
    batch_size: int = typer.Option(8),
    workers: int = typer.Option(6),
    limit: Optional[int] = typer.Option(None, help="только первые N изображений (для проверки)"),
    ema: bool = typer.Option(True, help="брать веса EMA; --no-ema возьмёт сырые"),
    zip_: bool = typer.Option(True, "--zip/--no-zip", help="упаковать в submission.zip"),
    check: bool = typer.Option(True, help="проверить результат сразу после сборки"),
) -> None:
    """Собрать сабмит: маски + submission.csv + zip, с проверкой формата."""
    from .submission import build_submission, pack_zip, validate_submission

    ensure_dirs()
    test_csv = test_csv or workspace_test_csv()
    template = template or submission_template()
    submit_name = name or (run_dirs[0].name if len(run_dirs) == 1
                           else f"ens_{len(run_dirs)}_{run_dirs[0].name}")
    out_dir = submissions_root() / submit_name

    stats = build_submission(
        run_dirs, out_dir,
        test_csv=test_csv, template=template, checkpoint=checkpoint,
        mask_threshold=thr, cls_threshold=cls_thr, min_area=min_area,
        tta=tuple(t.strip() for t in tta.split(",")),
        batch_size=batch_size, num_workers=workers, limit=limit, use_ema=ema,
        progress=typer.echo,
    )
    typer.echo(json.dumps(stats, indent=2, ensure_ascii=False))

    if stats["flagged_share"] > 0.9:
        typer.secho(
            f"ВНИМАНИЕ: у {stats['flagged_share'] * 100:.0f}% кадров площадь маски >= 1%. "
            "Если в тесте есть чистые кадры, FPR_neg будет высоким — подними cls_thr.",
            fg="yellow",
        )

    target = out_dir
    if zip_ and not limit:
        target = pack_zip(out_dir)
        typer.secho(f"архив: {target}", fg="green")
    elif limit:
        typer.secho("частичный прогон (--limit) — архив не собираю", fg="yellow")

    if check:
        report = validate_submission(target, test_csv=test_csv, partial=bool(limit))
        if report["ok"]:
            typer.secho("проверка формата пройдена", fg="green")
        else:
            typer.secho("ПРОБЛЕМЫ:", fg="red")
            for problem in report["problems"]:
                typer.echo(f"  - {problem}")


@app.command("check-submission")
def check_submission(
    path: Path = typer.Argument(..., help="папка сабмита или submission.zip"),
    test_csv: Optional[Path] = typer.Option(None, help="таблица тестовой выборки; по умолчанию из воркспейса"),
    max_problems: int = typer.Option(20),
    no_sizes: bool = typer.Option(False, help="не сверять размеры масок с изображениями (быстрее)"),
    partial: bool = typer.Option(False, help="не требовать полноты состава (для сборок с --limit)"),
) -> None:
    """Проверить готовый сабмит до загрузки: состав, формат PNG, размеры, значения."""
    from .submission import validate_submission

    report = validate_submission(
        path, test_csv=test_csv, max_problems=max_problems,
        check_sizes=not no_sizes, partial=partial,
    )
    problems = report.pop("problems")
    typer.echo(json.dumps(report, indent=2, ensure_ascii=False))
    if report["ok"]:
        typer.secho("\nсабмит валиден", fg="green")
    else:
        typer.secho("\nПРОБЛЕМЫ:", fg="red")
        for problem in problems:
            typer.echo(f"  - {problem}")
        raise typer.Exit(code=1)


# --------------------------------------------------------------------------
# сравнение и визуализация
# --------------------------------------------------------------------------

@app.command()
def board(
    sort: str = typer.Option("best_aic", help="по какой колонке сортировать"),
    diff: bool = typer.Option(True, help="показывать только различающиеся параметры конфигов"),
) -> None:
    """Таблица всех прогонов: AIC, Dice, FPR, пороги, чем конфиги отличаются."""
    from .analysis.leaderboard import leaderboard

    table = leaderboard(sort_by=sort, only_diff=diff)
    if table.empty:
        typer.secho("в runs/ пока пусто", fg="yellow")
        raise typer.Exit()
    typer.echo(table.to_string(index=False))


@app.command()
def report(
    run_dir: Path = typer.Argument(..., help="runs/<имя>"),
    vs: List[Path] = typer.Option([], "--vs", help="сравнить с другими прогонами"),
) -> None:
    """Где теряется Dice: корзины по площади GT, домены, генераторы, потолки.

    Итоговый AIC — одно число, и по нему не видно, сработала ли гипотеза:
    e1 целится в домены с большими картинками, e2 — в мелкие маски, e3 — в
    полные промахи. Все три двигают разные куски выборки.
    """
    from .analysis.oof_report import compare, report as build_report

    typer.echo(build_report(run_dir))
    if vs:
        typer.echo("\nсравнение по корзинам площади:")
        typer.echo(compare([run_dir, *vs]).to_string(index=False))


@app.command()
def viz(
    run_dir: Path = typer.Argument(...),
    n: int = typer.Option(12, help="сколько примеров"),
    out: Optional[Path] = typer.Option(None, help="куда сохранить png"),
    checkpoint: str = typer.Option("best"),
    worst: bool = typer.Option(True, help="показывать худшие по Dice, а не случайные"),
    scan: int = typer.Option(240, help="сколько кадров прогнать, чтобы выбрать из них n"),
    thr: Optional[float] = typer.Option(None, help="порог; по умолчанию из calib.json"),
) -> None:
    """Сетка «вход / GT / предсказание» — глазами понять, где модель ошибается."""
    from .analysis.viz import make_grid

    path = make_grid(
        run_dir, n=n, out=out, checkpoint=checkpoint, worst=worst, scan=scan, threshold=thr
    )
    typer.secho(f"сохранено: {path}", fg="green")


@app.command()
def history(
    runs: List[str] = typer.Argument(None, help="имена прогонов; без аргументов — все из runs/"),
    metrics: Optional[str] = typer.Option(
        None, help="панели через запятую; несколько величин на одной панели — через +"
    ),
    out: Optional[Path] = typer.Option(None, help="куда сохранить PNG"),
    table: bool = typer.Option(True, help="печатать ещё и итоговую таблицу"),
) -> None:
    """Кривые обучения и валидации по эпохам для нескольких прогонов на общих осях.

    Панели по умолчанию: train+val loss, AIC, Dice_pos, FPR_neg, подобранные
    пороги и learning rate. Прогоны без metrics.jsonl пропускаются.
    """
    from .analysis.compare_runs import DEFAULT_PANELS, compare_table, plot_history

    runs_dir = runs_root()
    requested = list(runs) if runs else sorted(d.name for d in runs_dir.iterdir() if d.is_dir())

    selected = []
    for name in requested:
        if (runs_dir / name / "metrics.jsonl").exists():
            selected.append(name)
        elif runs:  # молчим только про чужие папки в runs/, но не про то, что попросили явно
            typer.secho(f"пропускаю {name}: нет metrics.jsonl", fg="yellow")
    if not selected:
        typer.secho("не нашёл ни одного прогона с metrics.jsonl", fg="yellow")
        raise typer.Exit(code=1)

    panels = [p.strip() for p in metrics.split(",")] if metrics else list(DEFAULT_PANELS)
    figure = plot_history(selected, panels)

    out = Path(out) if out else runs_dir / "_history.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=120, bbox_inches="tight")
    typer.secho(f"график: {out}  ({len(selected)} прогонов)", fg="green")

    if table:
        typer.echo("\n" + compare_table(selected).to_string(index=False))


@app.command()
def curve(
    run_dir: Path = typer.Argument(...),
    cls_thr: float = typer.Option(0.0),
    min_area: float = typer.Option(0.0),
) -> None:
    """Кривая AIC/Dice/FPR по порогу бинаризации."""
    from .calibrate import threshold_curve

    frame = threshold_curve(Path(run_dir) / "oof" / "val.npz", cls_thr, min_area)
    typer.echo(
        frame[["mask_threshold", "aic", "dice_pos", "fpr_neg"]].to_string(index=False)
    )


if __name__ == "__main__":
    app()
