"""План — очередь экспериментов, описанная одним YAML-файлом.

Зачем. Прогон на 8 ГБ идёт часами, а гипотез за раз проверяется несколько.
Ставить их руками по одной значит либо сидеть у машины, либо терять ночь между
прогонами. План описывает всю серию сразу, запускается одной командой и
доводит очередь до конца, даже если какой-то прогон упал.

Формат:

    name: small-masks          # префикс имён прогонов, необязателен
    defaults:                  # переопределения, общие для всей очереди
      train.epochs: 6
    runs:
      - config: e0_control
      - config: e1_res768
      - config: e2_area
        name: e2-heavy         # иначе имя берётся из самого конфига
        set:
          loss.area.small_weight: 4.0
      - config: e2_area        # grid раскрывается в декартово произведение
        name: e2-sweep
        grid:
          data.small_area_fraction: [0.3, 0.4, 0.5]

Три вещи, ради которых это отдельный модуль, а не цикл в shell:

* **Префлайт.** Перед первым прогоном все конфиги загружаются и все модели
  собираются на CPU со случайными весами. Опечатка в имени энкодера в пятом
  прогоне вылезает сразу, а не через четыре часа.
* **Изоляция падений.** Упавший прогон не роняет очередь: он помечается
  `failed`, остальные идут дальше.
* **Возобновление.** Прогон, у которого уже есть `runs/<имя>/summary.json`,
  по умолчанию пропускается — очередь можно перезапустить после сбоя и она
  доделает остаток.

Порядок в очереди сохраняется: опорный прогон, стоящий в плане первым, и
посчитается первым.
"""

from __future__ import annotations

import itertools
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import yaml

from .config import load_config
from .workspace import plans_root, runs_root


@dataclass
class PlannedRun:
    """Один прогон очереди: конфиг плюс уже разрешённые переопределения."""

    config: str
    name: str
    overrides: dict[str, Any] = field(default_factory=dict)

    def as_cli_overrides(self) -> list[str]:
        """Переопределения в том же виде, в каком их принимает `-s`."""
        pairs = [f"{key}={_to_yaml_scalar(value)}" for key, value in self.overrides.items()]
        return pairs + [f"name={self.name}"]

    @property
    def run_dir(self) -> Path:
        return runs_root() / self.name

    @property
    def is_done(self) -> bool:
        """Прогон считается завершённым, если дописан summary.json."""
        return (self.run_dir / "summary.json").exists()


def _to_yaml_scalar(value: Any) -> str:
    """Обратно в строку так, чтобы `apply_override` разобрал то же значение."""
    if isinstance(value, str):
        return value
    return yaml.safe_dump(value, default_flow_style=True).strip().rstrip("...").strip()


def _slug(key: str, value: Any) -> str:
    """Кусок имени прогона для одной точки сетки: `small_weight4.0`."""
    tail = str(key).split(".")[-1]
    text = str(value).replace(".", "").replace(" ", "")
    return f"{tail}{text}"


def resolve_plan_path(spec: str | Path) -> Path:
    """Принимает 'series', 'series.yaml' или полный путь."""
    path = Path(spec)
    root = plans_root()
    candidates = [path, path.with_suffix(".yaml"), root / path,
                  root / path.with_suffix(".yaml")]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"план не найден: {spec} (искал в {root})")


def load_plan(spec: str | Path) -> tuple[str, list[PlannedRun]]:
    """YAML -> (имя плана, развёрнутая очередь прогонов)."""
    path = resolve_plan_path(spec)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: ожидался словарь на верхнем уровне")

    plan_name = str(data.get("name") or path.stem)
    defaults = dict(data.get("defaults", {}) or {})
    items = data.get("runs")
    if not items:
        raise ValueError(f"{path}: в плане нет ни одного прогона (ключ `runs`)")

    queue: list[PlannedRun] = []
    for position, item in enumerate(items):
        queue.extend(_expand(item, position, defaults))

    names = [run.name for run in queue]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(
            f"{path}: имена прогонов повторяются: {duplicates}. "
            "Два прогона с одним именем затрут результаты друг друга — "
            "задай `name` явно или разведи их через `grid`"
        )
    return plan_name, queue


def _expand(item: Any, position: int, defaults: dict) -> Iterator[PlannedRun]:
    if isinstance(item, str):
        item = {"config": item}
    if not isinstance(item, dict):
        raise ValueError(f"прогон #{position}: ожидалась строка или словарь, получено {item!r}")

    config = item.get("config")
    if not config:
        raise ValueError(f"прогон #{position}: не задан `config`")

    overrides = {**defaults, **dict(item.get("set", {}) or {})}
    grid = dict(item.get("grid", {}) or {})
    base_name = item.get("name")

    if not grid:
        yield PlannedRun(str(config), str(base_name or _name_from_config(config, overrides)),
                         overrides)
        return

    for values in itertools.product(*grid.values()):
        point = dict(zip(grid.keys(), values))
        suffix = "-".join(_slug(key, value) for key, value in point.items())
        stem = base_name or _name_from_config(config, overrides)
        yield PlannedRun(str(config), f"{stem}-{suffix}", {**overrides, **point})


def _name_from_config(config: str, overrides: dict) -> str:
    """Имя из самого конфига — то же, что дал бы обычный `train -c ...`."""
    if "name" in overrides:
        return str(overrides["name"])
    cfg = load_config(config)
    name = cfg.get("name")
    if not name:
        raise ValueError(
            f"у конфига {config} нет `name`, а в плане имя не задано — "
            "прогоны разъедутся по папкам со случайными хэшами"
        )
    return str(name)


def preflight(queue: list[PlannedRun]) -> list[str]:
    """Собрать каждый конфиг и каждую модель на CPU, проверить бюджет вычислений.

    Возвращает список проблем. Веса всегда None: проверяется форма архитектуры и
    разбор конфига, а не качество, и ходить в сеть за предобученными весами
    здесь незачем.

    Одинаковые (модель, лосс) собираются один раз: в свипе по `data.*` все
    точки сетки дают одну и ту же сеть, и пересобирать её десять раз незачем.
    А вот бюджет считается у КАЖДОЙ точки — сеть одна, но стоимость изображения
    зависит ещё и от `data.size`, и свип по разрешению разъезжается по лимиту.
    """
    from .budget import check as check_budget, rejection_text
    from .losses import build_loss
    from .models import build_model

    problems: list[str] = []
    seen: set[str] = set()
    for run in queue:
        try:
            cfg = load_config(run.config, run.as_cli_overrides() + ["model.encoder_weights=null"])
            signature = json.dumps([cfg.model, cfg.loss], sort_keys=True, default=str)
            if signature not in seen:
                build_model(cfg.model)
                build_loss(cfg.loss)
                seen.add(signature)

            verdict = check_budget(cfg)
            if not verdict.ok:
                problems.append(f"{run.name} ({run.config}): {rejection_text(cfg, verdict)}")
        except Exception as error:  # noqa: BLE001 — здесь важен не тип, а текст
            problems.append(f"{run.name} ({run.config}): {type(error).__name__}: {error}")
    return problems


def describe(plan_name: str, queue: list[PlannedRun], skip_done: bool = True) -> str:
    lines = [f"план `{plan_name}`: {len(queue)} прогонов", ""]
    for position, run in enumerate(queue, 1):
        mark = "готов" if (skip_done and run.is_done) else "будет запущен"
        overrides = " ".join(f"{k}={_to_yaml_scalar(v)}" for k, v in run.overrides.items())
        lines.append(f"  {position:>2}. {run.name:<28} -c {run.config:<16} {overrides}  [{mark}]")
    return "\n".join(lines)


def run_plan(
    spec: str | Path,
    *,
    skip_done: bool = True,
    stop_on_error: bool = False,
    only: list[str] | None = None,
    check_first: bool = True,
    logger=None,
) -> dict:
    """Пройти очередь до конца. Возвращает сводку по каждому прогону.

    `skip_done` пропускает прогоны с готовым summary.json — так очередь
    доделывается после сбоя без повторного счёта уже посчитанного.
    `stop_on_error` по умолчанию выключен: одна опечатка в середине плана не
    должна стоить всей ночи.
    `check_first` выключается, если префлайт уже сделан снаружи (так делает CLI,
    чтобы не собирать все модели дважды).
    """
    from .train import run as train_run

    say = logger or print
    plan_name, queue = load_plan(spec)
    if only:
        wanted = set(only)
        queue = [run for run in queue if run.name in wanted or run.config in wanted]
        if not queue:
            raise ValueError(f"фильтр {sorted(wanted)} не выбрал ни одного прогона плана")

    problems = preflight(queue) if check_first else []
    if problems:
        raise ValueError("префлайт не прошёл:\n  " + "\n  ".join(problems))

    results: list[dict] = []
    started = time.time()
    for position, run in enumerate(queue, 1):
        head = f"[{position}/{len(queue)}] {run.name}"
        if skip_done and run.is_done:
            say(f"{head}: уже посчитан, пропускаю")
            summary = json.loads((run.run_dir / "summary.json").read_text(encoding="utf-8"))
            results.append({"name": run.name, "status": "skipped",
                            "best_aic": summary.get("best_aic")})
            continue

        say(f"{head}: старт  (-c {run.config} {' '.join(run.as_cli_overrides())})")
        run_started = time.time()
        try:
            cfg = load_config(run.config, run.as_cli_overrides())
            summary = train_run(cfg)
            results.append({
                "name": run.name, "status": "ok",
                "best_aic": summary.get("best_aic"),
                "seconds": round(time.time() - run_started, 1),
            })
            say(f"{head}: готово, AIC={summary.get('best_aic'):.4f}")
        except Exception as error:  # noqa: BLE001 — очередь важнее одного прогона
            results.append({"name": run.name, "status": "failed",
                            "error": f"{type(error).__name__}: {error}",
                            "seconds": round(time.time() - run_started, 1)})
            say(f"{head}: УПАЛ — {type(error).__name__}: {error}")
            say(traceback.format_exc())
            if stop_on_error:
                break

    report = {
        "plan": plan_name,
        "total_seconds": round(time.time() - started, 1),
        "runs": results,
    }
    out_path = runs_root() / f"_plan_{plan_name}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
