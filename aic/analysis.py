"""Разбор накопленных прогонов: одна таблица по всей папке runs/.

Показывается не весь снапшот, а только те его ключи, которые между прогонами
РАЗЛИЧАЮТСЯ — иначе таблица становится нечитаемой уже на пятом эксперименте.

Модуль ничего не знает про способ обучения: он читает документированную папку
прогона и разворачивает снапшот точечными путями, чем бы тот ни был заполнен.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from .metric import EPS, FP_AREA_THRESHOLD, harmonic_aic
from .runs import Eval, Run

#: ключи снапшота, которые в таблицу не идут никогда: они различаются всегда и
#: ничего не объясняют
HIDE_PREFIXES = ("_source", "name", "seed")


def flatten(obj: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Плоский вид словаря: {'train.lr': 0.0003}."""
    flat: dict[str, Any] = {}
    for key, value in obj.items():
        full = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(flatten(value, prefix=f"{full}."))
        else:
            flat[full] = value
    return flat


def _row(run_dir: Path) -> dict | None:
    """Одна строка лидерборда или None, если это не папка прогона."""
    if not (run_dir / "config.yaml").exists():
        return None

    run = Run.open(run_dir)
    row: dict[str, Any] = {"run": run_dir.name, "best_aic": None}

    summary = run.summary
    if summary:
        best = summary.get("best") or {}
        # прогоны, посчитанные до появления гейта по FLOPs, бюджета не знают —
        # у них тут остаётся пусто, и это честнее, чем подставить ноль
        cost = summary.get("budget") or {}
        row.update({
            "best_aic": summary.get("best_aic"),
            "dice_pos": best.get("dice_pos"),
            "fpr_neg": best.get("fpr_neg"),
            "thr": best.get("mask_threshold"),
            "cls_thr": best.get("cls_threshold"),
            "min_area": best.get("min_area"),
            "epochs": summary.get("epochs_done"),
            "gflops": cost.get("gflops"),
            # именно «вне бюджета», а не «не помечен»: прогон с exempt всё равно
            # вне лимита, и в таблице это должно быть видно
            "over_budget": (not cost["within_limit"]) if "within_limit" in cost else None,
        })

    frame = run.history
    if not frame.empty:
        last = frame.iloc[-1]
        row["last_epoch"] = last.get("step")
        row["elapsed"] = round(float(last.get("elapsed_s", 0) or 0) / 60, 1)
        if row.get("best_aic") is None and "val/aic_tuned" in frame.columns:
            row["best_aic"] = float(frame["val/aic_tuned"].max())

    row["_cfg"] = flatten(run.snapshot)
    return row


def _spread_configs(rows: list[dict], only_diff: bool) -> None:
    """Разложить снапшоты по колонкам, оставив (по умолчанию) только различия."""
    configs = [row.pop("_cfg") for row in rows]
    keys = sorted({k for cfg in configs for k in cfg})
    if only_diff:
        keys = [
            k for k in keys
            if not k.startswith(HIDE_PREFIXES)
            and len({str(cfg.get(k)) for cfg in configs}) > 1
        ]

    # колонку зовём коротко, только если короткое имя ни с чем не сталкивается:
    # у `loss.seg.bce` и `loss.area.small_seg.bce` последний кусок общий, и одна
    # колонка молча показывала бы значение только второго из них
    leaves = Counter(key.rsplit(".", 1)[-1] for key in keys)
    labels = {key: (leaf if leaves[leaf] == 1 else key)
              for key, leaf in ((k, k.rsplit(".", 1)[-1]) for k in keys)}

    for row, cfg in zip(rows, configs):
        for key in keys:
            row[labels[key]] = cfg.get(key)


def leaderboard(
    runs_root: str | Path,
    *,
    sort_by: str = "best_aic",
    only_diff: bool = True,
) -> pd.DataFrame:
    """Таблица по всем прогонам в папке, отсортированная по метрике."""
    runs_root = Path(runs_root)
    if not runs_root.exists():
        return pd.DataFrame()

    rows = [r for r in (_row(d) for d in sorted(runs_root.iterdir()) if d.is_dir()) if r]
    if not rows:
        return pd.DataFrame()

    _spread_configs(rows, only_diff)
    table = pd.DataFrame(rows)
    if sort_by in table.columns:
        table = table.sort_values(sort_by, ascending=False, na_position="last")
    return table.reset_index(drop=True)


def compare_table(run_dirs: Sequence[str | Path]) -> pd.DataFrame:
    """Различия снапшотов бок о бок: строки — ключи, колонки — прогоны."""
    dirs = [Path(d) for d in run_dirs]
    configs = [flatten(Run.open(d).snapshot) for d in dirs]
    keys = sorted({k for cfg in configs for k in cfg})
    keys = [
        k for k in keys
        if not k.startswith(HIDE_PREFIXES)
        and len({str(cfg.get(k)) for cfg in configs}) > 1
    ]
    return pd.DataFrame(
        {d.name: [cfg.get(k) for k in keys] for d, cfg in zip(dirs, configs)},
        index=keys,
    )


def history(
    run_dirs: Sequence[str | Path],
    metric: str = "val/aic_tuned",
) -> pd.DataFrame:
    """Кривые нескольких прогонов в одной длинной таблице — для графика."""
    frames = []
    for run_dir in run_dirs:
        run = Run.open(run_dir)
        frame = run.history
        if frame.empty or metric not in frame.columns:
            continue
        part = frame[["step", metric]].copy()
        if "samples" in frame.columns:
            part["samples"] = frame["samples"]
        part.insert(0, "run", run.dir.name)
        frames.append(part)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


#: ниже этого Dice предсказание считаем полным промахом, а не неточностью
MISS_DICE = 0.05
#: границы корзин по доле площади GT. Главный разрез: у baseline кадры с маской
#: меньше 1% дают Dice 0.135 и 71% полных промахов, а больше 12% — 0.86+ и 2%
AREA_BINS = (0.0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1.01)


class OofView:
    """Развёрнутая валидация прогона: где теряется Dice и каков потолок.

    Ни модели, ни GPU, ни повторного прогона не нужно — в аккумуляторе лежат
    гистограммы по каждому кадру, из которых восстанавливается |P_t| и |P_t ∩ G|
    для любого порога.

    Таблицы берутся из `AICAccumulator.tables()`, а не разворачиваются заново.
    Второй реализации той же арифметики здесь быть не должно: разойдясь, они
    заставили бы считать вердикты не по той метрике, по которой отбирают модели.
    """

    def __init__(self, ev: Eval, rows: pd.DataFrame | None = None) -> None:
        self.ev = ev
        self.rows = rows
        pred, inter, gt_sum, n_pixels, cls_prob = ev.acc.tables()
        self.n_bins = ev.acc.n_bins
        self.pred = pred.astype(np.float64)
        self.inter = inter.astype(np.float64)
        self.gt_sum = gt_sum
        self.n_pixels = n_pixels
        self.cls_prob = cls_prob
        self.area = self.pred / self.n_pixels[:, None]
        self.dice = 2.0 * self.inter / (self.pred + self.gt_sum[:, None] + EPS)
        self.is_pos = self.gt_sum > 0
        self.gt_frac = self.gt_sum / self.n_pixels

    @classmethod
    def from_run(cls, run_dir: str | Path, name: str = "val") -> "OofView":
        run = Run.open(run_dir)
        return cls(run.load_eval(name), run.load_rows(name))

    def _bin_index(self, mask_threshold: float) -> int:
        # ровно как в AICAccumulator.sweep, иначе разрез считался бы в другой точке
        return int(np.clip(int(mask_threshold * self.n_bins), 0, self.n_bins - 1))

    def at(self, op: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
        """`(dice, area)` каждого кадра в операционной точке."""
        mask_threshold, cls_threshold, min_area = op
        k = self._bin_index(mask_threshold)
        keep = (self.cls_prob >= cls_threshold) & (self.area[:, k] >= min_area)
        return (
            np.where(keep, self.dice[:, k], 0.0),
            np.where(keep, self.area[:, k], 0.0),
        )

    def by_area(self, op: tuple[float, float, float]) -> pd.DataFrame:
        """Корзины по доле площади GT — главный разрез, только по позитивам."""
        dice, _ = self.at(op)
        pos = self.is_pos
        bucket = np.digitize(self.gt_frac[pos], AREA_BINS[1:-1])

        records = []
        for i in range(len(AREA_BINS) - 1):
            sel = bucket == i
            values = dice[pos][sel]
            records.append({
                "bucket": f"{AREA_BINS[i] * 100:g}-{AREA_BINS[i + 1] * 100:g}%",
                "n": int(sel.sum()),
                "dice": float(values.mean()) if values.size else 0.0,
                "miss_share": float((values < MISS_DICE).mean()) if values.size else 0.0,
            })
        return pd.DataFrame(records)

    def by_column(self, column: str, op: tuple[float, float, float]) -> pd.DataFrame:
        """Разрез по полю строк валидации: не выиграл ли прогон одним источником."""
        if self.rows is None or column not in self.rows.columns:
            available = [] if self.rows is None else list(self.rows.columns)
            raise ValueError(
                f"нет колонки {column!r} в таблице строк валидации (есть {available}). "
                "Передайте rows в OofView или сохраните их через Run.save_eval"
            )
        dice, area = self.at(op)
        frame = pd.DataFrame({
            column: self.rows[column].to_numpy(),
            "dice": dice,
            "alarm": (area >= FP_AREA_THRESHOLD).astype(float),
            "is_pos": self.is_pos,
        })

        records = {}
        for key, part in frame.groupby(column):
            pos, neg = part[part["is_pos"]], part[~part["is_pos"]]
            dice_pos = float(pos["dice"].mean()) if len(pos) else 0.0
            fpr_neg = float(neg["alarm"].mean()) if len(neg) else 0.0
            records[key] = {
                "n": len(part),
                "n_pos": len(pos),
                "dice_pos": dice_pos,
                "fpr_neg": fpr_neg,
                "aic": harmonic_aic(dice_pos, fpr_neg),
            }
        return pd.DataFrame(records).T

    def ceilings(self, op: tuple[float, float, float]) -> dict[str, float]:
        """Сколько ещё осталось у идеального классификатора и идеального порога.

        Если потолок близко, крутить эту ручку дальше бессмысленно — и это
        единственный способ узнать про неё, не потратив прогон.
        """
        k = self._bin_index(op[0])
        pos, neg = self.is_pos, ~self.is_pos

        dice_now, area_now = self.at(op)
        alarms_now = float((area_now[neg] >= FP_AREA_THRESHOLD).mean()) if neg.any() else 0.0
        current = harmonic_aic(
            float(dice_now[pos].mean()) if pos.any() else 0.0, alarms_now
        )

        # идеальный классификатор кадра: негативы обнулены, позитивы не тронуты
        perfect_cls = harmonic_aic(float(self.dice[pos, k].mean()) if pos.any() else 0.0, 0.0)

        # идеальный порог на кадр: у каждого позитива берётся его лучший Dice,
        # негативы остаются как есть в текущей точке
        best_per_frame = self.dice[pos].max(axis=1) if pos.any() else np.zeros(0)
        perfect_thr = harmonic_aic(
            float(best_per_frame.mean()) if best_per_frame.size else 0.0, alarms_now
        )
        return {"current": current, "perfect_cls": perfect_cls, "perfect_thr": perfect_thr}
