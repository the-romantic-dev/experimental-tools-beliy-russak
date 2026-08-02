"""Разбор валидации прогона: где именно теряется Dice и каков потолок.

Итоговый AIC — одно число, и по нему нельзя понять, сработала гипотеза или нет.
Прогон e2 целится в мелкие маски, e1 — в домены с большими картинками, e3 — в
кадры, которые модель промахивает полностью. Все три двигают РАЗНЫЕ куски
выборки, и сравнивать их надо по этим кускам.

Работает по `runs/<name>/oof/val.npz` (+ `val_rows.parquet`, если он есть):
ни модели, ни GPU, ни повторного прогона не нужно — в npz лежат гистограммы
вероятностей по каждому кадру, из которых восстанавливается |P_t| и |P_t ∩ G|
для любого порога.

Три блока:

* корзины по площади GT — главный разрез, на нём видно, сдвинулись ли мелкие
  маски (у baseline: <1% площади → Dice 0.135 и 71% полных промахов, >12% →
  Dice 0.86+ и 2%);
* домены и генераторы — не выиграл ли прогон на одном источнике за счёт другого;
* потолки — сколько ещё осталось у идеального классификатора кадра и у
  идеального порога на кадр. Если потолок близко, дальше крутить эту ручку
  бессмысленно.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..metrics import DEFAULT_AREA_GRID, FP_AREA_THRESHOLD, harmonic_aic

MISS_DICE = 0.05  # ниже этого предсказание считаем полным промахом, а не неточностью
AREA_BINS = (0.0, 0.01, 0.03, 0.06, 0.12, 0.25, 0.5, 1.01)


class OofView:
    """Развёрнутые из гистограмм таблицы |P_t|, |P_t ∩ G| и производные."""

    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        data = np.load(self.run_dir / "oof" / "val.npz")

        self.n_bins = int(data["n_bins"])
        hist_all = data["hist_all"].astype(np.int64)
        hist_gt = data["hist_gt"].astype(np.int64)
        # |P_t| для t = k / n_bins  <=>  сумма бинов с индексом >= k
        self.pred = np.cumsum(hist_all[:, ::-1], axis=1)[:, ::-1].astype(np.float64)
        self.inter = np.cumsum(hist_gt[:, ::-1], axis=1)[:, ::-1].astype(np.float64)

        self.gt_sum = data["gt_sum"].astype(np.float64)
        self.n_pixels = data["n_pixels"].astype(np.float64)
        self.cls_prob = data["cls_prob"].astype(np.float64)

        self.area = self.pred / self.n_pixels[:, None]
        self.dice = 2.0 * self.inter / (self.pred + self.gt_sum[:, None] + 1e-6)
        self.is_pos = self.gt_sum > 0
        self.gt_frac = self.gt_sum / self.n_pixels

        rows_path = self.run_dir / "oof" / "val_rows.parquet"
        self.rows = pd.read_parquet(rows_path) if rows_path.exists() else None

    def bin_of(self, threshold: float) -> int:
        return int(np.clip(int(threshold * self.n_bins), 0, self.n_bins - 1))

    def keep_mask(self, k: int, cls_threshold: float, min_area: float) -> np.ndarray:
        """Кадры, которые постобработка оставляет непустыми — как в `AICAccumulator.sweep`."""
        return (self.cls_prob >= cls_threshold) & (self.area[:, k] >= min_area)

    def score(
        self, mask_threshold: float, cls_threshold: float = 0.0, min_area: float = 0.0
    ) -> dict:
        k = self.bin_of(mask_threshold)
        keep = self.keep_mask(k, cls_threshold, min_area)
        dice = np.where(keep, self.dice[:, k], 0.0)
        area = np.where(keep, self.area[:, k], 0.0)
        dice_pos = float(dice[self.is_pos].mean()) if self.is_pos.any() else 0.0
        fpr = float((area[~self.is_pos] >= FP_AREA_THRESHOLD).mean()) if (~self.is_pos).any() else 0.0
        return {"aic": harmonic_aic(dice_pos, fpr), "dice_pos": dice_pos, "fpr_neg": fpr}

    def best_thresholds(self) -> tuple[float, float, float]:
        """Операционная точка из calib.json, а если его нет — подобранная здесь же.

        `min_area` возвращается наравне с порогами: без него отчёт считал бы AIC
        не в той точке, в которой отбиралась модель, и «текущий» сценарий в
        `ceilings` расходился бы с `best_aic` прогона.
        """
        calib_path = self.run_dir / "calib.json"
        if calib_path.exists():
            calib = json.loads(calib_path.read_text(encoding="utf-8"))
            return (
                float(calib["mask_threshold"]),
                float(calib["cls_threshold"]),
                float(calib.get("min_area", 0.0)),
            )

        best = max(
            (
                (self.score(k / self.n_bins, c, a)["aic"], k / self.n_bins, c, a)
                for k in range(1, self.n_bins)
                for c in (0.0, 0.3, 0.5, 0.6, 0.7, 0.8)
                for a in DEFAULT_AREA_GRID
            )
        )
        return best[1], best[2], best[3]


def _frame(
    view: OofView, mask_threshold: float, cls_threshold: float = 0.0, min_area: float = 0.0
) -> pd.DataFrame:
    k = view.bin_of(mask_threshold)
    keep = view.keep_mask(k, cls_threshold, min_area)
    table = pd.DataFrame({
        "dice": np.where(keep, view.dice[:, k], 0.0),
        "area": np.where(keep, view.area[:, k], 0.0),
        "gt_frac": view.gt_frac,
        "cls": view.cls_prob,
        "is_pos": view.is_pos,
    })
    table["miss"] = table.is_pos & (table.dice < MISS_DICE)
    if view.rows is not None and len(view.rows) >= len(table):
        extra = view.rows.iloc[: len(table)]
        for column in ("domain", "generator"):
            if column in extra.columns:
                table[column] = extra[column].to_numpy()
    return table


def by_area(table: pd.DataFrame) -> pd.DataFrame:
    """Главный разрез: Dice и доля полных промахов по размеру GT."""
    positives = table[table.is_pos].copy()
    positives["bucket"] = pd.cut(positives.gt_frac, list(AREA_BINS))
    grouped = positives.groupby("bucket", observed=True).agg(
        n=("dice", "size"), dice=("dice", "mean"), miss=("miss", "mean"), cls=("cls", "mean")
    )
    # сколько Dice_pos недобрано именно в этой корзине — на что смотреть в первую очередь
    grouped["lost"] = (
        positives.groupby("bucket", observed=True)["dice"].apply(lambda s: (1 - s).sum())
        / len(positives)
    )
    return grouped.round(3)


def by_column(table: pd.DataFrame, column: str) -> pd.DataFrame | None:
    if column not in table.columns:
        return None
    positives = table[table.is_pos]
    return positives.groupby(column).agg(
        n=("dice", "size"), dice=("dice", "mean"), miss=("miss", "mean"),
        gt_frac=("gt_frac", "median"),
    ).sort_values("dice").round(3)


def ceilings(view: OofView) -> pd.DataFrame:
    """Потолки: докуда можно дойти, не трогая саму модель."""
    oracle_gate_k = int(np.argmax(view.dice[view.is_pos].mean(axis=0)))
    oracle_gate = float(view.dice[view.is_pos, oracle_gate_k].mean())
    per_image = float(view.dice[view.is_pos].max(axis=1).mean())

    current = view.score(*view.best_thresholds())

    return pd.DataFrame([
        {"сценарий": "текущий (пороги из calib)", "dice_pos": round(current["dice_pos"], 4),
         "fpr_neg": round(current["fpr_neg"], 4), "aic": round(current["aic"], 4)},
        {"сценарий": "идеальный классификатор кадра", "dice_pos": round(oracle_gate, 4),
         "fpr_neg": 0.0, "aic": round(harmonic_aic(oracle_gate, 0.0), 4)},
        {"сценарий": "+ идеальный порог на кадр", "dice_pos": round(per_image, 4),
         "fpr_neg": 0.0, "aic": round(harmonic_aic(per_image, 0.0), 4)},
    ])


def report(run_dir: str | Path) -> str:
    view = OofView(run_dir)
    mask_threshold, cls_threshold, min_area = view.best_thresholds()
    table = _frame(view, mask_threshold, cls_threshold, min_area)

    n_pos, n_neg = int(view.is_pos.sum()), int((~view.is_pos).sum())
    misses = int(table.miss.sum())
    lines = [
        f"прогон: {Path(run_dir).name}   пороги: маска {mask_threshold:.3f}, "
        f"cls {cls_threshold:.2f}, min_area {min_area:.3f}",
        f"кадров: {n_pos} позитивов, {n_neg} негативов",
        f"полных промахов (Dice<{MISS_DICE}): {misses} ({misses / max(n_pos, 1) * 100:.1f}% позитивов)",
        "",
        "потолки без изменения модели:",
        ceilings(view).to_string(index=False),
        "",
        "по площади GT (lost = сколько Dice_pos недобрано в корзине):",
        by_area(table).to_string(),
    ]
    if n_neg:
        fa = float((table.loc[~table.is_pos, "area"] >= FP_AREA_THRESHOLD).mean())
        lines += ["", f"ложных тревог на негативах: {fa * 100:.1f}% "
                      f"({int(round(fa * n_neg))} из {n_neg})"]
    for column in ("domain", "generator"):
        sliced = by_column(table, column)
        if sliced is not None:
            lines += ["", f"по {column}:", sliced.to_string()]
    return "\n".join(lines)


def compare(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Одна таблица на несколько прогонов: AIC плюс Dice по корзинам площади.

    Ради этого разреза всё и затевалось — по нему видно, заплатил ли прогон за
    мелкие маски крупными, чего агрегатный AIC не показывает.
    """
    records = []
    for run_dir in run_dirs:
        view = OofView(run_dir)
        operating_point = view.best_thresholds()
        table = _frame(view, *operating_point)
        scores = view.score(*operating_point)

        record = {
            "run": Path(run_dir).name,
            "aic": round(scores["aic"], 4),
            "dice": round(scores["dice_pos"], 4),
            "fpr": round(scores["fpr_neg"], 4),
            "miss%": round(float(table.miss[table.is_pos].mean()) * 100, 1),
        }
        buckets = by_area(table)
        for bucket, row in buckets.iterrows():
            record[f"dice{bucket.right:g}"] = row["dice"]
        records.append(record)
    return pd.DataFrame(records)
