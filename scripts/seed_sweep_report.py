"""Сводка по свипу сидов: есть эффект или нет.

    python scripts/seed_sweep_report.py

Считает по каждому сиду парную разницу AIC между плечом и контролем, а по
набору сидов — среднее, интервал и вердикт.

Две вещи, ради которых это отдельный скрипт, а не чтение best_aic глазами:

* **Фиксированная операционная точка.** best_aic каждого прогона взят в своей
  тюненой точке, а она подбирается на тех же 639 негативах. У f7 это накрутило
  +0.0050 из видимых +0.0118. Здесь порог/cls/min_area одни и те же для всех
  прогонов и выбраны ДО свипа, по старому f0-control-768, поэтому ни одному
  плечу не подыгрывают.

* **Парность.** Сравниваются только прогоны с одинаковым сидом. Сид задаёт
  инициализацию декодера, порядок сэмплера и аугментации — общая для пары часть
  шума сокращается, и интервал получается вдвое уже, чем при сравнении средних.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

# запуск идёт как `python scripts/seed_sweep_report.py`, и тогда в sys.path
# попадает scripts/, а не корень воркспейса — пакет оттуда не виден
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experimental_tools_beliy_russak.metrics import AICAccumulator, harmonic_aic  # noqa: E402
from experimental_tools_beliy_russak.stats import align_by_stem, per_image  # noqa: E402
from experimental_tools_beliy_russak.workspace import runs_root  # noqa: E402

#: операционная точка старого f0-control-768. Зафиксирована ДО свипа и ни от
#: одного его прогона не зависит, поэтому сравнение остаётся честным
DEFAULT_OP = (0.275, 0.5, 0.0)


def load_run(run_dir: Path):
    accumulator = AICAccumulator.load(run_dir / "oof" / "val.npz")
    stems = pd.read_parquet(run_dir / "oof" / "val_rows.parquet")["stem"].to_numpy()
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    return accumulator, stems, summary


def aic_at(accumulator, op, idx) -> float:
    sample = per_image(accumulator, op)
    dice, alarm, is_pos = sample.dice[idx], sample.alarm[idx], sample.is_pos[idx]
    return harmonic_aic(dice[is_pos].mean(), alarm[~is_pos].mean())


def collect(control_prefix: str, arm_prefix: str, op) -> pd.DataFrame:
    root = runs_root()
    seeds = sorted(
        {d.name[len(control_prefix):] for d in root.glob(f"{control_prefix}*") if d.is_dir()}
        & {d.name[len(arm_prefix):] for d in root.glob(f"{arm_prefix}*") if d.is_dir()}
    )

    rows = []
    for seed in seeds:
        control_dir, arm_dir = root / f"{control_prefix}{seed}", root / f"{arm_prefix}{seed}"
        if not (control_dir / "summary.json").exists() or not (arm_dir / "summary.json").exists():
            print(f"  сид {seed}: пара не готова, пропускаю")
            continue

        control, control_stems, control_summary = load_run(control_dir)
        arm, arm_stems, arm_summary = load_run(arm_dir)
        idx_control, idx_arm = align_by_stem(control_stems, arm_stems)
        if idx_control.size < control_stems.size:
            print(
                f"  ВНИМАНИЕ сид {seed}: наборы val совпадают только на "
                f"{idx_control.size} из {control_stems.size} кадров"
            )

        rows.append({
            "seed": seed,
            "control": aic_at(control, op, idx_control),
            "arm": aic_at(arm, op, idx_arm),
            "control_tuned": control_summary["best_aic"],
            "arm_tuned": arm_summary["best_aic"],
        })

    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["delta"] = frame["arm"] - frame["control"]
        frame["delta_tuned"] = frame["arm_tuned"] - frame["control_tuned"]
    return frame


def report(frame: pd.DataFrame, alpha: float = 0.05) -> None:
    if frame.empty:
        print("ни одной готовой пары — считать нечего")
        return

    print(f"\n{'сид':>6}  {'контроль':>9}  {'плечо':>9}  {'дельта':>9}   {'дельта тюненая':>15}")
    for _, row in frame.iterrows():
        print(
            f"{row['seed']:>6}  {row['control']:9.4f}  {row['arm']:9.4f}  "
            f"{row['delta']:+9.4f}   {row['delta_tuned']:+15.4f}"
        )

    delta = frame["delta"].to_numpy()
    n = delta.size
    mean, sd = delta.mean(), delta.std(ddof=1)
    se = sd / np.sqrt(n)
    half = sps.t.ppf(1 - alpha / 2, n - 1) * se
    t_stat = mean / se if se > 0 else np.inf
    p_value = 2 * (1 - sps.t.cdf(abs(t_stat), n - 1)) if se > 0 else 0.0

    print(f"\nпар: {n}")
    print(f"средняя дельта: {mean:+.4f}   sd по парам: {sd:.4f}   se: {se:.4f}")
    print(f"{100 * (1 - alpha):.0f}% CI: [{mean - half:+.4f}, {mean + half:+.4f}]   "
          f"t({n - 1}) = {t_stat:+.2f}, p = {p_value:.3f}")
    print(f"для сравнения, по тюненым точкам: {frame['delta_tuned'].mean():+.4f} "
          f"(накрутка от подбора порогов {frame['delta_tuned'].mean() - mean:+.4f})")

    if mean - half > 0:
        print(f"\nЭФФЕКТ ПОДТВЕРЖДЁН: прирост {mean:+.4f} AIC, интервал не накрывает ноль")
    elif mean + half < 0:
        print(f"\nПЛЕЧО ХУЖЕ КОНТРОЛЯ: {mean:+.4f} AIC, интервал не накрывает ноль")
    else:
        print("\nЭФФЕКТ НЕ ПОДТВЕРЖДЁН: интервал накрывает ноль.")
        print(f"  Пар нужно примерно: {pairs_for_power(mean, sd, alpha, 0.5)} для шанса 50%, "
              f"{pairs_for_power(mean, sd, alpha, 0.8)} для шанса 80% "
              f"поймать эффект такого размера.")


def pairs_for_power(effect: float, sd: float, alpha: float, power: float) -> int:
    """Сколько пар нужно, чтобы с вероятностью `power` получить значимый результат.

    Правило «2 sigma» (то самое, что печатает stats.seeds_needed) отвечает на
    вопрос при power=0.5: оценка станет двухсигмовой, только если она ляжет ровно
    на истинное значение, а вбок она уезжает в половине случаев. Для реального
    планирования нужен второй член, z(power).
    """
    if abs(effect) < 1e-12:
        return 0
    z = sps.norm.ppf(1 - alpha / 2) + sps.norm.ppf(power)
    return int(np.ceil((z * sd / abs(effect)) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-prefix", default="sw-f0-s")
    parser.add_argument("--arm-prefix", default="sw-f7-s")
    parser.add_argument(
        "--op", default=",".join(str(x) for x in DEFAULT_OP),
        help="общая операционная точка: thr,cls_thr,min_area",
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    op = tuple(float(x) for x in args.op.split(","))
    if len(op) != 3:
        parser.error("--op ждёт три числа через запятую")

    print(f"операционная точка: thr={op[0]} cls={op[1]} min_area={op[2]}")
    report(collect(args.control_prefix, args.arm_prefix, op), alpha=args.alpha)


if __name__ == "__main__":
    main()
