"""Достоверность прироста: сравнение прогона с эталоном.

Зачем модуль. Два одинаковых прогона на текущем бюджете расходятся на 0.011 AIC,
а типичное плечо даёт 0.004–0.012. Читать такие числа глазами нельзя, поэтому
сравнение с эталоном считается прямо во время прогона.

Здесь только арифметика: ни torch, ни конфигов, ни обращений к диску. Это
единственная часть, которую надо покрыть тестами всерьёз, а тестировать её
можно, только если она не тянет за собой ни GPU, ни формат чужой папки.

Прогон с диска поднимает `runs.Run`; что делает два прогона несопоставимыми —
решает вызывающий, передавая список ключей в `diverged_keys`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .metric import EPS, FP_AREA_THRESHOLD, AICAccumulator, harmonic_aic

if TYPE_CHECKING:  # только для аннотации: stats не должен тянуть pandas
    from .runs import Eval


@dataclass(frozen=True)
class PerImage:
    """По-картиночные величины в фиксированной операционной точке."""

    #: Dice каждого кадра; для негативов бессмысленен и не используется
    dice: np.ndarray
    #: 1.0, если кадр даёт ложную тревогу по правилу 1% площади
    alarm: np.ndarray
    #: маска позитивов (в GT есть изменения)
    is_pos: np.ndarray


def per_image(accumulator: AICAccumulator, op: tuple[float, float, float]) -> PerImage:
    """Разложить накопленные гистограммы в по-картиночные величины.

    `op` — `(mask_threshold, cls_threshold, min_area)`. Средние по этим массивам
    обязаны совпадать с `accumulator.evaluate(*op)` до последнего знака: иначе
    вердикт будет считаться не по той метрике, по которой отбираются модели.
    """
    mask_threshold, cls_threshold, min_area = op
    pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = accumulator.tables()

    # индекс бина берётся ровно так же, как в AICAccumulator.sweep
    bin_index = int(np.clip(int(mask_threshold * accumulator.n_bins), 0, accumulator.n_bins - 1))
    pred = pred_counts[:, bin_index].astype(np.float64)
    inter = inter_counts[:, bin_index].astype(np.float64)
    area = pred / np.maximum(n_pixels, 1.0)

    keep = (cls_prob >= cls_threshold) & (area >= min_area)
    pred = np.where(keep, pred, 0.0)
    inter = np.where(keep, inter, 0.0)
    area = np.where(keep, area, 0.0)

    return PerImage(
        dice=2.0 * inter / (pred + gt_sum + EPS),
        alarm=(area >= FP_AREA_THRESHOLD).astype(np.float64),
        is_pos=gt_sum > 0,
    )


def align_by_stem(stems_a, stems_b) -> tuple[np.ndarray, np.ndarray]:
    """Индексы кадров, присутствующих в обоих наборах val.

    Сравнивать по позиции нельзя. У `f1-seed7` подвыборка позитивов бралась
    глобальным сидом, и с `f0-control-768` у него совпадает 1892 строки из 5664 —
    сравнение по индексу молча смешало бы разные кадры.
    """
    a = np.asarray(stems_a)
    b = np.asarray(stems_b)
    for name, values in (("первом", a), ("втором", b)):
        if len(np.unique(values)) != len(values):
            raise ValueError(f"stem'ы в {name} наборе повторяются — данные битые")

    common = np.intersect1d(a, b)
    order_a = np.argsort(a)
    order_b = np.argsort(b)
    idx_a = order_a[np.searchsorted(a, common, sorter=order_a)]
    idx_b = order_b[np.searchsorted(b, common, sorter=order_b)]
    return idx_a, idx_b


@dataclass(frozen=True)
class Boot:
    """Результат парного бутстрапа разницы AIC: плечо минус эталон."""

    delta: float
    lo: float
    hi: float
    sd: float


def _aic_of_draws(sample: PerImage, pos_idx: np.ndarray, neg_idx: np.ndarray) -> np.ndarray:
    """AIC для пачки бутстрап-выборок сразу; `*_idx` имеют форму (реплик, кадров)."""
    dice = sample.dice[pos_idx].mean(axis=1)
    fpr = sample.alarm[neg_idx].mean(axis=1)
    rest = 1.0 - fpr
    return 2.0 * dice * rest / np.maximum(dice + rest, 1e-12)


def paired_bootstrap(
    a: PerImage,
    b: PerImage,
    *,
    n: int = 2000,
    seed: int = 0,
    block: int = 250,
) -> Boot:
    """Парный бутстрап: обе выборки пересэмплируются ОДНИМИ И ТЕМИ ЖЕ индексами.

    Парность здесь не украшение: кадры общие, и совпадающая часть шума выборки
    сокращается. На независимых выборках интервал был бы вдвое шире и не мерил
    бы ничего полезного.

    Считается блоками по `block` реплик: матрица индексов на 2000 реплик и 5025
    позитивов заняла бы 160 МБ, блоками — около 10 МБ.
    """
    if a.dice.shape != b.dice.shape:
        raise ValueError("наборы кадров разной длины — сначала align_by_stem")
    if not np.array_equal(a.is_pos, b.is_pos):
        raise ValueError("разметка позитивов различается — сначала align_by_stem")

    pos = np.flatnonzero(a.is_pos)
    neg = np.flatnonzero(~a.is_pos)
    if pos.size == 0 or neg.size == 0:
        raise ValueError("для AIC нужны и позитивы, и негативы")

    rng = np.random.default_rng(seed)
    diffs = np.empty(n, dtype=np.float64)
    done = 0
    while done < n:
        size = min(block, n - done)
        pos_idx = rng.choice(pos, size=(size, pos.size), replace=True).astype(np.int32)
        neg_idx = rng.choice(neg, size=(size, neg.size), replace=True).astype(np.int32)
        diffs[done:done + size] = _aic_of_draws(b, pos_idx, neg_idx) - _aic_of_draws(a, pos_idx, neg_idx)
        done += size

    delta = (
        harmonic_aic(b.dice[pos].mean(), b.alarm[neg].mean())
        - harmonic_aic(a.dice[pos].mean(), a.alarm[neg].mean())
    )
    return Boot(
        delta=float(delta),
        lo=float(np.percentile(diffs, 2.5)),
        hi=float(np.percentile(diffs, 97.5)),
        sd=float(diffs.std(ddof=1)) if n > 1 else 0.0,
    )


def seeds_needed(delta: float, train_sigma: float) -> int | None:
    """Сколько сидов на плечо нужно, чтобы эффект размера `delta` стал 2-сигмовым.

    sigma разницы при k сидах на плечо равна `train_sigma * sqrt(2 / k)`; условие
    `|delta| >= 2 * sigma_разницы` даёт `k >= 8 * train_sigma^2 / delta^2`.

    Это и есть главный выход всей затеи: «внутри шума» превращается в «проверка
    стоила бы столько-то прогонов».
    """
    if abs(delta) < 1e-12:
        return None
    return max(1, math.ceil(8.0 * train_sigma ** 2 / delta ** 2))


@dataclass(frozen=True)
class Verdict:
    label: str
    reason: str
    seeds_needed: int | None


def verdict(
    delta: float,
    boot: Boot,
    train_sigma: float | None,
    *,
    diverged: tuple[str, ...] | list[str] = (),
) -> Verdict:
    """Вердикт по завершённому прогону.

    Асимметрия намеренная: заявка на выигрыш требует и превышения пола шума
    обучения, и чистого интервала по выборке val, а сигнал о провале — только
    первого. Ошибиться, объявив победу, дороже.
    """
    if diverged:
        return Verdict("несопоставимо", "разошлись ключи: " + ", ".join(diverged), None)
    if train_sigma is None:
        return Verdict(
            "пол шума не задан",
            "train_sigma не передан — вердикт вынести не по чему",
            None,
        )

    floor = 2.0 * train_sigma * math.sqrt(2.0)
    if delta <= -floor:
        return Verdict("хуже", f"|Δ| >= {floor:.4f} = 2*train_sigma*sqrt(2)", None)
    if delta >= floor:
        if boot.lo > 0.0:
            return Verdict(
                "подтверждено",
                f"Δ >= {floor:.4f} и 95% CI по выборке val не накрывает ноль",
                None,
            )
        return Verdict(
            "внутри шума обучения",
            f"Δ >= {floor:.4f}, но 95% CI по выборке val накрывает ноль",
            seeds_needed(delta, train_sigma),
        )
    return Verdict(
        "внутри шума обучения",
        f"|Δ| < {floor:.4f} = 2*train_sigma*sqrt(2)",
        seeds_needed(delta, train_sigma),
    )


@dataclass(frozen=True)
class Gate:
    """Решение онлайн-гейта на одной эпохе."""

    ref_aic: float | None
    delta: float | None
    fired: bool
    reason: str


def gate_check(
    curve: tuple[np.ndarray, np.ndarray],
    samples: int,
    aic: float,
    *,
    gate_delta: float = -0.05,
    after_samples: int = 24000,
) -> Gate:
    """Отстаёт ли плечо от эталона на том же числе показов.

    Сравнение идёт по ЧИСЛУ ПОКАЗОВ, а не по индексу эпохи: иначе плечо с
    `epoch_size: 4000, epochs: 12` нельзя было бы сопоставить с эталоном 8000x6.

    Оба значения берутся в своей тюненой точке. На ранних эпохах оптимумы
    порогов сильно разъезжаются (0.375 против 0.2), и общая точка давала бы
    ложные срабатывания. Гейт нарочно снисходительный: он ловит провалы, а не
    отличает соседние плечи.
    """
    if samples < after_samples:
        return Gate(None, None, False, f"рано судить: {samples} < {after_samples} показов")

    xs, ys = curve
    if len(xs) == 0 or samples < xs[0] or samples > xs[-1]:
        return Gate(None, None, False, f"{samples} показов вне кривой эталона")

    ref_aic = float(np.interp(samples, xs, ys))
    delta = float(aic - ref_aic)
    fired = delta <= gate_delta
    reason = (
        f"отставание {delta:+.4f} при пороге {gate_delta:+.4f}"
        if fired
        else f"отставание {delta:+.4f} в пределах порога {gate_delta:+.4f}"
    )
    return Gate(ref_aic, delta, fired, reason)


def gate_metrics(gate: Gate | None) -> dict:
    """Поля гейта для metrics.jsonl. Пустой словарь = машинерия выключена."""
    if gate is None:
        return {}
    return {
        "ref/aic_at_samples": gate.ref_aic,
        "ref/delta": gate.delta,
        "ref/gate": bool(gate.fired),
    }


@dataclass(frozen=True)
class Comparison:
    """Финальное сравнение прогона с эталоном."""

    delta_ref_op: float
    delta_own_op: float
    boot: Boot
    verdict: Verdict
    n_common: int
    n_pos: int
    n_neg: int
    ref_op: tuple[float, float, float]
    own_op: tuple[float, float, float]
    train_sigma: float | None
    warning: str | None

    def as_dict(self) -> dict:
        return {
            "delta_ref_op": self.delta_ref_op,
            "delta_own_op": self.delta_own_op,
            "ci_lo": self.boot.lo,
            "ci_hi": self.boot.hi,
            "sigma_val": self.boot.sd,
            "label": self.verdict.label,
            "reason": self.verdict.reason,
            "seeds_needed": self.verdict.seeds_needed,
            "n_common": self.n_common,
            "n_pos": self.n_pos,
            "n_neg": self.n_neg,
            "ref_op": list(self.ref_op),
            "own_op": list(self.own_op),
            "train_sigma": self.train_sigma,
            "warning": self.warning,
        }

    def report(self, ref_name: str, hours_per_run: float = 1.35) -> list[str]:
        """Строки для лога. Обе дельты печатаются всегда: их разрыв — накрутка
        от подбора порогов под 639 негативов, и её надо видеть."""
        lines = [f"вердикт против {ref_name}:"]
        if self.warning:
            lines.append(f"  ВНИМАНИЕ: {self.warning}")
        thr, cls_thr, min_area = self.ref_op
        lines.append(
            f"  Δ AIC = {self.delta_ref_op:+.4f}  "
            f"(в точке эталона thr={thr:.3f} cls={cls_thr:.3f} area={min_area:.3f})"
        )
        lines.append(
            f"  95% CI по выборке val: [{self.boot.lo:+.4f}, {self.boot.hi:+.4f}]   "
            f"sigma_val = {self.boot.sd:.4f}   кадров {self.n_pos} pos / {self.n_neg} neg"
        )
        if self.train_sigma is not None:
            lines.append(
                f"  пол шума обучения: train_sigma = {self.train_sigma:.4f} -> "
                f"sigma разницы {self.train_sigma * math.sqrt(2.0):.4f}"
            )
        tail = ""
        if self.verdict.seeds_needed:
            runs = 2 * self.verdict.seeds_needed
            # без склонений: числа тут любые, а «22 сида» и «24 прогонов» рядом
            # читаются как опечатка
            tail = (
                f" — чтобы подтвердить, сидов на плечо: {self.verdict.seeds_needed}, "
                f"прогонов: {runs}, ~{runs * hours_per_run:.0f} ч"
            )
        lines.append(f"  {self.verdict.label.upper()}{tail}")
        lines.append(f"  причина: {self.verdict.reason}")
        lines.append(f"  справочно, в своей тюненой точке: {self.delta_own_op:+.4f}")
        return lines





def diverged_keys(
    a: Mapping[str, Any],
    b: Mapping[str, Any],
    keys: Sequence[str],
) -> list[str]:
    """Ключи, по которым два снапшота разошлись. `keys` обязателен.

    Значения по умолчанию тут нет намеренно. Какие параметры делают прогоны
    несопоставимыми, знает только тот, кто их запускал: у одного это
    `train.epochs`, у другого `n_steps`, у третьего вообще ничего — он сравнивает
    два прогона одного своего скрипта. Список по умолчанию был бы тихой
    регламентацией способа обучения.

    Пути читаются точечной нотацией: `diverged_keys(a, b, ["train.epochs"])`.
    """
    def get(node: Any, dotted: str) -> Any:
        for part in dotted.split("."):
            if not isinstance(node, Mapping) or part not in node:
                return None
            node = node[part]
        return node

    return [key for key in keys if get(a, key) != get(b, key)]


def compare(
    own: "Eval",
    ref: "Eval",
    *,
    op: tuple[float, float, float],
    own_op: tuple[float, float, float] | None = None,
    train_sigma: float | None = None,
    bootstrap_n: int = 2000,
    bootstrap_seed: int = 0,
    diverged: Sequence[str] = (),
) -> Comparison:
    """Дельта, интервал и вердикт против эталона на общих кадрах.

    `op` — операционная точка ЭТАЛОНА: оба прогона меряются в ней, иначе разница
    включала бы в себя и разницу тюнинга порогов. `own_op` — своя тюненая точка,
    справочно; её разрыв с основной дельтой и есть накрутка от подбора порогов
    под свои негативы.
    """
    idx_self, idx_ref = align_by_stem(own.stems, ref.stems)

    def cut(sample: PerImage, idx: np.ndarray) -> PerImage:
        return PerImage(dice=sample.dice[idx], alarm=sample.alarm[idx], is_pos=sample.is_pos[idx])

    ref_at_op = cut(per_image(ref.acc, op), idx_ref)
    self_at_op = cut(per_image(own.acc, op), idx_self)
    boot = paired_bootstrap(ref_at_op, self_at_op, n=bootstrap_n, seed=bootstrap_seed)

    pos = ref_at_op.is_pos
    if own_op is None:
        delta_own = boot.delta
    else:
        self_at_own = cut(per_image(own.acc, own_op), idx_self)
        delta_own = harmonic_aic(
            self_at_own.dice[pos].mean(), self_at_own.alarm[~pos].mean()
        ) - harmonic_aic(ref_at_op.dice[pos].mean(), ref_at_op.alarm[~pos].mean())

    warning = None
    if idx_self.size < own.stems.size:
        warning = (
            f"val эталона совпадает с текущим только на {idx_self.size} из "
            f"{own.stems.size} кадров. Дельта считается на пересечении, CI будет шире."
        )

    return Comparison(
        delta_ref_op=boot.delta,
        delta_own_op=float(delta_own),
        boot=boot,
        verdict=verdict(boot.delta, boot, train_sigma, diverged=list(diverged)),
        n_common=int(idx_self.size),
        n_pos=int(pos.sum()),
        n_neg=int((~pos).sum()),
        ref_op=tuple(op),
        own_op=tuple(own_op) if own_op is not None else tuple(op),
        train_sigma=train_sigma,
        warning=warning,
    )
