# Достоверность прироста в логах прогона — план реализации

> **Для агентов:** ОБЯЗАТЕЛЬНЫЙ САБ-СКИЛЛ: используйте superpowers:subagent-driven-development (рекомендуется) или superpowers:executing-plans, чтобы выполнять план задача за задачей. Шаги размечены чекбоксами (`- [ ]`).

**Цель:** прогон сам печатает, отстаёт ли он от эталона настолько, что его пора снимать, и является ли финальный прирост чем-то большим, чем шум обучения.

**Архитектура:** новый модуль `experimental_tools_beliy_russak/stats.py` — чистые функции на numpy плюс одна функция с I/O (`load_reference`). В `train.py` добавляются только вызовы. Вся арифметика тестируется на синтетике, без GPU и без обучения.

**Стек:** Python 3.11+, numpy, pandas, pyyaml, pytest. Новых зависимостей нет.

Спека: [2026-08-01-run-stats-verdict-design.md](2026-08-01-run-stats-verdict-design.md).

## Global Constraints

- Новых зависимостей не добавлять: только numpy, pandas, pyyaml, уже стоящие в проекте.
- **Обучение в рамках этой работы не запускается ни разу**, включая smoke- и probe-прогоны. Все тесты — CPU, на синтетических данных или на временных папках.
- Каждый модуль начинается с `from __future__ import annotations`.
- Докстринги и комментарии — по-русски, как во всём пакете. Комментарий объясняет «почему», а не «что».
- Каждый тестовый файл первой строкой импорта делает `import experimental_tools_beliy_russak  # noqa: F401` — этот импорт ставит OpenMP-фикс.
- Тесты запускаются из корня репозитория интерпретатором окружения проекта: `D:\Apps\anaconda3\envs\challenges\python.exe -m pytest`. Ниже в командах он обозначен как `python`.
- `train_sigma` во всём коде означает σ **одного прогона**. σ разницы двух прогонов = `train_sigma * sqrt(2)`. Нигде не смешивать.
- При `stats.reference: null` поведение `train.py` должно быть побайтово прежним: ни новых полей в `metrics.jsonl`, ни новых строк в логе.

---

### Task 1: Извлечение по-картиночных величин и выравнивание наборов

**Files:**
- Create: `experimental_tools_beliy_russak/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `AICAccumulator`, `EPS`, `FP_AREA_THRESHOLD` из `experimental_tools_beliy_russak.metrics`.
- Produces: `PerImage(dice, alarm, is_pos)`; `per_image(accumulator, op) -> PerImage`; `align_by_stem(stems_a, stems_b) -> (idx_a, idx_b)`. `op` везде — кортеж `(mask_threshold, cls_threshold, min_area)` из трёх float.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_stats.py`:

```python
"""Статистика достоверности прироста.

Ошибка здесь тише, чем ошибка в метрике: неверный доверительный интервал не
ломает прогон, он просто убеждает, что шум — это результат.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest

from experimental_tools_beliy_russak.metrics import AICAccumulator
from experimental_tools_beliy_russak.stats import align_by_stem, per_image


def make_accumulator(seed: int = 0, n: int = 16) -> AICAccumulator:
    """Половина кадров — позитивы с квадратной маской, половина — негативы."""
    rng = np.random.default_rng(seed)
    probs = rng.random((n, 20, 20)).astype(np.float32)
    gts = np.zeros((n, 20, 20), dtype=np.float32)
    gts[::2, 5:15, 5:15] = 1.0
    cls = rng.random(n).astype(np.float32)

    accumulator = AICAccumulator(n_bins=256)
    accumulator.update(probs, gts, cls)
    return accumulator


@pytest.mark.parametrize("op", [(0.5, 0.0, 0.0), (0.25, 0.4, 0.0), (0.5, 0.0, 0.02)])
def test_per_image_reproduces_accumulator_evaluate(op):
    """Единственная проверка, которая ловит расхождение с настоящей метрикой."""
    accumulator = make_accumulator()
    expected = accumulator.evaluate(*op)
    got = per_image(accumulator, op)

    assert got.dice[got.is_pos].mean() == pytest.approx(expected.dice_pos, abs=1e-9)
    assert got.alarm[~got.is_pos].mean() == pytest.approx(expected.fpr_neg, abs=1e-9)
    assert got.is_pos.sum() == expected.n_pos
    assert (~got.is_pos).sum() == expected.n_neg


def test_align_by_stem_returns_intersection_in_matching_order():
    a = np.array(["x", "y", "z", "w"])
    b = np.array(["z", "w", "q", "x"])

    idx_a, idx_b = align_by_stem(a, b)

    assert list(a[idx_a]) == list(b[idx_b])
    assert sorted(a[idx_a]) == ["w", "x", "z"]


def test_align_by_stem_rejects_duplicates():
    with pytest.raises(ValueError, match="повторяются"):
        align_by_stem(np.array(["x", "x"]), np.array(["x"]))
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats.py -v`
Expected: FAIL с `ModuleNotFoundError: No module named 'experimental_tools_beliy_russak.stats'`

- [ ] **Step 3: Написать минимальную реализацию**

Создать `experimental_tools_beliy_russak/stats.py`:

```python
"""Достоверность прироста: сравнение прогона с эталоном.

Зачем модуль. Два одинаковых прогона на текущем бюджете расходятся на 0.011 AIC,
а типичное плечо даёт 0.004–0.012. Читать такие числа глазами нельзя, поэтому
сравнение с эталоном считается прямо во время прогона.

Здесь только арифметика: ни torch, ни обращения к диску (кроме `load_reference`).
Это единственная часть, которую надо покрыть тестами всерьёз, а тестировать её
можно, только если она не тянет за собой GPU.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .metrics import EPS, FP_AREA_THRESHOLD, AICAccumulator


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
    pred_counts, inter_counts, gt_sum, n_pixels, cls_prob = accumulator._tables()

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
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS, 5 тестов (три параметризации `per_image` плюс два теста выравнивания)

- [ ] **Step 5: Коммит**

```bash
git add experimental_tools_beliy_russak/stats.py tests/test_stats.py
git commit -m "stats: по-картиночные величины и выравнивание наборов val по stem"
```

---

### Task 2: Парный бутстрап разницы AIC

**Files:**
- Modify: `experimental_tools_beliy_russak/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `PerImage` из задачи 1, `harmonic_aic` из `metrics`.
- Produces: `Boot(delta, lo, hi, sd)`; `paired_bootstrap(a: PerImage, b: PerImage, *, n=2000, seed=0, block=250) -> Boot`. `delta` — это `AIC(b) − AIC(a)`, знак «плечо минус эталон».

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_stats.py`:

```python
from experimental_tools_beliy_russak.metrics import harmonic_aic
from experimental_tools_beliy_russak.stats import Boot, paired_bootstrap


def synthetic_pair(rng, n_pos: int, n_neg: int, gain: float):
    """Два прогона на одних кадрах: у второго Dice выше на `gain`."""
    is_pos = np.concatenate([np.ones(n_pos, bool), np.zeros(n_neg, bool)])
    dice_a = np.clip(rng.normal(0.70, 0.30, n_pos + n_neg), 0.0, 1.0)
    dice_b = np.clip(dice_a + gain, 0.0, 1.0)
    alarm = (rng.random(n_pos + n_neg) < 0.07).astype(float)
    return (
        PerImage(dice=dice_a, alarm=alarm, is_pos=is_pos),
        PerImage(dice=dice_b, alarm=alarm.copy(), is_pos=is_pos),
    )


def test_bootstrap_delta_equals_direct_difference():
    rng = np.random.default_rng(0)
    a, b = synthetic_pair(rng, 400, 100, gain=0.05)
    pos, neg = a.is_pos, ~a.is_pos

    expected = (
        harmonic_aic(b.dice[pos].mean(), b.alarm[neg].mean())
        - harmonic_aic(a.dice[pos].mean(), a.alarm[neg].mean())
    )
    got = paired_bootstrap(a, b, n=200, seed=1)

    assert got.delta == pytest.approx(expected, abs=1e-12)
    assert got.lo < got.delta < got.hi


def test_bootstrap_of_identical_runs_is_zero():
    rng = np.random.default_rng(2)
    a, _ = synthetic_pair(rng, 300, 80, gain=0.0)
    got = paired_bootstrap(a, a, n=200, seed=3)

    assert got.delta == pytest.approx(0.0, abs=1e-12)
    assert got.sd == pytest.approx(0.0, abs=1e-12)


def test_bootstrap_ci_covers_truth_about_95_percent_of_the_time():
    """Покрытие: интервал строится по выборке val, истина — по «популяции».

    Ради этого теста модуль и существует. Если покрытие уедет, все вердикты
    станут враньём, а заметить это иначе нечем.
    """
    rng = np.random.default_rng(7)
    population = synthetic_pair(rng, 5000, 1200, gain=0.05)
    pop_a, pop_b = population
    pop_pos, pop_neg = np.flatnonzero(pop_a.is_pos), np.flatnonzero(~pop_a.is_pos)
    truth = (
        harmonic_aic(pop_b.dice[pop_pos].mean(), pop_b.alarm[pop_neg].mean())
        - harmonic_aic(pop_a.dice[pop_pos].mean(), pop_a.alarm[pop_neg].mean())
    )

    covered = 0
    trials = 100
    for trial in range(trials):
        draw = np.random.default_rng(1000 + trial)
        take_pos = draw.choice(pop_pos, 400, replace=False)
        take_neg = draw.choice(pop_neg, 120, replace=False)
        take = np.concatenate([take_pos, take_neg])
        sample = [
            PerImage(dice=p.dice[take], alarm=p.alarm[take], is_pos=p.is_pos[take])
            for p in population
        ]
        boot = paired_bootstrap(sample[0], sample[1], n=400, seed=trial)
        covered += int(boot.lo <= truth <= boot.hi)

    assert covered / trials >= 0.85


def test_bootstrap_rejects_mismatched_sets():
    rng = np.random.default_rng(4)
    a, _ = synthetic_pair(rng, 100, 20, gain=0.0)
    b, _ = synthetic_pair(rng, 90, 20, gain=0.0)
    with pytest.raises(ValueError, match="align_by_stem"):
        paired_bootstrap(a, b, n=50, seed=0)
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats.py -k bootstrap -v`
Expected: FAIL с `ImportError: cannot import name 'Boot'`

- [ ] **Step 3: Написать минимальную реализацию**

Дописать в `experimental_tools_beliy_russak/stats.py` (импорт `harmonic_aic` добавить в существующую строку `from .metrics import ...`):

```python
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
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS, все тесты. Тест покрытия занимает порядка 10 с.

- [ ] **Step 5: Коммит**

```bash
git add experimental_tools_beliy_russak/stats.py tests/test_stats.py
git commit -m "stats: парный бутстрап разницы AIC блоками"
```

---

### Task 3: Вердикт и проверка сопоставимости конфигов

**Files:**
- Modify: `experimental_tools_beliy_russak/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `Boot` из задачи 2.
- Produces: `BUDGET_KEYS`; `comparable(cfg, cfg_ref, keys=BUDGET_KEYS) -> (bool, list[str])`; `seeds_needed(delta, train_sigma) -> int | None`; `Verdict(label, reason, seeds_needed)`; `verdict(delta, boot, train_sigma, *, diverged=()) -> Verdict`. Метки: `"подтверждено"`, `"внутри шума обучения"`, `"хуже"`, `"несопоставимо"`, `"пол шума не задан"`.

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_stats.py`:

```python
from experimental_tools_beliy_russak.stats import comparable, seeds_needed, verdict


def make_boot(delta: float, half_width: float = 0.003) -> Boot:
    return Boot(delta=delta, lo=delta - half_width, hi=delta + half_width, sd=half_width / 2)


def test_seeds_needed_matches_the_documented_formula():
    # k = ceil(8 * sigma^2 / delta^2); значения из спеки
    assert seeds_needed(0.005, 0.008) == 21
    assert seeds_needed(0.0068, 0.008) == 12
    assert seeds_needed(0.012, 0.008) == 4
    assert seeds_needed(0.0, 0.008) is None


def test_small_gain_is_called_noise_and_priced_in_seeds():
    got = verdict(0.005, make_boot(0.005), train_sigma=0.008)
    assert got.label == "внутри шума обучения"
    assert got.seeds_needed == 21


def test_large_gain_with_clean_interval_is_confirmed():
    got = verdict(0.05, make_boot(0.05), train_sigma=0.008)
    assert got.label == "подтверждено"
    assert got.seeds_needed is None


def test_large_gain_with_interval_over_zero_is_not_confirmed():
    """Порог по шуму обучения пройден, но выборка val сама по себе неубедительна."""
    got = verdict(0.05, make_boot(0.05, half_width=0.08), train_sigma=0.008)
    assert got.label == "внутри шума обучения"
    assert "CI" in got.reason


def test_large_loss_is_called_worse():
    got = verdict(-0.05, make_boot(-0.05), train_sigma=0.008)
    assert got.label == "хуже"


def test_missing_train_sigma_does_not_pretend_to_know():
    got = verdict(0.05, make_boot(0.05), train_sigma=None)
    assert got.label == "пол шума не задан"


def test_diverged_configs_block_the_verdict():
    got = verdict(0.05, make_boot(0.05), train_sigma=0.008, diverged=["data.epoch_size"])
    assert got.label == "несопоставимо"
    assert "data.epoch_size" in got.reason


def test_comparable_finds_the_key_that_diverged():
    base = {"data": {"size": 768, "epoch_size": 8000, "val_frac": 0.25, "val_seed": 42},
            "train": {"epochs": 6, "bs": 4, "accum_steps": 4, "fold": 0}}
    other = {"data": {"size": 768, "epoch_size": 4000, "val_frac": 0.25, "val_seed": 42},
             "train": {"epochs": 12, "bs": 4, "accum_steps": 4, "fold": 0}}

    ok, diverged = comparable(base, base)
    assert ok and diverged == []

    ok, diverged = comparable(base, other)
    assert not ok
    assert diverged == ["data.epoch_size", "train.epochs"]
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats.py -k "verdict or comparable or seeds" -v`
Expected: FAIL с `ImportError: cannot import name 'comparable'`

- [ ] **Step 3: Написать минимальную реализацию**

Дописать в `experimental_tools_beliy_russak/stats.py`; в начало файла добавить `import math`:

```python
#: ключи, определяющие бюджет прогона. Сравнивать между собой можно только
#: прогоны, у которых они совпадают: иначе разница мерит бюджет, а не гипотезу
BUDGET_KEYS = (
    "data.size",
    "data.epoch_size",
    "data.val_frac",
    "data.val_seed",
    "train.epochs",
    "train.bs",
    "train.accum_steps",
    "train.fold",
)


def _get_path(node, dotted: str):
    """Значение по пути `a.b.c`; отсутствующий ключ — None."""
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def comparable(cfg: dict, cfg_ref: dict, keys=BUDGET_KEYS) -> tuple[bool, list[str]]:
    """Совпадают ли бюджеты двух прогонов; вторым — список разошедшихся ключей."""
    diverged = [key for key in keys if _get_path(cfg, key) != _get_path(cfg_ref, key)]
    return not diverged, diverged


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
            "stats.train_sigma не заполнен — вердикт вынести не по чему",
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
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS, все тесты

- [ ] **Step 5: Коммит**

```bash
git add experimental_tools_beliy_russak/stats.py tests/test_stats.py
git commit -m "stats: вердикт по прогону и проверка сопоставимости бюджетов"
```

---

### Task 4: Онлайн-гейт по кривой эталона

**Files:**
- Modify: `experimental_tools_beliy_russak/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: ничего из предыдущих задач.
- Produces: `Gate(ref_aic, delta, fired, reason)`; `gate_check(curve, samples, aic, *, gate_delta, after_samples) -> Gate`. `curve` — кортеж из двух `np.ndarray`: число показов и `val/aic_tuned` эталона, отсортированные по показам.

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_stats.py`:

```python
from experimental_tools_beliy_russak.stats import gate_check

#: кривая f0-control-768: 8000 показов на эпоху, val/aic_tuned по эпохам
F0_CURVE = (
    np.array([8000, 16000, 24000, 32000, 40000, 48000], dtype=float),
    np.array([0.3490, 0.5207, 0.6563, 0.7361, 0.7794, 0.8030]),
)


def test_gate_stays_silent_before_the_minimum_budget():
    """На 16k показов даже провальное плечо не снимаем: кривые ещё не разошлись."""
    got = gate_check(F0_CURVE, samples=16000, aic=0.20, gate_delta=-0.05, after_samples=24000)
    assert not got.fired
    assert got.delta is None


def test_gate_fires_on_the_real_effnetv2_numbers():
    """g3-effnetv2-s на 24k показов: 0.5970 против 0.6563 у эталона."""
    got = gate_check(F0_CURVE, samples=24000, aic=0.5970, gate_delta=-0.05, after_samples=24000)
    assert got.fired
    assert got.ref_aic == pytest.approx(0.6563)
    assert got.delta == pytest.approx(-0.0593, abs=1e-4)


def test_gate_lets_a_normal_arm_through():
    """f7-aux-all на 24k: 0.6771 против 0.6563 — выше эталона, снимать нечего."""
    got = gate_check(F0_CURVE, samples=24000, aic=0.6771, gate_delta=-0.05, after_samples=24000)
    assert not got.fired
    assert got.delta == pytest.approx(0.0208, abs=1e-4)


def test_gate_interpolates_between_reference_epochs():
    """Плечо с epoch_size=4000 попадает на 28k — середину между эпохами эталона."""
    got = gate_check(F0_CURVE, samples=28000, aic=0.70, gate_delta=-0.05, after_samples=24000)
    assert got.ref_aic == pytest.approx((0.6563 + 0.7361) / 2)
    assert not got.fired


def test_gate_refuses_to_judge_outside_the_reference_curve():
    got = gate_check(F0_CURVE, samples=96000, aic=0.10, gate_delta=-0.05, after_samples=24000)
    assert not got.fired
    assert got.delta is None
    assert "кривой эталона" in got.reason
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats.py -k gate -v`
Expected: FAIL с `ImportError: cannot import name 'gate_check'`

- [ ] **Step 3: Написать минимальную реализацию**

Дописать в `experimental_tools_beliy_russak/stats.py`:

```python
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
    gate_delta: float,
    after_samples: int,
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
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS, все тесты

- [ ] **Step 5: Коммит**

```bash
git add experimental_tools_beliy_russak/stats.py tests/test_stats.py
git commit -m "stats: онлайн-гейт по кривой эталона на равном числе показов"
```

---

### Task 5: Чтение эталона с диска и сборка сравнения

**Files:**
- Modify: `experimental_tools_beliy_russak/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: всё из задач 1–4, `AICAccumulator`, `runs_root` из `workspace`, `DEFAULT_MASK_GRID`/`DEFAULT_CLS_GRID`/`DEFAULT_AREA_GRID` из `metrics`.
- Produces: `Reference(name, accumulator, stems, op, cfg, curve)`; `load_reference(run_dir) -> Reference`; `resolve_reference(name) -> Path`; `Comparison` с полями `delta_ref_op, delta_own_op, boot, verdict, n_common, n_pos, n_neg, ref_op, own_op, train_sigma, warning` и методами `as_dict()`, `report(ref_name) -> list[str]`; `compare_to_reference(accumulator, stems, cfg, reference, *, own_op, train_sigma, bootstrap_n, bootstrap_seed) -> Comparison`.

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_stats.py` (добавить импорты `json`, `yaml`, `pandas as pd`, `Path` в шапку файла):

```python
from experimental_tools_beliy_russak.stats import (
    Reference,
    compare_to_reference,
    load_reference,
)


def write_run(root, name: str, accumulator, stems, cfg: dict, aics: list[float]):
    """Минимальная папка прогона: столько, сколько читает load_reference."""
    run_dir = root / name
    (run_dir / "oof").mkdir(parents=True)
    accumulator.save(run_dir / "oof" / "val.npz")
    pd.DataFrame({"stem": stems}).to_parquet(run_dir / "oof" / "val_rows.parquet", index=False)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    (run_dir / "metrics.jsonl").write_text(
        "\n".join(json.dumps({"step": i, "val/aic_tuned": a}) for i, a in enumerate(aics)),
        encoding="utf-8",
    )
    return run_dir


def base_cfg(epoch_size: int = 4) -> dict:
    return {
        "data": {"size": 768, "epoch_size": epoch_size, "val_frac": 0.25, "val_seed": 42},
        "train": {"epochs": 3, "bs": 4, "accum_steps": 4, "fold": 0},
    }


def test_load_reference_reads_curve_in_samples(tmp_path):
    accumulator = make_accumulator(seed=5)
    stems = [f"кадр{i}" for i in range(len(accumulator))]
    run_dir = write_run(tmp_path, "эталон", accumulator, stems, base_cfg(), [0.30, 0.50, 0.65])

    reference = load_reference(run_dir)

    assert isinstance(reference, Reference)
    assert reference.name == "эталон"
    assert list(reference.curve[0]) == [4, 8, 12]     # (step + 1) * epoch_size
    assert list(reference.curve[1]) == [0.30, 0.50, 0.65]
    assert list(reference.stems) == stems
    assert len(reference.op) == 3


def test_compare_to_reference_on_identical_runs_gives_zero(tmp_path):
    accumulator = make_accumulator(seed=6)
    stems = [f"кадр{i}" for i in range(len(accumulator))]
    run_dir = write_run(tmp_path, "эталон", accumulator, stems, base_cfg(), [0.3, 0.5, 0.65])
    reference = load_reference(run_dir)

    got = compare_to_reference(
        accumulator, stems, base_cfg(), reference,
        own_op=reference.op, train_sigma=0.008, bootstrap_n=100, bootstrap_seed=0,
    )

    assert got.delta_ref_op == pytest.approx(0.0, abs=1e-12)
    assert got.delta_own_op == pytest.approx(0.0, abs=1e-12)
    assert got.verdict.label == "внутри шума обучения"
    assert got.n_common == len(accumulator)
    assert got.warning is None


def test_compare_warns_when_val_sets_only_partly_overlap(tmp_path):
    """Случай f1-seed7: у эталона другая подвыборка позитивов."""
    accumulator = make_accumulator(seed=8)
    stems = [f"кадр{i}" for i in range(len(accumulator))]
    shifted = [f"кадр{i + 4}" for i in range(len(accumulator))]
    run_dir = write_run(tmp_path, "эталон", accumulator, shifted, base_cfg(), [0.3, 0.5, 0.65])
    reference = load_reference(run_dir)

    got = compare_to_reference(
        accumulator, stems, base_cfg(), reference,
        own_op=reference.op, train_sigma=0.008, bootstrap_n=100, bootstrap_seed=0,
    )

    assert got.n_common < len(accumulator)
    assert "совпадает с текущим только" in got.warning


def test_compare_refuses_when_budgets_diverged(tmp_path):
    accumulator = make_accumulator(seed=9)
    stems = [f"кадр{i}" for i in range(len(accumulator))]
    run_dir = write_run(tmp_path, "эталон", accumulator, stems, base_cfg(), [0.3, 0.5, 0.65])
    reference = load_reference(run_dir)

    got = compare_to_reference(
        accumulator, stems, base_cfg(epoch_size=8), reference,
        own_op=reference.op, train_sigma=0.008, bootstrap_n=100, bootstrap_seed=0,
    )

    assert got.verdict.label == "несопоставимо"
    assert "data.epoch_size" in got.verdict.reason
    # report печатает метку в верхнем регистре
    assert any("НЕСОПОСТАВИМО" in line for line in got.report("эталон"))
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats.py -k "reference or compare" -v`
Expected: FAIL с `ImportError: cannot import name 'Reference'`

- [ ] **Step 3: Написать минимальную реализацию**

Дописать в `experimental_tools_beliy_russak/stats.py`; в шапку добавить `import json`, `from pathlib import Path`, `import pandas as pd`, `import yaml`, а импорт из `.metrics` расширить до `DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID`:

```python
@dataclass(frozen=True)
class Reference:
    """Опорный прогон, поднятый с диска."""

    name: str
    accumulator: AICAccumulator
    stems: np.ndarray
    op: tuple[float, float, float]
    cfg: dict
    curve: tuple[np.ndarray, np.ndarray]


def resolve_reference(name: str) -> Path:
    """Имя прогона или путь к его папке."""
    from .workspace import runs_root

    candidate = Path(name)
    return candidate if candidate.exists() else runs_root() / name


def load_reference(run_dir) -> Reference:
    """Единственная функция модуля, которая ходит на диск."""
    run_dir = Path(run_dir)
    accumulator = AICAccumulator.load(run_dir / "oof" / "val.npz")
    stems = pd.read_parquet(run_dir / "oof" / "val_rows.parquet")["stem"].to_numpy()
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8")) or {}

    # операционная точка эталона: из summary, иначе пересчитываем по сетке по умолчанию
    summary_path = run_dir / "summary.json"
    op = None
    if summary_path.exists():
        best = (json.loads(summary_path.read_text(encoding="utf-8")) or {}).get("best")
        if best:
            op = (
                float(best["mask_threshold"]),
                float(best["cls_threshold"]),
                float(best["min_area"]),
            )
    if op is None:
        best = accumulator.best(
            list(DEFAULT_MASK_GRID), list(DEFAULT_CLS_GRID), list(DEFAULT_AREA_GRID)
        )
        op = (best.mask_threshold, best.cls_threshold, best.min_area)

    epoch_size = int(_get_path(cfg, "data.epoch_size") or 0)
    rows = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rows = [r for r in rows if "val/aic_tuned" in r]
    xs = np.array([(int(r["step"]) + 1) * epoch_size for r in rows], dtype=float)
    ys = np.array([float(r["val/aic_tuned"]) for r in rows], dtype=float)

    return Reference(run_dir.name, accumulator, stems, op, cfg, (xs, ys))


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
            tail = (
                f": нужно {self.verdict.seeds_needed} сидов на плечо "
                f"({runs} прогонов, ~{runs * hours_per_run:.0f} ч)"
            )
        lines.append(f"  {self.verdict.label.upper()}{tail}")
        lines.append(f"  причина: {self.verdict.reason}")
        lines.append(f"  справочно, в своей тюненой точке: {self.delta_own_op:+.4f}")
        return lines


def compare_to_reference(
    accumulator: AICAccumulator,
    stems,
    cfg: dict,
    reference: Reference,
    *,
    own_op: tuple[float, float, float],
    train_sigma: float | None,
    bootstrap_n: int,
    bootstrap_seed: int,
) -> Comparison:
    """Собрать финальное сравнение: дельты, интервал, вердикт."""
    stems = np.asarray(stems)
    idx_self, idx_ref = align_by_stem(stems, reference.stems)

    def cut(sample: PerImage, idx: np.ndarray) -> PerImage:
        return PerImage(dice=sample.dice[idx], alarm=sample.alarm[idx], is_pos=sample.is_pos[idx])

    ref_at_ref_op = cut(per_image(reference.accumulator, reference.op), idx_ref)
    self_at_ref_op = cut(per_image(accumulator, reference.op), idx_self)
    boot = paired_bootstrap(ref_at_ref_op, self_at_ref_op, n=bootstrap_n, seed=bootstrap_seed)

    # справочная дельта: каждый прогон в СВОЕЙ тюненой точке. Разрыв с основной
    # дельтой и есть накрутка от подбора порогов под 639 негативов
    self_at_own_op = cut(per_image(accumulator, own_op), idx_self)
    pos = ref_at_ref_op.is_pos
    delta_own = harmonic_aic(
        self_at_own_op.dice[pos].mean(), self_at_own_op.alarm[~pos].mean()
    ) - harmonic_aic(ref_at_ref_op.dice[pos].mean(), ref_at_ref_op.alarm[~pos].mean())

    warning = None
    if idx_self.size < stems.size:
        warning = (
            f"val эталона совпадает с текущим только на {idx_self.size} из {stems.size} кадров. "
            f"Дельта считается на пересечении, CI будет шире."
        )

    _, diverged = comparable(cfg, reference.cfg)
    return Comparison(
        delta_ref_op=boot.delta,
        delta_own_op=float(delta_own),
        boot=boot,
        verdict=verdict(boot.delta, boot, train_sigma, diverged=diverged),
        n_common=int(idx_self.size),
        n_pos=int(pos.sum()),
        n_neg=int((~pos).sum()),
        ref_op=reference.op,
        own_op=own_op,
        train_sigma=train_sigma,
        warning=warning,
    )
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `python -m pytest tests/test_stats.py -v`
Expected: PASS, все тесты

- [ ] **Step 5: Коммит**

```bash
git add experimental_tools_beliy_russak/stats.py tests/test_stats.py
git commit -m "stats: чтение эталона с диска и сборка финального сравнения"
```

---

### Task 6: Блок `stats` в конфиге и его нормализация

**Files:**
- Modify: `configs/_base.yaml:106` (после блока `calib`)
- Modify: `experimental_tools_beliy_russak/schema.py:129-135` (после ветки `calib`)
- Modify: `experimental_tools_beliy_russak/stats.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `_get_path` из задачи 3.
- Produces: `StatsSettings(reference, train_sigma, bootstrap_n, bootstrap_seed, gate_delta, gate_after_samples, gate_action)`; `stats_settings(cfg) -> StatsSettings | None` — `None`, если `stats.reference` не задан.

- [ ] **Step 1: Написать падающий тест**

Дописать в `tests/test_stats.py`:

```python
from experimental_tools_beliy_russak.config import load_config
from experimental_tools_beliy_russak.schema import find_unknown_keys
from experimental_tools_beliy_russak.stats import stats_settings


def test_stats_block_is_known_to_the_schema():
    """Иначе проверка опечаток заругается на весь новый блок."""
    cfg = {"stats": {
        "reference": "f0-control-768", "train_sigma": 0.008,
        "bootstrap_n": 2000, "bootstrap_seed": 0,
        "gate_delta": -0.05, "gate_after_samples": 24000, "gate_action": "warn",
    }}
    assert find_unknown_keys(cfg) == []


def test_base_config_keeps_stats_switched_off():
    cfg = load_config("_base")
    assert cfg.get_path("stats.reference") is None
    assert stats_settings(cfg) is None


def test_stats_settings_reads_the_block():
    cfg = {"stats": {"reference": "f0-control-768", "train_sigma": 0.008,
                     "gate_action": "stop", "gate_after_samples": 12000}}
    got = stats_settings(cfg)

    assert got.reference == "f0-control-768"
    assert got.train_sigma == 0.008
    assert got.gate_action == "stop"
    assert got.gate_after_samples == 12000
    assert got.bootstrap_n == 2000        # значение по умолчанию
    assert got.gate_delta == -0.05        # значение по умолчанию


def test_stats_settings_rejects_unknown_gate_action():
    cfg = {"stats": {"reference": "f0-control-768", "gate_action": "убить"}}
    with pytest.raises(ValueError, match="gate_action"):
        stats_settings(cfg)
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats.py -k "stats_block or stats_settings or base_config" -v`
Expected: FAIL с `ImportError: cannot import name 'stats_settings'`

- [ ] **Step 3: Написать минимальную реализацию**

Дописать в конец `configs/_base.yaml`:

```yaml
# Достоверность прироста. Выключено, пока не задан reference.
stats:
  reference: null            # имя прогона в runs/ или путь к его папке
  # sigma ОДНОГО прогона, не разницы двух. По паре реплик это |A - B| / sqrt(2),
  # по трём и более — обычное стандартное отклонение. sigma разницы выводится
  # как train_sigma * sqrt(2). Замеры f-серии на 768/48k дают 0.008.
  train_sigma: null
  bootstrap_n: 2000
  bootstrap_seed: 0
  gate_delta: -0.05          # насколько ниже кривой эталона — повод снимать плечо
  gate_after_samples: 24000  # раньше этого числа показов не судим
  gate_action: warn          # warn | stop
```

Добавить в `SCHEMA` в `experimental_tools_beliy_russak/schema.py` после ветки `"calib"`:

```python
    "stats": {
        "reference": LEAF,
        "train_sigma": LEAF,
        "bootstrap_n": LEAF,
        "bootstrap_seed": LEAF,
        "gate_delta": LEAF,
        "gate_after_samples": LEAF,
        "gate_action": LEAF,
    },
```

Дописать в `experimental_tools_beliy_russak/stats.py`:

```python
GATE_ACTIONS = ("warn", "stop")


@dataclass(frozen=True)
class StatsSettings:
    reference: str
    train_sigma: float | None
    bootstrap_n: int
    bootstrap_seed: int
    gate_delta: float
    gate_after_samples: int
    gate_action: str


def stats_settings(cfg) -> StatsSettings | None:
    """Разобрать блок `stats`. None означает «машинерия выключена»."""
    reference = _get_path(cfg, "stats.reference")
    if not reference:
        return None

    action = str(_get_path(cfg, "stats.gate_action") or "warn")
    if action not in GATE_ACTIONS:
        raise ValueError(f"stats.gate_action={action!r}, допустимо: {GATE_ACTIONS}")

    train_sigma = _get_path(cfg, "stats.train_sigma")
    bootstrap_n = _get_path(cfg, "stats.bootstrap_n")
    bootstrap_seed = _get_path(cfg, "stats.bootstrap_seed")
    gate_delta = _get_path(cfg, "stats.gate_delta")
    after = _get_path(cfg, "stats.gate_after_samples")

    return StatsSettings(
        reference=str(reference),
        train_sigma=None if train_sigma is None else float(train_sigma),
        bootstrap_n=2000 if bootstrap_n is None else int(bootstrap_n),
        bootstrap_seed=0 if bootstrap_seed is None else int(bootstrap_seed),
        gate_delta=-0.05 if gate_delta is None else float(gate_delta),
        gate_after_samples=24000 if after is None else int(after),
        gate_action=action,
    )
```

- [ ] **Step 4: Запустить тесты и убедиться, что они проходят**

Run: `python -m pytest tests/test_stats.py tests/test_config_schema.py tests/test_configs_build.py -v`
Expected: PASS. `test_configs_build` проверяет, что все конфиги в `configs/` собираются — новый блок не должен его сломать.

- [ ] **Step 5: Коммит**

```bash
git add configs/_base.yaml experimental_tools_beliy_russak/schema.py experimental_tools_beliy_russak/stats.py tests/test_stats.py
git commit -m "stats: блок конфига, регистрация в схеме и нормализация настроек"
```

---

### Task 7: Врезка в train.py

**Files:**
- Modify: `experimental_tools_beliy_russak/train.py:27-35` (импорты), `:244-254` (загрузка эталона), `:275-293` (гейт), `:314-320` (вердикт в summary)
- Test: `tests/test_stats_integration.py`

**Interfaces:**
- Consumes: `stats_settings`, `load_reference`, `resolve_reference`, `gate_check`, `compare_to_reference` из `stats`.
- Produces: поля `ref/aic_at_samples`, `ref/delta`, `ref/gate` в `metrics.jsonl`; ключи `verdict` и `status` в `summary.json`.

- [ ] **Step 1: Написать падающий тест**

Создать `tests/test_stats_integration.py`:

```python
"""Врезка статистики в прогон: смотрим на то, что попадает в артефакты.

Обучение здесь не запускается — проверяются только те куски run(), которые
собирают поля метрик и summary.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import numpy as np
import pytest

from experimental_tools_beliy_russak.stats import Gate, gate_metrics


def test_gate_metrics_are_absent_when_stats_are_off():
    assert gate_metrics(None) == {}


def test_gate_metrics_carry_the_three_fields():
    gate = Gate(ref_aic=0.6563, delta=-0.0593, fired=True, reason="…")
    assert gate_metrics(gate) == {
        "ref/aic_at_samples": 0.6563,
        "ref/delta": -0.0593,
        "ref/gate": True,
    }


def test_gate_metrics_keep_none_when_gate_stayed_silent():
    gate = Gate(ref_aic=None, delta=None, fired=False, reason="рано судить")
    assert gate_metrics(gate) == {
        "ref/aic_at_samples": None,
        "ref/delta": None,
        "ref/gate": False,
    }
```

- [ ] **Step 2: Запустить тест и убедиться, что он падает**

Run: `python -m pytest tests/test_stats_integration.py -v`
Expected: FAIL с `ImportError: cannot import name 'gate_metrics'`

- [ ] **Step 3: Написать минимальную реализацию**

Дописать в `experimental_tools_beliy_russak/stats.py`:

```python
def gate_metrics(gate: Gate | None) -> dict:
    """Поля гейта для metrics.jsonl. Пустой словарь = машинерия выключена."""
    if gate is None:
        return {}
    return {
        "ref/aic_at_samples": gate.ref_aic,
        "ref/delta": gate.delta,
        "ref/gate": bool(gate.fired),
    }
```

- [ ] **Step 4: Запустить тест и убедиться, что он проходит**

Run: `python -m pytest tests/test_stats_integration.py -v`
Expected: PASS, 3 теста

- [ ] **Step 5: Врезать вызовы в train.py**

В `experimental_tools_beliy_russak/train.py` после существующих импортов из пакета (рядом со строкой 27, `from .engine import ...`) добавить:

```python
from .stats import (
    compare_to_reference,
    gate_check,
    gate_metrics,
    load_reference,
    resolve_reference,
    stats_settings,
)
```

После блока `resume` (сразу за строкой 243, `logger.info(f"продолжаю с эпохи ...")`) добавить:

```python
    settings = stats_settings(cfg)
    reference = None
    if settings is not None:
        reference = load_reference(resolve_reference(settings.reference))
        logger.info(
            f"эталон: {reference.name}, операционная точка {reference.op}, "
            f"train_sigma={settings.train_sigma}"
        )
    gate_stop = False
```

Внутри цикла эпох, между `tuned = accumulator.best(...)` (строка 273) и словарём `metrics` (строка 275), добавить:

```python
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
```

В словарь `metrics` (строки 275–288) добавить последним элементом перед закрывающей скобкой:

```python
            **gate_metrics(gate),
```

После `logger.info(f"эпоха {epoch}: ...")` (строка 293) добавить:

```python
        if gate is not None and gate.ref_aic is not None:
            logger.info(
                f"  эталон {reference.name} @{(epoch + 1) * per_epoch} показов = "
                f"{gate.ref_aic:.4f}   {gate.reason}"
                + ("   ГЕЙТ СРАБОТАЛ" if gate.fired else "")
            )
        if gate is not None and gate.fired and settings.gate_action == "stop":
            logger.info("снимаю прогон по гейту (stats.gate_action=stop)")
            gate_stop = True
            break
```

Блок `summary` (строки 314–320) заменить на:

```python
    summary = {
        "run": run_dir.name,
        "best_aic": best,
        "best": best_result.as_dict() if best_result else None,
        "config": str(cfg.get("_source", "")),
        "epochs_done": epoch + 1,
        "status": "killed" if gate_stop else "ok",
    }
    if gate_stop:
        summary["killed_reason"] = f"гейт по эталону {reference.name} на эпохе {epoch}"

    if reference is not None and best_result is not None and not gate_stop:
        comparison = compare_to_reference(
            AICAccumulator.load(run_dir / "oof" / "val.npz"),
            val_df["stem"].to_numpy(),
            dict(cfg),
            reference,
            own_op=(best_result.mask_threshold, best_result.cls_threshold, best_result.min_area),
            train_sigma=settings.train_sigma,
            bootstrap_n=settings.bootstrap_n,
            bootstrap_seed=settings.bootstrap_seed,
        )
        summary["verdict"] = comparison.as_dict()
        for line in comparison.report(reference.name):
            logger.info(line)
```

Строку 30 `from .metrics import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID` заменить на:

```python
from .metrics import DEFAULT_AREA_GRID, DEFAULT_CLS_GRID, DEFAULT_MASK_GRID, AICAccumulator
```

- [ ] **Step 6: Проверить, что выключенная машинерия ничего не меняет**

Дописать в `tests/test_stats_integration.py`:

```python
from experimental_tools_beliy_russak.config import load_config
from experimental_tools_beliy_russak.stats import stats_settings


def test_default_config_leaves_the_machinery_off():
    """При stats.reference: null в metrics.jsonl не должно появиться ни одного поля."""
    assert stats_settings(load_config("_base")) is None
    assert gate_metrics(None) == {}
```

Run: `python -m pytest tests/ -v`
Expected: PASS, весь набор тестов целиком — в том числе `test_pipeline_parts`, `test_reproducibility` и `test_resume_scheduler`, которые ходят в `train.py`

- [ ] **Step 7: Коммит**

```bash
git add experimental_tools_beliy_russak/train.py experimental_tools_beliy_russak/stats.py tests/test_stats_integration.py
git commit -m "train: гейт по эталону на каждой эпохе и вердикт в summary"
```

---

## После плана

Machinery включается только конфигом. Чтобы начать ей пользоваться, в конфиг плеча добавляется:

```yaml
stats:
  reference: f0-control-768
  train_sigma: 0.008
```

Значение `train_sigma: 0.008` — замер по двум парам реплик f-серии на 768/48k. При смене разрешения или бюджета его надо перемерить: `comparable` поймает расхождение бюджета с эталоном, но не то, что константа устарела при совпадающем бюджете.
