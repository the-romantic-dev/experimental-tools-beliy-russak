# `aic` за одну страницу

Библиотека не знает, как вы учите модель. Цикл, архитектура, лоссы и загрузка
данных — ваши. Она даёт то, что нужно всем и должно считаться одинаково: метрику
AIC, достоверность прироста, бюджет вычислений, папку прогона, данные и сабмит.

```bash
pip install git+https://github.com/the-romantic-dev/experimental-tools-beliy-russak.git
```

Ставятся только инструменты — без timm, smp и albumentations. torch не тянется:
он у вас уже стоит под свою CUDA. Нужен эталонный пайплайн с командой
`aic train` — добавьте extra:

```bash
pip install "aic[pipeline] @ git+https://github.com/the-romantic-dev/experimental-tools-beliy-russak.git"
```

## Где что лежит

```python
import aic

ws = aic.Workspace.find()          # AIC_WORKSPACE, иначе поиск вверх по data/
ws = aic.Workspace("D:/projects/aiijc")   # или явно

ws.train_csv, ws.runs, ws.test_csv, ws.index_path
```

Обычный объект: создают и передают аргументом. Два воркспейса рядом в одном
ноутбуке не конфликтуют.

## Данные

```python
df = aic.data.load_index(ws.index_path)       # соберётся: aic.data.build_index(ws)
folds = aic.data.load_folds(ws.split_path)    # соберутся: aic.data.make_folds(df, out_path=...)
```

Фолды групповые по `group_id`: один оригинал порождает несколько манипуляций, и,
разъехавшись по фолдам, они дают утечку.

## Бюджет — до первой эпохи

```python
model = build_my_model()
print(aic.budget.check(model, 768).text)
# 91.2 из 100 GFLOPs на изображение (768px) — в бюджете

print(aic.budget.check([model_a, model_b], 768, n_views=2).text)   # ансамбль x TTA
```

Строгие FLOPs через `torch.utils.flop_counter`: fvcore, thop и ptflops пишут
«FLOPs», а считают вдвое меньшие MACs. Не влезло — спросите, на чём влезет:

```python
aic.budget.largest_fitting_size(lambda size: build_my_model(size), start=768)
```

## Прогон

```python
run = aic.Run.create(ws.runs, "мой-эксперимент")
run.save_snapshot({"модель": "своя", "lr": 3e-4, "epoch_size": 8000})

for epoch in range(12):
    ...                                          # свой цикл обучения

    acc = aic.AICAccumulator()
    for probs, gts, cls_probs in my_validation():   # своя валидация
        acc.update(probs, gts, cls_probs)

    best = acc.best()                            # свип порогов без повторного прогона
    run.log(epoch, {"val/aic_tuned": best.aic, "samples": (epoch + 1) * 8000})
    run.save_state({"model": model.state_dict(), "epoch": epoch})

run.save_eval(acc, val_rows)                     # val_rows — таблица со столбцом stem
run.save_summary({"best_aic": best.aic, "best": best.as_dict()})
run.close()
```

Что класть в снапшот и в состояние — решаете вы: библиотека пишет их как есть и
внутрь не смотрит.

**Логируйте `samples`.** Сравнивать прогоны по номеру эпохи нельзя: плечо с
`epoch_size 4000 × 12 эпох` не сопоставить с эталоном `8000 × 6`. Ось — число
показов.

Продолжить упавший прогон:

```python
run = aic.Run.create(ws.runs, "мой-эксперимент", resume=True)
state = run.load_state("last.pt")
```

## Вердикт против эталона

Два одинаковых прогона расходятся на 0.011 AIC, а типичное плечо даёт
0.004–0.012. Глазами такие числа читать нельзя.

```python
ref = aic.Run.open(ws.runs / "f0-control-768")
cmp = aic.stats.compare(run.load_eval(), ref.load_eval(),
                        op=ref.operating_point(),      # оба меряются в точке эталона
                        train_sigma=0.0035)            # пол шума обучения
print("\n".join(cmp.report("f0-control-768")))
```

```
вердикт против f0-control-768:
  Δ AIC = +0.0121  (в точке эталона thr=0.275 cls=0.500 area=0.000)
  95% CI по выборке val: [+0.0043, +0.0198]   sigma_val = 0.0039   кадров 5025 pos / 639 neg
  пол шума обучения: train_sigma = 0.0035 -> sigma разницы 0.0049
  ПОДТВЕРЖДЕНО
```

Если прогоны несопоставимы по бюджету, скажите об этом явно — библиотека сама
не решает, какие ключи важны:

```python
diverged = aic.diverged_keys(run.snapshot, ref.snapshot, ["data.size", "train.epochs"])
cmp = aic.stats.compare(..., diverged=diverged)
```

Вердикт «внутри шума обучения» приходит с ценой проверки: сколько сидов на плечо
нужно, чтобы эффект стал двухсигмовым.

## Где именно теряется Dice

```python
view = aic.OofView.from_run(run.dir)
op = run.operating_point()

print(view.by_area(op))          # корзины по площади GT — главный разрез
print(view.by_column("domain", op))
print(view.ceilings(op))         # потолок идеального классификатора и идеального порога
```

Ни модели, ни GPU: всё восстанавливается из гистограмм в `oof/val.npz`.

## Все прогоны разом

```python
aic.leaderboard(ws.runs)                     # только различающиеся параметры
aic.compare_table([run.dir, ref.dir])
```

## Сабмит

```python
def my_predict(batch):                       # (B,3,H,W) в [0,1] -> (B,1,H,W) вероятностей
    return torch.sigmoid(model(batch))       # можно вернуть (probs, cls_probs)

paths = sorted(ws.test_img_dir.glob("*.png"))
items = aic.submit.predict_folder(paths, my_predict, size=768, tta=("none", "hflip"))

table, _ = aic.submit.load_test_table(ws.test_csv, ws.submission_template)
aic.submit.write("submissions/мой", items, table=table, mask_threshold=0.275)

print(aic.submit.validate("submissions/мой", ws.test_csv))
aic.submit.pack_zip("submissions/мой")
```

`validate` проверяет то, что иначе выясняется после загрузки, когда попытка уже
потрачена: состав относительно `test.csv`, размеры масок, значения строго 0/255.

## Чего в библиотеке нет намеренно

Цикла обучения, сборки моделей, лоссов, аугментаций, конфигов, реестров и
очереди экспериментов. Всё это — способ работы, а он у каждого свой. Эталонный
способ лежит рядом в `aic_pipeline` и никого ни к чему не обязывает.
