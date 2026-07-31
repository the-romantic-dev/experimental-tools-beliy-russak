# experimental_tools_beliy_russak — тулкит как переиспользуемый пакет

Дата: 2026-07-31

## Задача

Тулкит экспериментов лежит тремя кусками в корне репозитория: пакет `aic/`,
пакет `tools/` и скрипт `cli.py`. Работает он только из этой папки и только у
того, кто её клонировал целиком. Нужна библиотека, которую команда ставит и
импортирует как обычный пакет.

Решения, принятые до начала работы:

* переезжает **весь тулкит**, а не только task-agnostic ядро — вместе с
  датасетами, моделями, лоссами, метрикой AIC и сборкой сабмита;
* старые `aic/` и `tools/` **удаляются**, шимов обратной совместимости не
  остаётся; в корне сохраняется только `cli.py` как запускатель.

## Раскладка

```
gen_ai_detection/                       воркспейс: данные, конфиги, прогоны
├── cli.py                              запускатель, 3 строки
├── pyproject.toml                      NEW
├── requirements.txt                    схлопывается в `-e .[dev]`
├── configs/ plans/                     не переезжают
├── data/ artifacts/ runs/ submissions/ не переезжают
├── tests/                              импорты переписываются
└── experimental_tools_beliy_russak/
    ├── __init__.py       OpenMP-фикс + ленивый публичный API
    ├── workspace.py      NEW, вместо paths.py
    ├── config.py plans.py logging_utils.py utils.py
    ├── engine.py train.py losses.py metrics.py models/
    ├── datasets.py streams.py transforms.py imageio.py
    ├── indexing.py splits.py precache.py
    ├── inference.py calibrate.py submission.py probe.py
    ├── cli.py            бывший корневой
    └── analysis/         бывший tools/
        ├── compare_runs.py leaderboard.py oof_report.py
        └── profile_data.py viz.py
```

`configs/` и `plans/` внутрь пакета не кладутся. Это контент эксперимента, а не
инструмент; копия в пакете разошлась бы с корневой ровно так же, как разошлись
бы `aic` и его дубль.

## Workspace вместо PROJECT_ROOT

Сегодня `paths.py` вычисляет `PROJECT_ROOT = <папка пакета>/..` и раздаёт
готовые константы, которые пятнадцать модулей забирают через
`from .paths import RUNS_ROOT`. Для библиотеки это не годится по двум причинам:

* после переезда на уровень вниз `runs/`, `data/`, `artifacts/` уехали бы
  внутрь пакета;
* `from X import CONST` связывается на момент импорта, поэтому подменить корень
  снаружи потом нечем.

`paths.py` заменяется на `workspace.py`:

* класс `Workspace(root)` со свойствами `runs`, `configs`, `plans`, `artifacts`,
  `data`, `dataset_root`, `train_csv`, `src_dir`, `test_root`, `test_csv`,
  `submission_template`, `test_img_dir`, `submissions`, `cache`, `index_path`,
  `split_path`; методы `resolve(rel, root=None)` и `ensure_dirs()`;
* модульные функции `workspace()`, `set_workspace(path)`, `runs_root()`,
  `configs_root()`, `plans_root()` и остальные — по одной на свойство;
* `resolve(rel_path, root=None)` на уровне модуля, чтобы вызывающий код не
  менялся по смыслу.

Порядок поиска корня, первое сработавшее выигрывает:

1. явно заданный `set_workspace(path)`;
2. переменная среды `AIC_WORKSPACE`;
3. поиск вверх от текущей рабочей папки директории, содержащей `configs/`;
4. текущая рабочая папка.

Из корня репозитория срабатывает пункт 3, поэтому `python cli.py train -c
baseline` и запуск ноутбука из `notebooks/` ведут себя ровно как сегодня.

Правка механическая: около пятнадцати мест импорта и сорока обращений.
`ensure_dirs()` и `resolve()` сохраняют сигнатуру.

## Пакет

`pyproject.toml` на setuptools:

* `[project.scripts] aic = "experimental_tools_beliy_russak.cli:app"` — после
  `pip install -e .` команда `aic train -c baseline` работает из любой папки;
* torch в зависимости **не** пишется, только в `[project.optional-dependencies]`
  и в доке: иначе `pip install -e .` попробует переставить conda-сборку;
* `requirements.txt` становится однострочником `-e .[dev]`, реальный список
  переезжает в pyproject.

`__init__.py` первым делом выставляет `KMP_DUPLICATE_LIB_OK`,
`NO_ALBUMENTATIONS_UPDATE` и `OPENCV_LOG_LEVEL` — до любого импорта torch, как
и сейчас. Публичные имена отдаются лениво через `__getattr__` (PEP 562):
жадный импорт подтянул бы torch, smp и albumentations и превратил бы мгновенный
`import` в десятисекундный.

Соглашение об импорте для команды: `import experimental_tools_beliy_russak as etbr`.

## Что остаётся на месте

`aic.ps1`, `aic.cmd` и `scripts/setup.ps1` зовут корневой `cli.py`, который
никуда не девается, — эти три файла не трогаются.

Пустой `__init__.py` в корне репозитория удаляется: он делает пакетом сам
воркспейс и после установки настоящего пакета может дать конфликт имён.

## Проверка

* импорты в одиннадцати файлах `tests/` переписываются на новое имя;
* добавляется `tests/test_workspace.py` на все четыре ветки резолва корня:
  явный `set_workspace`, переменная среды, поиск маркера вверх, падение на CWD;
* `pytest tests -q` прогоняется на CPU — это единственное, что запускается.

GPU не занимается: `train`, `probe` и `smoke` остаются за пользователем, команды
отдаются текстом.

## Порядок работ

1. Физический переезд файлов, `tools/` → `analysis/`.
2. `workspace.py` и обновление всех обращений к путям.
3. `pyproject.toml`, корневой запускатель, `__init__.py`.
4. Тесты, зелёный прогон.
5. `README.md`, `GUIDE.md` (раздел «Карта проекта» и все упоминания `aic/…`
   и `tools/…`), ноутбук.

Документация идёт последней: пока код и тесты не сошлись, править 51 КБ гайда
преждевременно.
