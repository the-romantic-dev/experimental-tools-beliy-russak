# AIC — сегментация подделанных областей

Тулкит для экспериментов по задаче AI Challenge Stage 1: по RGB-изображению
предсказать маску изменённой области. Метрика — **AIC Score**, гармоническое
среднее `Dice_pos` и `1 − FPR_neg`.

Код собран в переиспользуемый пакет **`experimental_tools_beliy_russak`**.
Репозиторий делится надвое: пакет — это инструмент, а `configs/`, `data/`,
`runs/` и `artifacts/` — воркспейс, свой у каждого.

Условия задачи — [AI Challenge Stage 1.md](AI%20Challenge%20Stage%201.md).
Подробное руководство — **[GUIDE.md](GUIDE.md)**.

## Быстрый старт

PowerShell:

```powershell
.\scripts\setup.ps1          # зависимости, тесты, индекс, фолды, smoke
.\aic.ps1 train -c baseline  # первый эксперимент
```

cmd.exe (`.ps1` там не запускается — он просто откроется в редакторе):

```bat
scripts\setup.cmd
aic train -c baseline
```

Обе обёртки делают одно и то же — вызывают `cli.py` нужным интерпретатором.
Ищут они его так: `AIC_PYTHON`, если задан, иначе привычный путь conda-среды,
иначе `python` из PATH. На своей машине задаётся один раз:

```powershell
[Environment]::SetEnvironmentVariable("AIC_PYTHON", "C:/путь/python.exe", "User")
```

Если среда уже активирована, обёртки не нужны вовсе:
`python cli.py train -c baseline`.

## Установка

Только библиотека, без клона репозитория:

```bash
pip install git+https://github.com/ЗАМЕНИ-МЕНЯ/experimental-tools-beliy-russak.git
```

Так приезжает код и команда `aic`. Конфиги и планы в колесо **не входят** — это
контент, а не инструмент; свой воркспейс собирается отдельно (см. ниже).

Для работы над самим тулкитом — клон и editable-установка:

```bash
pip install -r requirements.txt
```

Правки в коде видны сразу, переустанавливать не нужно. **torch в зависимости
намеренно не входит**: иначе pip попробовал бы заменить conda-сборку колесом с
PyPI и другой версией CUDA. Ставится он отдельно, под свою карту.

Из Python:

```python
import experimental_tools_beliy_russak as etbr

cfg = etbr.load_config("baseline", ["train.lr=1e-4"])
board = etbr.leaderboard()
```

Где искать данные и куда писать прогоны, пакет определяет сам: переменная
`AIC_WORKSPACE`, иначе ближайшая вверх папка с `configs/`. Переключить руками —
`etbr.set_workspace(...)`. Подробнее — [GUIDE.md](GUIDE.md), раздел 2.

## Свои компоненты

Архитектуры и энкодеры открыты полностью — любая арка из smp, любой энкодер из
timm. Свой **вид** сущности (лосс, пресет аугментаций, оптимизатор, планировщик,
бэкенд) добавляется файлом `aic_plugins.py` в корне своего воркспейса, без
правки библиотеки:

```python
from experimental_tools_beliy_russak import register_loss

@register_loss("soft_iou")
def soft_iou(logits, targets, *, smooth=1.0):
    ...            # вернуть тензор (B,) — значение на каждый кадр
```

`.\aic.ps1 registry` показывает всё зарегистрированное и откуда оно взялось.
Справочник — **[docs/EXTENDING.md](docs/EXTENDING.md)**, образец для копирования —
[docs/aic_plugins.example.py](docs/aic_plugins.example.py).

## Основные команды

Ниже — форма для PowerShell. В cmd замени `.\aic.ps1` на `aic`.

| Команда | Что делает |
|---|---|
| `.\aic.ps1 env` | проверить воркспейс, интерпретатор, GPU, пакеты, артефакты |
| `.\aic.ps1 registry` | что можно писать в конфиге: лоссы, аугментации, оптимизаторы |
| `.\aic.ps1 index` | `train.csv` → `artifacts/index.parquet` |
| `.\aic.ps1 split` | групповые стратифицированные фолды без утечки |
| `.\aic.ps1 precache` | ресайз-кэш датасета для быстрых эпох |
| `.\aic.ps1 profile` | отчёт по данным |
| `.\aic.ps1 smoke` | весь пайплайн на 200 картинках, ~1 минута |
| `.\aic.ps1 train -c <конфиг>` | эксперимент |
| `.\aic.ps1 calibrate runs/<имя>` | подбор порогов по OOF, без GPU |
| `.\aic.ps1 eval runs/<имя>` | честная метрика в исходном разрешении |
| `.\aic.ps1 predict runs/<имя> --images <dir> --out <dir>` | PNG-маски для произвольной папки |
| `.\aic.ps1 submit runs/<имя>` | полный сабмит: маски + `submission.csv` + zip + проверка |
| `.\aic.ps1 check-submission <zip или папка>` | проверить сабмит до загрузки |
| `.\aic.ps1 board` | таблица всех прогонов |
| `.\aic.ps1 report runs/<имя>` | где теряется Dice: корзины площади, домены, потолки |
| `.\aic.ps1 viz runs/<имя>` | вход / GT / предсказание картинкой |
| `.\aic.ps1 curve runs/<имя>` | AIC по порогу бинаризации |

## Серия экспериментов

Готовые конфиги под конкретные гипотезы, общий бюджет, сравнимые между собой.
Подробности и как читать результат — [GUIDE.md](GUIDE.md), раздел 11.

Вся серия ставится одной командой — очередь идёт до конца, упавший прогон
не роняет остальные, после сбоя план доделывается повторным запуском:

```powershell
.\aic.ps1 plan series --dry-run    # посмотреть, что будет запущено
.\aic.ps1 plan series              # запустить очередь
.\aic.ps1 report runs/e2-area --vs runs/e0-control-512
```

Планы лежат в `plans/`: `series` (опорный прогон + три гипотезы), `followups`
(что ставить по результатам), `sweep_area` (пример сетки). Поштучно —
обычным `train -c e0_control`.

Любой параметр конфига правится из командной строки:

```powershell
.\aic.ps1 train -c baseline -s train.lr=1e-4 -s data.size=640 -s data.aug=heavy
```

## Тесты

```bash
pytest tests -q
```

204 теста, ~14 секунд, всё на CPU. Датасет не нужен: ни один тест не читает
`data/` и `artifacts/`. На каждый push их гоняет GitHub Actions.

## Лицензия

MIT — см. [LICENSE](LICENSE).
