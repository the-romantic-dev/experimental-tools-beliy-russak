#!/usr/bin/env bash
#
# Разовая подготовка проекта: зависимости -> индекс -> фолды -> smoke.
# После неё можно сразу запускать эксперименты. Аналог scripts/setup.ps1.
#
#     ./scripts/setup.sh
#     ./scripts/setup.sh --skip-install          # если пакеты уже стоят
#     ./scripts/setup.sh --precache              # ещё и ресайз-кэш (долго, ~10-15 ГБ)
#     ./scripts/setup.sh --cache-side 768 --workers 12
#
# Развёртывание с нуля на голом сервере — отдельный скрипт, setup_server.sh:
# он ставит системные пакеты, подбирает колесо torch под драйвер и качает
# датасет. Здесь предполагается, что репозиторий и данные уже на месте.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_python.sh
. "$ROOT/scripts/_python.sh"

SKIP_INSTALL=0
PRECACHE=0
CACHE_SIDE=768
# в PowerShell-версии по умолчанию 12; здесь берём по числу ядер — на ноутбуке
# с четырьмя это вдвое короче, а на сервере с 64 заметно быстрее
WORKERS="$(nproc 2>/dev/null || echo 8)"

while [ $# -gt 0 ]; do
    case "$1" in
        --skip-install) SKIP_INSTALL=1; shift ;;
        --precache)     PRECACHE=1; shift ;;
        --cache-side)   CACHE_SIDE="$2"; shift 2 ;;
        --workers)      WORKERS="$2"; shift 2 ;;
        -h|--help)      sed -n '3,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
done

PYTHON="$(aic_python_or_die "$ROOT")"
say() { printf '\n\033[36m=== %s\033[0m\n' "$*"; }

say "интерпретатор"
echo "$PYTHON"

if [ "$SKIP_INSTALL" -eq 0 ]; then
    say "зависимости"
    "$PYTHON" -m pip install --no-input -r "$ROOT/requirements.txt"
fi

say "проверка среды"
"$PYTHON" "$ROOT/cli.py" env

say "тесты"
"$PYTHON" -m pytest "$ROOT/tests" -q

say "индекс датасета"
"$PYTHON" "$ROOT/cli.py" index --workers "$WORKERS"

say "фолды"
"$PYTHON" "$ROOT/cli.py" split

if [ "$PRECACHE" -eq 1 ]; then
    say "ресайз-кэш s$CACHE_SIDE"
    "$PYTHON" "$ROOT/cli.py" precache --max-side "$CACHE_SIDE" --workers "$WORKERS"
fi

say "smoke"
"$PYTHON" "$ROOT/cli.py" smoke

printf '\n\033[32mготово. Дальше: ./aic.sh train -c baseline\033[0m\n'
