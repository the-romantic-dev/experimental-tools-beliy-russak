#!/usr/bin/env bash
#
# Обёртка над cli.py: подставляет интерпретатор, чтобы не писать путь к среде
# каждый раз. Аналог aic.ps1 для Linux и macOS.
#
#     ./aic.sh env
#     ./aic.sh train -c baseline -s train.lr=3e-4
#
# Какой Python берётся — см. scripts/_python.sh. Задать свой один раз:
#     export AIC_PYTHON=/путь/к/python
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_python.sh
. "$ROOT/scripts/_python.sh"

PYTHON="$(aic_python_or_die "$ROOT")"
exec "$PYTHON" "$ROOT/cli.py" "$@"
