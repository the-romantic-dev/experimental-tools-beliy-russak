#!/usr/bin/env bash
#
# Кривые обучения по всем прогонам сразу: http://localhost:6006
#
#     ./scripts/tensorboard.sh
#     ./scripts/tensorboard.sh --port 6007
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=_python.sh
. "$ROOT/scripts/_python.sh"

PORT=6006
while [ $# -gt 0 ]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        -h|--help) sed -n '3,7p' "${BASH_SOURCE[0]}"; exit 0 ;;
        *) echo "неизвестный аргумент: $1" >&2; exit 2 ;;
    esac
done

PYTHON="$(aic_python_or_die "$ROOT")"
exec "$PYTHON" -m tensorboard.main \
    --logdir "$ROOT/runs" --port "$PORT" --reload_multifile true
