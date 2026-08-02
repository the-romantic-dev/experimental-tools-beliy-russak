#!/usr/bin/env bash
#
# Развёртывание воркспейса AIC на чистом Linux-сервере с GPU.
#
#   bash setup_server.sh                  всё целиком
#   WS=~/aic bash setup_server.sh         другой корень воркспейса
#   bash setup_server.sh --skip-data      не трогать датасет (уже скачан)
#   bash setup_server.sh --skip-smoke     без проверочных прогонов
#
# Скрипт идемпотентен: повторный запуск пропускает уже готовые шаги.
# Что он НЕ делает — не приносит configs/, plans/ и artifacts/*.parquet:
# это приватная часть воркспейса, её копируют с рабочей машины (см. конец).
#
# Ключевые грабли, ради которых он и написан:
#   * колесо torch должно совпадать с CUDA драйвера. Правило minor-version-
#     compatibility на практике не работает: на драйвере 530 (CUDA 12.1)
#     сборки cu126 и cu130 дают cudaErrorDevicesUnavailable при живой карте,
#     а cu121 заводится. Версия берётся из nvidia-smi, а не наугад;
#   * пакету нужен python >= 3.10 и torch >= 2.4 (новая форма torch.amp);
#   * на Ubuntu 20.04 системный python 3.8, а `python3 -m venv` падает на
#     ensurepip — обходится своим 3.11 или uv.

set -euo pipefail

WS="${WS:-${AIC_WORKSPACE:-$HOME/aic}}"
REPO="${REPO:-$HOME/etbr}"
REPO_URL="${REPO_URL:-https://github.com/the-romantic-dev/experimental-tools-beliy-russak.git}"

DATA_URL="https://huggingface.co/datasets/QwertyNice/Digital_Detective_AIC2026/resolve/main/train_stage1.zip"
DATA_SHA="2ddd33c159d1497e97981e3f6051af4bbe0f5426ee2a16bdf64f4d198bf381be"

SKIP_DATA=0
SKIP_SMOKE=0
for arg in "$@"; do
    case "$arg" in
        --skip-data)  SKIP_DATA=1 ;;
        --skip-smoke) SKIP_SMOKE=1 ;;
        *) echo "неизвестный аргумент: $arg" >&2; exit 2 ;;
    esac
done

SUDO="$(command -v sudo || true)"
say() { printf '\n\033[36m=== %s\033[0m\n' "$*"; }
die() { printf '\n\033[31m%s\033[0m\n' "$*" >&2; exit 1; }

# --------------------------------------------------------------------------
say "системные пакеты"
# --------------------------------------------------------------------------
if [ -n "$SUDO" ] || [ "$(id -u)" = 0 ]; then
    $SUDO apt-get update -qq
    $SUDO apt-get install -y --no-install-recommends git wget curl unzip tmux ca-certificates
else
    echo "sudo нет — пропускаю; убедись, что стоят git wget curl unzip tmux"
fi

command -v nvidia-smi >/dev/null || die "нет nvidia-smi: на этой машине не видно GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# --------------------------------------------------------------------------
say "интерпретатор python >= 3.10"
# --------------------------------------------------------------------------
pick_python() {
    for cand in python3.13 python3.12 python3.11 python3.10 python3; do
        command -v "$cand" >/dev/null || continue
        if "$cand" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3,10) else 1)' 2>/dev/null; then
            echo "$cand"; return 0
        fi
    done
    return 1
}

make_venv_with_uv() {
    command -v uv >/dev/null || {
        curl -LsSf https://astral.sh/uv/install.sh | sh
        export PATH="$HOME/.local/bin:$PATH"
    }
    uv python install 3.11
    uv venv --python 3.11 --seed "$WS/venv"     # --seed кладёт pip внутрь venv
}

mkdir -p "$WS"
if [ -x "$WS/venv/bin/python" ]; then
    echo "venv уже есть: $WS/venv"
else
    if PY="$(pick_python)"; then
        echo "беру $PY ($("$PY" --version 2>&1))"
        if ! "$PY" -m venv "$WS/venv" 2>/dev/null; then
            echo "venv не создался (нет ensurepip) — ставлю ${PY}-venv"
            $SUDO apt-get install -y "${PY}-venv" 2>/dev/null || true
            "$PY" -m venv "$WS/venv" 2>/dev/null || make_venv_with_uv
        fi
    else
        echo "подходящего python в системе нет — ставлю через uv"
        make_venv_with_uv
    fi
fi

# shellcheck disable=SC1091
source "$WS/venv/bin/activate"
python -m pip install -qU pip wheel
python --version

# --------------------------------------------------------------------------
say "torch под версию драйвера"
# --------------------------------------------------------------------------
# «CUDA Version» в шапке nvidia-smi — это максимум, который тянет драйвер.
# Берём колесо ровно под него: более новое собирается, но контекст на карте
# не создаётся, и падает это уже на первом обучении, а не на установке.
cuda_index() {
    local ver num
    ver="$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1)"
    [ -n "$ver" ] || { echo ""; return; }
    num="$(awk -v v="$ver" 'BEGIN{split(v,a,"."); printf "%d%02d", a[1], a[2]}')"
    if   [ "$num" -ge 1300 ]; then echo cu130
    elif [ "$num" -ge 1208 ]; then echo cu128
    elif [ "$num" -ge 1206 ]; then echo cu126
    elif [ "$num" -ge 1204 ]; then echo cu124
    elif [ "$num" -ge 1201 ]; then echo cu121
    else echo cu118
    fi
}

IDX="$(cuda_index)"
[ -n "$IDX" ] || die "не смог прочитать версию CUDA из nvidia-smi"
echo "индекс колёс: $IDX"

if python -c 'import torch' 2>/dev/null && python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)' 2>/dev/null; then
    echo "torch уже рабочий: $(python -c 'import torch;print(torch.__version__)')"
else
    pip uninstall -qy torch torchvision 2>/dev/null || true
    pip install torch torchvision --index-url "https://download.pytorch.org/whl/$IDX"
fi

python - <<'PY'
import sys, torch
print(f"torch {torch.__version__}  cuda={torch.cuda.is_available()}")
# сравнивать версии строками нельзя: "2.13.0" < "2.4" лексикографически
version = tuple(int(part) for part in torch.__version__.split("+")[0].split(".")[:2])
if version < (2, 4):
    sys.exit("нужен torch >= 2.4: в engine.py используется torch.amp.GradScaler('cuda', ...)")
if not torch.cuda.is_available():
    sys.exit("torch не видит карту — колесо не совпало с драйвером")
torch.zeros(1, device="cuda")          # аллокация, а не только опрос драйвера
print(f"карта {torch.cuda.get_device_name(0)}  sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
PY

# --------------------------------------------------------------------------
say "тулкит"
# --------------------------------------------------------------------------
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" pull --ff-only
else
    git clone "$REPO_URL" "$REPO"
fi
pip install -q -e "$REPO[dev]"

# --------------------------------------------------------------------------
say "датасет"
# --------------------------------------------------------------------------
mkdir -p "$WS/data" "$WS/artifacts" "$WS/configs" "$WS/plans"
if [ "$SKIP_DATA" = 1 ]; then
    echo "пропускаю по флагу"
elif [ -f "$WS/data/train_stage1/stage1/train.csv" ]; then
    echo "данные на месте: $(du -sh "$WS/data/train_stage1" | cut -f1)"
else
    mkdir -p "$WS/dl"
    # 36 ГиБ; -c докачивает после обрыва, повторный запуск скрипта безопасен
    wget -c -O "$WS/dl/train_stage1.zip" "$DATA_URL"
    echo "считаю sha256 (несколько минут)..."
    got="$(sha256sum "$WS/dl/train_stage1.zip" | cut -d' ' -f1)"
    [ "$got" = "$DATA_SHA" ] || die "sha256 не совпал: $got вместо $DATA_SHA"
    mkdir -p "$WS/data/train_stage1"
    unzip -q "$WS/dl/train_stage1.zip" -d "$WS/data/train_stage1/"
    # у архива корень stage1/, но подстрахуемся от лишнего уровня вложенности
    if [ -d "$WS/data/train_stage1/train_stage1/stage1" ]; then
        mv "$WS/data/train_stage1/train_stage1/stage1" "$WS/data/train_stage1/"
        rmdir "$WS/data/train_stage1/train_stage1"
    fi
    [ -f "$WS/data/train_stage1/stage1/train.csv" ] || die "после распаковки нет stage1/train.csv"
    rm -rf "$WS/dl"
    echo "распаковано: $(du -sh "$WS/data/train_stage1" | cut -f1), файлов $(find "$WS/data/train_stage1" -type f | wc -l)"
fi

# --------------------------------------------------------------------------
say "окружение"
# --------------------------------------------------------------------------
cat > "$WS/env.sh" <<EOF
# источник правды по окружению: source $WS/env.sh
export AIC_WORKSPACE="$WS"
export NO_ALBUMENTATIONS_UPDATE=1
source "$WS/venv/bin/activate"
EOF
grep -qF "source $WS/env.sh" "$HOME/.bashrc" 2>/dev/null \
    || echo "source $WS/env.sh" >> "$HOME/.bashrc"

export AIC_WORKSPACE="$WS"
export NO_ALBUMENTATIONS_UPDATE=1

# --------------------------------------------------------------------------
say "проверка"
# --------------------------------------------------------------------------
aic env

missing=0
for f in configs/_base.yaml artifacts/index.parquet artifacts/folds.parquet; do
    [ -e "$WS/$f" ] || { echo "нет $WS/$f"; missing=1; }
done

if [ "$missing" = 1 ]; then
    cat <<EOF

Приватная часть воркспейса не приехала. С рабочей машины, из корня репозитория:

  scp -P <порт> -r configs plans <user>@<хост>:$WS/
  scp -P <порт> artifacts/index.parquet artifacts/folds.parquet <user>@<хост>:$WS/artifacts/

Индекс и фолды именно копируются, а не пересобираются: нарезка должна быть той
же, на которой посчитаны прошлые прогоны, иначе валидация несравнима.
EOF
    exit 0
fi

pytest "$REPO/tests" -q

if [ "$SKIP_SMOKE" = 0 ]; then
    say "smoke + прогрев предобученных весов"
    aic smoke
    # веса timm тянутся с HF при первом обращении: пусть это случится сейчас,
    # а не через час после старта очереди
    aic smoke -s model.encoder=tu-convnext_tiny -s model.encoder_weights=imagenet
    aic smoke -s model.encoder=tu-resnet34 -s model.encoder_weights=imagenet
fi

say "готово"
cat <<EOF
Воркспейс:   $WS
Тулкит:      $REPO
Окружение:   source $WS/env.sh   (уже добавлено в ~/.bashrc)

Дальше — очередь экспериментов, обязательно в tmux:

  tmux new -s aic
  source $WS/env.sh && cd $WS
  aic probe f0_control f2_srm_fixed f3_resnet34
  aic plan series_f --dry-run
  aic plan series_f 2>&1 | tee runs/plan_series_f.log
EOF
