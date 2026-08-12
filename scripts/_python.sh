# Поиск интерпретатора — одно место на все bash-обёртки.
# Подключается через `source`, самостоятельно не запускается.
#
# Порядок тот же, что в scripts/_python.ps1, но шаг с «привычным путём»
# заменён на переносимый: жёстко вписанных путей здесь нет намеренно —
# на Linux их роль играют активная среда и .venv в корне репозитория.
#
#   1. $AIC_PYTHON — как это настраивается у себя;
#   2. активная виртуальная среда ($VIRTUAL_ENV или $CONDA_PREFIX);
#   3. <корень репозитория>/.venv/bin/python;
#   4. conda-среда `challenges`, если conda есть в PATH;
#   5. python3 / python из PATH — когда среда уже активирована
#      (так это работает на Kaggle и в Colab).

aic_find_python() {
    local root="${1:-}"

    if [ -n "${AIC_PYTHON:-}" ] && [ -x "${AIC_PYTHON}" ]; then
        printf '%s\n' "$AIC_PYTHON"; return 0
    fi
    if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
        printf '%s\n' "${VIRTUAL_ENV}/bin/python"; return 0
    fi
    if [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
        printf '%s\n' "${CONDA_PREFIX}/bin/python"; return 0
    fi
    if [ -n "$root" ] && [ -x "${root}/.venv/bin/python" ]; then
        printf '%s\n' "${root}/.venv/bin/python"; return 0
    fi
    if command -v conda >/dev/null 2>&1; then
        local base
        base="$(conda info --base 2>/dev/null || true)"
        if [ -n "$base" ] && [ -x "${base}/envs/challenges/bin/python" ]; then
            printf '%s\n' "${base}/envs/challenges/bin/python"; return 0
        fi
    fi
    command -v python3 2>/dev/null || command -v python 2>/dev/null || return 1
}

aic_python_or_die() {
    local python
    if ! python="$(aic_find_python "${1:-}")" || [ -z "$python" ]; then
        # printf, а не cat: это builtin, и сообщение доходит даже тогда,
        # когда PATH сломан — а именно в таком случае мы сюда и попадаем
        printf '%s\n' \
            "не нашёл интерпретатор Python." \
            "Задай его один раз:" \
            "    export AIC_PYTHON=/путь/к/python      # в ~/.bashrc, чтобы не повторять" \
            "или активируй нужную среду перед запуском." >&2
        return 1
    fi
    printf '%s\n' "$python"
}
