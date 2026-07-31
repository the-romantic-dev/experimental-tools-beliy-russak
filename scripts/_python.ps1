<#
Поиск интерпретатора — одно место на все обёртки.

Раньше путь к conda-среде был вписан в четыре скрипта, и на любой машине, кроме
одной, они просто не запускались. Порядок теперь такой:

  1. $env:AIC_PYTHON — как это настраивается у себя;
  2. привычный путь среды `challenges` — на чужой машине его нет, шаг пропустится;
  3. `python` из PATH — когда среда уже активирована.
#>

# Локальные привычки, а не требование: на другой машине этих путей нет и
# сработает следующий шаг. Свой путь задавай через $env:AIC_PYTHON, а не правкой
# этого списка, иначе он приедет в чужой репозиторий.
$KnownPythons = @(
    "D:/Apps/anaconda3/envs/challenges/python.exe"
)

function Resolve-AicPython {
    $found = @($env:AIC_PYTHON) + $KnownPythons |
        Where-Object { $_ -and (Test-Path $_) } |
        Select-Object -First 1

    if ($found) { return $found }

    if (Get-Command python -ErrorAction SilentlyContinue) { return "python" }

    Write-Error @"
не нашёл интерпретатор Python.
Задай его один раз:
    [Environment]::SetEnvironmentVariable("AIC_PYTHON", "C:/путь/python.exe", "User")
или активируй нужную среду перед запуском.
"@
    exit 1
}
