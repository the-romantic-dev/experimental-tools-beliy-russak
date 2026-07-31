<#
Обёртка над cli.py: подставляет интерпретатор, чтобы не писать путь к среде
каждый раз.

    .\aic.ps1 env
    .\aic.ps1 train -c baseline -s train.lr=3e-4

Какой Python берётся, по порядку:
  1. $env:AIC_PYTHON, если задан — так это настраивается на своей машине;
  2. привычный путь conda-среды `challenges` (на чужой машине его просто нет);
  3. `python` из PATH — годится, когда среда уже активирована.

Задать свой интерпретатор один раз:
    [Environment]::SetEnvironmentVariable("AIC_PYTHON", "C:/путь/python.exe", "User")
#>
param([Parameter(ValueFromRemainingArguments = $true)] $Rest)

. "$PSScriptRoot/scripts/_python.ps1"
$python = Resolve-AicPython

& $python "$PSScriptRoot/cli.py" @Rest
exit $LASTEXITCODE
