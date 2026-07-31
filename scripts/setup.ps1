<#
Разовая подготовка проекта: зависимости -> индекс -> фолды -> smoke.
После неё можно сразу запускать эксперименты.

    .\scripts\setup.ps1
    .\scripts\setup.ps1 -SkipInstall     # если пакеты уже стоят
    .\scripts\setup.ps1 -Precache        # ещё и собрать ресайз-кэш (долго, ~10-15 ГБ)
#>
param(
    [switch] $SkipInstall,
    [switch] $Precache,
    [int] $CacheSide = 768,
    [int] $Workers = 12
)

$ErrorActionPreference = "Stop"
. "$PSScriptRoot/_python.ps1"
$python = Resolve-AicPython
$root = Split-Path $PSScriptRoot -Parent

if (-not $SkipInstall) {
    Write-Host "`n=== зависимости ===" -ForegroundColor Cyan
    & $python -m pip install --no-input -r "$root/requirements.txt"
}

Write-Host "`n=== проверка среды ===" -ForegroundColor Cyan
& $python "$root/cli.py" env

Write-Host "`n=== тесты ===" -ForegroundColor Cyan
& $python -m pytest "$root/tests" -q

Write-Host "`n=== индекс датасета ===" -ForegroundColor Cyan
& $python "$root/cli.py" index --workers $Workers

Write-Host "`n=== фолды ===" -ForegroundColor Cyan
& $python "$root/cli.py" split

if ($Precache) {
    Write-Host "`n=== ресайз-кэш s$CacheSide ===" -ForegroundColor Cyan
    & $python "$root/cli.py" precache --max-side $CacheSide --workers $Workers
}

Write-Host "`n=== smoke ===" -ForegroundColor Cyan
& $python "$root/cli.py" smoke

Write-Host "`nготово. Дальше: .\aic.ps1 train -c baseline" -ForegroundColor Green
