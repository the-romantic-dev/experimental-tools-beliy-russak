<#
Кривые обучения по всем прогонам сразу: http://localhost:6006

    .\scripts\tensorboard.ps1
#>
param([int] $Port = 6006)

. "$PSScriptRoot/_python.ps1"
$python = Resolve-AicPython
$root = Split-Path $PSScriptRoot -Parent

& $python -m tensorboard.main --logdir "$root/runs" --port $Port --reload_multifile true
