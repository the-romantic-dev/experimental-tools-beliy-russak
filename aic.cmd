@echo off
rem Wrapper around cli.py for cmd.exe (see aic.ps1 for PowerShell).
rem
rem     aic env
rem     aic train -c baseline -s train.lr=3e-4
rem
rem NOTE: this file must stay ASCII-only with CRLF line endings --
rem cmd.exe mis-parses LF-only batch files and garbles non-ASCII
rem comments under console code page 866.
rem
rem Interpreter lookup order:
rem   1. %AIC_PYTHON% if set -- this is how you configure your own machine
rem   2. the habitual conda env path (absent on any other machine)
rem   3. python from PATH -- fine when the env is already activated
setlocal
set "AIC_PY=%AIC_PYTHON%"
if not defined AIC_PY set "AIC_PY=D:\Apps\anaconda3\envs\challenges\python.exe"
if not exist "%AIC_PY%" set "AIC_PY=python"
"%AIC_PY%" "%~dp0cli.py" %*
exit /b %ERRORLEVEL%
