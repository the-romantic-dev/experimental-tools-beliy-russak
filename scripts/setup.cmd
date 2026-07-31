@echo off
rem One-time project setup from cmd.exe:
rem   deps, tests, dataset index, folds, smoke run.
rem
rem     scripts\setup.cmd
rem     scripts\setup.cmd -SkipInstall
rem
rem Arguments are passed through to setup.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1" %*
exit /b %ERRORLEVEL%
