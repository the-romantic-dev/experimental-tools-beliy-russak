"""Запуск тулкита из корня воркспейса без установки пакета.

    python cli.py train -c baseline

То же самое, что `aic train -c baseline` после `pip install -e .`. Обёртки
`aic.ps1` и `aic.cmd` зовут именно этот файл, поэтому он и остаётся в корне.
Вся реализация — в aic_pipeline/cli.py.
"""

from aic_pipeline.cli import app

if __name__ == "__main__":
    app()
