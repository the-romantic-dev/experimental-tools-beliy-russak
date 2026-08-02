"""Библиотека обязана импортироваться без торча и без пайплайна."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_version_is_a_string():
    import aic

    assert isinstance(aic.__version__, str)
    assert aic.__version__


def test_import_sets_env_before_anything_else():
    """Переменные среды выставлены к моменту, когда импорт вернул управление."""
    code = textwrap.dedent(
        """
        import os, sys
        assert "torch" not in sys.modules
        import aic
        assert os.environ["KMP_DUPLICATE_LIB_OK"] == "TRUE"
        assert os.environ["NO_ALBUMENTATIONS_UPDATE"] == "1"
        assert os.environ["OPENCV_LOG_LEVEL"] == "ERROR"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


def test_import_does_not_pull_torch():
    """`import aic` не тянет торч: половине задач он не нужен, а он тяжёлый."""
    code = textwrap.dedent(
        """
        import sys
        import aic
        assert "torch" not in sys.modules, "aic притащил torch на импорте"
        assert "timm" not in sys.modules, "aic притащил timm на импорте"
        print("ok")
        """
    )
    env = {**os.environ, "PYTHONPATH": str(REPO_ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=REPO_ROOT
    )
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout
