"""Чем именно посчитан прогон — снимок среды рядом с его результатами.

Регламент требует, чтобы код воспроизводил отправленный submission.csv. Снапшот
конфига для этого недостаточен: тот же конфиг на другой версии timm даёт другую
сеть, а на другом коммите — другой лосс. Поэтому в папке прогона лежит ещё и
env.json.
"""

from __future__ import annotations

import experimental_tools_beliy_russak  # noqa: F401

import json
import re
import subprocess

import pytest

from experimental_tools_beliy_russak.provenance import environment_info, write_environment


def test_environment_info_pins_the_versions_that_shape_the_network():
    info = environment_info()
    assert info["python"].startswith("3.")
    assert info["packages"]["torch"]
    # именно эти два переопределяют форму сети при том же конфиге
    assert "timm" in info["packages"]
    assert "segmentation_models_pytorch" in info["packages"]


def test_environment_info_reads_the_commit_of_the_toolkit():
    """Коммит берётся у кода, а не у текущей папки: пакет может стоять где угодно."""
    git = environment_info()["git"]

    assert re.fullmatch(r"[0-9a-f]{7,40}", git["commit"])
    assert isinstance(git["dirty"], bool)


def test_unknown_git_state_is_recorded_as_such_and_never_raises(monkeypatch):
    """Пакет, установленный из колеса, лежит вне репозитория — это не авария."""
    def refuse(*args, **kwargs):
        raise FileNotFoundError("git не установлен")

    monkeypatch.setattr(subprocess, "check_output", refuse)
    git = environment_info()["git"]

    assert git["commit"] is None
    assert git["dirty"] is None


def test_write_environment_leaves_readable_json(tmp_path):
    path = write_environment(tmp_path)

    assert path == tmp_path / "env.json"
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["packages"]["torch"] == environment_info()["packages"]["torch"]
    assert saved["command"]
