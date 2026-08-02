"""Воркспейс — обычный объект: создают и передают, глобального нет."""

from __future__ import annotations

from pathlib import Path

import pytest

from aic.paths import Workspace


def test_derived_paths_hang_off_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.root == tmp_path.resolve()
    assert ws.data == tmp_path.resolve() / "data"
    assert ws.runs == tmp_path.resolve() / "runs"
    assert ws.train_csv == ws.dataset_root / "stage1" / "train.csv"
    assert ws.index_path == ws.artifacts / "index.parquet"
    assert ws.test_csv == ws.test_root / "test.csv"


def test_two_workspaces_do_not_interfere(tmp_path):
    """Ровно то, чего не умел синглтон: два воркспейса рядом в одном процессе."""
    a = Workspace(tmp_path / "a")
    b = Workspace(tmp_path / "b")
    assert a.runs != b.runs
    assert a.runs.name == b.runs.name == "runs"


def test_find_walks_up_to_the_data_marker(tmp_path, monkeypatch):
    (tmp_path / "data").mkdir()
    deep = tmp_path / "notebooks" / "нора"
    deep.mkdir(parents=True)
    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    assert Workspace.find(deep).root == tmp_path.resolve()


def test_find_prefers_the_env_variable(tmp_path, monkeypatch):
    (tmp_path / "here" / "data").mkdir(parents=True)
    (tmp_path / "there").mkdir()
    monkeypatch.setenv("AIC_WORKSPACE", str(tmp_path / "there"))
    assert Workspace.find(tmp_path / "here").root == (tmp_path / "there").resolve()


def test_find_falls_back_to_start_when_no_marker(tmp_path, monkeypatch):
    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    assert Workspace.find(tmp_path).root == tmp_path.resolve()


def test_find_never_climbs_to_the_home_directory(tmp_path, monkeypatch):
    """`~/data` есть у многих, и без границы весь домашний каталог стал бы
    воркспейсом: `find()` вернул бы `~`, а ошибка всплыла бы только на чтении."""
    home = tmp_path / "дом"
    (home / "data").mkdir(parents=True)
    deep = home / "проекты" / "ноутбуки"
    deep.mkdir(parents=True)

    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert Workspace.find(deep).root == deep.resolve()


def test_find_still_sees_a_marker_below_home(tmp_path, monkeypatch):
    home = tmp_path / "дом"
    project = home / "проект"
    (project / "data").mkdir(parents=True)
    deep = project / "notebooks"
    deep.mkdir()

    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    assert Workspace.find(deep).root == project.resolve()


def test_find_accepts_another_marker(tmp_path, monkeypatch):
    """Пайплайн опознаёт свой воркспейс по configs/, а не по data/."""
    (tmp_path / "configs").mkdir()
    deep = tmp_path / "глубже"
    deep.mkdir()
    monkeypatch.delenv("AIC_WORKSPACE", raising=False)
    assert Workspace.find(deep, marker="configs").root == tmp_path.resolve()


def test_resolve_normalises_windows_separators(tmp_path):
    ws = Workspace(tmp_path)
    got = ws.resolve("stage1\\train\\img\\a.jpg")
    assert got == ws.dataset_root / "stage1" / "train" / "img" / "a.jpg"


def test_resolve_accepts_another_root(tmp_path):
    ws = Workspace(tmp_path)
    assert ws.resolve("a/b.png", root=ws.test_root) == ws.test_root / "a" / "b.png"


def test_ensure_dirs_creates_the_writable_ones(tmp_path):
    ws = Workspace(tmp_path)
    ws.ensure_dirs()
    for path in (ws.artifacts, ws.runs, ws.cache, ws.submissions):
        assert path.is_dir()


def test_frozen(tmp_path):
    ws = Workspace(tmp_path)
    with pytest.raises(Exception):
        ws.root = Path("/другое")


def test_equal_roots_compare_equal(tmp_path):
    assert Workspace(tmp_path) == Workspace(str(tmp_path))
    assert Workspace(tmp_path) != Workspace(tmp_path / "иной")
