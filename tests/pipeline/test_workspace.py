"""Поиск корня воркспейса — то, на чём держится переносимость пакета.

Пока код лежал в репозитории, корень считался как `<пакет>/..` и вопроса не
возникало. Теперь пакет ставится куда угодно, и от этих четырёх веток зависит,
найдёт ли команда `runs/` и `data/` вообще.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from experimental_tools_beliy_russak.workspace import (
    WORKSPACE_ENV,
    Workspace,
    configs_root,
    find_workspace_root,
    runs_root,
    set_workspace,
    use_workspace,
    workspace,
)


@pytest.fixture
def some_workspace(tmp_path):
    """Папка, похожая на воркспейс: по наличию configs/ её и опознают."""
    (tmp_path / "configs").mkdir()
    return tmp_path


# --- поиск корня ------------------------------------------------------------


def test_marker_directory_is_found_from_inside(some_workspace, monkeypatch):
    deep = some_workspace / "notebooks" / "черновики"
    deep.mkdir(parents=True)
    monkeypatch.chdir(deep)

    assert find_workspace_root() == some_workspace.resolve()


def test_without_marker_falls_back_to_current_directory(tmp_path, monkeypatch):
    lonely = tmp_path / "нет-конфигов"
    lonely.mkdir()
    monkeypatch.chdir(lonely)

    assert find_workspace_root() == lonely.resolve()


def test_environment_variable_wins_over_search(some_workspace, tmp_path, monkeypatch):
    elsewhere = tmp_path / "другой"
    (elsewhere / "configs").mkdir(parents=True)
    monkeypatch.chdir(some_workspace)
    monkeypatch.setenv(WORKSPACE_ENV, str(elsewhere))
    set_workspace(None)  # сбросить кэш, чтобы переменная перечиталась

    assert workspace().root == elsewhere.resolve()


def test_explicit_set_wins_over_environment(some_workspace, tmp_path, monkeypatch):
    monkeypatch.setenv(WORKSPACE_ENV, str(tmp_path / "из-переменной"))
    set_workspace(some_workspace)

    assert workspace().root == some_workspace.resolve()


# --- производные пути -------------------------------------------------------


def test_all_paths_hang_off_the_root(some_workspace):
    with use_workspace(some_workspace):
        current = workspace()
        assert current.runs == some_workspace / "runs"
        assert current.configs == some_workspace / "configs"
        assert current.plans == some_workspace / "plans"
        assert current.index_path == some_workspace / "artifacts" / "index.parquet"
        assert current.split_path == some_workspace / "artifacts" / "folds.parquet"
        assert current.train_csv == some_workspace / "data" / "train_stage1" / "stage1" / "train.csv"


def test_module_level_helpers_follow_the_switch(some_workspace):
    with use_workspace(some_workspace):
        assert runs_root() == some_workspace / "runs"
        assert configs_root() == some_workspace / "configs"


def test_resolve_prefixes_dataset_root_and_normalises_separators(some_workspace):
    with use_workspace(some_workspace):
        got = workspace().resolve("stage1\\train\\img\\a.jpg")

    assert got == some_workspace / "data" / "train_stage1" / "stage1" / "train" / "img" / "a.jpg"


def test_ensure_dirs_creates_what_the_run_will_need(some_workspace):
    with use_workspace(some_workspace):
        current = workspace()
        current.ensure_dirs()

        for path in (current.artifacts, current.runs, current.cache, current.submissions):
            assert path.is_dir()


# --- переключение -----------------------------------------------------------


def test_use_workspace_restores_previous_root(some_workspace, tmp_path):
    before = workspace().root
    with use_workspace(some_workspace):
        assert workspace().root == some_workspace.resolve()
    assert workspace().root == before


def test_use_workspace_restores_even_after_an_exception(some_workspace):
    before = workspace().root
    with pytest.raises(RuntimeError):
        with use_workspace(some_workspace):
            raise RuntimeError("что-то пошло не так внутри")

    assert workspace().root == before


def test_root_is_absolute_even_when_set_from_a_relative_path(some_workspace, monkeypatch):
    """Относительный корень тут же разворачивается: иначе пути поехали бы за chdir."""
    monkeypatch.chdir(some_workspace)

    assert Workspace(".").root == some_workspace.resolve()
    assert Path(Workspace(".").root).is_absolute()
