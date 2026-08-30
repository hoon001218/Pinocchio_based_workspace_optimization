from __future__ import annotations

from pathlib import Path

import pytest

from decanting_workspace.paths import (
    WORKSPACE_ROOT_ENV,
    portable_repository_reference,
    repository_root,
)


def test_source_checkout_root_is_discovered_independent_of_directory_name():
    assert repository_root() == Path(__file__).resolve().parents[1]
    assert portable_repository_reference(repository_root() / "config") == "config"


def test_explicit_workspace_root_supports_noneditable_python_install(
    tmp_path,
    monkeypatch,
):
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "cell_nominal.yaml").touch()
    (tmp_path / "assets" / "robots").mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(tmp_path))

    assert repository_root() == tmp_path.resolve()


def test_invalid_workspace_override_fails_clearly(tmp_path, monkeypatch):
    monkeypatch.setenv(WORKSPACE_ROOT_ENV, str(tmp_path))

    with pytest.raises(RuntimeError, match=WORKSPACE_ROOT_ENV):
        repository_root()
