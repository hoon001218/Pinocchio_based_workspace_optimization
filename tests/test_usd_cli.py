from __future__ import annotations

from importlib import import_module
from pathlib import Path
from types import SimpleNamespace

import pytest


_ENTRY_POINTS = (
    "cli",
    "candidate_cli",
    "live_cli",
    "precompute_cli",
    "playback_cli",
    "setup_cli",
)


def _arguments(module_name: str, tmp_path: Path) -> list[str]:
    args = ["--config", str(tmp_path / "task-settings.yaml")]
    if module_name == "precompute_cli":
        args.extend(
            [
                "--grid",
                str(tmp_path / "grid.yaml"),
                "--output-json",
                str(tmp_path / "result.json"),
            ]
        )
    return args


def _inject_loader(module, monkeypatch, loader) -> None:
    if hasattr(module, "_default_dependencies"):
        monkeypatch.setattr(
            module,
            "_default_dependencies",
            lambda: SimpleNamespace(spec_loader=loader),
        )
    else:
        monkeypatch.setattr(module, "load_scene_spec", loader)


@pytest.mark.parametrize("module_name", _ENTRY_POINTS)
def test_usd_override_reaches_scene_loader_before_heavy_work(
    module_name,
    tmp_path,
    monkeypatch,
    capsys,
):
    module = import_module(f"decanting_workspace.{module_name}")
    config = tmp_path / "task-settings.yaml"
    usd = tmp_path / "edited scene.usdc"
    received = []

    def loader(path, *, usd_path):
        received.append((path, usd_path))
        raise ValueError("USD scene could not be imported")

    _inject_loader(module, monkeypatch, loader)

    assert module.main(_arguments(module_name, tmp_path) + ["--usd", str(usd)]) == 2
    assert received == [(config, usd)]
    assert "USD scene could not be imported" in capsys.readouterr().err


@pytest.mark.parametrize("module_name", _ENTRY_POINTS)
def test_cli_without_usd_preserves_single_argument_loader_contract(
    module_name,
    tmp_path,
    monkeypatch,
    capsys,
):
    module = import_module(f"decanting_workspace.{module_name}")
    received = []

    def legacy_loader(path):
        received.append(path)
        raise ValueError("configuration validation failed")

    _inject_loader(module, monkeypatch, legacy_loader)

    assert module.main(_arguments(module_name, tmp_path)) == 2
    assert received == [tmp_path / "task-settings.yaml"]
    assert "configuration validation failed" in capsys.readouterr().err
