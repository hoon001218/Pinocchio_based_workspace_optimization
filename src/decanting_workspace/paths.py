"""Portable discovery for repository-owned configuration and assets."""

from __future__ import annotations

import os
from pathlib import Path


WORKSPACE_ROOT_ENV = "DECANTING_WORKSPACE_ROOT"


def repository_root() -> Path:
    """Return the data-bearing checkout root without depending on its name.

    Editable installs discover the root from this module. A non-editable Python
    install can be paired with any checkout by setting
    ``DECANTING_WORKSPACE_ROOT``. Markers prevent a mistyped override from
    silently loading another directory's data.
    """

    override = os.environ.get(WORKSPACE_ROOT_ENV)
    candidate = (
        Path(override).expanduser().resolve()
        if override
        else Path(__file__).resolve().parents[2]
    )
    required = (
        candidate / "config" / "cell_nominal.yaml",
        candidate / "assets" / "robots",
        candidate / "outputs",
    )
    if not all(path.exists() for path in required):
        source = f"${WORKSPACE_ROOT_ENV}" if override else "editable source layout"
        raise RuntimeError(
            f"invalid decanting workspace root from {source}: {candidate}; "
            f"set {WORKSPACE_ROOT_ENV} to a complete repository checkout"
        )
    return candidate


def repository_path(*parts: str) -> Path:
    return repository_root().joinpath(*parts)


def portable_repository_reference(path: str | Path) -> str:
    """Return a checkout-relative reference or a path-free external identity."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(repository_root()).as_posix()
    except ValueError:
        return f"external:{resolved.name}"
