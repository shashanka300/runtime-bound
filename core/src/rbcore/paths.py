"""Filesystem layout. All heavy data resolves under DATA_ROOT (default: <repo>/data)."""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path


@cache
def repo_root() -> Path:
    """Directory holding the workspace pyproject (walks up from this file)."""
    for parent in Path(__file__).resolve().parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file() and "[tool.uv.workspace]" in pyproject.read_text():
            return parent
    raise RuntimeError("workspace root not found; is rbcore installed editable from the repo?")


def data_root() -> Path:
    return Path(os.environ.get("DATA_ROOT", repo_root() / "data")).resolve()


def archive_root() -> Path:
    """Cold storage for finished runs, spare models and queue outputs.

    ``ARCHIVE_ROOT`` env var, else ``DATA_ROOT/archive`` (on this machine a symlink to
    <archive drive>, big but slower). Falls back to a plain directory on the fast disk
    for collaborators without a separate archive drive.
    """
    return Path(os.environ.get("ARCHIVE_ROOT", data_root() / "archive")).resolve()


def competition_dir() -> Path:
    return data_root() / "competition"


def experiments_dir() -> Path:
    return repo_root() / "experiments"


def paper_dir() -> Path:
    return repo_root() / "paper"
