"""araya_eye package bootstrap.

This file allows the local `araya_eye` workspace to be imported as a normal
Python package when the project is installed through uv.
"""

from __future__ import annotations

from importlib import metadata as _metadata

__all__ = ["__version__"]

try:
    __version__ = _metadata.version("araya-eye")
except _metadata.PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.0.0"
