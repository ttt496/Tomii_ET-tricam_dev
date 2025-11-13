"""Bootstrap module for the legacy `core` namespace."""

import sys as _sys

# Ensure modules that import the historic `core` namespace (without the
# `araya_eye.` prefix) continue to resolve correctly once the package is
# installed via uv.
_sys.modules.setdefault("core", _sys.modules[__name__])

from . import dtypes
from . import integrators
from . import io_utils
from . import pipeline

__all__ = [
    'dtypes',
    'integrators',
    'io_utils',
    'pipeline',
]
