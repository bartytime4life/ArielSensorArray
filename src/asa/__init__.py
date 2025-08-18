"""
ArielSensorArray (ASA) — Core Package Init

This module marks the `asa` directory as a Python package and provides
top-level metadata for the ArielSensorArray project.

Exposes:
    __version__ : str
        Package version identifier.
    __all__ : list[str]
        Publicly accessible submodules for clean imports.
"""

from __future__ import annotations

__version__ = "0.1.0"

# Explicitly list subpackages for clean import paths
__all__ = [
    "pipeline",
    "data",
    "calib",
    "diagnostics",
]

# Convenience imports (optional, keep light to avoid slow startup)
try:
    from . import pipeline, data, calib, diagnostics  # noqa: F401
except Exception:
    # Allow partial imports in minimal environments (e.g. docs, CI smoke tests)
    pass