"""
asa.pipeline — Core Modeling and Pipeline Orchestration

This subpackage defines the main machine learning / data processing
pipeline for the ArielSensorArray system. It contains model
definitions, training routines, inference logic, calibration, and
submission orchestration.

Exposes:
    __version__ : str
        Mirrors the top-level asa package version.
    __all__ : list[str]
        Publicly accessible pipeline modules.
"""

from __future__ import annotations

# Reuse version from top-level package
try:
    from .. import __version__  # noqa: F401
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "model_def",
    "train",
    "predict",
    "calibrate",
    "diagnostics",
]

# Convenience imports for top-level access
try:
    from . import model_def, train, predict, calibrate, diagnostics  # noqa: F401
except Exception:
    # Allow partial imports during early development or docs/CI runs
    pass