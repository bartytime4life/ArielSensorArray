"""Utility subpackage."""
"""
asa.utils — Utility Functions and Helpers

This subpackage provides shared utility functions used across the
ArielSensorArray pipeline, including logging, config handling,
file operations, and scientific helpers.

It is designed to be lightweight and dependency-minimal, so it can
be safely imported from any stage of the system (training, inference,
diagnostics, or CI checks).

Exposes:
    __version__ : str
        Mirrors the top-level asa package version for convenience.
    __all__ : list[str]
        Public utility modules explicitly exported.
"""

from __future__ import annotations

# Reuse version from top-level asa
try:
    from .. import __version__  # noqa: F401
except Exception:
    __version__ = "0.0.0"

__all__ = [
    "logging_utils",
    "config_utils",
    "io_utils",
    "math_utils",
]

# Optional: preload common utilities for convenience
try:
    from . import logging_utils, config_utils, io_utils, math_utils  # noqa: F401
except Exception:
    # Graceful degradation if some utils are missing during early dev/CI
    pass