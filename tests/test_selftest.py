"""
tests/test_selftest.py

Sanity tests for the SpectraMind V50 repository.

These tests verify that:
1. The repository structure is intact.
2. Core CLI entrypoints are importable.
3. Selftest can be invoked without crashing.
"""

import importlib
import subprocess
import sys
from pathlib import Path


def test_repo_structure():
    """Check that required top-level directories exist."""
    required = [
        "configs",
        "src",
        "logs",
        "outputs",
    ]
    for d in required:
        assert Path(d).exists(), f"Missing required directory: {d}"


def test_import_cli_module():
    """Ensure the root CLI module can be imported."""
    mod = importlib.import_module("spectramind")
    assert hasattr(mod, "app") or hasattr(mod, "main"), \
        "CLI module must expose a Typer app or main() entrypoint."


def test_selftest_cli_runs():
    """Run the selftest CLI in dry-run mode to ensure wiring works."""
    cmd = [sys.executable, "-m", "spectramind", "selftest", "--fast"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, f"Selftest failed: {result.stderr}"
    assert "SpectraMind" in result.stdout or "selftest" in result.stdout