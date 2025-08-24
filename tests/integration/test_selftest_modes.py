# tests/integration/test_selftest_modes.py
# -----------------------------------------------------------------------------
# SpectraMind V50 – Integration tests for `spectramind selftest` modes.
#
# What this validates (assumptions based on project conventions):
#   1) The CLI entrypoint is a Typer app exposed from the `spectramind` module.
#   2) A `selftest` subcommand exists and supports a `--mode` option with at
#      least these modes: "smoke", "quick", and "deep".
#   3) The command returns exit code 0 on success and produces human‑readable
#      progress/log lines without throwing exceptions.
#   4) Hydra is used under the hood to manage run directories; we can safely
#      override the output directory by passing `hydra.run.dir=<tmp>` so that
#      artifacts (logs, reports) are written into pytest’s tmp_path.
#
# Notes:
# * These are integration tests: they invoke the real CLI (not internal
#   functions) using Typer’s CliRunner. They are resilient to log wording by
#   avoiding brittle string matches (we only assert the absence of a traceback
#   and the presence of the run directory/artifacts).
# * The "deep" mode is marked slow and has a generous timeout. If your project
#   doesn’t provide a deep mode, skip that test or adapt the mode list below.
# * If your CLI uses a different module or subcommand naming, adjust
#   `from spectramind import app` and the `SELFTEST_CMD` accordingly.
# -----------------------------------------------------------------------------

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Iterable

import pytest

try:
    # Typer testing helper
    from typer.testing import CliRunner
except Exception as exc:  # pragma: no cover - only hit if dev env missing typer
    pytest.skip(f"typer is required to run CLI integration tests: {exc}", allow_module_level=True)

# Import the Typer app (CLI). If your project exposes it elsewhere, adjust this import.
try:
    from spectramind import app as spectramind_app  # type: ignore
except Exception as exc:
    pytest.skip(f"Cannot import spectramind Typer app: {exc}", allow_module_level=True)

# -----------------------------------------------------------------------------


# Utility / helpers
RUNNER = CliRunner(mix_stderr=False)

SELFTEST_CMD = "selftest"

SUPPORTED_MODES: tuple[str, ...] = ("smoke", "quick")  # keep "deep" separate/slow

# Text patterns we *don’t* want to see in successful runs.
TRACEBACK_PAT = re.compile(r"Traceback \(most recent call last\):", re.IGNORECASE)


def _run_cli(args: Iterable[str], *, env: dict[str, str] | None = None):
    """
    Run the Typer CLI with given args, returning the Click result object.
    We disable Typer's exception catching so pytest can surface real stack traces.
    """
    # Ensure full Hydra error output to make failures easier to debug.
    merged_env = os.environ.copy()
    merged_env.setdefault("HYDRA_FULL_ERROR", "1")
    if env:
        merged_env.update(env)

    return RUNNER.invoke(spectramind_app, list(args), env=merged_env, catch_exceptions=False)


def _assert_success(result, mode: str, out_dir: Path):
    # Exit code 0, no Python traceback in output, and an output dir exists.
    assert result.exit_code == 0, (
        f"`selftest --mode {mode}` failed "
        f"(exit_code={result.exit_code})\nSTDOUT:\n{result.stdout}"
    )
    assert not TRACEBACK_PAT.search(result.stdout), (
        f"Unexpected traceback in `selftest --mode {mode}` output:\n{result.stdout}"
    )
    # Verify the hydra.run.dir override actually created a directory for artifacts.
    assert out_dir.exists() and out_dir.is_dir(), (
        f"Expected run directory {out_dir} to exist for mode={mode}"
    )
    # Optionally: ensure at least one log or artifact was produced.
    # We don't assume specific filenames; look for something plausible.
    artifact_found = any(p.suffix in {".log", ".txt", ".json", ".yaml", ".yml", ".html"} for p in out_dir.rglob("*"))
    assert artifact_found, (
        f"No artifacts (*.log|*.txt|*.json|*.yaml|*.yml|*.html) were created in {out_dir} for mode={mode}\n"
        f"Contents: {[str(p) for p in out_dir.rglob('*')]}"
    )


# -----------------------------------------------------------------------------
# Happy-path: smoke & quick modes
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.parametrize("mode", SUPPORTED_MODES)
def test_selftest_modes_smoke_and_quick(tmp_path: Path, mode: str):
    """
    The selftest should succeed quickly in the lightweight modes
    and write artifacts into the overridden Hydra run dir.
    """
    out_dir = tmp_path / f"selftest_{mode}"
    args = [
        SELFTEST_CMD,
        "--mode",
        mode,
        # Ensure artifacts go to pytest's tmp dir (Hydra override)
        f"hydra.run.dir={out_dir}",
    ]

    result = _run_cli(args)
    _assert_success(result, mode=mode, out_dir=out_dir)


# -----------------------------------------------------------------------------
# Deep mode (slow / optional)
# -----------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.timeout(1200)  # 20 minutes; adjust if your deep selftest runs longer/shorter
def test_selftest_deep_mode(tmp_path: Path):
    """
    The deep selftest may run a fuller pipeline. We still assert that it completes without
    exceptions and that it emits artifacts into the requested run directory.
    """
    mode = "deep"
    out_dir = tmp_path / "selftest_deep"
    args = [
        SELFTEST_CMD,
        "--mode",
        mode,
        f"hydra.run.dir={out_dir}",
    ]
    result = _run_cli(args)
    _assert_success(result, mode=mode, out_dir=out_dir)


# -----------------------------------------------------------------------------
# Invalid mode should fail gracefully with a helpful message
# -----------------------------------------------------------------------------


@pytest.mark.integration
def test_selftest_invalid_mode(tmp_path: Path):
    out_dir = tmp_path / "selftest_invalid"
    args = [
        SELFTEST_CMD,
        "--mode",
        "not-a-real-mode",
        f"hydra.run.dir={out_dir}",
    ]
    result = _run_cli(args)

    # Click/Typer validation typically returns exit code 2 for bad parameter values,
    # but your CLI might return 1. Accept non-zero as failure.
    assert result.exit_code != 0, "Invalid mode should not succeed"
    # Expect a helpful error message instead of a raw traceback.
    assert "invalid" in result.stdout.lower() or "choose from" in result.stdout.lower() or "not a valid" in result.stdout.lower(), (
        "Expected a helpful validation error for invalid mode.\n"
        f"STDOUT:\n{result.stdout}"
    )


# -----------------------------------------------------------------------------
# Optional: verify environment flags that speed up selftests are honored
# (uncomment / adapt if your CLI supports these knobs)
# -----------------------------------------------------------------------------
#
# @pytest.mark.integration
# def test_selftest_respects_fast_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
#     """
#     If your selftest respects environment variables or flags for speed
#     (e.g., SPECTRAMIND_FAST=1, or --fast/--dry-run), validate that here.
#     """
#     out_dir = tmp_path / "selftest_fast"
#     monkeypatch.setenv("SPECTRAMIND_FAST", "1")
#     args = [SELFTEST_CMD, "--mode", "quick", f"hydra.run.dir={out_dir}", "--dry-run"]
#     result = _run_cli(args)
#     _assert_success(result, mode="quick(dry-run)", out_dir=out_dir)
#
# -----------------------------------------------------------------------------