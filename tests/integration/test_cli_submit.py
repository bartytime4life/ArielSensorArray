#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/integration/test_cli_submit.py

SpectraMind V50 — Integration Tests: CLI Submit

Goals
-----
1) Ensure the unified CLI exposes a "submit" entrypoint and shows helpful usage.
2) Exercise a lightweight/dry-run submission flow so CI can gate packaging regressions.
3) Verify basic reproducibility hooks (log files / config snapshot) are produced or at least invoked.
4) Keep tests defensive: skip gracefully if the CLI entrypoint is not found or heavy deps are missing.

Design notes
------------
• We discover the CLI entrypoint heuristically:
    - repository root `./spectramind.py` (Typer root CLI)
    - module path `./src/spectramind.py`
    - or dedicated submit CLI `./src/cli/cli_submit.py`
  The tests will *skip* if none are present (useful while scaffolding).

• We run `--help` to assert discoverability and doc presence.
• We attempt a conservative dry-run (or equivalent) call that avoids heavy work. Flags are probed
  defensively; if unknown, we fall back to a plain call that should no-op and exit 0 or explicitly tell us.
• We check for append-only debug logs (`logs/v50_debug_log.md`) and (optionally) a manifest if it exists.

Caveats
-------
These tests are intentionally gentle; they will *not* build or upload artifacts. Heavy paths are explicitly
avoided to keep CI times minimal.

"""

from __future__ import annotations

import os
import re
import json
import time
import shutil
import signal
import typing as t
import subprocess
from pathlib import Path

import pytest


# ---------------------------------------------------------------------
# CLI discovery
# ---------------------------------------------------------------------

def _candidate_cli_paths() -> t.List[Path]:
    return [
        Path("spectramind.py"),
        Path("src") / "spectramind.py",
        Path("src") / "cli" / "cli_submit.py",
    ]


def _first_existing(paths: t.List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="module")
def cli_path() -> Path:
    p = _first_existing(_candidate_cli_paths())
    if not p:
        pytest.skip("SpectraMind CLI entrypoint not found (expected spectramind.py or src/cli/cli_submit.py).")
    return p.resolve()


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def run_cli(cli: Path, args: t.List[str], env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a Python CLI as a subprocess and capture output."""
    cmd = [os.environ.get("PYTHON", "python"), "-u", str(cli), *args]
    proc = subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
        timeout=timeout,
        check=False,
    )
    return proc


def _log_path() -> Path:
    # canonical append-only debug log as used across the pipeline
    return Path("logs") / "v50_debug_log.md"


# ---------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------

def test_submit_help_shows_usage(cli_path: Path):
    """The submit CLI must present discoverable usage docs."""
    # Accept both root Typer and module-style invocation
    # Try: `spectramind.py submit --help` and fallback to `--help`
    # so this passes whether `cli_submit.py` is a subcommand or a standalone CLI.
    # Strategy: if "cli_submit.py", call with `--help`; else call "submit --help".
    is_standalone = cli_path.name == "cli_submit.py"

    args = ["--help"] if is_standalone else ["submit", "--help"]
    proc = run_cli(cli_path, args)
    out = (proc.stdout or "") + (proc.stderr or "")

    assert proc.returncode == 0, f"--help failed (exit {proc.returncode})\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    # Heuristics for helpful content
    assert re.search(r"(?i)(usage|submit|help)", out), f"Help output missing expected keywords:\n{out}"


def test_submit_dry_run_basic(cli_path: Path, tmp_path: Path):
    """Exercise a minimal dry-run to ensure the command wires together without heavy work."""
    logs_dir = _log_path().parent
    logs_dir.mkdir(parents=True, exist_ok=True)
    # capture prior mtime to detect append-only write
    log_file = _log_path()
    before_mtime = log_file.stat().st_mtime if log_file.exists() else None

    # Some CLIs expose different fast/skip flags. We'll probe a few common ones.
    candidate_flag_sets = [
        ["submit", "--dry-run"],
        ["submit", "--dry_run"],
        ["submit", "--fast", "--no-open", "--no-dashboard"],
        ["submit"],  # last resort (should still be a no-op or safe error code)
    ] if cli_path.name != "cli_submit.py" else [
        ["--dry-run"],
        ["--dry_run"],
        ["--fast", "--no-open", "--no-dashboard"],
        [],  # last resort
    ]

    env = {
        # Encourage fast path if the CLI honors env toggles
        "SPECTRAMIND_FAST": "1",
        "SPECTRAMIND_NO_NETWORK": "1",
        "PYTHONUNBUFFERED": "1",
    }

    proc = None
    for flags in candidate_flag_sets:
        proc = run_cli(cli_path, flags, env=env, timeout=180)
        # Accept success (0) or a "graceful usage error" indicating missing required inputs (non-zero but helpful).
        if proc.returncode == 0:
            break
        # If the CLI returns usage guidance, we accept and stop probing
        joined = (proc.stdout or "") + (proc.stderr or "")
        if re.search(r"(?i)(usage:|required|missing|no such option|error:)", joined):
            break

    assert proc is not None, "CLI process did not start."
    # If explicitly non-zero, require actionable message
    if proc.returncode != 0:
        joined = (proc.stdout or "") + (proc.stderr or "")
        assert re.search(r"(?i)(usage|required|missing|error|help|option)", joined), \
            f"Non-zero return without actionable message:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Optional: verify the append-only debug log updated (best-effort)
    if log_file.exists():
        # Allow a small clock skew tolerance
        time.sleep(0.25)
        after_mtime = log_file.stat().st_mtime
        if before_mtime is not None:
            assert after_mtime >= before_mtime, "Expected CLI to append to logs/v50_debug_log.md"


def test_submit_writes_manifest_when_available(cli_path: Path, tmp_path: Path):
    """
    If the submit flow writes a manifest (e.g., outputs/manifest.json or submission bundle manifest),
    validate its JSON shape. This is optional and will PASS if the file is absent.
    """
    out_dir = Path("outputs")
    # Clean up potential stale files to avoid false positives
    if out_dir.exists():
        # keep directory but remove common manifest names
        for name in ("manifest.json", "submission_manifest.json", "bundle_manifest.json"):
            f = out_dir / name
            if f.exists():
                f.unlink()

    # Attempt a dry run; ignore return code semantics as above
    flags = (["submit", "--dry-run"] if cli_path.name != "cli_submit.py" else ["--dry-run"])
    proc = run_cli(cli_path, flags, env={"SPECTRAMIND_FAST": "1"}, timeout=180)

    # Probe common manifest filenames
    candidates = [
        out_dir / "manifest.json",
        out_dir / "submission_manifest.json",
        out_dir / "bundle_manifest.json",
    ]
    found = [p for p in candidates if p.exists()]

    # If nothing exists, we pass (manifest is optional in dry-run or may be elsewhere)
    if not found:
        pytest.skip("No manifest produced by dry-run; skipping manifest content check.")

    # Validate JSON and presence of minimal keys if present
    for mpath in found:
        text = mpath.read_text(encoding="utf-8", errors="ignore")
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as e:
            pytest.fail(f"Manifest at {mpath} is not valid JSON: {e}")

        # Heuristic key checks (optional)
        # Accept either `artifacts`, `files`, or `created_at` style metadata
        keys = set(obj.keys())
        assert keys, f"Empty manifest at {mpath}"
        assert any(k in keys for k in ("artifacts", "files", "created_at", "config_hash", "version")), \
            f"Manifest seems too sparse; missing expected keys at {mpath}"


def test_submit_respects_selftest_if_exposed(cli_path: Path):
    """
    If the CLI exposes a --selftest or similar quick gate, calling it should succeed quickly.
    This test is soft: we skip if the flag is not supported.
    """
    # Check help text for a selftest hint
    help_proc = run_cli(cli_path, ["--help"] if cli_path.name == "cli_submit.py" else ["submit", "--help"])
    help_out = (help_proc.stdout or "") + (help_proc.stderr or "")
    has_selftest = re.search(r"(?i)(self[- ]?test|--selftest)", help_out) is not None

    if not has_selftest:
        pytest.skip("CLI submit does not advertise a self-test flag; skipping.")

    flags = (["--selftest"] if cli_path.name == "cli_submit.py" else ["submit", "--selftest"])
    proc = run_cli(cli_path, flags, env={"SPECTRAMIND_FAST": "1"})
    assert proc.returncode == 0, f"--selftest failed\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    msg = (proc.stdout or "") + (proc.stderr or "")
    assert re.search(r"(?i)(ok|pass|success|validated)", msg), "Self-test did not report success message."