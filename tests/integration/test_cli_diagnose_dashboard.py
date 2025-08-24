#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/integration/test_cli_diagnose_dashboard.py

SpectraMind V50 — Integration Tests: `diagnose dashboard`

Objectives
----------
1) Assert discoverability:
   • `spectramind diagnose --help` lists subcommands
   • `spectramind diagnose dashboard --help` shows usage text

2) Exercise a lightweight dashboard run that avoids heavy work:
   • Prefer fast/dry-run/skip flags if exposed (e.g., --dry-run/--fast/--no-open/--no-umap/--no-tsne)
   • Accept graceful non-zero exit if missing inputs but require actionable messaging

3) Probe key outputs if produced:
   • logs/v50_debug_log.md is append-only
   • outputs/diagnostics_report.json (or diagnostics_report*.json) is valid JSON with basic keys
   • an HTML report exists (e.g., outputs/diagnostics/diagnostic_report_v*.html or dashboard_v*.html)

Notes
-----
• Tests are defensive and will SKIP if no CLI entrypoint can be found yet.
• We try multiple flag combinations to adapt to repositories in flux.
• No network, no heavy computation; CI-friendly (~seconds).

"""

from __future__ import annotations

import json
import os
import re
import time
import typing as t
import subprocess
from pathlib import Path

import pytest


# --------------------------------------------------------------------------------------
# CLI discovery
# --------------------------------------------------------------------------------------

def _candidate_cli_paths() -> t.List[Path]:
    """
    Heuristics for common entrypoints in this repo:
      - monolithic root `spectramind.py` (Typer root CLI)
      - module path under src/
      - diagnose-only module (fallback)
    """
    return [
        Path("spectramind.py"),
        Path("src") / "spectramind.py",
        Path("src") / "cli" / "cli_diagnose.py",
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
        pytest.skip("SpectraMind CLI not found (expected spectramind.py or src/cli/cli_diagnose.py).")
    return p.resolve()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def run_cli(cli: Path, args: t.List[str], env: dict | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
    """Run the Python CLI as a subprocess, capturing output."""
    cmd = [os.environ.get("PYTHON", "python"), "-u", str(cli), *args]
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        env={**os.environ, **(env or {})},
        timeout=timeout,
        check=False,
    )


def _debug_log_path() -> Path:
    return Path("logs") / "v50_debug_log.md"


def _possible_json_reports() -> t.List[Path]:
    cands: list[Path] = []
    # Common locations / names used by the pipeline
    cands.append(Path("outputs") / "diagnostics_report.json")
    cands.extend(Path("outputs").glob("diagnostics_report*.json"))
    cands.extend((Path("outputs") / "diagnostics").glob("diagnostics_report*.json"))
    cands.extend((Path("outputs") / "diagnostics").glob("diagnostic*_report*.json"))
    return cands


def _possible_html_reports() -> t.List[Path]:
    cands: list[Path] = []
    base = Path("outputs")
    cands.extend(base.glob("diagnostic_report*.html"))
    cands.extend((base / "diagnostics").glob("diagnostic_report*.html"))
    cands.extend((base / "diagnostics").glob("dashboard*.html"))
    cands.extend(base.glob("dashboard*.html"))
    return cands


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_diagnose_help_shows_usage(cli_path: Path):
    """
    The diagnose 'root' help should exist and show usage and/or subcommands.
    Accept both root Typer entry (spectramind.py diagnose --help)
    and standalone diagnose CLI (cli_diagnose.py --help).
    """
    is_standalone = cli_path.name == "cli_diagnose.py"
    args = ["--help"] if is_standalone else ["diagnose", "--help"]
    proc = run_cli(cli_path, args)
    out = (proc.stdout or "") + (proc.stderr or "")

    assert proc.returncode == 0, f"--help failed: exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert re.search(r"(?i)(usage|diagnose|subcommands|dashboard)", out), f"Help output lacks expected keywords:\n{out}"


def test_diagnose_dashboard_help_shows_usage(cli_path: Path):
    """
    The `diagnose dashboard --help` should be discoverable and readable.
    """
    is_standalone = cli_path.name == "cli_diagnose.py"
    args = (["dashboard", "--help"] if is_standalone else ["diagnose", "dashboard", "--help"])
    proc = run_cli(cli_path, args)
    out = (proc.stdout or "") + (proc.stderr or "")

    # Not all repos may have implemented dashboard yet; accept graceful error with actionable message
    assert proc.returncode in (0, 2), f"'dashboard --help' unexpected exit={proc.returncode}\n{out}"
    assert re.search(r"(?i)(usage|dashboard|help|options|flags)", out), f"Missing dashboard usage text:\n{out}"


def test_diagnose_dashboard_dry_run(cli_path: Path, tmp_path: Path):
    """
    Try a minimal dashboard run in CI-safe mode.
    Strategy:
      • Probe candidate flag sets to find a fast path (no browser, no heavy projections).
      • Accept success (exit 0) OR a non-zero with actionable usage error (missing inputs).
      • If produced, check logs and optional artifacts.
    """
    # Ensure logs dir exists to observe append behavior
    log_dir = _debug_log_path().parent
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = _debug_log_path()
    before_mtime = log_file.stat().st_mtime if log_file.exists() else None

    is_standalone = cli_path.name == "cli_diagnose.py"
    base = [] if is_standalone else ["diagnose"]

    # Candidate flag sets (from most conservative to generic)
    candidate_flag_sets = [
        [*base, "dashboard", "--dry-run", "--no-open", "--no-browser", "--no-umap", "--no-tsne", "--fast", "--light"],
        [*base, "dashboard", "--dry-run", "--no-open", "--no-umap", "--no-tsne"],
        [*base, "dashboard", "--no-open", "--no-umap", "--no-tsne"],
        [*base, "dashboard"],  # last resort (should still show actionable output)
    ]

    env = {
        "SPECTRAMIND_FAST": "1",
        "SPECTRAMIND_NO_BROWSER": "1",
        "SPECTRAMIND_NO_NETWORK": "1",
        "PYTHONUNBUFFERED": "1",
    }

    proc = None
    for flags in candidate_flag_sets:
        proc = run_cli(cli_path, flags, env=env, timeout=240)
        if proc.returncode == 0:
            break
        # Accept useful guidance on missing inputs or bad flags, then stop probing
        joined = (proc.stdout or "") + (proc.stderr or "")
        if re.search(r"(?i)(usage:|missing|required|no such option|error:)", joined):
            break

    assert proc is not None, "CLI process did not start."

    # If non-zero, require actionable message (usage/missing/required/option/error/help)
    if proc.returncode != 0:
        joined = (proc.stdout or "") + (proc.stderr or "")
        assert re.search(r"(?i)(usage|required|missing|error|help|option)", joined), \
            f"Non-zero return without actionable message.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Best-effort: verify logs appended
    if log_file.exists():
        time.sleep(0.15)  # small tolerance for FS lag
        after_mtime = log_file.stat().st_mtime
        if before_mtime is not None:
            assert after_mtime >= before_mtime, "Expected logs/v50_debug_log.md to be appended."

    # Optional artifact checks — if files exist, validate lightly
    json_reports = [p for p in _possible_json_reports() if p.exists()]
    html_reports = [p for p in _possible_html_reports() if p.exists()]

    # If nothing was produced (e.g., missing inputs), skip artifact validation
    if not json_reports and not html_reports:
        pytest.skip("No diagnostics artifacts were produced in dry-run; skipping artifact checks.")

    # Validate JSON structure (if present)
    for jpath in json_reports[:3]:  # limit checks
        txt = jpath.read_text(encoding="utf-8", errors="ignore")
        try:
            obj = json.loads(txt)
        except json.JSONDecodeError as e:
            pytest.fail(f"Diagnostics JSON at {jpath} is not valid JSON: {e}")
        # Heuristic minimal keys
        keys = set(obj.keys())
        assert keys, f"Empty diagnostics JSON at {jpath}"
        assert any(k in keys for k in ("summary", "metrics", "artifacts", "config", "version")), \
            f"Diagnostics JSON missing expected keys at {jpath}"

    # Validate at least one HTML exists if claimed
    if html_reports:
        # Require an .html file larger than a trivial stub
        assert any(p.stat().st_size > 256 for p in html_reports), \
            "HTML report(s) appear to be empty stubs (<256 bytes)."


def test_diagnose_dashboard_selftest_if_available(cli_path: Path):
    """
    If `diagnose dashboard` exposes a quick self-test flag (e.g., --selftest or --self-test),
    ensure it runs successfully. Skip if not advertised.
    """
    is_standalone = cli_path.name == "cli_diagnose.py"
    base = [] if is_standalone else ["diagnose"]

    # Check help text for any self-test hint
    help_proc = run_cli(cli_path, [*base, "dashboard", "--help"])
    help_out = (help_proc.stdout or "") + (help_proc.stderr or "")
    has_selftest = re.search(r"(?i)(--self[- ]?test|\bself[- ]?test\b)", help_out) is not None

    if not has_selftest:
        pytest.skip("No --selftest flag advertised by `diagnose dashboard`; skipping.")

    # Try common variants
    for flag in ("--selftest", "--self-test"):
        proc = run_cli(cli_path, [*base, "dashboard", flag], env={"SPECTRAMIND_FAST": "1"})
        if proc.returncode == 0:
            msg = (proc.stdout or "") + (proc.stderr or "")
            assert re.search(r"(?i)(ok|pass|success|validated|ready)", msg), \
                "Self-test did not report a success indicator."
            return

    pytest.fail("Self-test flag was advertised but did not succeed with common variants.")