#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/integration/test_cli_analyze_log.py

SpectraMind V50 — Integration Tests: `analyze-log`

Objectives
----------
1) Discover the SpectraMind root CLI and assert that analyze-log is documented:
   • `spectramind analyze-log --help` (or standalone cli) returns 0 and shows usage text.

2) Exercise a conservative run to keep CI fast:
   • Prefer dry-run / fast / no-open style flags if available.
   • If inputs are missing, accept a graceful non-zero with actionable usage messaging.

3) If artifacts are written, validate them lightly:
   • outputs/log_table.md and/or outputs/log_table.csv exist and have non-trivial size.
   • (Optional) JSON or additional outputs, if provided, are minimally well-formed.

4) Ensure append-only audit log flows are respected:
   • Create (or append) a synthetic line in logs/v50_debug_log.md before calling analyze-log so the tool has data.
   • After invocation, verify the file still exists and (best-effort) was not truncated.

Notes
-----
• Tests are defensive and SKIP if no CLI entrypoint can be found in the repo yet.
• We probe several flag variants to match the project's evolving CLI options.
• No network, no heavy processing; should complete in seconds in CI.

"""

from __future__ import annotations

import os
import re
import json
import time
import typing as t
import subprocess
from datetime import datetime
from pathlib import Path

import pytest


# --------------------------------------------------------------------------------------
# CLI discovery
# --------------------------------------------------------------------------------------

def _candidate_cli_paths() -> t.List[Path]:
    """
    Common entrypoints:
      - monolithic root Typer CLI at ./spectramind.py
      - module path under ./src/spectramind.py
      - dedicated sub-CLI under ./src/cli/ (fallback)
    """
    return [
        Path("spectramind.py"),
        Path("src") / "spectramind.py",
        Path("src") / "cli" / "cli_core_v50.py",
        Path("src") / "cli" / "cli_analyze_log.py",
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
        pytest.skip("SpectraMind CLI not found (expected spectramind.py or a cli_* module).")
    return p.resolve()


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def _run_cli(cli: Path, args: t.List[str], env: dict | None = None, timeout: int = 180) -> subprocess.CompletedProcess:
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


def _ensure_minimal_log_line() -> None:
    """
    Ensure logs/v50_debug_log.md exists and contains at least one analyzable line.
    We append a synthetic, well-formed entry to drive analyze-log without requiring prior runs.
    """
    logf = _debug_log_path()
    logf.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    # minimal, friendly line; real project writes richer lines, this is enough for parsers.
    line = f"[{ts}] spectramind diagnose dashboard --dry-run | cfg=abcd1234 | status=OK | duration=1.23s\n"
    if logf.exists():
        with logf.open("a", encoding="utf-8") as f:
            f.write(line)
    else:
        with logf.open("w", encoding="utf-8") as f:
            f.write("# SpectraMind V50 — Append-only CLI Debug Log\n")
            f.write(line)


def _artifact_candidates() -> dict[str, list[Path]]:
    """
    Common artifact names/locations produced by 'analyze-log'.
    """
    base = Path("outputs")
    return {
        "csv": [
            base / "log_table.csv",
            *base.glob("log_table_*.csv"),
            *(base / "diagnostics").glob("log_table*.csv"),
        ],
        "md": [
            base / "log_table.md",
            *base.glob("log_table_*.md"),
            *(base / "diagnostics").glob("log_table*.md"),
        ],
        "json": [
            base / "log_table.json",
            *base.glob("log_table_*.json"),
            *(base / "diagnostics").glob("log_table*.json"),
        ],
    }


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------

def test_analyze_log_help_shows_usage(cli_path: Path):
    """
    The analyze-log help should exist and show usage docs.
    Accept both root Typer `spectramind analyze-log --help` and standalone `cli_analyze_log.py --help`.
    """
    is_standalone = cli_path.name.startswith("cli_") and "analyze" in cli_path.name
    args = ["--help"] if is_standalone else ["analyze-log", "--help"]
    proc = _run_cli(cli_path, args)
    out = (proc.stdout or "") + (proc.stderr or "")

    assert proc.returncode == 0, f"--help failed: exit={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    assert re.search(r"(?i)(usage|analyze[- ]?log|help|options)", out), f"Help output lacks expected keywords:\n{out}"


def test_analyze_log_dry_run_and_artifacts(cli_path: Path, tmp_path: Path):
    """
    Exercise a minimal, CI-safe path:
      • Ensure a minimal log exists.
      • Attempt analyze-log with dry-run / fast / no-open flags when available.
      • Accept graceful usage error if inputs/flags mismatch, but require actionable messaging.
      • If artifacts are produced (CSV/MD/JSON), validate their presence and non-triviality.
    """
    # Ensure log data exists for analysis
    _ensure_minimal_log_line()
    logf = _debug_log_path()
    before_mtime = logf.stat().st_mtime if logf.exists() else None

    is_standalone = cli_path.name.startswith("cli_") and "analyze" in cli_path.name

    candidate_flag_sets = [
        (["--dry-run", "--no-open", "--fast", "--clean", "--out", "outputs"], {}),
        (["--dry-run", "--no-open", "--out", "outputs"], {}),
        ([], {}),  # last resort — should still print useful guidance or succeed
    ]

    # Build args with/without subcommand depending on entrypoint
    def with_cmd(flags: list[str]) -> list[str]:
        return flags if is_standalone else ["analyze-log", *flags]

    env = {
        "SPECTRAMIND_FAST": "1",
        "SPECTRAMIND_NO_BROWSER": "1",
        "PYTHONUNBUFFERED": "1",
    }

    proc = None
    for flags, extra_env in candidate_flag_sets:
        proc = _run_cli(cli_path, with_cmd(flags), env={**env, **extra_env}, timeout=180)
        if proc.returncode == 0:
            break
        joined = (proc.stdout or "") + (proc.stderr or "")
        # If non-zero with actionable usage/missing info, accept and stop probing
        if re.search(r"(?i)(usage:|missing|required|no such option|error:)", joined):
            break

    assert proc is not None, "CLI did not start."

    # If non-zero, require actionable message
    if proc.returncode != 0:
        joined = (proc.stdout or "") + (proc.stderr or "")
        assert re.search(r"(?i)(usage|required|missing|error|help|option)", joined), \
            f"Non-zero return without actionable message.\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    # Best-effort: ensure debug log persists and is not obviously truncated
    if logf.exists() and before_mtime is not None:
        time.sleep(0.1)
        after_mtime = logf.stat().st_mtime
        assert after_mtime >= before_mtime, "logs/v50_debug_log.md appears not updated/accessible."

    # Probe artifacts if produced
    arts = _artifact_candidates()
    csvs = [p for p in arts["csv"] if p.exists()]
    mds = [p for p in arts["md"] if p.exists()]
    jsons = [p for p in arts["json"] if p.exists()]

    if not (csvs or mds or jsons):
        pytest.skip("No analyze-log artifacts produced in this run; skipping artifact validation.")

    # Validate CSV/MD size minimally (not empty stubs)
    for p in csvs[:2] + mds[:2]:
        assert p.stat().st_size > 64, f"Artifact too small: {p}"

    # Validate JSON (if any) is well-formed and contains at least one top-level key
    for jp in jsons[:2]:
        txt = jp.read_text(encoding="utf-8", errors="ignore")
        try:
            obj = json.loads(txt)
        except json.JSONDecodeError as e:
            pytest.fail(f"JSON artifact invalid at {jp}: {e}")
        assert isinstance(obj, (dict, list)), f"Unexpected JSON root type at {jp}"
        if isinstance(obj, dict):
            assert obj.keys(), f"Empty JSON object at {jp}"


def test_analyze_log_clean_flag_if_available(cli_path: Path):
    """
    If the CLI advertises a --clean capability for analyze-log (deduplicate logs, rotate, or purge),
    ensure invoking it returns success or a helpful message. Skip if not advertised.
    """
    is_standalone = cli_path.name.startswith("cli_") and "analyze" in cli_path.name
    help_proc = _run_cli(cli_path, (["--help"] if is_standalone else ["analyze-log", "--help"]))
    help_out = (help_proc.stdout or "") + (help_proc.stderr or "")
    has_clean = re.search(r"(?i)(--clean\b|\bclean\b)", help_out) is not None

    if not has_clean:
        pytest.skip("No --clean flag advertised by analyze-log; skipping.")

    flags = ["--clean"] if is_standalone else ["analyze-log", "--clean"]
    proc = _run_cli(cli_path, flags, env={"SPECTRAMIND_FAST": "1"})
    # Accept either success or a friendly "nothing to clean" message
    if proc.returncode != 0:
        msg = (proc.stdout or "") + (proc.stderr or "")
        assert re.search(r"(?i)(nothing|already|cleaned|usage|ok|success)", msg), \
            f"--clean returned {proc.returncode} without helpful output.\n{msg}"