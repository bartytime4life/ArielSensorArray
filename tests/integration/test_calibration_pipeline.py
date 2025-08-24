#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArielSensorArray/tests/integration/test_calibration_pipeline.py

SpectraMind V50 — Calibration Pipeline Integration Tests (pytest)

Goal
----
Exercise the repository's calibration stage end-to-end through the unified
Typer CLI in a fast, side‑effect‑safe manner, verifying:

  • The `calibrate` subcommand exists and runs with `--dry-run`.
  • Hydra overrides route all artifacts under a test `paths.output_dir`.
  • Append‑only audit log (logs/v50_debug_log.md) is written per run.
  • Optional structured JSONL stream (logs/v50_runs.jsonl) — tolerated if absent.
  • Optional run-hash snapshot (outputs/run_hash_summary_v50.json) — tolerated if absent.
  • Optional calibration artifacts land under canonical paths such as:
      - outputs/calibrated/ (e.g., lightcurves.h5, *.npy, *.json)
      - outputs/diagnostics/ (e.g., calibration plots/reports)
    Tests tolerate absences in strict `--dry-run`, but validate shape if present.

Additionally, we probe for the calibration checker help path(s) to ensure wiring
exists (e.g., `diagnose check-calibration --help`). We don't depend on real data.

If the Typer CLI cannot be imported in the test environment, tests are skipped
gracefully to keep CI green for partial environments.

Requirements
-----------
• pytest
• typer (for CliRunner)
• The repo's Python package importable in the test environment
"""

from __future__ import annotations

import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# ---------------------------------------------------------------------
# Attempt to import the unified Typer CLI app. We support common layouts:
#   1) top-level `spectramind.py` exposing `app`
#   2) packaged module at `src/cli/spectramind.py` exposing `app`
# If unavailable (or typer.testing missing), tests will be skipped.
# ---------------------------------------------------------------------

_TyperApp = None
_CliRunner = None

try:
    from spectramind import app as _TyperApp  # type: ignore[attr-defined]
except Exception:
    try:
        from src.cli.spectramind import app as _TyperApp  # type: ignore[attr-defined]
    except Exception:
        _TyperApp = None

if _TyperApp is not None:
    try:
        from typer.testing import CliRunner as _CliRunner  # type: ignore
    except Exception:
        _TyperApp = None  # Mark as unavailable if Typer test harness missing


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

@dataclass
class CalibrationArtifacts:
    """
    Canonical paths under a given run's output directory.

    We keep expectations flexible to accommodate strict dry-run modes:
    - If a file/dir is present, we validate its basic shape.
    - If absent under dry-run, we do not fail the test.
    """
    outdir: Path
    logs_dir: Path
    debug_log: Path
    runs_jsonl: Path
    run_hash_json: Path
    outputs_dir: Path
    calibrated_dir: Path
    diagnostics_dir: Path
    # Common calibrated outputs (optional)
    lightcurves_h5: Path
    lc_json: Path

    @staticmethod
    def locate(outdir: Path) -> "CalibrationArtifacts":
        logs_dir = outdir / "logs"
        outputs = outdir / "outputs"
        calibrated = outputs / "calibrated"
        diagnostics = outputs / "diagnostics"
        return CalibrationArtifacts(
            outdir=outdir,
            logs_dir=logs_dir,
            debug_log=logs_dir / "v50_debug_log.md",
            runs_jsonl=logs_dir / "v50_runs.jsonl",
            run_hash_json=outputs / "run_hash_summary_v50.json",
            outputs_dir=outputs,
            calibrated_dir=calibrated,
            diagnostics_dir=diagnostics,
            lightcurves_h5=calibrated / "lightcurves.h5",
            lc_json=calibrated / "lightcurves.json",
        )


def _mk_clean_dir(p: Path) -> Path:
    """Ensure a clean directory exists at path p."""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _env_for_cli(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Construct a clean environment for CLI execution with Hydra friendliness.

    Notes:
    • HYDRA_FULL_ERROR=1 gives cleaner tracebacks if something goes wrong.
    • TERM and PYTHONUNBUFFERED stabilize output/log behavior in CI.
    """
    env = dict(os.environ)
    env["HYDRA_FULL_ERROR"] = "1"
    env["TERM"] = "xterm-256color"
    env["PYTHONUNBUFFERED"] = "1"
    if extra:
        env.update(extra)
    return env


def _run_cli(
    runner,  # typer.testing.CliRunner
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    cwd: Optional[Path] = None,
):
    """Invoke the Typer app with CliRunner and return the result object."""
    kwargs = {}
    if env is not None:
        kwargs["env"] = env
    if cwd is not None:
        kwargs["cwd"] = str(cwd)
    return runner.invoke(_TyperApp, cmd, **kwargs)


# ---------------------------------------------------------------------
# Pytest fixtures
# ---------------------------------------------------------------------

@pytest.fixture(scope="session")
def cli_available() -> bool:
    """Skip all tests if the unified Typer CLI is not importable."""
    if _TyperApp is None or _CliRunner is None:
        pytest.skip("Spectramind unified CLI (Typer app) not importable in this environment.")
    return True


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """Provide a clean per-test workspace directory."""
    return _mk_clean_dir(tmp_path / "workspace")


@pytest.fixture()
def runner(cli_available) -> "_CliRunner":  # type: ignore[name-defined]
    """Provide a Typer CliRunner instance."""
    return _CliRunner(mix_stderr=False)


# ---------------------------------------------------------------------
# Tests — Core calibration flow
# ---------------------------------------------------------------------

def test_calibrate_cli_dry_run_executes_and_logs(runner, workdir: Path):
    """
    Smoke: `spectramind calibrate` runs with `--dry-run` + fast overrides,
    exits code 0, and writes the append-only audit log and (optionally)
    the structured JSONL event stream. It must not write outside the
    configured `paths.output_dir`.
    """
    outdir = _mk_clean_dir(workdir / "cal_run_A")

    cmd = [
        "calibrate",
        "--dry-run",
        # Route artifacts to temp outdir:
        f"paths.output_dir={outdir}",
        # Keep it fast, minimal:
        "calibration.fast_mode=true",   # tolerant: if unknown, Hydra will ignore or map if configured
        "data.sample_limit=8",
        # Deterministic:
        "repro.seed=9090",
    ]

    res = _run_cli(runner, cmd, env=_env_for_cli(), cwd=workdir)
    assert res.exit_code == 0, f"CLI exited non-zero.\nSTDOUT:\n{res.stdout}"

    arts = CalibrationArtifacts.locate(outdir)

    # Logs dir + audit log required
    assert arts.logs_dir.exists(), "logs/ directory not created under output_dir"
    assert arts.debug_log.is_file(), "v50_debug_log.md not found; audit trail is required"

    # Audit log should mention 'calibrate' and include some metadata markers
    dbg = arts.debug_log.read_text(encoding="utf-8")
    assert re.search(r"\bcalibrate\b", dbg, flags=re.IGNORECASE), "Audit log should mention 'calibrate' subcommand"
    assert re.search(r"\bconfig\b|\bhash\b|\btimestamp\b|\bseed\b", dbg, flags=re.IGNORECASE), \
        "Audit log should include config/hash/timestamp/seed markers"

    # Optional structured JSONL stream: if present, ensure not empty
    if arts.runs_jsonl.exists():
        text = arts.runs_jsonl.read_text(encoding="utf-8").strip()
        assert text, "v50_runs.jsonl exists but is empty"

    # If strict dry-run still writes placeholder dirs/files, sanity-check their placement.
    if arts.calibrated_dir.exists():
        assert arts.calibrated_dir.is_dir(), "outputs/calibrated exists but is not a directory"
        # Heuristic: at least one plausible artifact is present
        some = list(arts.calibrated_dir.glob("*"))
        assert any(p.suffix.lower() in {".h5", ".npy", ".json", ".csv"} for p in some), \
            "outputs/calibrated exists but contains no recognizable calibration artifacts"

    if arts.diagnostics_dir.exists():
        assert arts.diagnostics_dir.is_dir(), "outputs/diagnostics exists but is not a directory"


def test_calibrate_cli_respects_output_routing_and_paths(runner, workdir: Path):
    """
    Verify artifacts (when generated) are confined under `paths.output_dir`
    and that any calibration outputs reside under `outputs/calibrated/`.
    """
    outdir = _mk_clean_dir(workdir / "cal_run_paths")

    res = _run_cli(
        runner,
        [
            "calibrate",
            "--dry-run",
            f"paths.output_dir={outdir}",
            "calibration.fast_mode=true",
            "repro.seed=7",
        ],
        env=_env_for_cli(),
        cwd=workdir,
    )
    assert res.exit_code == 0, f"CLI failed:\n{res.stdout}"

    arts = CalibrationArtifacts.locate(outdir)

    # Ensure the output dir is present and no unexpected top-level dirs appear
    siblings = [p for p in workdir.iterdir() if p.is_dir()]
    assert outdir in siblings, "Configured output directory missing"
    assert len(siblings) <= 2, (
        "Unexpected sibling directories created; all artifacts should route under the configured output_dir."
    )

    # If calibrated outputs exist, ensure they are under outputs/calibrated
    if arts.calibrated_dir.exists():
        assert arts.calibrated_dir.is_dir(), "outputs/calibrated exists but is not a directory"
        # Any known file type indicates at least a stub
        found_types = set(p.suffix.lower() for p in arts.calibrated_dir.glob("*.*"))
        assert found_types & {".h5", ".json", ".npy", ".csv"}, \
            "outputs/calibrated contains no recognized calibration artifacts (.h5/.json/.npy/.csv)"


def test_calibrate_run_hash_seed_sensitivity_when_available(runner, workdir: Path):
    """
    If the pipeline writes `outputs/run_hash_summary_v50.json` during calibrate,
    then with all else equal a seed change should typically alter the hash.

    In strict dry-run modes, the file may be omitted — xfail that case.
    """
    out1 = _mk_clean_dir(workdir / "cal_hash_seed_A")
    out2 = _mk_clean_dir(workdir / "cal_hash_seed_B")

    common = [
        "calibration.fast_mode=true",
        "data.sample_limit=4",
    ]

    r1 = _run_cli(
        runner,
        ["calibrate", "--dry-run", f"paths.output_dir={out1}", "repro.seed=111", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )
    r2 = _run_cli(
        runner,
        ["calibrate", "--dry-run", f"paths.output_dir={out2}", "repro.seed=222", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )

    assert r1.exit_code == 0 and r2.exit_code == 0, "Calibrate dry-run failed."

    meta1 = out1 / "outputs" / "run_hash_summary_v50.json"
    meta2 = out2 / "outputs" / "run_hash_summary_v50.json"

    if not (meta1.exists() and meta2.exists()):
        pytest.xfail("run_hash_summary_v50.json not emitted during calibrate dry-run; cannot validate seed sensitivity.")

    j1 = json.loads(meta1.read_text(encoding="utf-8"))
    j2 = json.loads(meta2.read_text(encoding="utf-8"))

    key = "config_hash"
    if key in j1 and key in j2:
        equal = (j1[key] == j2[key])
    else:
        sig1 = json.dumps({k: j1.get(k) for k in sorted(j1.keys()) if "hash" in k.lower() or "seed" in k.lower()}, sort_keys=True)
        sig2 = json.dumps({k: j2.get(k) for k in sorted(j2.keys()) if "hash" in k.lower() or "seed" in k.lower()}, sort_keys=True)
        equal = (sig1 == sig2)

    assert equal is False, "Hashes/signatures should differ when seeds differ during calibrate (if hashing includes seed)."


def test_calibration_checker_help_is_wired_somewhere(runner, workdir: Path):
    """
    Probe common wiring for a calibration checker validator. We don't require
    a specific command layout; instead, we try several plausible entrypoints
    and pass if any prints help/usage without error.

    Examples we probe:
      - spectramind diagnose check-calibration --help
      - spectramind diagnose calibration-check --help
      - spectramind check-calibration --help
      - spectramind diagnose calibration --help
    """
    candidates = [
        ["diagnose", "check-calibration", "--help"],
        ["diagnose", "calibration-check", "--help"],
        ["check-calibration", "--help"],
        ["diagnose", "calibration", "--help"],
    ]
    for cmd in candidates:
        res = _run_cli(runner, cmd, env=_env_for_cli(), cwd=workdir)
        if res.exit_code == 0 and ("usage:" in res.stdout.lower() or "options:" in res.stdout.lower()):
            return
    pytest.xfail("No calibration-checker help path found among common entrypoints.")


def test_calibration_help_is_discoverable(runner, workdir: Path):
    """
    Regression: `spectramind calibrate --help` should print usage/options text.
    """
    res = _run_cli(runner, ["calibrate", "--help"], env=_env_for_cli(), cwd=workdir)
    assert res.exit_code == 0
    out = res.stdout.lower()
    assert "usage:" in out or "options:" in out or "help" in out, "Help output missing expected sections"
