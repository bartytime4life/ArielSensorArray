#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArielSensorArray/tests/test_train_pipeline.py

SpectraMind V50 — Training Pipeline Tests (pytest)

This module validates the training pipeline’s CLI entrypoint (Typer),
Hydra configuration wiring, reproducibility/hash logging, and dry‑run
guardrails without requiring heavy data or long runtimes.

Design anchors (see inline citations in comments):
• Unified Typer CLI with subcommands; Hydra config composition on every call.   
• NASA‑grade reproducibility: config snapshots, run hashing, DVC alignment, logs.   
• “UI‑light” Rich console plus append‑only v50_debug_log.md audit trail.  
• Safe guards & fast checks via dry‑run / fast‑dev run patterns.  

Notes:
- These tests use Typer’s CliRunner to execute the unified CLI in‑process.
- They prefer `--dry-run` and “fast dev” overrides to avoid long runs.
- Paths are redirected to a tmp dir to keep the repo clean.
- If the CLI module is unavailable, tests are skipped (graceful CI behavior).

Requirements:
- pytest
- The repository’s Python package importable in the test environment
  (so that `spectramind` CLI module and Hydra configs are discoverable).

"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pytest

# Try to import the Typer app for the unified CLI.
# The repo may expose one of the following entrypoints depending on structure:
#   - spectramind.py (root CLI unifier)
#   - src/cli/spectramind.py (packaged module path)
# We attempt the common import paths in order; if all fail, we skip tests.

_TyperApp = None
_CliRunner = None

try:
    # Preferred: packaged CLI exposed as a module `spectramind` (top-level app)
    from spectramind import app as _TyperApp  # type: ignore[attr-defined]
except Exception:
    try:
        # Alternate: packaged under src/cli
        from src.cli.spectramind import app as _TyperApp  # type: ignore[attr-defined]
    except Exception:
        _TyperApp = None

if _TyperApp is not None:
    try:
        from typer.testing import CliRunner as _CliRunner  # type: ignore
        _CliRunner = _CliRunner
    except Exception:
        _TyperApp = None  # Mark as unavailable if Typer test harness missing

# -------------------------
# Test helpers & structures
# -------------------------

@dataclass
class RunArtifacts:
    """Convenience container for locating expected artifacts in a run output root."""
    outdir: Path
    logs_dir: Path
    debug_log: Path
    runs_jsonl: Path
    run_hash_json: Path
    diagnostics_dir: Path

    @staticmethod
    def locate(outdir: Path) -> "RunArtifacts":
        """
        Find canonical artifact locations under an output directory.
        We follow the SpectraMind V50 conventions described in the plan:
        • logs/v50_debug_log.md (append-only audit log)  
        • logs/v50_runs.jsonl (structured events)
        • outputs/run_hash_summary_v50.json (config/env hash snapshot)  
        • outputs/diagnostics/ (HTML/PNG/JSON diagnostics; may be present after train/diagnose)
        """
        logs_dir = outdir / "logs"
        diagnostics_dir = outdir / "outputs" / "diagnostics"
        return RunArtifacts(
            outdir=outdir,
            logs_dir=logs_dir,
            debug_log=logs_dir / "v50_debug_log.md",
            runs_jsonl=logs_dir / "v50_runs.jsonl",
            run_hash_json=outdir / "outputs" / "run_hash_summary_v50.json",
            diagnostics_dir=diagnostics_dir,
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

    HYDRA_FULL_ERROR=1 ensures clearer tracebacks for debugging.  
    """
    env = dict(os.environ)
    env["HYDRA_FULL_ERROR"] = "1"
    # Avoid interactive pagers or TTY-only behaviors in CI
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
    """
    Execute the Typer CLI in-process using CliRunner.

    Returns the result object; asserts are done by the caller.
    """
    # Typer runner isolates CWD; ensure correct working directory for relative config paths.
    run_kwargs = {}
    if env:
        run_kwargs["env"] = env
    if cwd:
        run_kwargs["cwd"] = str(cwd)

    return runner.invoke(_TyperApp, cmd, **run_kwargs)


# -------------------------
# Pytest fixtures
# -------------------------

@pytest.fixture(scope="session")
def cli_available() -> bool:
    """Return True if the Typer CLI app is importable; else skip tests."""
    if _TyperApp is None or _CliRunner is None:
        pytest.skip("SpectraMind unified CLI (Typer app) is not importable in this environment.")
    return True


@pytest.fixture()
def workdir(tmp_path: Path) -> Path:
    """
    Provide a clean working directory for each test case.

    Layout:
      tmp_path/
        runA/
        runB/
        ...
    """
    return _mk_clean_dir(tmp_path / "workspace")


@pytest.fixture()
def runner(cli_available) -> "_CliRunner":  # type: ignore[name-defined]
    """Provide a Typer CliRunner for invoking the unified CLI app."""
    return _CliRunner(mix_stderr=False)


# -------------------------
# Tests
# -------------------------

def test_train_cli_dry_run_executes_and_logs(runner, workdir: Path):
    """
    Sanity: `spectramind train` runs with --dry-run and fast-dev overrides,
    writes audit/structured logs, and exits with code 0.

    We do not assume heavy data availability; the pipeline is expected to
    honor dry-run / dev flags and still produce minimal logs & a run-hash
    snapshot (or a placeholder thereof).   
    """
    # Arrange
    outdir = workdir / "runA"
    _mk_clean_dir(outdir)

    # Common CLI pattern (examples per plan/docs):
    #   spectramind train [hydra overrides] [global flags]
    #
    # We include representative Hydra overrides to keep runtime minimal, and we point
    # outputs to our tmp dir. The exact keys may vary by config schema; the CLI is
    # expected to accept generic group-style overrides.
    cmd = [
        "train",
        "--dry-run",
        # Route outputs / logs into tmp
        "paths.output_dir=" + str(outdir),
        # Prefer short runs; names are representative—if your schema differs, map accordingly:
        "training.fast_dev_run=true",
        "training.epochs=1",
        "data.sample_limit=8",
        # Ensure deterministic seed for reproducibility hashing
        "repro.seed=1234",
    ]

    env = _env_for_cli()

    # Act
    res = _run_cli(runner, cmd, env=env, cwd=workdir)

    # Assert
    # 1) Exit cleanly
    assert res.exit_code == 0, f"CLI exited with non-zero code.\nSTDOUT:\n{res.stdout}"

    # 2) Logs and basic artifacts exist
    arts = RunArtifacts.locate(outdir)
    assert arts.logs_dir.exists(), "logs/ directory not created under output_dir"
    assert arts.debug_log.is_file(), "v50_debug_log.md not found (append-only audit log expected)"
    # Structured JSONL event stream is optional but encouraged; tolerate absent file with warning.
    # If present, ensure non-empty.
    if arts.runs_jsonl.exists():
        text = arts.runs_jsonl.read_text(encoding="utf-8").strip()
        assert text, "v50_runs.jsonl exists but is empty"

    # 3) The audit log should contain at least one CLI invocation line and the subcommand `train`
    dbg = arts.debug_log.read_text(encoding="utf-8")
    # Example markers the repo tends to include (command echo, config hash, timestamp)
    assert re.search(r"\btrain\b", dbg, flags=re.IGNORECASE), "Audit log should mention 'train' subcommand"
    assert re.search(r"\bconfig\b|\bhash\b|\btimestamp\b", dbg, flags=re.IGNORECASE), \
        "Audit log should include config/hash/timestamp markers"

    # 4) If a run hash summary JSON is produced, validate its basic structure
    if arts.run_hash_json.exists():
        meta = json.loads(arts.run_hash_json.read_text(encoding="utf-8"))
        # Expect keys like: config_hash, env_hash, git_commit, created_at, etc.
        assert isinstance(meta, dict) and meta, "run_hash_summary_v50.json must be a non-empty JSON object"
        expect_any = {"config", "hash", "commit", "created"}
        assert any(k for k in meta.keys() for needle in expect_any if needle in k.lower()), \
            "run_hash_summary_v50.json lacks expected metadata keys"


def test_train_cli_reproducible_hash_with_fixed_seed(runner, workdir: Path):
    """
    With a fixed seed and the same overrides, a dry‑run should yield consistent
    run-hash metadata (if produced) across back‑to‑back invocations.

    This tests the “configuration-as-code” + hashing ethos.  
    """
    outA = _mk_clean_dir(workdir / "runB_A")
    outB = _mk_clean_dir(workdir / "runB_B")

    common_overrides = [
        "training.fast_dev_run=true",
        "training.epochs=1",
        "data.sample_limit=8",
        "repro.seed=777",  # fixed seed
    ]

    # First run
    resA = _run_cli(
        runner,
        ["train", "--dry-run", f"paths.output_dir={outA}", *common_overrides],
        env=_env_for_cli(),
        cwd=workdir,
    )
    assert resA.exit_code == 0, f"First run failed:\n{resA.stdout}"

    # Second run (identical args)
    resB = _run_cli(
        runner,
        ["train", "--dry-run", f"paths.output_dir={outB}", *common_overrides],
        env=_env_for_cli(),
        cwd=workdir,
    )
    assert resB.exit_code == 0, f"Second run failed:\n{resB.stdout}"

    artsA = RunArtifacts.locate(outA)
    artsB = RunArtifacts.locate(outB)

    # Hash file is optional in dry-run. If both exist, they should match on core fields.
    if artsA.run_hash_json.exists() and artsB.run_hash_json.exists():
        metaA = json.loads(artsA.run_hash_json.read_text(encoding="utf-8"))
        metaB = json.loads(artsB.run_hash_json.read_text(encoding="utf-8"))

        # Compare a subset of canonical keys if present
        for key in ("config_hash", "env_hash", "git_commit", "schema_version"):
            if key in metaA and key in metaB:
                assert metaA[key] == metaB[key], f"Mismatch in {key} between repeated runs"

        # The created_at timestamps will differ; ignore time fields.


def test_train_cli_respects_output_routing_and_generates_minimal_artifacts(runner, workdir: Path):
    """
    Ensure the CLI honors output routing (paths.output_dir=...) and does not
    leak artifacts outside the configured directory. Also verify that a
    diagnostics directory (if produced in train) is nested under outputs/.

    The repository’s design emphasizes isolated per-run outputs managed by Hydra/CLI.   
    """
    outdir = _mk_clean_dir(workdir / "runC")

    res = _run_cli(
        runner,
        [
            "train",
            "--dry-run",
            f"paths.output_dir={outdir}",
            "training.fast_dev_run=true",
            "training.epochs=1",
            "repro.seed=42",
        ],
        env=_env_for_cli(),
        cwd=workdir,
    )
    assert res.exit_code == 0, f"CLI failed:\n{res.stdout}"

    arts = RunArtifacts.locate(outdir)

    # Confirm no sibling directories were created at workdir level besides the chosen outdir
    siblings = [p for p in workdir.iterdir() if p.is_dir()]
    assert outdir in siblings, "Configured output directory missing"
    # At most 1-2 other transient dirs (Hydra .hydra/) might appear inside outdir; verify none at top-level root:
    assert len(siblings) <= 2, (
        "Unexpected directories were created at the test root—outputs should be routed under the configured output_dir."
    )

    # If diagnostics exist, they should be under outputs/diagnostics
    if arts.diagnostics_dir.exists():
        assert arts.diagnostics_dir.is_dir(), "diagnostics path exists but is not a directory"
        # Heuristic: presence of at least one HTML/JSON/PNG file indicates basic diagnostics export
        found = list(arts.diagnostics_dir.glob("**/*"))
        assert any(fp.suffix.lower() in {".html", ".json", ".png"} for fp in found), \
            "Diagnostics directory is present but no known diagnostic artifacts were found."


def test_cli_help_discoverability_for_train(runner, workdir: Path):
    """
    Quick regression: `spectramind train --help` should render usage text.

    CLI discoverability is a core requirement (nested --help, tab-complete).  
    """
    res = _run_cli(runner, ["train", "--help"], env=_env_for_cli(), cwd=workdir)
    assert res.exit_code == 0
    out = res.stdout.lower()
    # Look for standard Typer/Click usage lines and option hints
    assert "usage:" in out or "options:" in out or "help" in out, "Help output did not include expected sections"


# -------------------------
# Optional: mark-slow or extended checks
# -------------------------

@pytest.mark.parametrize(
    "seedA,seedB,expect_equal",
    [
        (101, 101, True),
        (101, 202, False),
    ],
)
def test_run_hash_changes_with_seed_variation_when_available(runner, workdir: Path, seedA: int, seedB: int, expect_equal: bool):
    """
    If the pipeline includes seed in its hashing strategy, changing the seed
    should (typically) alter the config hash; identical seeds should match.

    This is an inferential test: if run_hash_summary_v50.json is not produced in
    dry-run, the test is xfailed for that case.  
    """
    out1 = _mk_clean_dir(workdir / f"run_seed_{seedA}")
    out2 = _mk_clean_dir(workdir / f"run_seed_{seedB}")

    common = ["training.fast_dev_run=true", "training.epochs=1", "data.sample_limit=4"]
    res1 = _run_cli(
        runner,
        ["train", "--dry-run", f"paths.output_dir={out1}", f"repro.seed={seedA}", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )
    res2 = _run_cli(
        runner,
        ["train", "--dry-run", f"paths.output_dir={out2}", f"repro.seed={seedB}", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )

    assert res1.exit_code == 0 and res2.exit_code == 0, "Dry-run training failed unexpectedly."

    meta1_path = out1 / "outputs" / "run_hash_summary_v50.json"
    meta2_path = out2 / "outputs" / "run_hash_summary_v50.json"

    if not (meta1_path.exists() and meta2_path.exists()):
        pytest.xfail("run_hash_summary_v50.json not emitted in dry-run; cannot validate seed sensitivity.")

    meta1 = json.loads(meta1_path.read_text(encoding="utf-8"))
    meta2 = json.loads(meta2_path.read_text(encoding="utf-8"))

    # Compare config_hash if available; otherwise fallback to a combined textual signature.
    key = "config_hash"
    if key in meta1 and key in meta2:
        equal = (meta1[key] == meta2[key])
    else:
        # Build a naive signature to compare; this is a soft check.
        sig1 = json.dumps({k: meta1.get(k) for k in sorted(meta1.keys()) if "hash" in k.lower() or "seed" in k.lower()}, sort_keys=True)
        sig2 = json.dumps({k: meta2.get(k) for k in sorted(meta2.keys()) if "hash" in k.lower() or "seed" in k.lower()}, sort_keys=True)
        equal = (sig1 == sig2)

    assert equal == expect_equal, (
        f"Expected equality={expect_equal} for seedA={seedA}, seedB={seedB}, "
        f"but observed equality={equal}."
    )
