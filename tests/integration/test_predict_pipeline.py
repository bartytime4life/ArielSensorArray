#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArielSensorArray/tests/test_predict_pipeline.py

SpectraMind V50 — Prediction Pipeline Tests (pytest)

This suite validates the prediction pipeline’s CLI entrypoint (Typer),
Hydra configuration wiring, output routing, and audit logging using
a fast, side‑effect‑safe dry‑run mode. It does NOT require the real
challenge dataset; instead it relies on the repository’s contract
that `--dry-run` (and/or equivalent dev overrides) executes the full
orchestrator path without heavy I/O or long runtimes.

Key expectations the repository should satisfy:
  • Unified Typer CLI exposes a `predict` subcommand.
  • Global flags: `--dry-run`, `--help` (and similar) exist and work.
  • Hydra‑style overrides route all artifacts under `paths.output_dir=...`.
  • Append‑only audit log at `logs/v50_debug_log.md` is written per run.
  • Optional structured log `logs/v50_runs.jsonl` may be present.
  • Optional prediction artifacts live under `outputs/predictions/`
    (e.g., `mu.npy`, `sigma.npy`, `submission.csv`) — but tests tolerate
    their absence in pure dry‑run mode as long as logs + stubs exist.

If the Typer CLI cannot be imported in the test environment, tests are
skipped gracefully so CI on partial environments remains green.

Requires:
  • pytest
  • typer (for CliRunner)
  • The repo’s Python package importable in the test environment
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
class PredictArtifacts:
    """Pointers to canonical logs and prediction outputs for a given run root."""
    outdir: Path
    logs_dir: Path
    debug_log: Path
    runs_jsonl: Path
    run_hash_json: Path
    predictions_dir: Path
    submission_csv: Path
    mu_npy: Path
    sigma_npy: Path

    @staticmethod
    def locate(outdir: Path) -> "PredictArtifacts":
        """
        Infer standard artifact locations under an output root. The repository
        may choose different exact filenames; the tests will tolerate absences
        for pure `--dry-run` modes and only enforce the logging contract.
        """
        logs_dir = outdir / "logs"
        outputs = outdir / "outputs"
        preds = outputs / "predictions"
        return PredictArtifacts(
            outdir=outdir,
            logs_dir=logs_dir,
            debug_log=logs_dir / "v50_debug_log.md",
            runs_jsonl=logs_dir / "v50_runs.jsonl",
            run_hash_json=outputs / "run_hash_summary_v50.json",
            predictions_dir=preds,
            submission_csv=preds / "submission.csv",
            mu_npy=preds / "mu.npy",
            sigma_npy=preds / "sigma.npy",
        )


def _mk_clean_dir(p: Path) -> Path:
    """Ensure a clean directory exists at path p."""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _env_for_cli(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Construct a clean environment for CLI execution. We set HYDRA_FULL_ERROR
    for clearer tracebacks and sanitize TTY/Python buffering for CI.
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
        pytest.skip("SpectraMind unified CLI (Typer app) not importable in this environment.")
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
# Tests
# ---------------------------------------------------------------------

def test_predict_cli_dry_run_executes_and_logs(runner, workdir: Path):
    """
    Smoke: `spectramind predict` runs with `--dry-run` + fast overrides,
    exits code 0, and writes the append-only audit log and (optionally)
    the structured JSONL event stream. It must not write outside the
    configured `paths.output_dir`.
    """
    outdir = _mk_clean_dir(workdir / "pred_run_A")

    cmd = [
        "predict",
        "--dry-run",
        # Route artifacts to temp outdir:
        f"paths.output_dir={outdir}",
        # Keep it extremely fast and data-light:
        "predict.fast_dev_run=true",
        "data.sample_limit=8",
        # Deterministic:
        "repro.seed=2025",
    ]

    res = _run_cli(runner, cmd, env=_env_for_cli(), cwd=workdir)
    assert res.exit_code == 0, f"CLI exited non-zero.\nSTDOUT:\n{res.stdout}"

    arts = PredictArtifacts.locate(outdir)

    # Logs dir + audit log required
    assert arts.logs_dir.exists(), "logs/ directory not created under output_dir"
    assert arts.debug_log.is_file(), "v50_debug_log.md not found; audit trail is required"

    # Audit log should mention 'predict' and include some metadata markers
    dbg = arts.debug_log.read_text(encoding="utf-8")
    assert re.search(r"\bpredict\b", dbg, flags=re.IGNORECASE), "Audit log should mention 'predict' subcommand"
    assert re.search(r"\bconfig\b|\bhash\b|\btimestamp\b|\bseed\b", dbg, flags=re.IGNORECASE), \
        "Audit log should include config/hash/timestamp/seed markers"

    # Optional structured JSONL stream: if present, ensure not empty
    if arts.runs_jsonl.exists():
        text = arts.runs_jsonl.read_text(encoding="utf-8").strip()
        assert text, "v50_runs.jsonl exists but is empty"

    # In pure dry-run, predictions may be skipped; tolerate absence if dry-run is active.
    # However, if predictions dir exists, basic shape of outputs should look sane.
    if arts.predictions_dir.exists():
        assert arts.predictions_dir.is_dir(), "outputs/predictions exists but is not a directory"
        # Heuristic: at least one of the canonical artifacts appears
        found_any = any(p.exists() for p in (arts.submission_csv, arts.mu_npy, arts.sigma_npy))
        assert found_any, "outputs/predictions exists but contains none of the canonical prediction artifacts"


@pytest.mark.parametrize(
    "seedA,seedB,expect_equal",
    [
        (123, 123, True),
        (123, 456, False),
    ],
)
def test_predict_run_hash_seed_sensitivity_when_available(
    runner, workdir: Path, seedA: int, seedB: int, expect_equal: bool
):
    """
    If the pipeline writes `outputs/run_hash_summary_v50.json` during predict,
    then with all else equal a seed change should typically alter the hash (and
    identical seeds should yield identical hashes). In strict dry-run modes,
    the file may be omitted — xfail that case.
    """
    out1 = _mk_clean_dir(workdir / f"pred_hash_seed_{seedA}")
    out2 = _mk_clean_dir(workdir / f"pred_hash_seed_{seedB}")

    common = [
        "predict.fast_dev_run=true",
        "data.sample_limit=4",
    ]

    r1 = _run_cli(
        runner,
        ["predict", "--dry-run", f"paths.output_dir={out1}", f"repro.seed={seedA}", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )
    r2 = _run_cli(
        runner,
        ["predict", "--dry-run", f"paths.output_dir={out2}", f"repro.seed={seedB}", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )
    assert r1.exit_code == 0 and r2.exit_code == 0, "Predict dry-run failed."

    meta1 = out1 / "outputs" / "run_hash_summary_v50.json"
    meta2 = out2 / "outputs" / "run_hash_summary_v50.json"

    if not (meta1.exists() and meta2.exists()):
        pytest.xfail("run_hash_summary_v50.json not emitted during predict dry-run; cannot validate seed sensitivity.")

    j1 = json.loads(meta1.read_text(encoding="utf-8"))
    j2 = json.loads(meta2.read_text(encoding="utf-8"))

    key = "config_hash"
    if key in j1 and key in j2:
        equal = (j1[key] == j2[key])
    else:
        sig1 = json.dumps({k: j1.get(k) for k in sorted(j1.keys()) if "hash" in k.lower() or "seed" in k.lower()}, sort_keys=True)
        sig2 = json.dumps({k: j2.get(k) for k in sorted(j2.keys()) if "hash" in k.lower() or "seed" in k.lower()}, sort_keys=True)
        equal = (sig1 == sig2)

    assert equal == expect_equal, (
        f"Expected equality={expect_equal} for seedA={seedA}, seedB={seedB}, observed={equal}"
    )


def test_predict_cli_respects_output_routing_and_pred_paths(runner, workdir: Path):
    """
    Verify artifacts (when generated) are confined under `paths.output_dir`
    and that any prediction outputs reside under `outputs/predictions/`.
    """
    outdir = _mk_clean_dir(workdir / "pred_run_paths")

    res = _run_cli(
        runner,
        [
            "predict",
            "--dry-run",
            f"paths.output_dir={outdir}",
            "predict.fast_dev_run=true",
            "repro.seed=7",
        ],
        env=_env_for_cli(),
        cwd=workdir,
    )
    assert res.exit_code == 0, f"CLI failed:\n{res.stdout}"

    arts = PredictArtifacts.locate(outdir)

    # Ensure the output dir is present and no unexpected top-level dirs appear
    siblings = [p for p in workdir.iterdir() if p.is_dir()]
    assert outdir in siblings, "Configured output directory missing"
    assert len(siblings) <= 2, (
        "Unexpected sibling directories created; all artifacts should route under the configured output_dir."
    )

    # If predictions exist, ensure they are under outputs/predictions
    if arts.predictions_dir.exists():
        assert arts.predictions_dir.is_dir(), "outputs/predictions exists but is not a directory"
        # Any known file type (CSV/NPY/JSON) indicates at least a stub
        found_types = set(p.suffix.lower() for p in arts.predictions_dir.glob("*.*"))
        assert found_types & {".csv", ".npy", ".json"}, \
            "outputs/predictions contains no recognized prediction artifacts (.csv/.npy/.json)"


def test_predict_cli_help_and_usage(runner, workdir: Path):
    """
    Regression: `spectramind predict --help` should print usage/options text.
    """
    res = _run_cli(runner, ["predict", "--help"], env=_env_for_cli(), cwd=workdir)
    assert res.exit_code == 0
    out = res.stdout.lower()
    assert "usage:" in out or "options:" in out or "help" in out, "Help output missing expected sections"


def test_predict_then_validate_submission_if_available(runner, workdir: Path):
    """
    If the repository exposes a `validate` or `validate-submission` CLI path
    (e.g., `spectramind submit validate` or `spectramind diagnose validate-submission`
    or `spectramind validate`), try invoking it in a no-op/dry-run style to
    ensure the validator wiring is present. We don't assume any actual files;
    we simply verify that the subcommand exists and prints help without error.
    This test is resilient to repos that do not expose such a command.
    """
    # Probe a few plausible validator entrypoints; run `--help` to avoid I/O.
    candidate_cmds = [
        ["submit", "validate", "--help"],
        ["diagnose", "validate-submission", "--help"],
        ["validate", "--help"],
    ]
    for cmd in candidate_cmds:
        res = _run_cli(runner, cmd, env=_env_for_cli(), cwd=workdir)
        if res.exit_code == 0 and ("usage:" in res.stdout.lower() or "options:" in res.stdout.lower()):
            # Found a working validator entrypoint; test passes.
            return
    # If none worked, mark as xfail to avoid breaking CI on repos that route validation differently.
    pytest.xfail("No submission-validator help path found among common entrypoints (submit/diagnose/validate).")
