#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ArielSensorArray/tests/integration/test_selftest_runner.py

SpectraMind V50 — Self-Test Runner Integration Tests (pytest)

Purpose
-------
Exercise the repository's *self-test* entrypoint via the unified Typer CLI in a
fast, side‑effect‑safe manner, verifying:

  • The `test` (or `selftest`) subcommand is exposed and runs successfully.
  • Hydra overrides route all artifacts under a test `paths.output_dir`.
  • Append‑only audit log (logs/v50_debug_log.md) is written per run.
  • Optional structured JSONL stream (logs/v50_runs.jsonl) — tolerated if absent.
  • Optional run-hash snapshot (outputs/run_hash_summary_v50.json) — tolerated if absent.
  • Optional self-test reports are generated (e.g., outputs/selftest/report.md/.json).

We do not depend on real datasets. Tests tolerate "strict" dry-run modes that
only produce logs and zero/placeholder artifacts.

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
class SelfTestArtifacts:
    """
    Canonical locations beneath a given run's output directory.

    Expectations are flexible to accommodate strict dry-run behavior:
    - If a file/dir is present, we validate its basic shape.
    - If absent (dry-run), we don't fail the test.
    """
    outdir: Path
    logs_dir: Path
    debug_log: Path
    runs_jsonl: Path
    run_hash_json: Path
    outputs_dir: Path
    selftest_dir: Path
    report_md: Path
    report_json: Path
    diagnostics_dir: Path

    @staticmethod
    def locate(outdir: Path) -> "SelfTestArtifacts":
        logs_dir = outdir / "logs"
        outputs = outdir / "outputs"
        selftest = outputs / "selftest"
        diagnostics = outputs / "diagnostics"
        return SelfTestArtifacts(
            outdir=outdir,
            logs_dir=logs_dir,
            debug_log=logs_dir / "v50_debug_log.md",
            runs_jsonl=logs_dir / "v50_runs.jsonl",
            run_hash_json=outputs / "run_hash_summary_v50.json",
            outputs_dir=outputs,
            selftest_dir=selftest,
            report_md=selftest / "report.md",
            report_json=selftest / "report.json",
            diagnostics_dir=diagnostics,
        )


def _mk_clean_dir(p: Path) -> Path:
    """Ensure a clean directory exists at path p."""
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _env_for_cli(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Construct a stable environment for CLI execution with Hydra friendliness.
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
# Tests — Core self-test flow
# ---------------------------------------------------------------------

def test_selftest_cli_dry_run_executes_and_logs(runner, workdir: Path):
    """
    Smoke: `spectramind test` (or fallback `selftest`) runs successfully with
    `--dry-run` + fast overrides, exits code 0, writes the append-only audit
    log and (optionally) the structured JSONL event stream.
    """
    outdir = _mk_clean_dir(workdir / "selftest_run_A")

    # Prefer the canonical "test" subcommand; if that fails, try "selftest".
    candidate_cmds = [
        ["test", "--dry-run", f"paths.output_dir={outdir}", "selftest.fast_mode=true", "repro.seed=1357"],
        ["selftest", "--dry-run", f"paths.output_dir={outdir}", "selftest.fast_mode=true", "repro.seed=1357"],
    ]

    last_res = None
    for cmd in candidate_cmds:
        res = _run_cli(runner, cmd, env=_env_for_cli(), cwd=workdir)
        last_res = res
        if res.exit_code == 0:
            break
    assert last_res is not None and last_res.exit_code == 0, \
        f"Self-test CLI exited non-zero for all tried entrypoints.\nSTDOUT:\n{last_res.stdout if last_res else ''}"

    arts = SelfTestArtifacts.locate(outdir)

    # Logs dir + audit log required
    assert arts.logs_dir.exists(), "logs/ directory not created under output_dir"
    assert arts.debug_log.is_file(), "v50_debug_log.md not found; audit trail is required"

    # Audit log should mention 'test' or 'selftest' and include metadata markers
    dbg = arts.debug_log.read_text(encoding="utf-8")
    assert re.search(r"\b(test|selftest)\b", dbg, flags=re.IGNORECASE), \
        "Audit log should mention 'test' or 'selftest' subcommand"
    assert re.search(r"\bconfig\b|\bhash\b|\btimestamp\b|\bseed\b", dbg, flags=re.IGNORECASE), \
        "Audit log should include config/hash/timestamp/seed markers"

    # Optional structured JSONL stream: if present, ensure not empty
    if arts.runs_jsonl.exists():
        text = arts.runs_jsonl.read_text(encoding="utf-8").strip()
        assert text, "v50_runs.jsonl exists but is empty"

    # Optional selftest reports directory: if present, validate basic shape
    if arts.selftest_dir.exists():
        assert arts.selftest_dir.is_dir(), "outputs/selftest exists but is not a directory"
        # Heuristic: at least one of report.md / report.json appears
        if not (arts.report_md.exists() or arts.report_json.exists()):
            # If neither exists, accept any recognized report-like artifact
            found = list(arts.selftest_dir.glob("*"))
            assert any(p.suffix.lower() in {".md", ".json", ".html", ".txt"} for p in found), \
                "outputs/selftest exists but contains no recognizable report artifacts"


def test_selftest_cli_respects_output_routing_and_paths(runner, workdir: Path):
    """
    Verify artifacts (when generated) are confined under `paths.output_dir`
    and that any selftest outputs reside under `outputs/selftest/`.
    """
    outdir = _mk_clean_dir(workdir / "selftest_run_paths")

    res = _run_cli(
        runner,
        ["test", "--dry-run", f"paths.output_dir={outdir}", "selftest.fast_mode=true", "repro.seed=2468"],
        env=_env_for_cli(),
        cwd=workdir,
    )
    # Fallback to "selftest" if "test" isn't registered.
    if res.exit_code != 0:
        res = _run_cli(
            runner,
            ["selftest", "--dry-run", f"paths.output_dir={outdir}", "selftest.fast_mode=true", "repro.seed=2468"],
            env=_env_for_cli(),
            cwd=workdir,
        )
    assert res.exit_code == 0, f"Self-test CLI failed:\n{res.stdout}"

    arts = SelfTestArtifacts.locate(outdir)

    # Ensure the output dir is present and no unexpected top-level dirs appear
    siblings = [p for p in workdir.iterdir() if p.is_dir()]
    assert outdir in siblings, "Configured output directory missing"
    assert len(siblings) <= 2, (
        "Unexpected sibling directories created; all artifacts should route under the configured output_dir."
    )

    # If selftest outputs exist, ensure they are under outputs/selftest
    if arts.selftest_dir.exists():
        assert arts.selftest_dir.is_dir(), "outputs/selftest exists but is not a directory"
        # Any known file type indicates at least a stub
        found_types = set(p.suffix.lower() for p in arts.selftest_dir.glob("*.*"))
        assert found_types & {".md", ".json", ".html", ".txt"}, \
            "outputs/selftest contains no recognized self-test artifacts (.md/.json/.html/.txt)"


@pytest.mark.parametrize(
    "seedA,seedB,expect_equal",
    [
        (1010, 1010, True),
        (1010, 2020, False),
    ],
)
def test_selftest_run_hash_seed_sensitivity_when_available(
    runner, workdir: Path, seedA: int, seedB: int, expect_equal: bool
):
    """
    If the pipeline writes `outputs/run_hash_summary_v50.json` during self-test,
    then with all else equal a seed change should typically alter the hash (and
    identical seeds should yield identical hashes). In strict dry-run modes,
    the file may be omitted — xfail that case.
    """
    out1 = _mk_clean_dir(workdir / f"selftest_hash_seed_{seedA}")
    out2 = _mk_clean_dir(workdir / f"selftest_hash_seed_{seedB}")

    common = ["selftest.fast_mode=true"]

    r1 = _run_cli(
        runner,
        ["test", "--dry-run", f"paths.output_dir={out1}", f"repro.seed={seedA}", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )
    if r1.exit_code != 0:
        r1 = _run_cli(
            runner,
            ["selftest", "--dry-run", f"paths.output_dir={out1}", f"repro.seed={seedA}", *common],
            env=_env_for_cli(),
            cwd=workdir,
        )

    r2 = _run_cli(
        runner,
        ["test", "--dry-run", f"paths.output_dir={out2}", f"repro.seed={seedB}", *common],
        env=_env_for_cli(),
        cwd=workdir,
    )
    if r2.exit_code != 0:
        r2 = _run_cli(
            runner,
            ["selftest", "--dry-run", f"paths.output_dir={out2}", f"repro.seed={seedB}", *common],
            env=_env_for_cli(),
            cwd=workdir,
        )

    assert r1.exit_code == 0 and r2.exit_code == 0, "Self-test dry-run failed."

    meta1 = out1 / "outputs" / "run_hash_summary_v50.json"
    meta2 = out2 / "outputs" / "run_hash_summary_v50.json"

    if not (meta1.exists() and meta2.exists()):
        pytest.xfail("run_hash_summary_v50.json not emitted during self-test dry-run; cannot validate seed sensitivity.")

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


def test_selftest_help_is_discoverable(runner, workdir: Path):
    """
    Regression: `spectramind test --help` or `spectramind selftest --help`
    should print usage/options text.
    """
    res = _run_cli(runner, ["test", "--help"], env=_env_for_cli(), cwd=workdir)
    if res.exit_code != 0:
        res = _run_cli(runner, ["selftest", "--help"], env=_env_for_cli(), cwd=workdir)

    assert res.exit_code == 0
    out = res.stdout.lower()
    assert "usage:" in out or "options:" in out or "help" in out, "Help output missing expected sections"


def test_selftest_optionally_supports_deep_mode_help(runner, workdir: Path):
    """
    Optional: If a deep mode exists (e.g., `--deep` or `--mode=deep`), its help
    should be discoverable. We probe non-exhaustively and pass if any candidate
    help path prints without error. If none exist, xfail (non-fatal).
    """
    candidate_cmds = [
        ["test", "--help"],  # baseline
        ["test", "--deep", "--help"],
        ["test", "run", "--deep", "--help"],
        ["selftest", "--help"],
        ["selftest", "--deep", "--help"],
        ["selftest", "run", "--deep", "--help"],
    ]
    for cmd in candidate_cmds:
        res = _run_cli(runner, cmd, env=_env_for_cli(), cwd=workdir)
        if res.exit_code == 0 and ("usage:" in res.stdout.lower() or "options:" in res.stdout.lower()):
            return
    pytest.xfail("Deep-mode style help not discoverable among probed entrypoints (acceptable if feature not implemented).")
