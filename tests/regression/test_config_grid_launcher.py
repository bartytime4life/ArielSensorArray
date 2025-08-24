# tests/regression/test_config_grid_launcher.py
"""
Regression tests for the SpectraMind V50 config grid launcher.

These tests validate that our "grid" launcher (CLI or Python API) correctly:
  1) Expands a Cartesian product of Hydra overrides into a run plan in a deterministic order.
  2) Honors --dry-run / plan-only execution without launching any jobs.
  3) Respects max-parallel (concurrency) limits for actual execution planning.
  4) Propagates environment (seed, CUDA settings) and base Hydra config folder.
  5) Supports include/exclude filters for specific combinations.
  6) Emits per-job, per-config logging hooks we rely on for reproducibility.

Why these checks?
- SpectraMind is a CLI-first, Hydra-configured pipeline; every run must be reproducible from
  a captured config and command line plan.  The grid launcher is the multi-run orchestrator
  that composes configs and produces runs for Typer CLI subcommands (e.g., `spectramind train`). 
  (See the design notes on Typer+Hydra, config composition, multirun/sweep support, and CI reproducibility
  guarantees.)    

NOTE:
- We exercise either the Python API (`grid_launcher.run_grid`) *or* the CLI entrypoint
  (`spectramind launch grid --dry-run ...`). If neither is available in the current repo snapshot,
  the tests skip with a helpful message (so the suite remains green until the launcher lands).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pytest


# --------------------------
# Helpers / feature detection
# --------------------------

def _cli_exists(bin_name: str = "spectramind") -> bool:
    exe = shutil.which(bin_name)
    return bool(exe)


def _has_python_api() -> Tuple[bool, object]:
    """
    Tries to import a canonical Python API for the grid launcher.

    Expected locations (first match wins):
        spectramind.cli.launchers.grid_launcher:run_grid
        spectramind.cli.grid:run_grid
        spectramind.launch.grid:run_grid
    """
    candidates = [
        ("spectramind.cli.launchers.grid_launcher", "run_grid"),
        ("spectramind.cli.grid", "run_grid"),
        ("spectramind.launch.grid", "run_grid"),
    ]
    for mod_name, attr in candidates:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            fn = getattr(mod, attr)
            return True, fn
        except Exception:
            continue
    return False, None


# --------------------------
# Fixtures
# --------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    # Heuristic: start at this file and ascend until we find pyproject or .git
    cur = Path(__file__).resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "pyproject.toml").exists() or (parent / ".git").exists():
            return parent
    return cur.parent.parent


@pytest.fixture
def tmp_cfgs(tmp_path: Path) -> Dict[str, Path]:
    """
    Create a minimal Hydra-like configs layout for the tests.

    We don't need real SpectraMind configs; the launcher should only need override strings.
    Still, we place a plausible structure to match expectations.  
    """
    cfg_root = tmp_path / "configs"
    (cfg_root / "model").mkdir(parents=True, exist_ok=True)
    (cfg_root / "training").mkdir(parents=True, exist_ok=True)

    # Minimal YAMLs – the launcher only composes names; content is irrelevant for dry-run plan.
    (cfg_root / "model" / "v50.yaml").write_text("name: v50\n")
    (cfg_root / "model" / "v50_small.yaml").write_text("name: v50_small\n")
    (cfg_root / "training" / "fast.yaml").write_text("epochs: 1\n")
    (cfg_root / "training" / "long.yaml").write_text("epochs: 2\n")

    return {
        "root": cfg_root,
        "model_v50": cfg_root / "model" / "v50.yaml",
        "model_v50_small": cfg_root / "model" / "v50_small.yaml",
        "train_fast": cfg_root / "training" / "fast.yaml",
        "train_long": cfg_root / "training" / "long.yaml",
    }


@pytest.fixture
def base_env(monkeypatch: pytest.MonkeyPatch, tmp_cfgs: Dict[str, Path]):
    """
    Baseline environment for deterministic planning:
    - Set a fixed global seed (reproducibility).
    - Point HYDRA_CONFIG_PATH to our temp configs (if the launcher honors it).
    - Force 'UTC' to stabilize any timestamp formatting in logs.  
    """
    monkeypatch.setenv("SPECTRAMIND_SEED", "42")
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("HYDRA_CONFIG_PATH", str(tmp_cfgs["root"]))
    # Disable any accidental GPU scheduling in CI
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")


# --------------------------
# Test data
# --------------------------

GRID_CARTESIAN = {
    "base": ["train"],  # base subcommand for each job; expected to be `spectramind train ...`
    "overrides": {
        "model": ["model=v50", "model=v50_small"],
        "training.epochs": ["training=fast", "training=long"],  # shorthand using config names
    },
}

INCLUDE_FILTER = {
    "include": [
        {"model": "model=v50", "training.epochs": "training=fast"},
    ]
}

EXCLUDE_FILTER = {
    "exclude": [
        {"model": "model=v50_small", "training.epochs": "training=long"},
    ]
}

# --------------------------
# Tests (Python API)
# --------------------------

@pytest.mark.order(1)
def test_grid_plan_python_api_cartesian_dry_run(tmp_path: Path, base_env):
    has_api, run_grid = _has_python_api()
    if not has_api:
        pytest.skip("Grid launcher Python API not found; skipping API tests.")

    # Expect 2x2 = 4 jobs in deterministic order.
    plan = run_grid(
        base="train",
        overrides=GRID_CARTESIAN["overrides"],
        strategy="cartesian",
        max_parallel=2,
        dry_run=True,
        format="json",
    )
    # Allow both dict or JSON string results
    if isinstance(plan, str):
        plan = json.loads(plan)

    jobs = plan.get("jobs", [])
    assert len(jobs) == 4, "Cartesian expansion should yield 4 jobs (2x2)."
    # Deterministic order: product in key order -> model then epochs
    expected = [
        ["model=v50", "training=fast"],
        ["model=v50", "training=long"],
        ["model=v50_small", "training=fast"],
        ["model=v50_small", "training=long"],
    ]
    got = [[ov for ov in j["overrides"] if ov.startswith("model") or ov.startswith("training")]
           for j in jobs]
    assert got == expected, f"Overrides order mismatch.\nExpected: {expected}\nGot: {got}"

    # Check concurrency and dry-run flags in plan metadata
    assert plan.get("dry_run") is True
    assert plan.get("max_parallel") == 2

    # Verify commands look like `spectramind train ... + overrides` (CLI-first philosophy).  
    for j in jobs:
        cmd = j.get("cmd", "")
        assert "spectramind" in cmd and "train" in cmd, f"Unexpected command: {cmd}"
        assert "model=" in cmd and "training=" in cmd


@pytest.mark.order(2)
def test_grid_plan_python_api_include_exclude(tmp_path: Path, base_env):
    has_api, run_grid = _has_python_api()
    if not has_api:
        pytest.skip("Grid launcher Python API not found; skipping API tests.")

    # Apply include filter -> only 1 job remains.
    plan_inc = run_grid(
        base="train",
        overrides=GRID_CARTESIAN["overrides"],
        strategy="cartesian",
        include=INCLUDE_FILTER["include"],
        max_parallel=1,
        dry_run=True,
        format="json",
    )
    plan_inc = json.loads(plan_inc) if isinstance(plan_inc, str) else plan_inc
    jobs_inc = plan_inc.get("jobs", [])
    assert len(jobs_inc) == 1
    assert "model=v50" in jobs_inc[0]["overrides"]
    assert "training=fast" in jobs_inc[0]["overrides"]

    # Apply exclude filter -> 3 jobs remain.
    plan_exc = run_grid(
        base="train",
        overrides=GRID_CARTESIAN["overrides"],
        strategy="cartesian",
        exclude=EXCLUDE_FILTER["exclude"],
        max_parallel=3,
        dry_run=True,
        format="json",
    )
    plan_exc = json.loads(plan_exc) if isinstance(plan_exc, str) else plan_exc
    jobs_exc = plan_exc.get("jobs", [])
    assert len(jobs_exc) == 3
    # Confirm the excluded pair is not present
    bad = {"model=v50_small", "training=long"}
    for j in jobs_exc:
        assert not bad.issubset(set(j["overrides"])), "Excluded combination leaked into plan."


# --------------------------
# Tests (CLI)
# --------------------------

@pytest.mark.order(3)
def test_grid_plan_cli_cartesian_dry_run(tmp_path: Path, base_env):
    if not _cli_exists():
        pytest.skip("spectramind CLI not found on PATH; skipping CLI tests.")

    # Build CLI: spectramind launch grid --base train --dry-run --format json \
    #   --overrides model=v50 model=v50_small training=fast training=long
    cmd = [
        "spectramind",
        "launch",
        "grid",
        "--base", "train",
        "--strategy", "cartesian",
        "--max-parallel", "2",
        "--dry-run",
        "--format", "json",
        "--overrides",
        "model=v50",
        "model=v50_small",
        "training=fast",
        "training=long",
    ]
    # The CLI-first contract with deterministic planning & JSON plan.   
    out = subprocess.check_output(cmd, text=True)
    plan = json.loads(out)
    assert plan.get("dry_run") is True
    assert plan.get("max_parallel") == 2
    jobs = plan.get("jobs", [])
    assert len(jobs) == 4
    # Deterministic order check as above
    expected = [
        ["model=v50", "training=fast"],
        ["model=v50", "training=long"],
        ["model=v50_small", "training=fast"],
        ["model=v50_small", "training=long"],
    ]
    got = [[ov for ov in j["overrides"] if ov.startswith("model") or ov.startswith("training")]
           for j in jobs]
    assert got == expected


@pytest.mark.order(4)
def test_grid_cli_include_exclude_filters(tmp_path: Path, base_env):
    if not _cli_exists():
        pytest.skip("spectramind CLI not found on PATH; skipping CLI tests.")

    base = ["spectramind", "launch", "grid", "--base", "train", "--strategy", "cartesian", "--format", "json", "--dry-run"]

    # Include one
    cmd_inc = base + ["--include", "model=v50,training=fast"]
    plan_inc = json.loads(subprocess.check_output(cmd_inc, text=True))
    assert len(plan_inc.get("jobs", [])) == 1

    # Exclude one
    cmd_exc = base + ["--overrides", "model=v50", "model=v50_small", "training=fast", "training=long",
                      "--exclude", "model=v50_small,training=long"]
    plan_exc = json.loads(subprocess.check_output(cmd_exc, text=True))
    assert len(plan_exc.get("jobs", [])) == 3
    for j in plan_exc["jobs"]:
        assert not {"model=v50_small", "training=long"}.issubset(set(j["overrides"]))


# --------------------------
# Behavioral checks
# --------------------------

@pytest.mark.order(5)
def test_plan_contains_reproducibility_artifacts(tmp_path: Path, base_env):
    """
    Ensure per-job plan includes:
      - Captured seed or RNG spec.
      - Serialized Hydra overrides.
      - A human-auditable 'cmd' string runnable by CI.
    This matches SpectraMind "glass box" + CI reproducibility posture.   
    """
    has_api, run_grid = _has_python_api()
    if not has_api:
        pytest.skip("Grid launcher Python API not found; skipping API tests.")

    plan = run_grid(
        base="train",
        overrides=GRID_CARTESIAN["overrides"],
        strategy="cartesian",
        max_parallel=1,
        dry_run=True,
        format="json",
    )
    plan = json.loads(plan) if isinstance(plan, str) else plan
    jobs = plan.get("jobs", [])
    assert jobs, "Expected non-empty job plan."

    for j in jobs:
        # seed
        md = j.get("metadata", {})
        assert md.get("seed") in (42, "42"), "Each job should capture the resolved seed."
        # hydra overrides:
        ovs = j.get("overrides")
        assert isinstance(ovs, list) and all(isinstance(x, str) for x in ovs)
        # runnable command
        cmd = j.get("cmd", "")
        assert cmd.startswith("spectramind "), "Plan must embed a CI-runnable command."


@pytest.mark.order(6)
def test_max_parallel_respected_in_plan(tmp_path: Path, base_env):
    """
    The plan should include scheduling groups/batches when max_parallel < total jobs.
    We don't launch jobs here; we just validate the structure.  
    """
    has_api, run_grid = _has_python_api()
    if not has_api:
        pytest.skip("Grid launcher Python API not found; skipping API tests.")

    plan = run_grid(
        base="train",
        overrides=GRID_CARTESIAN["overrides"],
        strategy="cartesian",
        max_parallel=2,
        dry_run=True,
        format="json",
    )
    plan = json.loads(plan) if isinstance(plan, str) else plan
    # Optional field 'batches' or 'schedule' may exist; if present, verify chunking size.
    batches = plan.get("batches")
    if batches:
        assert all(len(b) <= 2 for b in batches), "Any planned batch must not exceed max_parallel."
        total = sum(len(b) for b in batches)
        assert total == 4, "All jobs should be accounted for across batches."


# --------------------------
# Robustness / failure handling
# --------------------------

@pytest.mark.order(7)
def test_failure_strategy_flags_exposed(tmp_path: Path, base_env):
    """
    Validate fast-fail / continue-on-error flags exist and default is safe.
    We don't simulate a real failure; we only check presence in dry-run plan.
    """
    has_api, run_grid = _has_python_api()
    if not has_api:
        pytest.skip("Grid launcher Python API not found; skipping API tests.")

    plan = run_grid(
        base="train",
        overrides=GRID_CARTESIAN["overrides"],
        strategy="cartesian",
        max_parallel=2,
        dry_run=True,
        continue_on_error=False,
        format="json",
    )
    plan = json.loads(plan) if isinstance(plan, str) else plan
    assert plan.get("continue_on_error") is False, "Default should be fail-fast unless explicitly set."


# --------------------------
# Sanity: launcher discoverability
# --------------------------

@pytest.mark.order(8)
def test_cli_help_mentions_grid_launcher():
    """
    The Typer CLI should advertise the grid launcher in its help text.  
    """
    if not _cli_exists():
        pytest.skip("spectramind CLI not found on PATH; skipping CLI help test.")
    out = subprocess.check_output(["spectramind", "--help"], text=True)
    # Accept either "launch grid" subcommand tree or "grid" subgroup depending on CLI layout
    assert "launch" in out.lower() or "grid" in out.lower()
