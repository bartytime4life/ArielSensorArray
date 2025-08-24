# tests/diagnostics/test_config_grid_launcher.py
"""
Upgraded tests for the Config Grid Launcher used in SpectraMind V50.

These tests are intentionally defensive:
- They discover the target module from a few plausible import paths.
- They gracefully skip when optional pieces (e.g., the Typer CLI) are missing.
- They validate both "library-level" helpers (parsing/expansion/seed-assign)
  and "CLI-level" behaviors (dry-run, fail-fast) when the CLI is available.

Test matrix (will be skipped if symbols are absent in the repo):
  1) parse_param_grid() – turns a compact string into a dict of lists
  2) expand_grid()     – Cartesian expansion to a list of override dicts
  3) to_hydra_overrides() – override dict -> list["key=value", ...]
  4) assign_seeds()    – deterministic seeds for each expanded run
  5) launch_grid()     – orchestration with dry-run and fail-fast features
  6) Typer CLI ("spectramind grid") dry-run behavior via CliRunner
  7) Typer CLI fail-fast behavior (simulated one-run failure)

Authoring guidance for the production code this test expects:
  - Module path (any one of these is fine):
        spectramind.tools.config_grid_launcher
        src.tools.config_grid_launcher
        tools.config_grid_launcher
  - Suggested functions (duck-typed; names must match):
        parse_param_grid(spec: str) -> dict[str, list[str]]
        expand_grid(param_dict: dict[str, list[str]]) -> list[dict[str, str]]
        to_hydra_overrides(one: dict[str, str]) -> list[str]
        assign_seeds(num: int, base_seed: int | None = None) -> list[int]
        launch_grid(
            expanded: list[dict[str, str]],
            base_cmd: list[str] | None = None,
            dry_run: bool = False,
            fail_fast: bool = False,
            seeds: list[int] | None = None,
            log_root: str | None = None,
        ) -> "LaunchResult"
      where LaunchResult has fields: planned:int, succeeded:int, failed:int, logs_dir:str
  - Optional Typer command group:
        spectramind.app (Typer)
        with command: grid
        required options:
           --params / -p  (str) like "trainer.epochs=1,2;optimizer.lr=1e-3,1e-4"
           --dry-run / -n (flag)
           --fail-fast / -F (flag)
           --base-seed (int)
        optional:
           --log-root (path)
           --base-cmd (multi option) (advanced)

The tests will skip if imports fail so they never block your pipeline
before the feature lands, but they will execute fully once the launcher exists.
"""

from __future__ import annotations

import os
import sys
import types
import itertools
import importlib
from pathlib import Path
from typing import Any

import pytest


# ---------- dynamic imports with graceful skips ----------

def _import_any(*module_names: str) -> types.ModuleType | None:
    for name in module_names:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError:
            continue
    return None


LAUNCHER_MOD = _import_any(
    "spectramind.tools.config_grid_launcher",
    "src.tools.config_grid_launcher",
    "tools.config_grid_launcher",
)

# Try to import the Typer app if present
SPECTRAMIND_MOD = _import_any("spectramind")
TYPER_MOD = _import_any("typer")
CLICK_TESTING_MOD = _import_any("typer.testing") or _import_any("click.testing")

# Helper to fetch callable or skip
def _sym_or_skip(mod: types.ModuleType | None, name: str):
    if mod is None or not hasattr(mod, name):
        pytest.skip(f"Missing symbol {name} in {mod}")
    return getattr(mod, name)


# ---------- fixtures ----------

@pytest.fixture(autouse=True)
def _hydra_full_error_env(monkeypatch: pytest.MonkeyPatch):
    """Ensure Hydra errors are verbose when the launcher spawns runs."""
    monkeypatch.setenv("HYDRA_FULL_ERROR", "1")


# ---------- unit tests for helpers ----------

@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_parse_param_grid_basic():
    parse_param_grid = _sym_or_skip(LAUNCHER_MOD, "parse_param_grid")

    spec = "optimizer.lr=1e-3,1e-4;training.batch_size=32,64;trainer.epochs=1"
    parsed = parse_param_grid(spec)
    # Expected normalized dict-of-lists
    assert isinstance(parsed, dict)
    assert set(parsed) == {"optimizer.lr", "training.batch_size", "trainer.epochs"}
    assert parsed["optimizer.lr"] == ["1e-3", "1e-4"]
    assert parsed["training.batch_size"] == ["32", "64"]
    assert parsed["trainer.epochs"] == ["1"]  # singletons become one-item lists


@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_expand_grid_cartesian_count_and_content():
    expand_grid = _sym_or_skip(LAUNCHER_MOD, "expand_grid")
    parse_param_grid = _sym_or_skip(LAUNCHER_MOD, "parse_param_grid")

    spec = "a=1,2;b=10,20,30;c=Z"
    parsed = parse_param_grid(spec)
    expanded = expand_grid(parsed)
    # 2 * 3 * 1 = 6 combos
    assert len(expanded) == 6
    # Each item is a dict[str,str]
    for one in expanded:
        assert isinstance(one, dict)
        assert set(one) == {"a", "b", "c"}
    # Content sanity: set of tuples should match full cartesian product
    expected = set(itertools.product(["1", "2"], ["10", "20", "30"], ["Z"]))
    got = set((d["a"], d["b"], d["c"]) for d in expanded)
    assert got == expected


@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_to_hydra_overrides_shape_and_order():
    to_hydra_overrides = _sym_or_skip(LAUNCHER_MOD, "to_hydra_overrides")

    sample = {"optimizer.lr": "1e-3", "training.batch_size": "64", "trainer.epochs": "5"}
    overrides = to_hydra_overrides(sample)
    assert isinstance(overrides, list)
    assert all(isinstance(s, str) and "=" in s for s in overrides)
    # Order-insensitive check – convert to set
    assert set(overrides) == {"optimizer.lr=1e-3", "training.batch_size=64", "trainer.epochs=5"}


@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_assign_seeds_deterministic_and_unique():
    assign_seeds = _sym_or_skip(LAUNCHER_MOD, "assign_seeds")

    seeds_a = assign_seeds(5, base_seed=12345)
    seeds_b = assign_seeds(5, base_seed=12345)
    assert seeds_a == seeds_b, "Deterministic assignment must be stable with same base_seed"
    assert len(seeds_a) == 5
    assert len(set(seeds_a)) == 5, "Seeds must be unique per run"
    # If base_seed is None, still stable by count/uniqueness
    seeds_c = assign_seeds(3, base_seed=None)
    assert len(seeds_c) == 3
    assert len(set(seeds_c)) == 3


# ---------- integration-like tests (launcher orchestration) ----------

@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_launch_grid_dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    launch_grid = _sym_or_skip(LAUNCHER_MOD, "launch_grid")
    expand_grid = _sym_or_skip(LAUNCHER_MOD, "expand_grid")
    parse_param_grid = _sym_or_skip(LAUNCHER_MOD, "parse_param_grid")

    # Fake the actual "execute single run" path if production launcher calls a function like _exec_one()
    # We don't assume internals; launch_grid(dry_run=True) should never execute anything.
    # Still, we provide a planted base_cmd to show how a real invocation might look (not used here).
    spec = "trainer.epochs=1,2;optimizer.lr=1e-3,1e-4"
    expanded = expand_grid(parse_param_grid(spec))

    result = launch_grid(
        expanded,
        base_cmd=["python", "-m", "spectramind", "train"],
        dry_run=True,
        fail_fast=True,
        seeds=None,
        log_root=str(tmp_path),
    )
    # Expect just a plan, nothing executed
    assert hasattr(result, "planned") and result.planned == 4
    assert hasattr(result, "succeeded") and result.succeeded == 0
    assert hasattr(result, "failed") and result.failed == 0
    assert hasattr(result, "logs_dir") and Path(result.logs_dir).exists()


@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_launch_grid_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Simulate one failing run and assert that fail_fast stops the remaining schedule.
    This monkeypatch relies on a private hook only if it exists; otherwise we skip.
    """
    expand_grid = _sym_or_skip(LAUNCHER_MOD, "expand_grid")
    parse_param_grid = _sym_or_skip(LAUNCHER_MOD, "parse_param_grid")
    launch_grid = _sym_or_skip(LAUNCHER_MOD, "launch_grid")

    # If the module exposes an injectable single-run exec function, patch it.
    # For example, a function like "run_one" or "_exec_one". If none present, we skip.
    run_one_name = None
    for candidate in ("run_one", "_run_one", "_exec_one", "exec_one"):
        if hasattr(LAUNCHER_MOD, candidate):
            run_one_name = candidate
            break
    if run_one_name is None:
        pytest.skip("No injectable single-run function found to simulate failure")

    calls = {"count": 0}

    def fake_run_one(*args, **kwargs):
        calls["count"] += 1
        # Make the first run fail; subsequent runs should not be attempted when fail_fast=True
        if calls["count"] == 1:
            raise RuntimeError("Simulated failure on first run")
        return 0  # success exit code

    monkeypatch.setattr(LAUNCHER_MOD, run_one_name, fake_run_one)

    spec = "trainer.epochs=1,2,3"  # three planned runs
    expanded = expand_grid(parse_param_grid(spec))

    res = launch_grid(
        expanded,
        base_cmd=["python", "-m", "spectramind", "train"],
        dry_run=False,
        fail_fast=True,
        seeds=[111, 222, 333],
        log_root=str(tmp_path),
    )
    assert res.planned == 3
    assert res.failed == 1
    # With fail_fast, the launcher should stop after the first failure:
    assert calls["count"] == 1 or res.succeeded == 0, "Fail-fast must prevent subsequent executions"


# ---------- CLI tests via Typer CliRunner (optional) ----------

@pytest.mark.skipif(
    any(x is None for x in (SPECTRAMIND_MOD, TYPER_MOD, CLICK_TESTING_MOD)),
    reason="Typer CLI or testing harness not available",
)
def test_cli_grid_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    spectramind grid --params "a=1,2;b=Q" --dry-run should:
      - exit 0
      - print planned run count (2)
      - NOT attempt to execute the underlying train command
    """
    app = getattr(SPECTRAMIND_MOD, "app", None)
    if app is None:
        pytest.skip("spectramind.app (Typer) not found")

    # Prevent any real executions from happening inside CLI by disabling spawn method if exposed
    # (We guard with hasattr to avoid failing if the CLI uses a different code path.)
    if LAUNCHER_MOD is not None and hasattr(LAUNCHER_MOD, "run_one"):
        monkeypatch.setattr(LAUNCHER_MOD, "run_one", lambda *a, **k: 0)

    CliRunner = getattr(CLICK_TESTING_MOD, "CliRunner", None)
    if CliRunner is None:
        pytest.skip("CliRunner not available")

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "grid",
            "--params",
            "trainer.epochs=1,2;b=Q",
            "--dry-run",
            "--log-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    # Sanity on output (don't bind to exact phrasing):
    assert "planned" in result.output.lower() or "plan" in result.output.lower()
    assert "2" in result.output  # 2 combinations


@pytest.mark.skipif(
    any(x is None for x in (SPECTRAMIND_MOD, TYPER_MOD, CLICK_TESTING_MOD)),
    reason="Typer CLI or testing harness not available",
)
def test_cli_grid_fail_fast(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    """
    Make the first run fail via monkeypatch and assert the CLI reports a failure and stops.
    """
    app = getattr(SPECTRAMIND_MOD, "app", None)
    if app is None:
        pytest.skip("spectramind.app (Typer) not found")

    # If the launcher exposes run_one, monkeypatch to raise on first call.
    if LAUNCHER_MOD is None or not hasattr(LAUNCHER_MOD, "run_one"):
        pytest.skip("Launcher 'run_one' not found for simulated failure")

    calls = {"count": 0}

    def fail_first(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("Boom")
        return 0

    monkeypatch.setattr(LAUNCHER_MOD, "run_one", fail_first)

    CliRunner = getattr(CLICK_TESTING_MOD, "CliRunner", None)
    if CliRunner is None:
        pytest.skip("CliRunner not available")

    runner = CliRunner()
    res = runner.invoke(
        app,
        [
            "grid",
            "--params",
            "trainer.epochs=1,2,3",
            "--fail-fast",
            "--log-root",
            str(tmp_path),
        ],
    )
    # Exit code non-zero is acceptable here (CLI may exit 1 upon failure).
    assert res.exit_code != 0 or "failed" in res.output.lower()
    # Ensure we did not proceed to all runs:
    assert calls["count"] == 1, "Fail-fast CLI should stop after first failing run"


# ---------- tiny utility tests (pathing and logging) ----------

@pytest.mark.skipif(LAUNCHER_MOD is None, reason="config_grid_launcher module not available yet")
def test_logs_dir_contains_grid_token(tmp_path: Path):
    """
    If the launcher creates a logs root, we expect it to include a recognizable grid token (e.g., 'grid', 'sweep').
    """
    launch_grid = _sym_or_skip(LAUNCHER_MOD, "launch_grid")
    result = launch_grid([], base_cmd=None, dry_run=True, fail_fast=False, seeds=None, log_root=str(tmp_path))
    # If logs_dir is returned, it should exist and have a sensible name.
    logs_dir = Path(getattr(result, "logs_dir", str(tmp_path)))
    assert logs_dir.exists()
    tokenized = logs_dir.name.lower()
    assert any(tok in tokenized for tok in ("grid", "sweep", "multirun", "runs", "launcher")), \
        f"logs_dir name '{logs_dir.name}' should convey grid/multirun semantics"
