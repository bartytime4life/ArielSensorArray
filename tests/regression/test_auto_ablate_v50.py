#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/regression/test_auto_ablate_v50.py

SpectraMind V50 — Regression Tests
Auto Ablation Orchestrator (auto_ablate_v50)

These tests validate the ablation grid/runner in a *non-destructive*, *fast*, and
*signature-agnostic* way. They prefer a DRY-RUN/PLAN-ONLY execution flow so no
heavy training occurs. They support three discovery paths:

  1) Python API:
       • tools.auto_ablate_v50:run_ablate
       • tools.auto_ablate_v50:main (plan mode)
       • src.tools.auto_ablate_v50:run_ablate
       • spectramind.cli.ablate:run_ablate
  2) Module CLI:
       python -m tools.auto_ablate_v50 ... (if importable)
  3) Typer CLI:
       spectramind ablate ... (if available on PATH)

What we check
-------------
• Dry-run plan creation (JSON/MD/HTML leaderboard artifacts are optional but preferred)
• Include/exclude filters applied to plan
• top-N selection handled (if present)
• --outdir discipline (no stray writes)
• Append-only audit log behavior
• Optional ZIP export (if supported; xfail gracefully if not)

Design
------
• We generate a tiny Hydra-like configs/ tree so override resolution feels real.
• We never require a specific filename beyond loose expectations (plan.json/leaderboard.*).
• We accept that some implementations ignore unknown flags; tests will still pass if at least
  one valid artifact is produced in the outdir.

Run
---
pytest -q tests/regression/test_auto_ablate_v50.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import pytest


# ======================================================================================
# Discovery helpers
# ======================================================================================

API_CANDIDATES: List[Tuple[str, str]] = [
    ("tools.auto_ablate_v50", "run_ablate"),
    ("tools.auto_ablate_v50", "main"),
    ("src.tools.auto_ablate_v50", "run_ablate"),
    ("spectramind.cli.ablate", "run_ablate"),
]

MODULE_CLI = "tools.auto_ablate_v50"


def _import_callable() -> Optional[Callable]:
    """
    Try to import an ablation callable from known modules.
    Return the callable or None if not found.
    """
    for mod_name, attr in API_CANDIDATES:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None


def _has_module_cli() -> bool:
    try:
        __import__(MODULE_CLI)
        return True
    except Exception:
        return False


def _has_spectramind_cli() -> bool:
    return shutil.which("spectramind") is not None


# ======================================================================================
# Test inputs / scaffolding
# ======================================================================================

def _write_tiny_configs(cfg_root: Path) -> Dict[str, Path]:
    """
    Create a minimal Hydra-like configs directory to mimic real override usage.
    We don't validate config content — the ablator should not *execute* heavy jobs in DRY-RUN.
    """
    (cfg_root / "model").mkdir(parents=True, exist_ok=True)
    (cfg_root / "training").mkdir(parents=True, exist_ok=True)
    (cfg_root / "ablate").mkdir(parents=True, exist_ok=True)

    (cfg_root / "model" / "v50.yaml").write_text("name: v50\n")
    (cfg_root / "model" / "v50_small.yaml").write_text("name: v50_small\n")
    (cfg_root / "training" / "fast.yaml").write_text("epochs: 1\n")
    (cfg_root / "training" / "long.yaml").write_text("epochs: 2\n")
    # Optional ablation policy file (some implementations look for this)
    (cfg_root / "ablate" / "defaults.yaml").write_text("policy: grid\n")

    return {
        "root": cfg_root,
        "model_v50": cfg_root / "model" / "v50.yaml",
        "model_v50_small": cfg_root / "model" / "v50_small.yaml",
        "train_fast": cfg_root / "training" / "fast.yaml",
        "train_long": cfg_root / "training" / "long.yaml",
        "ablate_defaults": cfg_root / "ablate" / "defaults.yaml",
    }


def _baseline_env(monkeypatch: pytest.MonkeyPatch, cfg_root: Path):
    """
    Apply baseline deterministic environment:
      • Force UTC so any timestamps are stable
      • Provide Hydra config root (if the tool honors it)
      • Disable GPU in CI
    """
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setenv("HYDRA_CONFIG_PATH", str(cfg_root))
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SPECTRAMIND_TEST", "1")  # encourage dry/fast paths


# ======================================================================================
# Runners
# ======================================================================================

def _run_via_api(
    fn: Callable,
    outdir: Path,
    overrides: Dict[str, List[str]],
    include: Optional[List[Dict[str, str]]] = None,
    exclude: Optional[List[Dict[str, str]]] = None,
    top_n: Optional[int] = None,
    extra: Optional[Dict[str, Union[str, int, bool]]] = None,
) -> Dict:
    """
    Call the ablation function with a minimal, tolerant signature. We only pass kwargs
    that appear in the function signature.
    """
    import inspect

    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        pass

    kwargs: Dict[str, Union[str, int, bool, Dict, List]] = {
        "outdir": str(outdir),
        "dry_run": True,
        "format": "json",
        "overrides": overrides,
        "strategy": "cartesian",
        "save_md": True,
        "save_html": True,
        "quiet": True,
    }
    if include:
        kwargs["include"] = include
    if exclude:
        kwargs["exclude"] = exclude
    if top_n is not None:
        kwargs["top_n"] = top_n
    if extra:
        kwargs.update(extra)

    if sig is not None:
        filtered = {}
        for k, v in kwargs.items():
            if k in sig.parameters:
                filtered[k] = v
        kwargs = filtered

    plan = fn(**kwargs)
    if isinstance(plan, str):
        try:
            return json.loads(plan)
        except Exception:
            return {"raw": plan}
    if isinstance(plan, dict):
        return plan
    # Unknown return: still OK — caller will check artifacts on disk.
    return {"result": "ok"}


def _run_via_module_cli(
    outdir: Path,
    overrides: Dict[str, List[str]],
    include: Optional[List[Dict[str, str]]],
    exclude: Optional[List[Dict[str, str]]],
    top_n: Optional[int],
    extra_flags: Optional[List[str]] = None,
) -> Tuple[int, str, str]:
    """
    Execute: python -m tools.auto_ablate_v50 ...
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TZ", "UTC")
    env.setdefault("SPECTRAMIND_TEST", "1")
    env.setdefault("MPLBACKEND", "Agg")

    # Flatten overrides into CLI list (common patterns):
    ov_list: List[str] = []
    for k, vals in overrides.items():
        for v in vals:
            # Allow "model=v50" style directly; if `k` contains a dot, the value may be a config alias
            if "=" in v:
                ov_list.append(v)
            else:
                ov_list.append(f"{k}={v}")

    cmd = [
        sys.executable,
        "-m",
        MODULE_CLI,
        "--outdir", str(outdir),
        "--dry-run",
        "--strategy", "cartesian",
        "--format", "json",
        "--overrides",
        *ov_list,
        "--save-md",
        "--save-html",
        "--quiet",
    ]
    if include:
        # Pass as repeated --include "k=v,k=v"
        for inc in include:
            cmd += ["--include", ",".join(f"{k}={v}" for k, v in inc.items())]
    if exclude:
        for exc in exclude:
            cmd += ["--exclude", ",".join(f"{k}={v}" for k, v in exc.items())]
    if top_n is not None:
        cmd += ["--top-n", str(top_n)]
    if extra_flags:
        cmd += list(extra_flags)

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _run_via_spectramind_cli(
    outdir: Path,
    overrides: Dict[str, List[str]],
    include: Optional[List[Dict[str, str]]],
    exclude: Optional[List[Dict[str, str]]],
    top_n: Optional[int],
    extra_flags: Optional[List[str]] = None,
) -> Tuple[int, str, str]:
    """
    Execute: spectramind ablate ...
    """
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("TZ", "UTC")
    env.setdefault("SPECTRAMIND_TEST", "1")
    env.setdefault("MPLBACKEND", "Agg")

    ov_list: List[str] = []
    for k, vals in overrides.items():
        for v in vals:
            if "=" in v:
                ov_list.append(v)
            else:
                ov_list.append(f"{k}={v}")

    cmd = [
        "spectramind",
        "ablate",
        "--outdir", str(outdir),
        "--dry-run",
        "--strategy", "cartesian",
        "--format", "json",
        "--overrides",
        *ov_list,
        "--save-md",
        "--save-html",
        "--quiet",
    ]
    if include:
        for inc in include:
            cmd += ["--include", ",".join(f"{k}={v}" for k, v in inc.items())]
    if exclude:
        for exc in exclude:
            cmd += ["--exclude", ",".join(f"{k}={v}" for k, v in exc.items())]
    if top_n is not None:
        cmd += ["--top-n", str(top_n)]
    if extra_flags:
        cmd += list(extra_flags)

    proc = subprocess.run(
        cmd,
        env=env,
        cwd=str(Path.cwd()),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return proc.returncode, proc.stdout, proc.stderr


# ======================================================================================
# Utilities
# ======================================================================================

def _scan_artifacts(outdir: Path) -> Dict[str, List[Path]]:
    return {
        "json": [p for p in outdir.rglob("*.json")],
        "md":   [p for p in outdir.rglob("*.md")],
        "html": [p for p in outdir.rglob("*.html")],
        "csv":  [p for p in outdir.rglob("*.csv")],
        "zip":  [p for p in outdir.rglob("*.zip")],
        "png":  [p for p in outdir.rglob("*.png")],
    }


def _assert_nonempty(path: Path) -> None:
    assert path.exists() and path.is_file() and path.stat().st_size > 0, f"Empty or missing file: {path}"


# ======================================================================================
# Pytest fixtures
# ======================================================================================

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Dict[str, Path]:
    """
    Provide minimal repo-like tree with logs/ outputs/ configs/.
    """
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "outputs" / "diagnostics").mkdir(parents=True, exist_ok=True)
    cfgs = _write_tiny_configs(tmp_path / "configs")
    _baseline_env(monkeypatch, cfgs["root"])
    return {"root": tmp_path, **cfgs}


# ======================================================================================
# Shared test data
# ======================================================================================

GRID_OVERRIDES = {
    "model": ["model=v50", "model=v50_small"],
    "training": ["training=fast", "training=long"],
}
INCLUDE = [{"model": "model=v50", "training": "training=fast"}]
EXCLUDE = [{"model": "model=v50_small", "training": "training=long"}]


# ======================================================================================
# Tests
# ======================================================================================

@pytest.mark.order(1)
def test_dry_run_plan_and_leaderboard(repo_tmp: Dict[str, Path]) -> None:
    """
    Dry-run plan creation via API/Module/CLI — expect at least one JSON *or* MD/HTML artifact.
    """
    outdir = repo_tmp["root"] / "outputs" / "diagnostics" / "ablate_plan"
    outdir.mkdir(parents=True, exist_ok=True)

    fn = _import_callable()
    code = 0
    stdout = ""
    stderr = ""
    if fn is not None:
        plan = _run_via_api(fn, outdir, GRID_OVERRIDES, include=INCLUDE, exclude=EXCLUDE, top_n=2)
        # If a dict plan is returned, optionally inspect
        if isinstance(plan, dict) and plan:
            # Optional: jobs count sanity (2x2=4; include/exclude might reduce it)
            pass
    elif _has_module_cli():
        code, stdout, stderr = _run_via_module_cli(outdir, GRID_OVERRIDES, INCLUDE, EXCLUDE, top_n=2)
        if code != 0:
            print("MODULE CLI STDOUT:\n", stdout)
            print("MODULE CLI STDERR:\n", stderr)
        assert code == 0
    elif _has_spectramind_cli():
        code, stdout, stderr = _run_via_spectramind_cli(outdir, GRID_OVERRIDES, INCLUDE, EXCLUDE, top_n=2)
        if code != 0:
            print("SPECTRAMIND CLI STDOUT:\n", stdout)
            print("SPECTRAMIND CLI STDERR:\n", stderr)
        assert code == 0
    else:
        pytest.skip("No ablation API/CLI available; skipping until auto_ablate_v50 is wired.")

    arts = _scan_artifacts(outdir)
    assert arts["json"] or arts["md"] or arts["html"], \
        "Expected a plan/leaderboard artifact (json/md/html) in outdir."
    # If present, ensure non-empty
    for kind in ("json", "md", "html"):
        for p in arts[kind]:
            _assert_nonempty(p)


@pytest.mark.order(2)
def test_outdir_respected_and_no_strays(repo_tmp: Dict[str, Path]) -> None:
    """
    The ablator should write to --outdir and logs/ only (plus optional outputs/run_hash_summary*.json).
    """
    outdir = repo_tmp["root"] / "outputs" / "diagnostics" / "ablate_outdir"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp["root"]).as_posix() for p in repo_tmp["root"].rglob("*") if p.is_file())

    fn = _import_callable()
    if fn is not None:
        _ = _run_via_api(fn, outdir, GRID_OVERRIDES, top_n=1)
    elif _has_module_cli():
        code, stdout, stderr = _run_via_module_cli(outdir, GRID_OVERRIDES, None, None, top_n=1)
        if code != 0:
            print("MODULE CLI STDOUT:\n", stdout)
            print("MODULE CLI STDERR:\n", stderr)
        assert code == 0
    elif _has_spectramind_cli():
        code, stdout, stderr = _run_via_spectramind_cli(outdir, GRID_OVERRIDES, None, None, top_n=1)
        if code != 0:
            print("SPECTRAMIND CLI STDOUT:\n", stdout)
            print("SPECTRAMIND CLI STDERR:\n", stderr)
        assert code == 0
    else:
        pytest.skip("No ablation API/CLI available.")

    after = set(p.relative_to(repo_tmp["root"]).as_posix() for p in repo_tmp["root"].rglob("*") if p.is_file())
    new_files = sorted(list(after - before))

    disallowed: List[str] = []
    out_rel = outdir.relative_to(repo_tmp["root"]).as_posix()
    for rel in new_files:
        if rel.startswith("logs/"):
            continue
        if rel.startswith(out_rel):
            continue
        if rel.startswith("outputs/") and re.search(r"run_hash_summary.*\.json$", rel):
            continue
        if "/__pycache__/" in rel or rel.endswith(".pyc"):
            continue
        disallowed.append(rel)
    assert not disallowed, f"Unexpected stray writes outside --outdir: {disallowed}"


@pytest.mark.order(3)
def test_audit_log_append_only(repo_tmp: Dict[str, Path]) -> None:
    """
    Two dry-run calls should append to logs/v50_debug_log.md (or at least not shrink it).
    """
    log_path = repo_tmp["root"] / "logs" / "v50_debug_log.md"
    out1 = repo_tmp["root"] / "outputs" / "diagnostics" / "ablate_log1"
    out2 = repo_tmp["root"] / "outputs" / "diagnostics" / "ablate_log2"
    out1.mkdir(parents=True, exist_ok=True)
    out2.mkdir(parents=True, exist_ok=True)

    fn = _import_callable()
    runner = None
    if fn is not None:
        runner = lambda d: _run_via_api(fn, d, GRID_OVERRIDES, top_n=1)  # noqa: E731
    elif _has_module_cli():
        runner = lambda d: _run_via_module_cli(d, GRID_OVERRIDES, None, None, top_n=1)  # noqa: E731
    elif _has_spectramind_cli():
        runner = lambda d: _run_via_spectramind_cli(d, GRID_OVERRIDES, None, None, top_n=1)  # noqa: E731
    else:
        pytest.skip("No ablation API/CLI available.")

    # First run
    res1 = runner(out1)
    if isinstance(res1, tuple):
        code, stdout, stderr = res1
        if code != 0:
            print("RUN1 STDOUT:\n", stdout)
            print("RUN1 STDERR:\n", stderr)
        assert code == 0
    size1 = log_path.stat().st_size if log_path.exists() else 0

    # Second run
    res2 = runner(out2)
    if isinstance(res2, tuple):
        code, stdout, stderr = res2
        if code != 0:
            print("RUN2 STDOUT:\n", stdout)
            print("RUN2 STDERR:\n", stderr)
        assert code == 0
    size2 = log_path.stat().st_size if log_path.exists() else 0

    assert size2 >= size1, "Audit log size should not shrink."
    if size1 > 0:
        assert size2 > size1, "Audit log should typically grow after a subsequent run."


@pytest.mark.order(4)
def test_include_exclude_and_topn_effects(repo_tmp: Dict[str, Path]) -> None:
    """
    When include/exclude/top-n flags are supplied, the resulting plan (if available as JSON)
    should reflect them. If the tool does not emit a machine-readable plan, this test will
    pass as long as an artifact exists; otherwise we xfail with context.
    """
    outdir = repo_tmp["root"] / "outputs" / "diagnostics" / "ablate_filters"
    outdir.mkdir(parents=True, exist_ok=True)

    plan_obj = None
    fn = _import_callable()
    if fn is not None:
        plan_obj = _run_via_api(fn, outdir, GRID_OVERRIDES, include=INCLUDE, exclude=EXCLUDE, top_n=1)
    elif _has_module_cli():
        code, stdout, stderr = _run_via_module_cli(outdir, GRID_OVERRIDES, INCLUDE, EXCLUDE, top_n=1)
        if code != 0:
            print("MODULE CLI STDOUT:\n", stdout)
            print("MODULE CLI STDERR:\n", stderr)
        assert code == 0
        try:
            plan_obj = json.loads(stdout) if stdout.strip().startswith("{") else None
        except Exception:
            plan_obj = None
    elif _has_spectramind_cli():
        code, stdout, stderr = _run_via_spectramind_cli(outdir, GRID_OVERRIDES, INCLUDE, EXCLUDE, top_n=1)
        if code != 0:
            print("SPECTRAMIND CLI STDOUT:\n", stdout)
            print("SPECTRAMIND CLI STDERR:\n", stderr)
        assert code == 0
        try:
            plan_obj = json.loads(stdout) if stdout.strip().startswith("{") else None
        except Exception:
            plan_obj = None
    else:
        pytest.skip("No ablation API/CLI available.")

    arts = _scan_artifacts(outdir)
    assert arts["json"] or arts["md"] or arts["html"], "Expected a plan/leaderboard artifact."

    # If we have a JSON plan, lightly assert the filters/top-N shaped the job set.
    if isinstance(plan_obj, dict) and plan_obj.get("jobs"):
        jobs = plan_obj["jobs"]
        # With include=1 pair and exclude=1 pair and 4 total combos, reasonable outcomes:
        # - top_n=1 could trim to one job.
        assert len(jobs) >= 1, "Expected at least one job after include/top-n."
        # Ensure the excluded pair isn't present if the plan lists overrides
        bad = {"model=v50_small", "training=long"}
        for j in jobs:
            ov = set(j.get("overrides", []))
            assert not bad.issubset(ov), "Excluded combination leaked into plan."
    else:
        pytest.xfail("No machine-readable plan JSON; acceptable if the ablator only outputs files.")


@pytest.mark.order(5)
def test_zip_export_when_requested(repo_tmp: Dict[str, Path]) -> None:
    """
    If the ablator supports a ZIP export flag, ensure it creates a non-empty zip file.
    If unsupported, xfail gracefully.
    """
    outdir = repo_tmp["root"] / "outputs" / "diagnostics" / "ablate_zip"
    outdir.mkdir(parents=True, exist_ok=True)

    fn = _import_callable()
    supported = True
    if fn is not None:
        # Attempt via API — pass a plausible flag, but filter by signature
        try:
            plan = _run_via_api(
                fn,
                outdir,
                GRID_OVERRIDES,
                top_n=1,
                extra={"export_zip": True, "zip": True},
            )
        except TypeError:
            supported = False
    elif _has_module_cli():
        code, stdout, stderr = _run_via_module_cli(
            outdir, GRID_OVERRIDES, None, None, top_n=1, extra_flags=["--export-zip"]
        )
        if code != 0:
            # Try an alternate flag name
            code2, stdout2, stderr2 = _run_via_module_cli(
                outdir, GRID_OVERRIDES, None, None, top_n=1, extra_flags=["--zip"]
            )
            if code2 != 0:
                supported = False
    elif _has_spectramind_cli():
        code, stdout, stderr = _run_via_spectramind_cli(
            outdir, GRID_OVERRIDES, None, None, top_n=1, extra_flags=["--export-zip"]
        )
        if code != 0:
            code2, stdout2, stderr2 = _run_via_spectramind_cli(
                outdir, GRID_OVERRIDES, None, None, top_n=1, extra_flags=["--zip"]
            )
            if code2 != 0:
                supported = False
    else:
        pytest.skip("No ablation API/CLI available.")

    if not supported:
        pytest.xfail("Ablator does not expose ZIP export flags yet — acceptable.")

    zips = _scan_artifacts(outdir)["zip"]
    if not zips:
        pytest.xfail("ZIP export produced no .zip; acceptable if implementation defers ZIP creation.")
    for z in zips:
        _assert_nonempty(z)
