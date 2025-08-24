#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tests/regression/test_calibration_checker.py

SpectraMind V50 — Regression Tests
Calibration Checker (σ vs residuals): coverage, z-scores, quantiles

This suite validates the upgraded calibration checker in a *fast*, *non-destructive*,
and *signature-agnostic* way.

It supports three discovery paths:
  1) Python API:
       • tools.check_calibration:run_check / check_calibration / main
       • src.tools.check_calibration:run_check / check_calibration
       • diagnostics.check_calibration:run_check / check_calibration
  2) Module CLI:
       python -m tools.check_calibration ... (if importable)
       (also tries python -m tools.calibration_checker / tools.eval_calibration)
  3) Typer/CLI script:
       direct path execution of:
         tools/check_calibration.py
         tools/calibration_checker.py
         tools/eval_calibration.py

What we check
-------------
• End-to-end run on tiny synthetic μ/σ and y_true with known noise scale.
• Summary JSON produced with finite numeric metrics (coverage, z hist, etc.).
• At least one plot artifact (PNG/SVG/PDF) is emitted (if plotting enabled).
• Optional CSV artifacts (per-bin coverage/z histogram) if supported.
• --outdir discipline (no stray writes outside outdir except logs/).
• Append-only audit log behavior across runs.
• Robustness to NaN/Inf and near-zero σ (sanitization/clamping).
• Optional quantile evaluation flags (e.g., --quantiles 0.8,0.95) if supported.

Design
------
• Synthetic data: y_true ~ μ + N(0, σ_true). Predictions have σ_pred ≈ σ_true * s (scale).
• We avoid brittle assumptions about exact filenames; we only require "reasonable" outputs.
• If neither API nor CLI is present yet, tests skip with clear messaging.

Run
---
pytest -q tests/regression/test_calibration_checker.py
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Headless plotting for CI
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("SPECTRAMIND_TEST", "1")


# ======================================================================================
# Synthetic data
# ======================================================================================

def _make_synthetic_calibration(
    n_planets: int = 32,
    n_bins: int = 61,
    seed: int = 123,
    sigma_scale: float = 1.0,
) -> Dict[str, np.ndarray]:
    """
    Create tiny arrays (μ_pred, σ_pred, y_true):
      • μ_pred = smooth baseline + small spectral structure + noise
      • σ_true = baseline 0.03 with curvature-based variation
      • y_true = μ_pred + eps, eps ~ N(0, σ_true)
      • σ_pred = σ_true * sigma_scale (to test under/over-confidence)
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 2*np.pi, n_bins, dtype=np.float64)

    mu_list, sigma_list, y_list = [], [], []
    for _ in range(n_planets):
        base = 0.2*np.sin(x) + 0.05*np.cos(2*x)
        wiggle = rng.normal(0, 0.01, size=n_bins)
        this_mu = base + wiggle

        # curvature-proportional sigma_true with lower floor
        curv = np.abs(np.gradient(np.gradient(this_mu)))
        sigma_true = 0.03 + 0.05 * (curv / (curv.max() + 1e-6))

        eps = rng.normal(0, sigma_true, size=n_bins)
        y = this_mu + eps

        mu_list.append(this_mu)
        sigma_list.append(sigma_true * sigma_scale)
        y_list.append(y)

    mu = np.stack(mu_list, axis=0).astype(np.float64)
    sigma = np.stack(sigma_list, axis=0).astype(np.float64)
    y_true = np.stack(y_list, axis=0).astype(np.float64)

    return {"mu": mu, "sigma": sigma, "y_true": y_true}


def _inject_pathologies(arrs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    mu = arrs["mu"].copy()
    sigma = arrs["sigma"].copy()
    y_true = arrs["y_true"].copy()

    # NaN/Inf/near-zero in σ; NaN in μ
    if mu.size >= 10:
        mu[1, 3] = np.nan
    if sigma.size >= 20:
        sigma[2, 5] = np.inf
    if sigma.size >= 30:
        sigma[3, 7] = 1e-12  # near-zero (should be clamped inside tool)
    return {"mu": mu, "sigma": sigma, "y_true": y_true}


# ======================================================================================
# Discovery
# ======================================================================================

# Candidate script paths (under tools/)
CANDIDATE_SCRIPTS = [
    "check_calibration.py",
    "calibration_checker.py",
    "eval_calibration.py",
]

# Candidate module names for python -m
CANDIDATE_MODULES = [
    "tools.check_calibration",
    "tools.calibration_checker",
    "tools.eval_calibration",
]

# Candidate API callables
CANDIDATE_API = [
    ("tools.check_calibration", "run_check"),
    ("tools.check_calibration", "check_calibration"),
    ("tools.check_calibration", "main"),
    ("src.tools.check_calibration", "run_check"),
    ("diagnostics.check_calibration", "run_check"),
]


def _discover_script(repo_root: Path) -> Optional[Path]:
    for name in CANDIDATE_SCRIPTS:
        p = repo_root / "tools" / name
        if p.exists():
            return p
    return None


def _discover_module() -> Optional[str]:
    for mod in CANDIDATE_MODULES:
        try:
            __import__(mod)
            return mod
        except Exception:
            continue
    return None


def _discover_api() -> Optional[Callable]:
    for mod_name, attr in CANDIDATE_API:
        try:
            mod = __import__(mod_name, fromlist=[attr])
            fn = getattr(mod, attr, None)
            if callable(fn):
                return fn
        except Exception:
            continue
    return None


# ======================================================================================
# Runners
# ======================================================================================

def _run_via_api(
    fn: Callable,
    mu: np.ndarray,
    sigma: np.ndarray,
    y_true: np.ndarray,
    outdir: Path,
    extra_kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Call API with flexible kwargs; only pass accepted parameters.
    """
    import inspect
    sig = None
    try:
        sig = inspect.signature(fn)
    except Exception:
        pass

    kwargs: Dict[str, Any] = {
        "mu": mu,
        "sigma": sigma,
        "y_true": y_true,
        "outdir": str(outdir),
        "save_plots": True,
        "save_csv": True,
        "save_json": True,
        "quiet": True,
        "no_browser": True,
        "quantiles": [0.8, 0.95],
        "sanitize": True,
        "clamp_min_sigma": 1e-6,
        "dpi": 120,
    }
    if extra_kwargs:
        kwargs.update(extra_kwargs)

    if sig is not None:
        kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}

    return fn(**kwargs)


def _run_via_module_cli(
    module_name: str,
    mu_path: Path,
    sigma_path: Path,
    y_true_path: Path,
    outdir: Path,
    extra_flags: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SPECTRAMIND_TEST", "1")

    cmd = [
        sys.executable, "-m", module_name,
        "--mu", str(mu_path),
        "--sigma", str(sigma_path),
        "--y-true", str(y_true_path),
        "--outdir", str(outdir),
        "--save-plots",
        "--save-csv",
        "--save-json",
        "--no-browser",
        "--quantiles", "0.8,0.95",
        "--dpi", "120",
    ]
    if extra_flags:
        cmd += list(extra_flags)

    return subprocess.run(
        cmd, env=env, cwd=str(Path.cwd()),
        capture_output=True, text=True, timeout=90
    )


def _run_via_script(
    script_path: Path,
    mu_path: Path,
    sigma_path: Path,
    y_true_path: Path,
    outdir: Path,
    extra_flags: Optional[List[str]] = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("SPECTRAMIND_TEST", "1")

    cmd = [
        sys.executable, str(script_path),
        "--mu", str(mu_path),
        "--sigma", str(sigma_path),
        "--y-true", str(y_true_path),
        "--outdir", str(outdir),
        "--save-plots",
        "--save-csv",
        "--save-json",
        "--no-browser",
        "--quantiles", "0.8,0.95",
        "--dpi", "120",
    ]
    if extra_flags:
        cmd += list(extra_flags)

    return subprocess.run(
        cmd, env=env, cwd=str(Path.cwd()),
        capture_output=True, text=True, timeout=90
    )


# ======================================================================================
# Utilities
# ======================================================================================

def _scan_artifacts(outdir: Path) -> Dict[str, List[Path]]:
    return {
        "json": [p for p in outdir.rglob("*.json")],
        "csv":  [p for p in outdir.rglob("*.csv")],
        "png":  [p for p in outdir.rglob("*.png")],
        "svg":  [p for p in outdir.rglob("*.svg")],
        "pdf":  [p for p in outdir.rglob("*.pdf")],
    }


def _assert_nonempty(path: Path) -> None:
    assert path.exists() and path.is_file() and path.stat().st_size > 0, f"Empty or missing file: {path}"


def _ensure_repo_scaffold(repo_root: Path) -> None:
    (repo_root / "tools").mkdir(parents=True, exist_ok=True)
    (repo_root / "logs").mkdir(parents=True, exist_ok=True)
    (repo_root / "outputs" / "diagnostics").mkdir(parents=True, exist_ok=True)


# ======================================================================================
# Pytest fixtures
# ======================================================================================

@pytest.fixture(scope="function")
def repo_tmp(tmp_path: Path) -> Path:
    _ensure_repo_scaffold(tmp_path)
    return tmp_path


@pytest.fixture(scope="function")
def tiny_inputs(repo_tmp: Path) -> Dict[str, Path]:
    """
    Write tiny μ/σ/y_true to disk for CLI paths.
    """
    arrs = _make_synthetic_calibration(n_planets=24, n_bins=41, seed=777, sigma_scale=1.1)
    inputs_dir = repo_tmp / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    mu_p = inputs_dir / "mu.npy"
    sg_p = inputs_dir / "sigma.npy"
    yt_p = inputs_dir / "y_true.npy"
    np.save(mu_p, arrs["mu"])
    np.save(sg_p, arrs["sigma"])
    np.save(yt_p, arrs["y_true"])
    return {"mu": mu_p, "sigma": sg_p, "y_true": yt_p}


# ======================================================================================
# Tests
# ======================================================================================

@pytest.mark.integration
def test_calibration_checker_basic(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Basic end-to-end run via API or CLI/module.
    Expect diagnostic summary JSON, at least one plot, and audit log appended.
    """
    outdir = repo_tmp / "outputs" / "diagnostics" / "cal_check_basic"
    outdir.mkdir(parents=True, exist_ok=True)

    api_fn = _discover_api()
    mod = _discover_module() if api_fn is None else None
    script = _discover_script(repo_tmp) if (api_fn is None and mod is None) else None

    if api_fn is not None:
        mu = np.load(tiny_inputs["mu"])
        sigma = np.load(tiny_inputs["sigma"])
        y_true = np.load(tiny_inputs["y_true"])
        _ = _run_via_api(
            api_fn, mu, sigma, y_true, outdir,
            extra_kwargs={"random_state": 123, "seed": 123}
        )
    elif mod is not None:
        proc = _run_via_module_cli(
            mod, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir
        )
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        assert proc.returncode == 0
    elif script is not None:
        proc = _run_via_script(script, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir)
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        assert proc.returncode == 0
    else:
        pytest.skip("No calibration checker API or CLI found. Add tools/check_calibration.py to enable this test.")

    arts = _scan_artifacts(outdir)
    assert arts["json"], "Expected a JSON summary artifact."
    assert arts["png"] or arts["svg"] or arts["pdf"], "Expected at least one plot artifact."

    # Validate JSON minimally
    # (Use the first json file; implementations may produce multiple)
    data = json.loads(arts["json"][0].read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    # Common fields we expect: metrics/calibration/coverage or similar
    keys = set(k.lower() for k in data.keys())
    assert any(k in keys for k in ("metrics", "summary", "calibration")), \
        "Summary JSON should include a metrics/calibration/summary object."
    # Ensure no NaN/Inf tokens in serialized JSON
    raw = arts["json"][0].read_text(encoding="utf-8")
    assert not any(tok in raw for tok in ("NaN", "Infinity", "-Infinity")), "JSON must not contain NaN/Inf literals."

    # Audit log
    log_path = repo_tmp / "logs" / "v50_debug_log.md"
    assert log_path.exists(), "Expected audit log logs/v50_debug_log.md"
    log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    assert "calibration" in log_text or "z-score" in log_text or "coverage" in log_text, \
        "Audit log should mention calibration/z-score/coverage."


@pytest.mark.integration
def test_outdir_respected_and_no_strays(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Ensure the checker writes under --outdir (plus logs/) and does not create stray files.
    """
    outdir = repo_tmp / "outputs" / "diagnostics" / "cal_check_outdir"
    outdir.mkdir(parents=True, exist_ok=True)

    before = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())

    api_fn = _discover_api()
    mod = _discover_module() if api_fn is None else None
    script = _discover_script(repo_tmp) if (api_fn is None and mod is None) else None

    if api_fn is not None:
        mu = np.load(tiny_inputs["mu"])
        sigma = np.load(tiny_inputs["sigma"])
        y_true = np.load(tiny_inputs["y_true"])
        _ = _run_via_api(api_fn, mu, sigma, y_true, outdir)
    elif mod is not None:
        assert _run_via_module_cli(mod, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir).returncode == 0
    elif script is not None:
        assert _run_via_script(script, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir).returncode == 0
    else:
        pytest.skip("No calibration checker available.")

    after = set(p.relative_to(repo_tmp).as_posix() for p in repo_tmp.rglob("*") if p.is_file())
    new_files = sorted(list(after - before))

    disallowed: List[str] = []
    out_rel = outdir.relative_to(repo_tmp).as_posix()
    for rel in new_files:
        if rel.startswith("logs/"):
            continue
        if rel.startswith(out_rel):
            continue
        if rel.startswith("outputs/") and re.search(r"run_hash_summary.*\.json$", rel):
            continue
        if rel.endswith(".pyc") or "/__pycache__/" in rel:
            continue
        disallowed.append(rel)

    assert not disallowed, f"Unexpected stray writes outside --outdir: {disallowed}"


@pytest.mark.integration
def test_handles_nan_inf_and_tiny_sigma(repo_tmp: Path) -> None:
    """
    Robustness: inject NaN/Inf and near-zero σ, expect clean completion and artifacts.
    """
    clean = _make_synthetic_calibration(n_planets=16, n_bins=33, seed=991, sigma_scale=1.0)
    arrs = _inject_pathologies(clean)

    inputs_dir = repo_tmp / "inputs" / "edge"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    mu_p, sg_p, yt_p = inputs_dir / "mu.npy", inputs_dir / "sigma.npy", inputs_dir / "y_true.npy"
    np.save(mu_p, arrs["mu"])
    np.save(sg_p, arrs["sigma"])
    np.save(yt_p, arrs["y_true"])

    outdir = repo_tmp / "outputs" / "diagnostics" / "cal_check_edge"
    outdir.mkdir(parents=True, exist_ok=True)

    api_fn = _discover_api()
    mod = _discover_module() if api_fn is None else None
    script = _discover_script(repo_tmp) if (api_fn is None and mod is None) else None

    if api_fn is not None:
        _ = _run_via_api(
            api_fn, arrs["mu"], arrs["sigma"], arrs["y_true"], outdir,
            extra_kwargs={"sanitize": True, "clamp_min_sigma": 1e-6}
        )
    elif mod is not None:
        proc = _run_via_module_cli(
            mod, mu_p, sg_p, yt_p, outdir, extra_flags=["--sanitize", "--clamp-min-sigma", "1e-6"]
        )
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        assert proc.returncode == 0
    elif script is not None:
        proc = _run_via_script(
            script, mu_p, sg_p, yt_p, outdir, extra_flags=["--sanitize", "--clamp-min-sigma", "1e-6"]
        )
        if proc.returncode != 0:
            print("STDOUT:\n", proc.stdout)
            print("STDERR:\n", proc.stderr)
        assert proc.returncode == 0
    else:
        pytest.skip("No calibration checker available.")

    arts = _scan_artifacts(outdir)
    assert arts["json"], "Expected JSON outputs even on edge-case inputs."


@pytest.mark.integration
def test_quantile_and_coverage_fields_present(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    If quantile evaluation is supported (e.g., 0.8 / 0.95), expect corresponding fields
    in the summary JSON. If not, we xfail gracefully.
    """
    outdir = repo_tmp / "outputs" / "diagnostics" / "cal_check_quantiles"
    outdir.mkdir(parents=True, exist_ok=True)

    api_fn = _discover_api()
    mod = _discover_module() if api_fn is None else None
    script = _discover_script(repo_tmp) if (api_fn is None and mod is None) else None

    if api_fn is not None:
        mu = np.load(tiny_inputs["mu"]); sigma = np.load(tiny_inputs["sigma"]); y_true = np.load(tiny_inputs["y_true"])
        _ = _run_via_api(api_fn, mu, sigma, y_true, outdir, extra_kwargs={"quantiles": [0.8, 0.95]})
    elif mod is not None:
        assert _run_via_module_cli(mod, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir,
                                   extra_flags=["--quantiles", "0.8,0.95"]).returncode == 0
    elif script is not None:
        assert _run_via_script(script, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir,
                               extra_flags=["--quantiles", "0.8,0.95"]).returncode == 0
    else:
        pytest.skip("No calibration checker available.")

    arts = _scan_artifacts(outdir)
    if not arts["json"]:
        pytest.xfail("No JSON summary to check for quantile/coverage fields (acceptable if tool only plots).")
    data = json.loads(arts["json"][0].read_text(encoding="utf-8"))

    # Loosely check for quantile/coverage keys
    blob = json.dumps(data).lower()
    if ("quantile" not in blob) and ("coverage" not in blob):
        pytest.xfail("Calibration checker did not emit quantile/coverage fields (acceptable for minimal version).")


@pytest.mark.integration
def test_audit_log_append_only(repo_tmp: Path, tiny_inputs: Dict[str, Path]) -> None:
    """
    Two calls should append to logs/v50_debug_log.md (or at least not shrink it).
    """
    log_path = repo_tmp / "logs" / "v50_debug_log.md"

    out1 = repo_tmp / "outputs" / "diagnostics" / "cal_check_log1"
    out2 = repo_tmp / "outputs" / "diagnostics" / "cal_check_log2"
    out1.mkdir(parents=True, exist_ok=True); out2.mkdir(parents=True, exist_ok=True)

    # First run
    api_fn = _discover_api()
    mod = _discover_module() if api_fn is None else None
    script = _discover_script(repo_tmp) if (api_fn is None and mod is None) else None

    def _do(outdir: Path):
        if api_fn is not None:
            mu = np.load(tiny_inputs["mu"]); sigma = np.load(tiny_inputs["sigma"]); y_true = np.load(tiny_inputs["y_true"])
            _ = _run_via_api(api_fn, mu, sigma, y_true, outdir)
        elif mod is not None:
            assert _run_via_module_cli(mod, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir).returncode == 0
        elif script is not None:
            assert _run_via_script(script, tiny_inputs["mu"], tiny_inputs["sigma"], tiny_inputs["y_true"], outdir).returncode == 0
        else:
            pytest.skip("No calibration checker available.")

    _do(out1)
    size1 = log_path.stat().st_size if log_path.exists() else 0
    _do(out2)
    size2 = log_path.stat().st_size if log_path.exists() else 0

    assert size2 >= size1, "Audit log size should not shrink."
    if size1 > 0:
        assert size2 > size1, "Audit log should typically grow after a second run."
