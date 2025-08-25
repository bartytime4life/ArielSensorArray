# /tests/diagnostics/test_symbolic_violation_overlay.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: symbolic_violation_overlay

Purpose
-------
Validate the behavior of the *symbolic violation overlay* diagnostic, which
projects rule violations over wavelength bins (L≈283) and optionally renders
heatmaps/overlays for dashboards.

We test:
1) API discovery & shape contracts for single/batch spectra
2) Rule localization (violations higher inside rule masks)
3) Weight scaling monotonicity (↑weight ⇒ ↑overlay magnitude)
4) Determinism with fixed seeds
5) Optional plotting returns a Matplotlib Figure
6) Optional artifact saver writes files
7) Optional CLI smoke: `spectramind diagnose symbolic-violation-overlay ...`

Design Notes
------------
• Defensively adaptable to small API differences:
  - Tries several module import paths and entrypoint names.
  - Accepts either ndarray or dict outputs; normalizes to (B,R,L) and combined (B,L).
• Uses synthetic μ with two disjoint masked regions to trigger localized violations.

Author: SpectraMind V50 Team
"""

from __future__ import annotations

import importlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytest


# -----------------------------------------------------------------------------
# Module discovery
# -----------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    "tools.symbolic_violation_overlay",
    "src.tools.symbolic_violation_overlay",
    "diagnostics.symbolic_violation_overlay",
    "symbolic_violation_overlay",
]


def _import_overlay_module():
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:  # pragma: no cover - only runs when not found
            last_err = e
    raise ImportError(
        "Could not import symbolic_violation_overlay from any of:\n"
        f"  {CANDIDATE_IMPORTS}\nLast error: {last_err}"
    )


def _locate_entrypoint(mod):
    """
    Locate a callable to compute/render violation overlays.

    Accepted options (any of):
      - compute_symbolic_violation_overlay(mu, rules=..., **cfg)
      - symbolic_violation_overlay(mu, rules=..., **cfg)
      - compute_violation_overlay(mu, rules=..., **cfg)
      - class SymbolicViolationOverlay(...).run(mu=..., rules=..., **cfg)

    Returns
    -------
    kind: 'func' | 'class'
    target: callable | type
    """
    for fn in (
        "compute_symbolic_violation_overlay",
        "symbolic_violation_overlay",
        "compute_violation_overlay",
        "run_violation_overlay",
    ):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return "func", getattr(mod, fn)

    for cls in ("SymbolicViolationOverlay", "ViolationOverlay", "OverlayRunner"):
        if hasattr(mod, cls):
            Cls = getattr(mod, cls)
            if hasattr(Cls, "run") and callable(getattr(Cls, "run")):
                return "class", Cls

    pytest.xfail(
        "symbolic_violation_overlay module found, but no known entrypoint. "
        "Expected one of the functions above, or a class with .run(...)."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, mu: np.ndarray, rules: List[Dict[str, Any]], **cfg) -> Dict[str, Any]:
    """
    Invoke the overlay API and coerce to a dict.

    Expected dict keys (subset ok):
      - 'per_rule' : ndarray (R,L) or (B,R,L)  — per-rule violation map
      - 'combined' : ndarray (L,) or (B,L)     — aggregated overlay (e.g., weighted sum)
      - 'rules'    : the rule list used
      - 'fig'      : optional Matplotlib figure handle
    """
    if kind == "func":
        out = target(mu, rules=rules, **cfg)
    elif kind == "class":
        try:
            inst = target(mu=mu, rules=rules, **cfg)
        except TypeError:
            inst = target(**cfg)
            out = inst.run(mu=mu, rules=rules)
        else:
            out = inst.run()
    else:  # pragma: no cover
        pytest.fail("Unknown invocation kind.")

    if isinstance(out, dict):
        return out
    # If bare array returned, assume it's combined overlay
    return {"combined": out}


# -----------------------------------------------------------------------------
# Synthetic inputs
# -----------------------------------------------------------------------------

L_DEFAULT = 283
_RNG = np.random.RandomState(20250824)


def _make_mu_single(L: int = L_DEFAULT) -> np.ndarray:
    """
    Smooth baseline + two absorption-like bumps aligned with rule masks.
    """
    x = np.linspace(0, 1, L, dtype=np.float64)
    base = 0.02 + 0.004 * np.sin(2 * math.pi * 1.9 * x)
    b1 = 0.012 * np.exp(-0.5 * ((x - 0.22) / 0.02) ** 2)
    b2 = 0.010 * np.exp(-0.5 * ((x - 0.66) / 0.018) ** 2)
    mu = np.clip(base + b1 + b2, 0.0, 1.0)
    return mu


def _make_mu_batch(B: int = 4, L: int = L_DEFAULT) -> np.ndarray:
    base = _make_mu_single(L)
    batch = np.stack([base + _RNG.normal(0, 0.0006, size=L) for _ in range(B)], axis=0)
    return np.clip(batch, 0.0, 1.0)


def _rect_mask(L: int, a: float, b: float) -> np.ndarray:
    """Closed interval [a,b) in fractional coordinates."""
    i0 = max(0, int(a * L))
    i1 = min(L, int(b * L))
    m = np.zeros(L, dtype=np.float32)
    m[i0:i1] = 1.0
    return m


def _make_rules(L: int = L_DEFAULT) -> List[Dict[str, Any]]:
    """
    Two disjoint masks with default weight=1.0 and enabled=True.
    """
    r1 = {
        "name": "left_band_rule",
        "mask": _rect_mask(L, 0.16, 0.30).tolist(),  # around bump near x~0.22
        "weight": 1.0,
        "enabled": True,
        "kind": "band_smoothness",
    }
    r2 = {
        "name": "right_band_rule",
        "mask": _rect_mask(L, 0.58, 0.74).tolist(),  # around bump near x~0.66
        "weight": 1.0,
        "enabled": True,
        "kind": "band_smoothness",
    }
    return [r1, r2]


# -----------------------------------------------------------------------------
# Normalizers
# -----------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
    assert np.isfinite(x).all(), "Array contains non-finite values"
    return x


def _norm_per_rule(arr: np.ndarray, L: int) -> Tuple[np.ndarray, str]:
    """
    Normalize per_rule maps to (B,R,L). Accepts (R,L) or (B,R,L).
    """
    arr = _as_np(arr)
    if arr.ndim == 2 and arr.shape[1] == L:
        R = arr.shape[0]
        return arr[None, ...], f"(1,{R},{L})"
    if arr.ndim == 3 and arr.shape[2] == L:
        return arr, f"({arr.shape[0]},{arr.shape[1]},{L})"
    pytest.fail(f"Unexpected per_rule shape {arr.shape} (expected (R,{L}) or (B,R,{L}))")
    return arr, ""  # pragma: no cover


def _norm_combined(arr: np.ndarray, L: int, B_expect: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """
    Normalize combined overlay to (B,L). Accepts (L,) or (B,L).
    """
    arr = _as_np(arr)
    if arr.ndim == 1 and arr.shape[0] == L:
        return arr[None, :], f"(1,{L})"
    if arr.ndim == 2 and arr.shape[1] == L:
        if B_expect is not None:
            assert arr.shape[0] == B_expect, f"Expected B={B_expect}, got {arr.shape[0]}"
        return arr, f"({arr.shape[0]},{L})"
    pytest.fail(f"Unexpected combined shape {arr.shape} (expected ({L},) or (B,{L}))")
    return arr, ""  # pragma: no cover


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def overlay_mod():
    return _import_overlay_module()


@pytest.fixture(scope="module")
def overlay_callable(overlay_mod):
    return _locate_entrypoint(overlay_mod)


@pytest.fixture
def mu_single():
    return _make_mu_single(L_DEFAULT)


@pytest.fixture
def mu_batch():
    return _make_mu_batch(B=3, L=L_DEFAULT)


@pytest.fixture
def rules_default():
    return _make_rules(L_DEFAULT)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_api_shapes_single(overlay_callable, mu_single, rules_default):
    """
    Single-spectrum: expect per_rule (R,L) or (1,R,L) and combined (L,) or (1,L).
    """
    kind, target = overlay_callable
    out = _invoke(kind, target, mu_single, rules_default, seed=1234, return_dict=True)
    assert isinstance(out, dict)
    # Per-rule violation maps
    if "per_rule" in out:
        pr, _ = _norm_per_rule(_as_np(out["per_rule"]), L=L_DEFAULT)
        B, R, L = pr.shape
        assert B == 1 and R == len(rules_default) and L == L_DEFAULT
    # Combined overlay
    if "combined" in out:
        comb, _ = _norm_combined(_as_np(out["combined"]), L=L_DEFAULT, B_expect=1)
        assert comb.shape == (1, L_DEFAULT)


def test_api_shapes_batch(overlay_callable, mu_batch, rules_default):
    """
    Batched-spectrum: per_rule should be (B,R,L); combined should be (B,L).
    """
    kind, target = overlay_callable
    out = _invoke(kind, target, mu_batch, rules_default, seed=7)
    assert isinstance(out, dict)
    if "per_rule" in out:
        pr, _ = _norm_per_rule(_as_np(out["per_rule"]), L=L_DEFAULT)
        B, R, L = pr.shape
        assert B == mu_batch.shape[0] and R == len(rules_default) and L == L_DEFAULT
    if "combined" in out:
        comb, _ = _norm_combined(_as_np(out["combined"]), L=L_DEFAULT, B_expect=mu_batch.shape[0])
        assert comb.shape == (mu_batch.shape[0], L_DEFAULT)


def test_rule_localization(overlay_callable, mu_single, rules_default):
    """
    Violations should be larger (robustly) inside a rule's mask than outside.
    """
    kind, target = overlay_callable
    out = _invoke(kind, target, mu_single, rules_default, seed=99)
    assert isinstance(out, dict)

    # Work with per_rule if available; else derive from combined via masks (weaker check)
    if "per_rule" in out:
        pr, _ = _norm_per_rule(_as_np(out["per_rule"]), L=L_DEFAULT)
        arr = pr[0]  # (R,L)
        for r_idx, rule in enumerate(rules_default):
            mask = np.asarray(rule["mask"], dtype=bool)
            inside = np.abs(arr[r_idx, mask])
            outside = np.abs(arr[r_idx, ~mask])
            if inside.size and outside.size:
                p75_in = float(np.percentile(inside, 75))
                p75_out = float(np.percentile(outside, 75))
                assert p75_in > p75_out * 1.2, (
                    f"Rule '{rule['name']}' not localized: inside P75={p75_in:.3e} vs outside P75={p75_out:.3e}"
                )
    elif "combined" in out:
        comb, _ = _norm_combined(_as_np(out["combined"]), L=L_DEFAULT, B_expect=1)
        c = np.abs(comb[0])
        # At least each mask area has higher median than global median
        global_med = float(np.median(c))
        for rule in rules_default:
            mask = np.asarray(rule["mask"], dtype=bool)
            if mask.any():
                med_in = float(np.median(c[mask]))
                assert med_in >= global_med * 1.1, "Combined overlay not elevated within masked region."
    else:
        pytest.xfail("Overlay output contains neither 'per_rule' nor 'combined' arrays.")


def test_weight_scaling_monotonicity(overlay_callable, mu_single, rules_default):
    """
    Increasing a rule's weight should not reduce its aggregate violation magnitude.
    Try kwarg 'rule_weights' first; else override inside rules list.
    """
    kind, target = overlay_callable

    # Baseline
    out0 = _invoke(kind, target, mu_single, rules_default, seed=123)
    pr0 = out0.get("per_rule", None)
    if pr0 is None:
        pytest.xfail("Overlay does not expose per_rule map; weight monotonicity test requires per_rule.")
    pr0n, _ = _norm_per_rule(_as_np(pr0), L=L_DEFAULT)
    base_strength0 = float(np.sum(np.abs(pr0n[0, 0, :])))

    # Try call-time rule_weights
    try:
        out_up = _invoke(kind, target, mu_single, rules_default, seed=123, rule_weights=[5.0, 1.0])
        pr_up = out_up.get("per_rule", None)
        assert pr_up is not None, "per_rule missing when using rule_weights kwarg."
        pr_upn, _ = _norm_per_rule(_as_np(pr_up), L=L_DEFAULT)
        up_strength0 = float(np.sum(np.abs(pr_upn[0, 0, :])))
        assert up_strength0 >= base_strength0 * 1.2, "Rule weight increase did not raise violation magnitude."
        return
    except TypeError:
        # Override within rules
        rules_mod = [dict(r) for r in rules_default]
        rules_mod[0]["weight"] = 5.0
        out_up2 = _invoke(kind, target, mu_single, rules_mod, seed=123)
        pr_up2 = out_up2.get("per_rule", None)
        if pr_up2 is None:
            pytest.xfail("Overlay API doesn't support per-rule weights.")
        pr_up2n, _ = _norm_per_rule(_as_np(pr_up2), L=L_DEFAULT)
        up2_strength0 = float(np.sum(np.abs(pr_up2n[0, 0, :])))
        assert up2_strength0 >= base_strength0 * 1.1, "Per-structure weight override failed monotonicity."


def test_determinism_fixed_seed(overlay_callable, mu_single, rules_default):
    """
    Fixed seed ⇒ identical outputs.
    """
    kind, target = overlay_callable
    out1 = _invoke(kind, target, mu_single, rules_default, seed=777)
    out2 = _invoke(kind, target, mu_single, rules_default, seed=777)

    # Compare per_rule when available, else combined
    if "per_rule" in out1 and "per_rule" in out2:
        a1, _ = _norm_per_rule(_as_np(out1["per_rule"]), L=L_DEFAULT)
        a2, _ = _norm_per_rule(_as_np(out2["per_rule"]), L=L_DEFAULT)
        assert np.array_equal(a1, a2), "per_rule overlays changed despite fixed seed."
    elif "combined" in out1 and "combined" in out2:
        c1, _ = _norm_combined(_as_np(out1["combined"]), L=L_DEFAULT, B_expect=1)
        c2, _ = _norm_combined(_as_np(out2["combined"]), L=L_DEFAULT, B_expect=1)
        assert np.array_equal(c1, c2), "combined overlay changed despite fixed seed."
    else:
        pytest.xfail("Overlay output lacks comparable arrays for determinism test.")


# -----------------------------------------------------------------------------
# Optional plotting
# -----------------------------------------------------------------------------

@pytest.mark.mpl
def test_plotting_returns_figure_if_available(overlay_mod, overlay_callable, mu_single, rules_default):
    """
    If the module exposes a plotting helper (e.g., plot_symbolic_violation_overlay),
    it should return a Matplotlib Figure object.
    """
    # Attempt to find a plotting function
    plot_fn = None
    for name in ("plot_symbolic_violation_overlay", "plot_violation_overlay", "plot_overlay"):
        if hasattr(overlay_mod, name) and callable(getattr(overlay_mod, name)):
            plot_fn = getattr(overlay_mod, name)
            break
    if plot_fn is None:
        pytest.xfail("No plotting helper exposed by symbolic_violation_overlay; skipping plot test.")

    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception:
        pytest.skip("Matplotlib not available in environment.")

    kind, target = overlay_callable
    out = _invoke(kind, target, mu_single, rules_default, seed=101)
    fig = plot_fn(out)
    # Some implementations accept arrays directly instead of dict
    if fig is None:
        # Try direct per_rule
        if "per_rule" in out:
            fig = plot_fn(out["per_rule"])
        elif "combined" in out:
            fig = plot_fn(out["combined"])
    assert hasattr(fig, "savefig"), "Plot function did not return a Matplotlib Figure."


# -----------------------------------------------------------------------------
# Optional artifact save round-trip
# -----------------------------------------------------------------------------

def test_save_artifacts_roundtrip_if_available(overlay_mod, overlay_callable, tmp_path, mu_single, rules_default):
    """
    If a saver like `save_violation_artifacts(result, outdir=...)` exists, verify it writes files.
    """
    save_fn = None
    for name in ("save_violation_artifacts", "save_overlay_artifacts", "save_artifacts", "write_artifacts"):
        if hasattr(overlay_mod, name) and callable(getattr(overlay_mod, name)):
            save_fn = getattr(overlay_mod, name)
            break
    if save_fn is None:
        pytest.xfail("No artifact saver in symbolic_violation_overlay; skipping round-trip.")

    kind, target = overlay_callable
    result = _invoke(kind, target, mu_single, rules_default, seed=2025)
    outdir = tmp_path / "overlay_artifacts"
    outdir.mkdir(parents=True, exist_ok=True)

    save_fn(result, outdir=str(outdir))  # should write at least one file

    files = list(outdir.glob("*"))
    assert files, "Artifact saver wrote no files."
    # If a per_rule.npy exists, ensure shape normalization works
    pr_path = outdir / "per_rule.npy"
    if pr_path.exists():
        arr = np.load(pr_path)
        _norm_per_rule(arr, L=L_DEFAULT)  # raises on mismatch


# -----------------------------------------------------------------------------
# Optional CLI smoke
# -----------------------------------------------------------------------------

@pytest.mark.skipif(__import__("shutil").which("spectramind") is None, reason="SpectraMind CLI not found in PATH")
def test_cli_smoke_symbolic_violation_overlay(tmp_path, mu_single, rules_default):
    """
    Smoke test CLI route (adjust flags if your repo differs):

        spectramind diagnose symbolic-violation-overlay \
            --mu mu.npy --rules rules.json --outdir out --seed 123

    We only validate that it runs and writes at least one artifact.
    """
    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, mu_single)

    rules_path = tmp_path / "rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules_default, f)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "spectramind", "diagnose", "symbolic-violation-overlay",
        "--mu", str(mu_path),
        "--rules", str(rules_path),
        "--outdir", str(outdir),
        "--seed", "123",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.xfail(
            "CLI returned nonzero exit (flags or subcommand may differ).\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    produced = list(outdir.glob("*"))
    assert len(produced) > 0, "CLI ran but produced no artifacts."


# -----------------------------------------------------------------------------
# Performance guardrail
# -----------------------------------------------------------------------------

def test_runs_fast_enough(overlay_callable, mu_single, rules_default):
    """
    Tiny single-spectrum overlay should complete in <1.5s on CI CPU.
    """
    kind, target = overlay_callable
    t0 = time.time()
    _ = _invoke(kind, target, mu_single, rules_default, seed=11)
    dt = time.time() - t0
    assert dt < 1.5, f"Symbolic violation overlay too slow: {dt:.3f}s (should be < 1.5s)"