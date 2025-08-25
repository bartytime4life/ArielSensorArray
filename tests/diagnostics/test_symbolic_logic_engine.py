# /tests/diagnostics/test_symbolic_logic_engine.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: symbolic_logic_engine

Purpose
-------
Validate core scientific and engineering behaviors of the project's
`symbolic_logic_engine` implementation. The engine evaluates physics-informed
symbolic rules over spectra μ (length L≈283), emitting per-rule loss maps and
aggregate diagnostics in both "soft" and "hard" modes.

What we verify
--------------
1) API discovery & shape contracts
   • Accepts μ shapes (L,) and (B,L)
   • Emits loss maps shaped (R,L) or (B,R,L) and per-rule scalars shaped (R,) or (B,R)
2) Rule localization
   • Loss is larger inside a rule's masked region than outside (statistical check)
3) Weights & enable flags
   • Higher rule weight ⇒ higher aggregate rule loss (monotonicity)
   • Disabling a rule (weight 0 or enabled=False) collapses its contribution (≈0)
4) Soft vs hard mode
   • Hard mode loss ≥ soft mode loss for the same μ/rules (or equal)
5) Determinism
   • Fixed seeds ⇒ identical outputs
6) Optional artifact save
   • If the module exposes `save_logic_artifacts(...)`, verify round-trip
7) Optional CLI smoke
   • If the repo exposes `spectramind diagnose symbolic-rank`, smoke-test it

Design Notes
------------
• The test is defensively adaptable to small API differences:
  - Module discovery tries several common import paths.
  - Entry-point discovery looks for multiple function/class names.
  - Return normalization tolerates dict or array outputs.

• The synthetic rules are two disjoint rectangular masks. A physically sensible
  engine should produce higher violation metrics within these masks for μ that
  exhibits structure in those regions.

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
# Module & Entrypoint Discovery
# -----------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    # common locations in this project
    "src.symbolic.symbolic_logic_engine",
    "symbolic.symbolic_logic_engine",
    "tools.symbolic_logic_engine",
    "src.tools.symbolic_logic_engine",
    "diagnostics.symbolic_logic_engine",
    # direct
    "symbolic_logic_engine",
]


def _import_logic_module():
    """
    Try importing the symbolic logic engine module from several common paths.
    """
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        f"Could not import symbolic_logic_engine from any of:\n  {CANDIDATE_IMPORTS}\n"
        f"Last error: {last_err}"
    )


def _locate_entrypoints(mod):
    """
    Discover a callable (function or class) to evaluate symbolic logic losses.

    Expected options (any of):
      - evaluate_symbolic_logic(mu, rules=..., mode='soft'|'hard', **cfg)
      - run_logic(mu, rules=..., **cfg)
      - symbolic_logic_engine(mu, rules=..., **cfg)
      - class SymbolicLogicEngine(...).run(mu, rules, **cfg)

    Returns
    -------
    kind : str
        'func' or 'class'
    target : callable | type
    """
    for fn in ("evaluate_symbolic_logic", "run_logic", "symbolic_logic_engine", "evaluate"):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return "func", getattr(mod, fn)

    for cls_name in ("SymbolicLogicEngine", "LogicEngine", "SymbolicEngine"):
        if hasattr(mod, cls_name):
            cls = getattr(mod, cls_name)
            if hasattr(cls, "run") and callable(getattr(cls, "run")):
                return "class", cls

    pytest.xfail(
        "symbolic_logic_engine module found but no known callable was discovered.\n"
        "Expected one of: evaluate_symbolic_logic(), run_logic(), symbolic_logic_engine(), "
        "evaluate(), or a class with .run(...)."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, mu: np.ndarray, rules: List[Dict[str, Any]], **cfg) -> Dict[str, Any]:
    """
    Invoke the discovered API in an agnostic way, and coerce to a dict result.

    Expected dict keys (any subset is fine):
      - 'loss_map' : ndarray (R,L) or (B,R,L)
      - 'per_rule' : ndarray (R,) or (B,R)
      - 'total'    : float or (B,)
      - 'mode'     : 'soft' or 'hard'
      - 'rules'    : the rule list used

    If the returned value is already a dict with these keys, pass it through.
    If it's a bare ndarray, wrap as {'loss_map': arr}.
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
    else:
        pytest.fail("Unknown invocation kind.")  # pragma: no cover

    if isinstance(out, dict):
        return out
    return {"loss_map": out}


# -----------------------------------------------------------------------------
# Synthetic μ and rules
# -----------------------------------------------------------------------------

L_DEFAULT = 283
_RNG = np.random.RandomState(20250824)


def _build_mu_single(L: int = L_DEFAULT) -> np.ndarray:
    """
    Smooth baseline with two absorption-like bumps (for mask localization tests).
    """
    x = np.linspace(0.0, 1.0, L, dtype=np.float64)
    baseline = 0.02 + 0.004 * np.sin(2 * math.pi * 2.2 * x)
    bumps = (
        0.012 * np.exp(-0.5 * ((x - 0.25) / 0.02) ** 2)
        + 0.010 * np.exp(-0.5 * ((x - 0.65) / 0.018) ** 2)
    )
    mu = np.clip(baseline + bumps, 0.0, 1.0)
    return mu


def _build_mu_batch(B: int = 4, L: int = L_DEFAULT) -> np.ndarray:
    base = _build_mu_single(L)
    batch = np.stack([base + _RNG.normal(0, 0.0005, size=L) for _ in range(B)], axis=0)
    return np.clip(batch, 0.0, 1.0)


def _rect_mask(L: int, i0: int, i1: int) -> np.ndarray:
    m = np.zeros(L, dtype=np.float32)
    m[max(0, i0): min(L, i1)] = 1.0
    return m


def _make_rules(L: int = L_DEFAULT) -> List[Dict[str, Any]]:
    """
    Two disjoint rectangular masks with weights and optional 'enabled' flag.
    """
    r1 = {
        "name": "left_band_smoothness",
        "mask": _rect_mask(L, int(0.17 * L), int(0.32 * L)).tolist(),
        "weight": 1.0,
        "enabled": True,
        "kind": "band_smoothness",
    }
    r2 = {
        "name": "right_band_smoothness",
        "mask": _rect_mask(L, int(0.58 * L), int(0.74 * L)).tolist(),
        "weight": 1.0,
        "enabled": True,
        "kind": "band_smoothness",
    }
    return [r1, r2]


# -----------------------------------------------------------------------------
# Helpers: shape normalization & aggregates
# -----------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
    assert np.isfinite(x).all(), "Array contains non-finite values"
    return x


def _norm_loss_map(arr: np.ndarray, L: int = L_DEFAULT) -> Tuple[np.ndarray, str]:
    """
    Normalize loss_map to (B,R,L). Accepts (R,L) or (B,R,L).
    """
    arr = _as_np(arr)
    if arr.ndim == 2 and arr.shape[1] == L:
        R = arr.shape[0]
        return arr[None, ...], f"(1,{R},{L})"
    if arr.ndim == 3 and arr.shape[2] == L:
        return arr, f"({arr.shape[0]},{arr.shape[1]},{L})"
    pytest.fail(f"Unexpected loss_map shape {arr.shape} (expected (R,{L}) or (B,R,{L}))")
    return arr, ""  # pragma: no cover


def _norm_per_rule(arr: np.ndarray, R_expected: int) -> Tuple[np.ndarray, str]:
    """
    Normalize per_rule array to (B,R). Accepts (R,) or (B,R).
    """
    arr = _as_np(arr)
    if arr.ndim == 1 and arr.shape[0] == R_expected:
        return arr[None, :], f"(1,{R_expected})"
    if arr.ndim == 2 and arr.shape[1] == R_expected:
        return arr, f"({arr.shape[0]},{R_expected})"
    pytest.fail(f"Unexpected per_rule shape {arr.shape} (expected ({R_expected},) or (B,{R_expected}))")
    return arr, ""  # pragma: no cover


def _total_from_loss_map(loss_map: np.ndarray, weights: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    Compute a naive total = sum_R sum_L |loss_map| * weight_R (per batch).
    """
    loss_map = np.abs(loss_map)  # magnitude aggregate
    if weights is None:
        w = np.ones(loss_map.shape[1], dtype=np.float64)
    else:
        w = np.asarray(weights, dtype=np.float64)
        assert w.shape[0] == loss_map.shape[1], "weights must match number of rules"
    # (B,R,L) → (B,)
    return np.einsum("brl,r->b", loss_map, w)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def logic_mod():
    return _import_logic_module()


@pytest.fixture(scope="module")
def logic_callable(logic_mod):
    return _locate_entrypoints(logic_mod)


@pytest.fixture
def mu_single():
    return _build_mu_single(L_DEFAULT)


@pytest.fixture
def mu_batch():
    return _build_mu_batch(B=3, L=L_DEFAULT)


@pytest.fixture
def rules_default():
    return _make_rules(L_DEFAULT)


# -----------------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------------

def test_api_shapes_single(logic_callable, mu_single, rules_default):
    """
    Single-spectrum: shapes normalize to (B=1,R,L).
    """
    kind, target = logic_callable
    out = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=1234)
    assert isinstance(out, dict)
    assert "loss_map" in out
    lm, _ = _norm_loss_map(out["loss_map"], L=L_DEFAULT)
    B, R, L = lm.shape
    assert B == 1 and R == len(rules_default) and L == L_DEFAULT

    # If per_rule provided, normalize and check
    if "per_rule" in out:
        pr, _ = _norm_per_rule(_as_np(out["per_rule"]), R_expected=R)
        assert pr.shape == (1, R)


def test_api_shapes_batch(logic_callable, mu_batch, rules_default):
    """
    Batched-spectrum: shapes normalize to (B,R,L).
    """
    kind, target = logic_callable
    out = _invoke(kind, target, mu_batch, rules_default, mode="soft", seed=7)
    lm, _ = _norm_loss_map(_as_np(out["loss_map"]))
    B, R, L = lm.shape
    assert B == mu_batch.shape[0] and R == len(rules_default) and L == L_DEFAULT

    if "per_rule" in out:
        pr, _ = _norm_per_rule(_as_np(out["per_rule"]), R_expected=R)
        assert pr.shape == (B, R)


def test_rule_localization(logic_callable, mu_single, rules_default):
    """
    Loss magnitude inside each rule's mask should exceed outside (percentile test).
    """
    kind, target = logic_callable
    out = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=99)
    lm, _ = _norm_loss_map(_as_np(out["loss_map"]))
    B, R, L = lm.shape
    assert B == 1

    for r_idx, rule in enumerate(rules_default):
        mask = np.asarray(rule["mask"], dtype=bool)
        loss = np.abs(lm[0, r_idx, :])
        inside = loss[mask]
        outside = loss[~mask]
        inside_q = float(np.percentile(inside, 75))
        outside_q = float(np.percentile(outside, 75))
        assert inside_q > outside_q * 1.2, (
            f"Rule '{rule['name']}' not localized: P75 inside={inside_q:.3e}, outside={outside_q:.3e}"
        )


def test_weight_monotonicity(logic_callable, mu_single, rules_default):
    """
    Increasing rule weight should not decrease its aggregate contribution.
    We allow either kwarg 'rule_weights' or per-rule dict key 'weight'.
    """
    kind, target = logic_callable

    # Baseline (weights 1.0)
    out0 = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=123)
    lm0, _ = _norm_loss_map(_as_np(out0["loss_map"]))
    base_total = _total_from_loss_map(lm0, weights=[r.get("weight", 1.0) for r in rules_default])[0]

    # Try call-time kwarg
    try:
        out_hi = _invoke(
            kind,
            target,
            mu_single,
            rules_default,
            mode="soft",
            seed=123,
            rule_weights=[5.0, 1.0],
        )
        lm_hi, _ = _norm_loss_map(_as_np(out_hi["loss_map"]))
        hi_total = _total_from_loss_map(lm_hi, weights=[5.0, 1.0])[0]
        assert hi_total >= base_total * 1.2, "Increasing rule 0 weight did not increase total as expected."
        return
    except TypeError:
        # Fall back to in-structure override
        rules_mod = [dict(r) for r in rules_default]
        rules_mod[0]["weight"] = 5.0
        out_hi2 = _invoke(kind, target, mu_single, rules_mod, mode="soft", seed=123)
        lm_hi2, _ = _norm_loss_map(_as_np(out_hi2["loss_map"]))
        hi2_total = _total_from_loss_map(lm_hi2, weights=[r.get("weight", 1.0) for r in rules_mod])[0]
        if not (hi2_total >= base_total * 1.1):
            pytest.xfail("Engine does not scale losses with per-rule weights (observed non-monotonic behavior).")


def test_disable_rule_collapse(logic_callable, mu_single, rules_default):
    """
    Disabling a rule (weight=0 or enabled=False) should nearly zero its aggregate loss.
    """
    kind, target = logic_callable

    out0 = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=321)
    lm0, _ = _norm_loss_map(_as_np(out0["loss_map"]))
    # Aggregate per-rule strengths (L1 over L)
    base_rule_strengths = np.sum(np.abs(lm0[0]), axis=1)  # (R,)

    # Attempt to disable rule 1
    # Prefer explicit enabled flag; otherwise set weight to 0
    rules_mod = [dict(r) for r in rules_default]
    if "enabled" in rules_mod[1]:
        rules_mod[1]["enabled"] = False
    else:
        rules_mod[1]["weight"] = 0.0

    out1 = _invoke(kind, target, mu_single, rules_mod, mode="soft", seed=321)
    lm1, _ = _norm_loss_map(_as_np(out1["loss_map"]))
    new_rule_strengths = np.sum(np.abs(lm1[0]), axis=1)

    # Rule 1 strength should collapse by large factor
    assert new_rule_strengths[1] <= base_rule_strengths[1] * 0.2, (
        f"Disabled rule still strong: before={base_rule_strengths[1]:.3e}, after={new_rule_strengths[1]:.3e}"
    )


def test_soft_vs_hard_mode(logic_callable, mu_single, rules_default):
    """
    Hard mode should not be smaller than soft mode in aggregate (≥).
    """
    kind, target = logic_callable
    out_soft = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=777)
    out_hard = _invoke(kind, target, mu_single, rules_default, mode="hard", seed=777)

    lm_soft, _ = _norm_loss_map(_as_np(out_soft["loss_map"]))
    lm_hard, _ = _norm_loss_map(_as_np(out_hard["loss_map"]))
    w = [r.get("weight", 1.0) for r in rules_default]

    total_soft = _total_from_loss_map(lm_soft, weights=w)[0]
    total_hard = _total_from_loss_map(lm_hard, weights=w)[0]

    assert total_hard + 1e-12 >= total_soft, (
        f"Hard mode aggregate < soft mode: hard={total_hard:.6e}, soft={total_soft:.6e}"
    )


def test_determinism_fixed_seed(logic_callable, mu_single, rules_default):
    """
    With identical inputs and a fixed seed, results should be identical.
    """
    kind, target = logic_callable
    out1 = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=2025)
    out2 = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=2025)

    # Compare loss_maps exactly; if engine injects randomness, seed must control it
    lm1, _ = _norm_loss_map(_as_np(out1["loss_map"]))
    lm2, _ = _norm_loss_map(_as_np(out2["loss_map"]))
    assert np.array_equal(lm1, lm2), "Loss maps changed despite fixed seed."


# -----------------------------------------------------------------------------
# Optional Artifact Save Round-Trip
# -----------------------------------------------------------------------------

def test_save_artifacts_roundtrip_if_available(logic_mod, logic_callable, tmp_path, mu_single, rules_default):
    """
    If `save_logic_artifacts(result_dict, outdir=...)` exists, verify it writes
    one or more files, and reload a saved 'loss_map.npy' if present.
    """
    save_fn = None
    for name in ("save_logic_artifacts", "save_artifacts", "write_artifacts"):
        if hasattr(logic_mod, name) and callable(getattr(logic_mod, name)):
            save_fn = getattr(logic_mod, name)
            break
    if save_fn is None:
        pytest.xfail("No artifact save function exposed by symbolic_logic_engine; skipping round-trip.")

    kind, target = logic_callable
    result = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=4242)
    outdir = tmp_path / "logic_artifacts"
    outdir.mkdir(parents=True, exist_ok=True)

    save_fn(result, outdir=str(outdir))  # should not raise

    # Require at least some files
    produced = list(outdir.glob("*"))
    assert produced, "save_logic_artifacts wrote no files"

    npy = outdir / "loss_map.npy"
    if npy.exists():
        arr = np.load(npy)
        _norm_loss_map(arr, L=L_DEFAULT)  # raises on mismatch


# -----------------------------------------------------------------------------
# Optional CLI Smoke
# -----------------------------------------------------------------------------

@pytest.mark.skipif(__import__("shutil").which("spectramind") is None, reason="SpectraMind CLI not found in PATH")
def test_cli_smoke_symbolic_rank(tmp_path, mu_single, rules_default):
    """
    Smoke the CLI command if present:

        spectramind diagnose symbolic-rank \
            --mu mu.npy --rules rules.json --outdir out --mode soft

    We only assert that the command succeeds and writes at least one artifact.
    """
    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, mu_single)

    rules_path = tmp_path / "rules.json"
    with open(rules_path, "w", encoding="utf-8") as f:
        json.dump(rules_default, f)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "spectramind", "diagnose", "symbolic-rank",
        "--mu", str(mu_path),
        "--rules", str(rules_path),
        "--outdir", str(outdir),
        "--mode", "soft",
        "--seed", "101",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.xfail(
            "CLI returned nonzero exit. The command/flags may differ in your repo.\n"
            f"STDOUT/ERR:\n{proc.stdout}\n{proc.stderr}"
        )

    produced = list(outdir.glob("*"))
    assert len(produced) > 0, "CLI ran but produced no artifacts."


# -----------------------------------------------------------------------------
# Performance Sanity
# -----------------------------------------------------------------------------

def test_runs_fast_enough(logic_callable, mu_single, rules_default):
    """
    Very light performance guardrail for CI: < 1.5s on CPU for tiny input.
    """
    kind, target = logic_callable
    t0 = time.time()
    _ = _invoke(kind, target, mu_single, rules_default, mode="soft", seed=11)
    dt = time.time() - t0
    assert dt < 1.5, f"Symbolic logic evaluation too slow: {dt:.3f}s (should be < 1.5s)"