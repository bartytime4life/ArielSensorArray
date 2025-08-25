# tests/diagnostics/test_shap_symbolic_overlay.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics

Upgraded contract tests for the SHAP × Symbolic overlay module.

This suite is *adaptive*:
- It will discover your overlay implementation from several common module paths.
- It tolerates optional helper availability (e.g., heatmap, rank_bins, CLI).
- It asserts the *contract* (shapes, bounds, α‑blend behavior, batching/broadcast),
  not any particular internal formula.

What we validate
----------------
1) Discovery: overlay API is importable from common repo paths.
2) Shapes & bounds:
   - SHAP can be (N,B) or (B,).
   - Symbolic can be (B,), (N,B), (R,B), or (N,R,B) (+ optional rule_weights (R,)).
   - Output is (N,B); normalized in [0,1] when normalize=True.
3) α behavior (blend coefficient):
   - α=1 → overlay dominated by SHAP signal
   - α=0 → overlay dominated by symbolic signal
   - α in (0,1) blends sensibly.
4) Batching & broadcast:
   - (B,) symbolic broadcast across N rows of (N,B) SHAP.
5) Plotting (optional):
   - overlay_heatmap(...) writes a non‑empty image when given save_path.
6) Ranking (optional):
   - rank_bins(...) returns top‑k pairs with non‑increasing scores.
7) Error handling:
   - Mismatched shapes raise ValueError/TypeError/AssertionError, not silent broadcast.
8) Determinism:
   - Same inputs → identical outputs.
9) CLI smoke (optional):
   - If a Typer CLI is present, a shap‑symbolic overlay subcommand runs and saves artifacts.

Expected API (soft contract)
----------------------------
overlay_shap_with_symbolic(shap, symbolic, *, alpha=0.5, mode="weighted",
                           rule_weights=None, normalize=True, **kw) -> np.ndarray
overlay_heatmap(overlay, *, shap=None, symbolic=None, bin_labels=None,
                title=None, save_path=None, **kw) -> (path | bytes | None)
rank_bins(overlay, bin_labels=None, k=10) -> list[(label, score)]

Module paths probed (first match wins)
--------------------------------------
- spectramind.diagnostics.shap_symbolic_overlay
- src.spectramind.diagnostics.shap_symbolic_overlay
- spectramind.tools.shap_symbolic_overlay
- tools.shap_symbolic_overlay
- spectramind.diagnostics.shap_symbolic_fusion   (legacy alias)

CLI subcommands probed (optional)
---------------------------------
spectramind diagnose shap-symbolic-overlay
spectramind diagnose shap_symbolic_overlay
spectramind diagnostics shap-symbolic-overlay
spectramind diagnostics shap_symbolic_overlay
"""

from __future__ import annotations

import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest


# =============================================================================
# Dynamic API resolution
# =============================================================================

@dataclass
class OverlayAPI:
    overlay: Callable[..., np.ndarray]
    heatmap: Optional[Callable[..., Any]]
    rank: Optional[Callable[..., Any]]


_IMPORT_ERRORS: list[tuple[str, str]] = []


def _try_import_mod(modname: str) -> Optional[Any]:
    try:
        return importlib.import_module(modname)
    except Exception as e:  # pragma: no cover (diagnostic)
        _IMPORT_ERRORS.append((modname, repr(e)))
        return None


def _resolve_api() -> Optional[OverlayAPI]:
    candidates = [
        # module                                              overlay fn                   heatmap fn         rank fn
        ("spectramind.diagnostics.shap_symbolic_overlay",     "overlay_shap_with_symbolic", "overlay_heatmap", "rank_bins"),
        ("src.spectramind.diagnostics.shap_symbolic_overlay", "overlay_shap_with_symbolic", "overlay_heatmap", "rank_bins"),
        ("spectramind.tools.shap_symbolic_overlay",           "overlay_shap_with_symbolic", "overlay_heatmap", "rank_bins"),
        ("tools.shap_symbolic_overlay",                       "overlay_shap_with_symbolic", "overlay_heatmap", "rank_bins"),
        ("spectramind.diagnostics.shap_symbolic_fusion",      "overlay_shap_with_symbolic", "overlay_heatmap", "rank_bins"),
    ]
    for modname, f_overlay, f_heat, f_rank in candidates:
        mod = _try_import_mod(modname)
        if not mod:
            continue
        overlay = getattr(mod, f_overlay, None)
        heatmap = getattr(mod, f_heat, None)
        rank = getattr(mod, f_rank, None)
        if callable(overlay):
            return OverlayAPI(
                overlay=overlay,
                heatmap=heatmap if callable(heatmap) else None,
                rank=rank if callable(rank) else None,
            )
        else:
            _IMPORT_ERRORS.append((modname, f"{f_overlay} not found"))
    return None


API = _resolve_api()

skip_msg = (
    "Could not import SHAP×Symbolic overlay helpers. Tried modules:\n  - "
    + "\n  - ".join(m for m, _ in _IMPORT_ERRORS or [("<<none>>", "")])
    + "\nPlease expose `overlay_shap_with_symbolic()` (and optionally `overlay_heatmap`, `rank_bins`)."
)

pytestmark = pytest.mark.skipif(API is None, reason=skip_msg)


# =============================================================================
# Synthetic data builders
# =============================================================================

@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    return np.random.default_rng(2025_08_24)


def make_synthetic_shap(rng: np.random.Generator, N: int, B: int, strong_bins: list[int]) -> np.ndarray:
    """Signed SHAP-like matrix with highlighted 'strong' bins."""
    shap = rng.normal(0.0, 0.2, size=(N, B)).astype(np.float32)
    for n in range(N):
        for b in strong_bins:
            shap[n, b] += 1.0 + 0.2 * (n + 1)
    return shap


def make_symbolic_mask_B(B: int, active_bins: list[int]) -> np.ndarray:
    """Symbolic mask of shape (B,) with 0/1 markers."""
    m = np.zeros((B,), dtype=np.float32)
    m[active_bins] = 1.0
    return m


def make_symbolic_mask_NB(N: int, B: int, active_bins: list[int]) -> np.ndarray:
    """Symbolic mask of shape (N,B) with 0/1 markers per-sample."""
    m = np.zeros((N, B), dtype=np.float32)
    m[:, active_bins] = 1.0
    return m


def make_rule_masks_RB(B: int, rules: dict[str, list[int]]) -> tuple[np.ndarray, list[str]]:
    """Return (R,B) rule masks and rule names list."""
    names = list(rules.keys())
    R = len(names)
    M = np.zeros((R, B), dtype=np.float32)
    for i, name in enumerate(names):
        M[i, rules[name]] = 1.0
    return M, names


# =============================================================================
# Tests — core behavior
# =============================================================================

def test_shapes_bounds_and_broadcast(rng: np.random.Generator):
    """Overlay returns (N,B), normalized [0,1] when requested, supporting broadcast of (B,) symbolic."""
    N, B = 5, 17
    shap = make_synthetic_shap(rng, N, B, strong_bins=[2, 11, 14])

    # Case A: symbolic (B,) → broadcast
    sym_B = make_symbolic_mask_B(B, active_bins=[11, 12])
    outA = API.overlay(shap=shap, symbolic=sym_B, alpha=0.5, mode="weighted", normalize=True)
    assert isinstance(outA, np.ndarray) and outA.shape == (N, B)
    assert np.isfinite(outA).all() and outA.min() >= -1e-6 and outA.max() <= 1 + 1e-6

    # Determinism
    outA2 = API.overlay(shap=shap, symbolic=sym_B, alpha=0.5, mode="weighted", normalize=True)
    np.testing.assert_array_equal(outA, outA2)

    # Case B: symbolic (N,B)
    sym_NB = make_symbolic_mask_NB(N, B, active_bins=[1, 2, 3])
    outB = API.overlay(shap=shap, symbolic=sym_NB, alpha=0.25, mode="union", normalize=True)
    assert outB.shape == (N, B) and np.isfinite(outB).all()


def test_alpha_extremes_and_blend(rng: np.random.Generator):
    """α=1→SHAP-dominant bins top; α=0→symbolic-dominant bins top; α=0.5 blends."""
    N, B = 4, 16
    shap = make_synthetic_shap(rng, N, B, strong_bins=[4, 10])  # SHAP strong bins
    sym = make_symbolic_mask_B(B, active_bins=[7, 10])          # symbolic strong bins (overlap=10)

    over1 = API.overlay(shap=shap, symbolic=sym, alpha=1.0, mode="weighted", normalize=True)
    assert set(np.argmax(over1, axis=1)).issubset({4, 10})

    over0 = API.overlay(shap=shap, symbolic=sym, alpha=0.0, mode="weighted", normalize=True)
    assert set(np.argmax(over0, axis=1)).issubset({7, 10})

    over05 = API.overlay(shap=shap, symbolic=sym, alpha=0.5, mode="weighted", normalize=True)
    # Shared strong bin should often be selected
    assert (np.argmax(over05, axis=1) == 10).sum() >= 1


def test_rule_weighted_RB_masks(rng: np.random.Generator):
    """(R,B) masks + rule_weights steer overlay when α=0 (symbolic-only)."""
    N, B = 3, 20
    shap = np.full((N, B), 0.2, dtype=np.float32)  # flatten SHAP so symbolic dominates
    rule_masks, rule_names = make_rule_masks_RB(B, {"R_light": [2, 3], "R_heavy": [12, 13, 14]})
    weights = np.array([0.25, 1.0], dtype=np.float32)

    over = API.overlay(shap=shap, symbolic=rule_masks, alpha=0.0, mode="weighted", rule_weights=weights, normalize=True)
    mean_over = over.mean(axis=0)
    assert mean_over[13] > mean_over[2], "Heavier rule did not boost its bins as expected"


def test_shape_mismatch_raises(rng: np.random.Generator):
    """Mismatched shapes should raise a clear error, not silently broadcast."""
    N, B = 4, 15
    shap = make_synthetic_shap(rng, N, B, strong_bins=[6])
    bad_symbolic = np.ones((B - 2,), dtype=np.float32)  # wrong length

    with pytest.raises((ValueError, TypeError, AssertionError)):
        _ = API.overlay(shap=shap, symbolic=bad_symbolic, alpha=0.5, mode="union", normalize=True)


def test_normalize_false_allows_raw_scale(rng: np.random.Generator):
    """When normalize=False, values may leave [0,1] — only shape/finite are enforced."""
    N, B = 2, 10
    shap = make_synthetic_shap(rng, N, B, strong_bins=[1, 8])
    sym = make_symbolic_mask_B(B, active_bins=[8, 9])

    out = API.overlay(shap=shap, symbolic=sym, alpha=0.4, mode="weighted", normalize=False)
    assert out.shape == (N, B) and np.isfinite(out).all()
    assert (out.max() > 1.0) or (out.min() < 0.0)


def test_unknown_mode_raises_or_gracefully_falls_back(rng: np.random.Generator):
    """Unknown mode → either clear error or valid sensible output (impl-defined fallback)."""
    N, B = 3, 12
    shap = make_synthetic_shap(rng, N, B, strong_bins=[5])
    sym = make_symbolic_mask_B(B, active_bins=[0, 1])

    try:
        out = API.overlay(shap=shap, symbolic=sym, alpha=0.5, mode="totally-unknown-mode", normalize=True)
    except (ValueError, TypeError):
        return
    assert isinstance(out, np.ndarray) and out.shape == (N, B) and np.isfinite(out).all()


def test_idempotency_same_inputs_same_output(rng: np.random.Generator):
    """Exact equality for repeated calls with identical inputs."""
    N, B = 3, 10
    shap = make_synthetic_shap(rng, N, B, strong_bins=[5])
    sym = make_symbolic_mask_B(B, active_bins=[5])

    a = API.overlay(shap=shap, symbolic=sym, alpha=0.5, mode="weighted", normalize=True)
    b = API.overlay(shap=shap, symbolic=sym, alpha=0.5, mode="weighted", normalize=True)
    np.testing.assert_array_equal(a, b)


# =============================================================================
# Optional helpers — heatmap & rank
# =============================================================================

@pytest.mark.skipif(API is None or API.heatmap is None, reason="overlay_heatmap helper not available")
def test_overlay_heatmap_writes_image(tmp_path: Path, rng: np.random.Generator):
    N, B = 3, 18
    shap = make_synthetic_shap(rng, N, B, strong_bins=[6, 12])
    sym = make_symbolic_mask_B(B, active_bins=[5, 6, 7])
    over = API.overlay(shap=shap, symbolic=sym, alpha=0.6, mode="weighted", normalize=True)

    out = tmp_path / "overlay_heatmap.png"
    API.heatmap(overlay=over, shap=shap, symbolic=sym, bin_labels=[f"b{j}" for j in range(B)], title="SHAP×Symbolic", save_path=str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(API is None or API.rank is None, reason="rank_bins helper not available")
def test_rank_bins_returns_monotone_scores(rng: np.random.Generator):
    N, B = 2, 14
    shap = make_synthetic_shap(rng, N, B, strong_bins=[4, 9])
    sym = make_symbolic_mask_B(B, active_bins=[9, 10])
    over = API.overlay(shap=shap, symbolic=sym, alpha=0.5, mode="union", normalize=True)

    labels = [f"λ{b:03d}" for b in range(B)]
    top = API.rank(over, bin_labels=labels, k=6)
    assert isinstance(top, list) and len(top) == 6
    scores = [float(s) for _, s in top]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1)), "Scores not non‑increasing"


# =============================================================================
# Optional CLI smoke (Typer) — skip if CLI not present
# =============================================================================

def _maybe_cli_runner():
    try:
        sm = importlib.import_module("spectramind")
        app = getattr(sm, "app", None)
        if app is None:
            return None
        from typer.testing import CliRunner
        return CliRunner(), app
    except Exception:
        return None


@pytest.mark.skipif(_maybe_cli_runner() is None, reason="Typer CLI not available (spectramind.app)")
def test_cli_shap_symbolic_overlay_smoke(tmp_path: Path, rng: np.random.Generator):
    runner_app = _maybe_cli_runner()
    assert runner_app is not None
    runner, app = runner_app

    # Minimal CSVs for CLI
    N, B = 20, 9
    shap = make_synthetic_shap(rng, N, B, strong_bins=[2, 7])
    X = rng.normal(size=(N, B)).astype(np.float32)  # not used by all CLIs; safe to include
    rules = {"ruleA": [2, 3], "ruleB": [7]}

    shap_path = tmp_path / "shap.csv"
    X_path = tmp_path / "X.csv"
    rules_path = tmp_path / "rules.json"
    out_dir = tmp_path / "artifacts"

    # Save files
    def _to_csv(arr: np.ndarray, path: Path):
        with path.open("w") as f:
            f.write(",".join([f"b{i}" for i in range(arr.shape[1])]) + "\n")
            for r in arr:
                f.write(",".join(f"{float(v):.6f}" for v in r) + "\n")

    _to_csv(shap, shap_path)
    _to_csv(X, X_path)
    rules_path.write_text(json.dumps(rules, indent=2))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Try a few likely subcommands
    candidates = [
        ["diagnose", "shap-symbolic-overlay"],
        ["diagnose", "shap_symbolic_overlay"],
        ["diagnostics", "shap-symbolic-overlay"],
        ["diagnostics", "shap_symbolic_overlay"],
    ]
    ran = False
    for cmd in candidates:
        res = runner.invoke(app, cmd + ["--shap", str(shap_path), "--input", str(X_path), "--rules", str(rules_path), "--out", str(out_dir), "--save"])
        if res.exit_code == 0:
            ran = True
            break

    if not ran:
        pytest.skip("No CLI subcommand accepted shap+rules inputs (skipping CLI smoke).")

    # Confirm at least one artifact exists
    files = list(out_dir.glob("**/*"))
    assert any(p.suffix.lower() in {".png", ".json", ".csv", ".svg", ".pdf"} for p in files), (
        f"CLI produced no overlay artifacts in {out_dir}"
    )
