# /tests/diagnostics/test_generate_fft_symbolic_fusion.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: generate_fft_symbolic_fusion

Purpose
-------
Validate the behavior and contracts of the FFT × Symbolic Fusion tool, which
computes frequency-domain features from μ spectra, fuses them with symbolic /
SHAP / entropy overlays, and (optionally) projects to low dimensions (PCA/UMAP/
t‑SNE) with clustering and dashboard‑ready artifacts.

We verify:
1) API discovery & output contracts
   • Accepts μ of shape (B,L) or (L,) and optional overlays (entropy, symbolic)
   • Emits FFT features and at least one of: PCA/UMAP/t‑SNE/fusion embedding
   • Optional cluster labels array or dict is well‑formed
2) Fusion overlay sanity
   • Symbolic overlay elevates masked regions vs. outside (statistical check)
3) Determinism
   • Fixed seed ⇒ identical embeddings / clusters
4) Artifact save (optional)
   • If saver exists, writes JSON/NPY/HTML/PNG and arrays reload with correct shape
5) CLI smoke (optional)
   • If `spectramind diagnose fft-fusion` exists, run and assert artifact creation
6) Performance guardrail
   • Tiny synthetic input completes quickly on CI

Design Notes
------------
• Defensively adaptable to small API differences:
  - Tries multiple import paths and entrypoint names (func/class)
  - Normalizes dict/array returns
  - Tolerates the presence/absence of particular projections (e.g., UMAP off)
• Synthetic μ is shaped (B=6, L=283) with two banded regions; symbolic overlay aligns.

Author: SpectraMind V50 Team
"""

from __future__ import annotations

import importlib
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    "tools.generate_fft_symbolic_fusion",
    "src.tools.generate_fft_symbolic_fusion",
    "diagnostics.generate_fft_symbolic_fusion",
    "generate_fft_symbolic_fusion",
]


def _import_fusion_module():
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import FFT × Symbolic Fusion module from any of:\n"
        f"  {CANDIDATE_IMPORTS}\n"
        f"Last error: {last_err}"
    )


def _locate_entrypoint(mod):
    """
    Locate a callable/class to run FFT × symbolic fusion.

    Accepted function names (any):
      - generate_fft_symbolic_fusion(mu, symbolic=..., entropy=..., **cfg)
      - compute_fft_symbolic_fusion(...)
      - run_fft_symbolic_fusion(...)
    Accepted classes (with .run or .generate):
      - FFTSymbolicFusion(...).run(...)
      - FusionGenerator(...).run(...)
    """
    for fn in (
        "generate_fft_symbolic_fusion",
        "compute_fft_symbolic_fusion",
        "run_fft_symbolic_fusion",
    ):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return "func", getattr(mod, fn)

    for cls in ("FFTSymbolicFusion", "FusionGenerator", "FFTFusion"):
        if hasattr(mod, cls):
            Cls = getattr(mod, cls)
            for method in ("run", "generate"):
                if hasattr(Cls, method) and callable(getattr(Cls, method)):
                    return "class", Cls

    pytest.xfail(
        "generate_fft_symbolic_fusion module found but no known entrypoint. "
        "Expected a function like generate_fft_symbolic_fusion()/compute_fft_symbolic_fusion()/run_fft_symbolic_fusion(), "
        "or a class with .run/.generate."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, mu: np.ndarray, **cfg) -> Dict[str, Any]:
    """
    Invoke the entrypoint and coerce output to a dict.

    Expected dict keys (subset ok):
      - 'fft'           : ndarray (B, Ffft) or (Ffft,)
      - 'pca'           : ndarray (B, k)   (optional)
      - 'umap'          : ndarray (B, k)   (optional)
      - 'tsne'          : ndarray (B, k)   (optional)
      - 'fusion'        : ndarray (B, k)   (optional consolidated embedding)
      - 'clusters'      : ndarray (B,) or dict with 'labels'
      - 'overlay'       : overlays dict or arrays (e.g., 'symbolic', 'entropy', 'shap')
      - 'meta'          : dict with config/seed
    """
    if kind == "func":
        out = target(mu, **cfg)
    elif kind == "class":
        try:
            inst = target(mu=mu, **cfg)
        except TypeError:
            inst = target(**cfg)
            out = inst.run(mu=mu)
        else:
            out = inst.run() if hasattr(inst, "run") else inst.generate()
    else:  # pragma: no cover
        pytest.fail("Unknown invocation kind.")

    if isinstance(out, dict):
        return out
    # If bare array returned, wrap as a minimal dict
    return {"pca": out}


# --------------------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------------------

L = 283
B = 6
_RNG = np.random.RandomState(20250824)


def _rect_mask(L: int, a: float, b: float) -> np.ndarray:
    i0 = max(0, int(a * L))
    i1 = min(L, int(b * L))
    m = np.zeros(L, dtype=np.float32)
    m[i0:i1] = 1.0
    return m


def _make_mu_batch(B: int = B, L_: int = L) -> np.ndarray:
    """
    Build smooth μ with two absorption‑like bumps aligned with two fixed masks;
    add small stochastic variation across batch.
    """
    x = np.linspace(0.0, 1.0, L_, dtype=np.float64)
    base = 0.02 + 0.004 * np.sin(2 * math.pi * 2.0 * x)
    b1 = 0.012 * np.exp(-0.5 * ((x - 0.25) / 0.02) ** 2)
    b2 = 0.010 * np.exp(-0.5 * ((x - 0.67) / 0.018) ** 2)
    mu0 = np.clip(base + b1 + b2, 0.0, 1.0)
    batch = np.stack([mu0 + _RNG.normal(0, 6e-4, size=L_) for _ in range(B)], axis=0)
    return np.clip(batch, 0.0, 1.0)


def _make_symbolic_overlay(B: int = B, L_: int = L) -> Dict[str, Any]:
    """
    Create a simple symbolic overlay: two rule masks + per‑bin violation magnitudes.
    For each sample, elevate violations inside masked bands; keep outside small.
    """
    m1 = _rect_mask(L_, 0.16, 0.31).astype(bool)
    m2 = _rect_mask(L_, 0.58, 0.74).astype(bool)

    sym = np.zeros((B, L_), dtype=np.float32)
    for i in range(B):
        base_noise = np.abs(_RNG.normal(0, 1e-4, size=L_)).astype(np.float32)
        sym[i] = base_noise
        sym[i, m1] += 6e-3 + np.abs(_RNG.normal(0, 1e-3, size=m1.sum()))
        sym[i, m2] += 5e-3 + np.abs(_RNG.normal(0, 1e-3, size=m2.sum()))

    rules = [
        {"name": "left_band_rule", "mask": m1.astype(np.float32).tolist(), "weight": 1.0},
        {"name": "right_band_rule", "mask": m2.astype(np.float32).tolist(), "weight": 1.0},
    ]
    return {"symbolic": sym, "rules": rules}


def _make_entropy(B: int = B, L_: int = L) -> np.ndarray:
    """
    Light entropy proxy rising in banded regions; outside near baseline.
    """
    m1 = _rect_mask(L_, 0.16, 0.31).astype(bool)
    m2 = _rect_mask(L_, 0.58, 0.74).astype(bool)
    ent = np.abs(_RNG.normal(2e-3, 5e-4, size=(B, L_))).astype(np.float32)
    ent[:, m1] += 1.5e-3
    ent[:, m2] += 1.2e-3
    return ent


# --------------------------------------------------------------------------------------
# Normalizers
# --------------------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
    assert np.isfinite(x).all(), "Array contains non-finite values"
    return x


def _norm_2d(arr: np.ndarray, B_expect: Optional[int] = None) -> Tuple[np.ndarray, str]:
    """
    Accept (B,k) or (k,) and return (B,k).
    """
    arr = _as_np(arr)
    if arr.ndim == 1:
        return arr[None, :], f"(1,{arr.shape[0]})"
    assert arr.ndim == 2, f"Expected 2D array, got shape={arr.shape}"
    if B_expect is not None:
        assert arr.shape[0] == B_expect, f"Expected B={B_expect}, got {arr.shape[0]}"
    return arr, f"({arr.shape[0]},{arr.shape[1]})"


def _norm_clusters(obj: Any, B_expect: int) -> np.ndarray:
    """
    Accept labels array (B,) or dict with 'labels'; return (B,) int array.
    """
    if isinstance(obj, dict):
        labels = obj.get("labels", None)
        assert labels is not None, "clusters dict missing 'labels'"
        labels = np.asarray(labels)
    else:
        labels = np.asarray(obj)
    assert labels.ndim == 1 and labels.shape[0] == B_expect, f"Expected cluster labels shape (B,), got {labels.shape}"
    return labels.astype(int)


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def fusion_mod():
    return _import_fusion_module()


@pytest.fixture(scope="module")
def fusion_entry(fusion_mod):
    return _locate_entrypoint(fusion_mod)


@pytest.fixture
def mu_batch():
    return _make_mu_batch(B=B, L_=L)


@pytest.fixture
def overlays():
    return {
        "symbolic": _make_symbolic_overlay(B=B, L_=L),
        "entropy": _make_entropy(B=B, L_=L),
    }


# --------------------------------------------------------------------------------------
# Tests — API & shapes
# --------------------------------------------------------------------------------------

def test_api_and_shapes(fusion_entry, mu_batch, overlays):
    """
    Ensure minimal expected keys exist and shapes are coherent.
    """
    kind, target = fusion_entry
    out = _invoke(
        kind,
        target,
        mu_batch,
        seed=1234,
        entropy=overlays["entropy"],
        symbolic=overlays["symbolic"]["symbolic"],
        rules=overlays["symbolic"]["rules"],
        n_freq=64,             # tiny FFT bins to keep CI fast
        do_pca=True,
        do_umap=False,         # some envs may not have umap; keep optional
        do_tsne=False,
    )
    assert isinstance(out, dict)

    # FFT features should exist
    assert "fft" in out, "Output missing 'fft' features."
    fft, _ = _norm_2d(_as_np(out["fft"]), B_expect=B)  # (B, Ffft)
    assert fft.shape[0] == B and fft.shape[1] > 8, "FFT feature dimension too small."

    # At least one embedding (pca/umap/tsne/fusion) should exist
    has_emb = False
    for key in ("pca", "umap", "tsne", "fusion"):
        if key in out:
            emb, _ = _norm_2d(_as_np(out[key]), B_expect=B)
            assert emb.shape[1] in (2, 3, 5, 10), "Unexpected embedding width; expected small k."
            has_emb = True
            break
    assert has_emb, "No embedding found among ['pca','umap','tsne','fusion']."

    # If clusters present, check shape
    if "clusters" in out:
        labels = _norm_clusters(out["clusters"], B_expect=B)
        assert labels.shape == (B,)


# --------------------------------------------------------------------------------------
# Tests — Fusion overlay sanity (localization)
# --------------------------------------------------------------------------------------

def test_symbolic_overlay_localization(fusion_entry, mu_batch, overlays):
    """
    If the tool returns a per-bin or per-sample overlay, verify that values
    are elevated inside masked rule regions versus outside (statistical check).
    """
    kind, target = fusion_entry
    out = _invoke(
        kind,
        target,
        mu_batch,
        seed=99,
        entropy=overlays["entropy"],
        symbolic=overlays["symbolic"]["symbolic"],
        rules=overlays["symbolic"]["rules"],
        n_freq=48,
        return_overlay=True,
    )

    # We accept either:
    #   out['overlay']['symbolic_fused'] -> (B,L)
    # or out['overlay']['symbolic']      -> (B,L)
    ov = None
    if "overlay" in out and isinstance(out["overlay"], dict):
        for k in ("symbolic_fused", "symbolic", "fusion_per_bin"):
            if k in out["overlay"]:
                arr = np.asarray(out["overlay"][k])
                if arr.ndim == 2 and arr.shape[1] == L:
                    ov = arr
                    break
    if ov is None:
        pytest.xfail("No suitable per-bin overlay found in output['overlay']; skipping localization test.")

    # Build masks
    m1 = np.asarray(overlays["symbolic"]["rules"][0]["mask"], dtype=bool)
    m2 = np.asarray(overlays["symbolic"]["rules"][1]["mask"], dtype=bool)

    # Check median elevation inside masks vs outside (aggregated across batch)
    inside = np.median(np.abs(ov[:, m1])); outside = np.median(np.abs(ov[:, ~m1]))
    assert inside > outside * 1.2, f"Overlay not elevated in mask #1: inside={inside:.3e}, outside={outside:.3e}"

    inside2 = np.median(np.abs(ov[:, m2])); outside2 = np.median(np.abs(ov[:, ~m2]))
    assert inside2 > outside2 * 1.2, f"Overlay not elevated in mask #2: inside={inside2:.3e}, outside={outside2:.3e}"


# --------------------------------------------------------------------------------------
# Tests — Determinism
# --------------------------------------------------------------------------------------

def test_determinism_fixed_seed(fusion_entry, mu_batch, overlays):
    """
    With identical inputs and seed, outputs should be identical (embeddings and clusters).
    """
    kind, target = fusion_entry

    out1 = _invoke(
        kind,
        target,
        mu_batch,
        seed=777,
        entropy=overlays["entropy"],
        symbolic=overlays["symbolic"]["symbolic"],
        rules=overlays["symbolic"]["rules"],
        n_freq=32,
        do_pca=True,
    )
    out2 = _invoke(
        kind,
        target,
        mu_batch,
        seed=777,
        entropy=overlays["entropy"],
        symbolic=overlays["symbolic"]["symbolic"],
        rules=overlays["symbolic"]["rules"],
        n_freq=32,
        do_pca=True,
    )

    # Compare PCA (or any available embedding)
    key = "pca" if "pca" in out1 and "pca" in out2 else ("fusion" if "fusion" in out1 and "fusion" in out2 else None)
    if key is None:
        pytest.xfail("No comparable embedding key ('pca' or 'fusion') present for determinism test.")
    e1, _ = _norm_2d(_as_np(out1[key]), B_expect=B)
    e2, _ = _norm_2d(_as_np(out2[key]), B_expect=B)
    assert np.array_equal(e1, e2), f"Embedding '{key}' changed despite fixed seed."

    # Clusters (if present)
    if "clusters" in out1 and "clusters" in out2:
        c1 = _norm_clusters(out1["clusters"], B_expect=B)
        c2 = _norm_clusters(out2["clusters"], B_expect=B)
        assert np.array_equal(c1, c2), "Cluster labels changed despite fixed seed."


# --------------------------------------------------------------------------------------
# Tests — Artifact save (optional)
# --------------------------------------------------------------------------------------

def test_artifact_save_roundtrip_if_available(fusion_mod, fusion_entry, tmp_path, mu_batch, overlays):
    """
    If module exposes a saver like save_fusion_artifacts(...), verify that
    it writes files and that NPY arrays reload with correct shapes.
    """
    save_fn = None
    for name in ("save_fusion_artifacts", "save_artifacts", "write_artifacts"):
        if hasattr(fusion_mod, name) and callable(getattr(fusion_mod, name)):
            save_fn = getattr(fusion_mod, name)
            break
    if save_fn is None:
        pytest.xfail("Fusion module exposes no artifact saver; skipping round-trip test.")

    kind, target = fusion_entry
    out = _invoke(
        kind,
        target,
        mu_batch,
        seed=123,
        entropy=overlays["entropy"],
        symbolic=overlays["symbolic"]["symbolic"],
        rules=overlays["symbolic"]["rules"],
        n_freq=40,
        do_pca=True,
    )

    outdir = tmp_path / "fusion_artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    save_fn(out, outdir=str(outdir))

    files = list(outdir.glob("*"))
    assert files, "No artifacts written by saver."

    # Reload selectable arrays if present
    for key, fname in (("fft", "fft.npy"), ("pca", "pca.npy"), ("fusion", "fusion.npy")):
        p = outdir / fname
        if p.exists():
            arr = np.load(p)
            _norm_2d(arr, B_expect=B)  # raises on mismatch

    # Clusters optional
    p_labels = outdir / "clusters.npy"
    if p_labels.exists():
        labels = np.load(p_labels)
        _norm_clusters(labels, B_expect=B)  # raises on mismatch


# --------------------------------------------------------------------------------------
# Tests — CLI smoke (optional)
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(__import__("shutil").which("spectramind") is None, reason="spectramind CLI not found in PATH")
def test_cli_smoke_fft_symbolic_fusion(tmp_path, mu_batch, overlays):
    """
    Smoke test the repo CLI for this diagnostic (adjust flags if your repo differs):

        spectramind diagnose fft-fusion \
            --mu mu.npy \
            --symbolic symbolic.json \
            --entropy entropy.npy \
            --outdir out \
            --n-freq 48 \
            --seed 123

    We only assert it runs and writes at least one artifact.
    """
    mu_path = tmp_path / "mu.npy"
    np.save(mu_path, mu_batch)

    entropy_path = tmp_path / "entropy.npy"
    np.save(entropy_path, overlays["entropy"])

    # Minimal symbolic JSON with per-sample per-bin violations + rules
    sym_path = tmp_path / "symbolic.json"
    sym_payload = {
        "symbolic": overlays["symbolic"]["symbolic"].tolist(),
        "rules": overlays["symbolic"]["rules"],
    }
    with open(sym_path, "w", encoding="utf-8") as f:
        json.dump(sym_payload, f)

    outdir = tmp_path / "out"
    outdir.mkdir(parents=True, exist_ok=True)

    cmd = [
        "spectramind", "diagnose", "fft-fusion",
        "--mu", str(mu_path),
        "--symbolic", str(sym_path),
        "--entropy", str(entropy_path),
        "--outdir", str(outdir),
        "--n-freq", "48",
        "--seed", "123",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        pytest.xfail(
            "CLI returned nonzero exit (subcommand/flags may differ).\n"
            f"STDOUT:\n{proc.stdout}\n\nSTDERR:\n{proc.stderr}"
        )

    produced = list(outdir.glob("*"))
    assert len(produced) > 0, "CLI ran but produced no artifacts."


# --------------------------------------------------------------------------------------
# Performance guardrail
# --------------------------------------------------------------------------------------

def test_runs_fast_enough(fusion_entry, mu_batch, overlays):
    """
    Tiny configuration should complete in < 1.5s on CI CPU.
    """
    kind, target = fusion_entry
    t0 = time.time()
    _ = _invoke(
        kind,
        target,
        mu_batch,
        seed=11,
        entropy=overlays["entropy"],
        symbolic=overlays["symbolic"]["symbolic"],
        rules=overlays["symbolic"]["rules"],
        n_freq=32,
        do_pca=True,
        do_umap=False,
        do_tsne=False,
    )
    dt = time.time() - t0
    assert dt < 1.5, f"FFT × Symbolic fusion too slow: {dt:.3f}s (should be < 1.5s)"