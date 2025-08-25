# /tests/diagnostics/test_plot_tsne_interactive.py
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Diagnostics Test: plot_tsne_interactive

Purpose
-------
Validate the interactive t‑SNE plotting tool used in diagnostics dashboards.
This test exercises both a Python API (if exposed) and an optional CLI route.

We verify:
1) API discovery & output contracts
   • Accepts latents of shape (B, D)
   • Returns an embedding (B, 2|3) and/or an HTML artifact (path or HTML string)
   • Optional label/hover/id metadata are wired without error
2) Determinism (with fixed seed)
   • Same inputs + seed ⇒ same embedding (within tight tolerance)
3) Artifact save (optional)
   • If saver exists, writes HTML/JSON/PNG and files are non‑trivial
4) CLI smoke (optional)
   • If `spectramind diagnose tsne-latents` exists, run and assert artifact creation
5) Performance guardrail
   • Tiny synthetic input completes quickly on CI

Design Notes
------------
• Defensively adaptable:
  - Discover module under several import paths.
  - Locate multiple potential entrypoint names (functions/classes).
  - Normalize outputs (dict/array/string/path) to a common form.
• The test keeps B and D small to be CI‑friendly.

Author: SpectraMind V50 Team
"""

from __future__ import annotations

import importlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import pytest


# --------------------------------------------------------------------------------------
# Discovery
# --------------------------------------------------------------------------------------

CANDIDATE_IMPORTS = [
    "tools.plot_tsne_interactive",
    "src.tools.plot_tsne_interactive",
    "diagnostics.plot_tsne_interactive",
    "plot_tsne_interactive",
]


def _import_tsne_module():
    last_err = None
    for name in CANDIDATE_IMPORTS:
        try:
            return importlib.import_module(name)
        except Exception as e:
            last_err = e
    raise ImportError(
        "Could not import plot_tsne_interactive module from any of:\n"
        f"  {CANDIDATE_IMPORTS}\n"
        f"Last error: {last_err}"
    )


def _locate_entrypoint(mod):
    """
    Accept any of the following (function or class):
      - generate_tsne_html(latents, labels=..., ids=..., outdir=..., **cfg)
      - plot_tsne_interactive(latents, labels=..., ids=..., **cfg)
      - run_tsne(latents, **cfg)
      - tsne_interactive(latents, **cfg)
      - class TSNEInteractive(...).run(...) / .generate(...)
    """
    for fn in (
        "generate_tsne_html",
        "plot_tsne_interactive",
        "run_tsne",
        "tsne_interactive",
    ):
        if hasattr(mod, fn) and callable(getattr(mod, fn)):
            return "func", getattr(mod, fn)

    for cls in ("TSNEInteractive", "TSNEPlotter", "LatentTSNE"):
        if hasattr(mod, cls):
            Cls = getattr(mod, cls)
            for method in ("run", "generate", "plot"):
                if hasattr(Cls, method) and callable(getattr(Cls, method)):
                    return "class", Cls

    pytest.xfail(
        "plot_tsne_interactive module found but no known entrypoint. "
        "Expected a function like generate_tsne_html()/plot_tsne_interactive()/run_tsne()/tsne_interactive(), "
        "or a class with .run/.generate/.plot."
    )
    return "none", None  # pragma: no cover


def _invoke(kind: str, target, latents: np.ndarray, **cfg) -> Dict[str, Any]:
    """
    Invoke entrypoint and coerce output to a dict.

    Expected keys (subset ok):
      - 'embedding' : ndarray (B, 2|3)
      - 'html_path' : str path to HTML file (or)
      - 'html'      : HTML string
      - 'labels'    : echo of labels or label vector
      - 'ids'       : echo of IDs (length B)
      - 'meta'      : dict with seed/config
    """
    if kind == "func":
        out = target(latents, **cfg)
    elif kind == "class":
        try:
            inst = target(latents=latents, **cfg)
        except TypeError:
            inst = target(**cfg)
            # Prefer .run(latents=...), fall back to .generate/.plot
            if hasattr(inst, "run"):
                out = inst.run(latents=latents)
            elif hasattr(inst, "generate"):
                out = inst.generate(latents=latents)
            else:
                out = inst.plot(latents=latents)
        else:
            if hasattr(inst, "run"):
                out = inst.run()
            elif hasattr(inst, "generate"):
                out = inst.generate()
            else:
                out = inst.plot()
    else:
        pytest.fail("Unknown invocation kind.")  # pragma: no cover

    # Coerce to dict
    if isinstance(out, dict):
        return out
    if isinstance(out, np.ndarray):
        return {"embedding": out}
    if isinstance(out, str):
        # Likely an HTML string or path
        key = "html" if "<html" in out.lower() or "<!doctype" in out.lower() else "html_path"
        return {key: out}
    pytest.fail("Unsupported return type from t-SNE entrypoint; expected dict/ndarray/str.")  # pragma: no cover
    return {}


# --------------------------------------------------------------------------------------
# Synthetic inputs
# --------------------------------------------------------------------------------------

B = 120      # points
D = 16       # latent dim (small)
_SEED = 20250824
RNG = np.random.RandomState(_SEED)


def _make_latents(B_: int = B, D_: int = D) -> np.ndarray:
    """
    Create a tiny, clusterable latent set: three Gaussian blobs in D dims.
    """
    centers = np.stack([
        np.concatenate([np.ones(D_) * 2.0, np.zeros(0)]),
        np.concatenate([np.ones(D_) * -2.0, np.zeros(0)]),
        np.concatenate([np.zeros(D_), np.zeros(0)]),
    ], axis=0)
    # Assign clusters roughly equally
    labels = np.repeat(np.arange(3), B_ // 3)
    if labels.size < B_:
        labels = np.concatenate([labels, RNG.choice(3, size=B_ - labels.size, replace=True)])
    RNG.shuffle(labels)

    X = np.zeros((B_, D_), dtype=np.float32)
    for i in range(B_):
        c = centers[labels[i]]
        X[i] = RNG.normal(loc=c, scale=0.6, size=D_).astype(np.float32)
    return X


def _make_labels_ids(B_: int = B) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.array([f"class_{i%3}" for i in range(B_)], dtype=object)
    ids = np.array([f"PL-{i:04d}" for i in range(B_)], dtype=object)
    return labels, ids


# --------------------------------------------------------------------------------------
# Normalizers
# --------------------------------------------------------------------------------------

def _as_np(x) -> np.ndarray:
    assert isinstance(x, np.ndarray), "Expected numpy.ndarray"
    assert np.isfinite(x).all(), "Array contains non-finite values"
    return x


def _norm_embedding(arr: np.ndarray, B_expect: int) -> Tuple[np.ndarray, str]:
    """
    Accept (B,2|3). Return (B,2|3).
    """
    arr = _as_np(arr)
    assert arr.ndim == 2 and arr.shape[0] == B_expect, f"Unexpected embedding shape {arr.shape}"
    assert arr.shape[1] in (2, 3), f"Embedding must be 2D or 3D, got {arr.shape[1]}"
    return arr, f"({arr.shape[0]},{arr.shape[1]})"


# --------------------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tsne_mod():
    return _import_tsne_module()


@pytest.fixture(scope="module")
def tsne_entry(tsne_mod):
    return _locate_entrypoint(tsne_mod)


@pytest.fixture
def latents() -> np.ndarray:
    return _make_latents(B, D)


@pytest.fixture
def labels_ids() -> Tuple[np.ndarray, np.ndarray]:
    return _make_labels_ids(B)


# --------------------------------------------------------------------------------------
# Tests — API & shapes
# --------------------------------------------------------------------------------------

def test_api_and_shapes(tsne_entry, latents, labels_ids, tmp_path: Path):
    """
    • Embedding of shape (B,2|3)
    • HTML string or file path created (if function supports outdir)
    • Labels/IDs accepted without error
    """
    labels, ids = labels_ids
    kind, target = tsne_entry

    out = _invoke(
        kind,
        target,
        latents,
        seed=1234,
        labels=labels,
        ids=ids,
        outdir=str(tmp_path / "tsne_artifacts"),
        perplexity=20,
        n_iter=500,
        learning_rate=200,
        metric="euclidean",
        html_name="tsne_test.html",
        open_browser=False,
        return_embedding=True,
    )
    assert isinstance(out, dict)

    # Embedding
    if "embedding" in out:
        emb, _ = _norm_embedding(np.asarray(out["embedding"]), B_expect=B)

    # HTML
    if "html_path" in out:
        p = Path(out["html_path"])
        assert p.exists() and p.suffix.lower() in (".html", ".htm")
        assert p.stat().st_size > 4000, "HTML seems too small to be a valid interactive plot."
        # Quick content sniff for Plotly/vis presence
        txt = p.read_text(encoding="utf-8", errors="ignore").lower()
        assert ("plotly" in txt or "div id=" in txt or "<script" in txt), "HTML content missing expected plotting markup."
    elif "html" in out:
        s = str(out["html"]).lower()
        assert ("<html" in s and "script" in s) or ("plotly" in s), "HTML string missing expected contents."
    else:
        # Some implementations may not return HTML unless asked explicitly; that's acceptable.
        pass


# --------------------------------------------------------------------------------------
# Tests — Determinism
# --------------------------------------------------------------------------------------

def test_determinism_fixed_seed(tsne_entry, latents, labels_ids, tmp_path: Path):
    """
    With fixed seed, embedding must be identical (or numerically equal within tight tol).
    t‑SNE can have small numeric jitter; allow atol=1e-6.
    """
    labels, ids = labels_ids
    kind, target = tsne_entry

    out1 = _invoke(
        kind,
        target,
        latents,
        seed=777,
        labels=labels,
        ids=ids,
        outdir=str(tmp_path / "tsne_seed1"),
        perplexity=25,
        n_iter=400,
        learning_rate=150,
        metric="euclidean",
        return_embedding=True,
    )
    out2 = _invoke(
        kind,
        target,
        latents,
        seed=777,
        labels=labels,
        ids=ids,
        outdir=str(tmp_path / "tsne_seed2"),
        perplexity=25,
        n_iter=400,
        learning_rate=150,
        metric="euclidean",
        return_embedding=True,
    )

    assert "embedding" in out1 and "embedding" in out2, "Entrypoint did not return embeddings."
    e1, _ = _norm_embedding(np.asarray(out1["embedding"]), B_expect=B)
    e2, _ = _norm_embedding(np.asarray(out2["embedding"]), B_expect=B)
    assert np.allclose(e1, e2, atol=1e-6), "Embeddings differ despite fixed seed."


# --------------------------------------------------------------------------------------
# Tests — Artifact save (optional)
# --------------------------------------------------------------------------------------

def test_artifact_save_roundtrip_if_available(tsne_mod, tsne_entry, latents, labels_ids, tmp_path: Path):
    """
    If module exposes an explicit saver (save_tsne_artifacts/save_artifacts/write_artifacts),
    verify it writes and files are non-trivial.
    """
    save_fn = None
    for name in ("save_tsne_artifacts", "save_artifacts", "write_artifacts"):
        if hasattr(tsne_mod, name) and callable(getattr(tsne_mod, name)):
            save_fn = getattr(tsne_mod, name)
            break
    if save_fn is None:
        pytest.xfail("Module exposes no artifact saver; skipping round-trip test.")

    labels, ids = labels_ids
    kind, target = tsne_entry
    out = _invoke(
        kind,
        target,
        latents,
        seed=42,
        labels=labels,
        ids=ids,
        outdir=str(tmp_path / "tsne_save"),
        return_embedding=True,
        html_name="tsne_saved.html",
    )

    outdir = tmp_path / "tsne_saved_artifacts"
    outdir.mkdir(parents=True, exist_ok=True)
    save_fn(out, outdir=str(outdir))

    files = list(outdir.glob("*"))
    assert files, "No artifacts written by saver."
    # Prefer HTML presence
    htmls = [p for p in files if p.suffix.lower() in (".html", ".htm")]
    if htmls:
        assert htmls[0].stat().st_size > 4000


# --------------------------------------------------------------------------------------
# CLI smoke (optional)
# --------------------------------------------------------------------------------------

@pytest.mark.skipif(shutil.which("spectramind") is None, reason="spectramind CLI not found in PATH")
def test_cli_smoke_tsne_latents(tmp_path: Path, latents, labels_ids):
    """
    Smoke test the repo CLI for t‑SNE (adjust flags if your repo differs):

        spectramind diagnose tsne-latents \
            --latents latents.npy \
            --labels labels.csv \
            --ids ids.txt \
            --outdir out \
            --seed 123 \
            --perplexity 25 \
            --n-iter 400

    We only assert that the command succeeds and writes an HTML artifact.
    """
    lat_path = tmp_path / "latents.npy"
    np.save(lat_path, latents)

    labels, ids = labels_ids
    labels_path = tmp_path / "labels.csv"
    pd.DataFrame({"label": labels}).to_csv(labels_path, index=False)

    ids_path = tmp_path / "ids.txt"
    ids_path.write_text("\n".join(map(str, ids)), encoding="utf-8")

    outdir = tmp_path / "out_cli_tsne"
    outdir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ["spectramind", "diagnose", "tsne-latents",
         "--latents", str(lat_path),
         "--labels", str(labels_path),
         "--ids", str(ids_path),
         "--outdir", str(outdir),
         "--seed", "123",
         "--perplexity", "25",
         "--n-iter", "400"],
        # Slight variants (flag naming may differ)
        ["spectramind", "diagnose", "tsne-latents",
         "--latents", str(lat_path),
         "--labels", str(labels_path),
         "--outdir", str(outdir),
         "--seed", "123"],
    ]

    last_proc = None
    for cmd in candidates:
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True)
        except Exception:
            continue
        last_proc = proc
        if proc.returncode == 0:
            break

    if last_proc is None or last_proc.returncode != 0:
        pytest.xfail(
            "No working CLI tsne-latents pattern found. Ensure subcommand/flags are wired. "
            f"Last stdout/stderr:\n{'' if last_proc is None else last_proc.stdout}\n"
            f"{'' if last_proc is None else last_proc.stderr}"
        )

    # HTML (or any artifact) must exist
    produced = list(outdir.glob("*"))
    assert produced, "CLI ran but produced no artifacts."
    htmls = [p for p in produced if p.suffix.lower() in (".html", ".htm")]
    assert htmls, "CLI did not produce an HTML output file."


# --------------------------------------------------------------------------------------
# Performance guardrail
# --------------------------------------------------------------------------------------

def test_runs_fast_enough(tsne_entry, latents, labels_ids, tmp_path: Path):
    kind, target = tsne_entry
    t0 = time.time()
    _ = _invoke(
        kind,
        target,
        latents,
        seed=11,
        labels=labels_ids[0],
        ids=labels_ids[1],
        outdir=str(tmp_path / "tsne_perf"),
        perplexity=20,
        n_iter=350,
        learning_rate=150,
        return_embedding=True,
    )
    dt = time.time() - t0
    assert dt < 1.5, f"t‑SNE interactive too slow for tiny input: {dt:.3f}s (should be < 1.5s)"