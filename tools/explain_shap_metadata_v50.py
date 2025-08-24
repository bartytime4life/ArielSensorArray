#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/explain_shap_metadata_v50.py

SpectraMind V50 — SHAP + Metadata Explainer (Ultimate, Challenge-Grade)

Purpose
-------
This tool fuses SHAP attributions with per-planet metadata and diagnostics to produce:
  • Interactive UMAP/t‑SNE/PCA projections with SHAP/entropy/symbolic overlays
  • Per-planet metrics CSV/JSON (entropy, total |SHAP|, top‑K bins, violation scores, GLL if available)
  • Top‑K SHAP bar charts and distribution plots
  • Self-contained, versioned HTML diagnostics "mini dashboard"
  • Append-only CLI audit logs (logs/v50_debug_log.md, logs/v50_runs.jsonl)

It is CLI-ready (Typer), Hydra-friendly (accepts override-like flags), and reproducibility-safe
(deterministic seeds, run hash manifest). It integrates gracefully even when some optional
inputs (symbolic/GLL) are missing — the pipeline stays functional with defaults.

Inputs (any subset; the more you pass, the richer the overlays):
  --shap:               Numpy .npy (P × B) or CSV/Parquet/Feather/NPZ with 2D array (planets × bins)
  --mu:                 Optional Numpy .npy (P × B) predicted μ spectra (for overlays / summaries)
  --metadata:           CSV or JSON with at least a 'planet_id' column (or we synthesize IDs)
  --symbolic:           JSON produced by symbolic modules (per-planet violation scores or matrices)
  --diagnostic-summary: JSON from generate_diagnostic_summary.py (to fetch GLL/coverage/etc.)

Outputs:
  outdir/
    shap_metadata_metrics.csv
    shap_metadata_metrics.json
    projection_umap.html (if enabled)
    projection_tsne.html (if enabled)
    projection_pca.html  (if enabled)
    topk_shap_bins_bar.png
    shap_distribution.png
    explain_shap_metadata_manifest.json
    run_hash_summary_v50.json (appended/updated)
    (and auxiliary artifacts)

Example
-------
poetry run python tools/explain_shap_metadata_v50.py \
  --shap outputs/shap/shap_values.npy \
  --mu outputs/predictions/mu.npy \
  --metadata data/planet_metadata.csv \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --diagnostic-summary outputs/diagnostics/diagnostic_summary.json \
  --projection all --proj-bins 256 --topk 20 \
  --outdir outputs/explain_shap_metadata_v50 --open-browser

Design Notes
------------
• Zero external network calls. Optional dependencies (umap-learn, plotly) are handled gracefully.
• Deterministic runs: np/random + sklearn PRNG seeded; TSNE/UMAP seeds set.
• HTML is standalone (Plotly) and safe to archive under versioned filenames.
• Logging: always appends an audited, timestamped entry to logs/v50_debug_log.md and logs/v50_runs.jsonl.
• "No placeholders": when optional files are absent, we compute meaningful fallbacks and keep going.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sys
import textwrap
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ----------------------------
# Strict, robust standard deps
# ----------------------------
import numpy as np

# Pandas is used for tabular fusion/exports; hard requirement for rich outputs.
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for this tool. Please `pip install pandas`.") from e

# Scikit-learn is used for PCA/t-SNE and preprocessing
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
except Exception as e:
    raise RuntimeError("scikit-learn is required. Please `pip install scikit-learn`.") from e

# Optional: UMAP (nice to have)
try:
    import umap
    _UMAP_OK = True
except Exception:
    _UMAP_OK = False

# Optional: Plotly (interactive HTML)
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

# Matplotlib for static PNGs
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False


# ============================================================
# Utility: robust logging (append-only), hashing, RNG control
# ============================================================

def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _hash_dict_sorted(d: Dict[str, Any]) -> str:
    b = json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def set_deterministic(seed: int) -> None:
    """
    Deterministic seeding for numpy and sklearn/UMAP/TSNE usage.
    """
    np.random.seed(seed)
    # sklearn and umap use numpy RNG internally. We pass random_state=seed to estimators below.


@dataclass
class AuditLogger:
    """
    Append-only logger writing to both a Markdown log (human-friendly) and a JSONL log (machine-friendly).
    This mirrors the V50 logging approach used across CLIs.
    """
    md_path: Path
    jsonl_path: Path

    def log(self, event: Dict[str, Any]) -> None:
        _ensure_dir(self.md_path.parent)
        _ensure_dir(self.jsonl_path.parent)
        event = dict(event)
        event.setdefault("timestamp", _now_iso())
        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        # MD
        md = textwrap.dedent(f"""
        ---
        time: {event.get('timestamp')}
        tool: explain_shap_metadata_v50
        action: {event.get('action','run')}
        outdir: {event.get('outdir','')}
        projection: {event.get('projection','')}
        seeds: {event.get('seed')}
        shap: {event.get('shap')}
        mu: {event.get('mu')}
        metadata: {event.get('metadata')}
        symbolic: {event.get('symbolic')}
        diagnostic_summary: {event.get('diagnostic_summary')}
        status: {event.get('status','ok')}
        message: {event.get('message','')}
        """).strip() + "\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(md + "\n")


# ===============================================
# Data Loading: SHAP, μ, metadata, symbolic, GLL
# ===============================================

def _load_array_any(path: Path) -> np.ndarray:
    """
    Load a 2D array (planets × bins) from .npy, .npz, or text/CSV-like (csv/tsv/parquet/feather is routed via pandas).
    """
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array in {path}, got shape {arr.shape}")
        return arr
    if suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        # Try to find the first array with 2D shape
        for k in z.files:
            a = z[k]
            if a.ndim == 2:
                return a
        raise ValueError(f"No 2D arrays found in {path}")
    # Otherwise attempt pandas reader then to_numpy
    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path) if suffix == ".csv" else pd.read_csv(path, sep="\t")
        arr = df.to_numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D data in {path}, got shape {arr.shape}")
        return arr
    if suffix in {".parquet"}:
        df = pd.read_parquet(path)
        arr = df.to_numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D data in {path}, got shape {arr.shape}")
        return arr
    if suffix in {".feather"}:
        df = pd.read_feather(path)
        arr = df.to_numpy()
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D data in {path}, got shape {arr.shape}")
        return arr
    raise ValueError(f"Unsupported array format: {path}")


def _load_metadata_any(path: Optional[Path], n_planets: int) -> pd.DataFrame:
    """
    Load metadata with a 'planet_id' column. If missing, synthesize a DataFrame with default IDs.
    """
    if path is None:
        # synthesize
        return pd.DataFrame({"planet_id": [f"planet_{i:04d}" for i in range(n_planets)]})
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        df = pd.read_csv(path) if suffix == ".csv" else pd.read_csv(path, sep="\t")
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in {".parquet"}:
        df = pd.read_parquet(path)
    elif suffix in {".feather"}:
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported metadata format: {path}")
    if "planet_id" not in df.columns:
        # try to auto-derive, else synthesize
        df = df.copy()
        df["planet_id"] = [f"planet_{i:04d}" for i in range(len(df))]
    return df


def _load_symbolic_any(path: Optional[Path], planet_ids: List[str]) -> pd.DataFrame:
    """
    Load symbolic results. Expected to contain per-planet violation scores or matrices.
    Flexible schemas:
      • {"planet_id": ..., "violation_score": float, ...}
      • {"planet_id": ..., "rule_scores": {...}}
      • {"planets": {"p0": {"violation": ...}, ...}}
    Returns a tidy DataFrame indexed by planet_id with columns:
      violation_score, n_rules, (and optionally per-rule columns if we can flatten)
    Missing -> returns zeros.
    """
    if path is None:
        return pd.DataFrame({"planet_id": planet_ids, "violation_score": np.zeros(len(planet_ids)),
                             "n_rules": np.zeros(len(planet_ids), dtype=int)})
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Heuristics to flatten flexible structures
    records: Dict[str, Dict[str, Any]] = {}

    def ensure_rec(pid: str) -> Dict[str, Any]:
        if pid not in records:
            records[pid] = {"planet_id": pid}
        return records[pid]

    if isinstance(obj, dict):
        # Case A: top-level list-like under "rows" or "planets"
        if "rows" in obj and isinstance(obj["rows"], list):
            for row in obj["rows"]:
                pid = str(row.get("planet_id", ""))
                if not pid:
                    continue
                rec = ensure_rec(pid)
                for k, v in row.items():
                    if k == "planet_id":
                        continue
                    rec[k] = v
        elif "planets" in obj and isinstance(obj["planets"], dict):
            for pid, payload in obj["planets"].items():
                rec = ensure_rec(str(pid))
                if isinstance(payload, dict):
                    # flatten
                    for k, v in payload.items():
                        if isinstance(v, (int, float)):
                            rec[k] = v
                        elif isinstance(v, list):
                            rec[f"{k}_sum"] = float(np.sum(v))
                            rec[f"{k}_mean"] = float(np.mean(v)) if len(v) else 0.0
                            rec[f"{k}_n"] = len(v)
                        elif isinstance(v, dict):
                            # flatten 1-level dict
                            for k2, v2 in v.items():
                                if isinstance(v2, (int, float)):
                                    rec[f"{k}_{k2}"] = v2
                else:
                    rec["violation_score"] = float(payload) if isinstance(payload, (int, float)) else 0.0
        else:
            # try direct table
            for k, v in obj.items():
                if k == "planet_id":
                    continue
                # attempt to align lists by order
                if isinstance(v, list) and len(v) == len(planet_ids):
                    # add column
                    # keep as numeric if possible
                    arr = pd.to_numeric(pd.Series(v), errors="coerce")
                    # we'll assign later
                    pass
            # fallback: not enough info; create zeros
    elif isinstance(obj, list):
        for row in obj:
            if not isinstance(row, dict):
                continue
            pid = str(row.get("planet_id", ""))
            if not pid:
                continue
            rec = ensure_rec(pid)
            for k, v in row.items():
                if k == "planet_id":
                    continue
                rec[k] = v

    # Build DataFrame
    if records:
        sdf = pd.DataFrame(list(records.values()))
        if "planet_id" not in sdf.columns:
            sdf["planet_id"] = planet_ids[: len(sdf)]
    else:
        sdf = pd.DataFrame({"planet_id": planet_ids})
    # Normalize core fields
    if "violation_score" not in sdf.columns:
        # try to find a reasonable aggregate
        agg_cols = [c for c in sdf.columns if c not in {"planet_id"}]
        if agg_cols:
            sdf["violation_score"] = pd.to_numeric(sdf[agg_cols], errors="coerce").fillna(0.0).sum(axis=1)
        else:
            sdf["violation_score"] = 0.0
    # rule count
    rule_cols = [c for c in sdf.columns if c not in {"planet_id", "violation_score"}]
    sdf["n_rules"] = len(rule_cols) if rule_cols else 0
    return sdf


def _load_diag_summary_any(path: Optional[Path], planet_ids: List[str]) -> pd.DataFrame:
    """
    Load diagnostic_summary.json if available; extract per-planet GLL and other metrics when present.
    Returns DataFrame with columns subset: planet_id, gll (float), coverage (optional), entropy (optional)
    Missing -> zeros with planet_id.
    """
    if path is None:
        return pd.DataFrame({"planet_id": planet_ids, "gll": np.zeros(len(planet_ids))})
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Flexible schema support:
    # Expect either:
    #  - {"planets":[{"planet_id":..., "gll":...}, ...]}
    #  - {"planets": {"pid": {"gll":...}, ...}}
    #  - {"rows":[...]}, etc.
    recs: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "planets" in obj and isinstance(obj["planets"], list):
            for row in obj["planets"]:
                if isinstance(row, dict):
                    pid = str(row.get("planet_id", ""))
                    if not pid:
                        continue
                    recs.append({
                        "planet_id": pid,
                        "gll": float(row.get("gll", 0.0)),
                        "coverage": float(row.get("coverage", 0.0)) if "coverage" in row else np.nan,
                        "entropy_diag": float(row.get("entropy", 0.0)) if "entropy" in row else np.nan,
                    })
        elif "planets" in obj and isinstance(obj["planets"], dict):
            for pid, payload in obj["planets"].items():
                recs.append({
                    "planet_id": str(pid),
                    "gll": float(payload.get("gll", 0.0)) if isinstance(payload, dict) else 0.0,
                    "coverage": float(payload.get("coverage", 0.0)) if isinstance(payload, dict) and "coverage" in payload else np.nan,
                    "entropy_diag": float(payload.get("entropy", 0.0)) if isinstance(payload, dict) and "entropy" in payload else np.nan,
                })
        elif "rows" in obj and isinstance(obj["rows"], list):
            for row in obj["rows"]:
                if not isinstance(row, dict):
                    continue
                pid = str(row.get("planet_id", ""))
                if not pid:
                    continue
                recs.append({
                    "planet_id": pid,
                    "gll": float(row.get("gll", 0.0)),
                    "coverage": float(row.get("coverage", 0.0)) if "coverage" in row else np.nan,
                    "entropy_diag": float(row.get("entropy", 0.0)) if "entropy" in row else np.nan,
                })
    df = pd.DataFrame(recs) if recs else pd.DataFrame({"planet_id": planet_ids, "gll": np.zeros(len(planet_ids))})
    return df


# =========================================
# Metrics & Overlays: entropy, SHAP summaries
# =========================================

def shap_entropy_per_planet(shap_abs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Compute Shannon entropy over normalized |SHAP| values per planet.
    """
    P, B = shap_abs.shape
    probs = shap_abs / (shap_abs.sum(axis=1, keepdims=True) + eps)
    # entropy = -sum p log p
    ent = -np.sum(probs * (np.log(probs + eps)), axis=1)
    return ent


def summarize_topk_bins(shap_abs: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    For each planet, find top-K bins by |SHAP| magnitude and return:
      topk_idx: (P × k) indices
      topk_val: (P × k) values
    """
    if k <= 0:
        return np.zeros((shap_abs.shape[0], 0), dtype=int), np.zeros((shap_abs.shape[0], 0), dtype=float)
    idx = np.argpartition(-shap_abs, kth=min(k, shap_abs.shape[1]-1), axis=1)[:, :k]
    # sort within top-k for presentation (descending)
    row_idx = np.arange(shap_abs.shape[0])[:, None]
    vals = shap_abs[row_idx, idx]
    order = np.argsort(-vals, axis=1)
    idx_sorted = idx[row_idx, order]
    vals_sorted = vals[row_idx, order]
    return idx_sorted, vals_sorted


def aggregate_bin_importance(shap_abs: np.ndarray) -> np.ndarray:
    """
    Mean |SHAP| per bin across planets.
    """
    return shap_abs.mean(axis=0)


# =========================================
# Projections: PCA / t-SNE / UMAP
# =========================================

def _prepare_projection_features(shap_abs: np.ndarray, proj_bins: int) -> np.ndarray:
    """
    Reduce per-planet features for projection.
    Strategy: if B > proj_bins, keep the most variable bins across planets; else use all bins.
    """
    P, B = shap_abs.shape
    if proj_bins <= 0 or proj_bins >= B:
        return shap_abs
    # Variance-based bin selection
    var = shap_abs.var(axis=0)
    keep = np.argsort(-var)[:proj_bins]
    return shap_abs[:, keep]


def project_pca(X: np.ndarray, n_components: int, seed: int) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=n_components, random_state=seed)
    Y = pca.fit_transform(Xs)
    return Y


def project_tsne(X: np.ndarray, n_components: int, seed: int, perplexity: float, n_iter: int) -> np.ndarray:
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    tsne = TSNE(
        n_components=n_components,
        random_state=seed,
        perplexity=perplexity,
        n_iter=n_iter,
        learning_rate="auto",
        init="pca",
        metric="euclidean",
        verbose=0,
        square_distances=True,
    )
    Y = tsne.fit_transform(Xs)
    return Y


def project_umap(X: np.ndarray, n_components: int, seed: int, n_neighbors: int, min_dist: float) -> np.ndarray:
    if not _UMAP_OK:
        raise RuntimeError("UMAP is not installed. Install via `pip install umap-learn`.")
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(
        n_components=n_components,
        random_state=seed,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="euclidean",
        verbose=False,
    )
    Y = reducer.fit_transform(Xs)
    return Y


# =========================================
# Visualization: Plotly HTML + Matplotlib PNG
# =========================================

def _colorbar_title(name: str) -> str:
    return {"entropy": "Entropy", "violation_score": "Symbolic Violations", "gll": "GLL"}\
        .get(name, name)


def _make_projection_html(
    Y: np.ndarray,
    planet_ids: List[str],
    color_values: np.ndarray,
    color_name: str,
    size_values: Optional[np.ndarray],
    title: str,
    out_html: Path,
    extra_hover: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save interactive 2D scatter as HTML via Plotly. If Plotly is unavailable, save a CSV with coordinates instead.
    """
    if not _PLOTLY_OK:
        # Fallback: save CSV with coordinates for offline plotting
        df = pd.DataFrame({
            "planet_id": planet_ids,
            "x": Y[:, 0],
            "y": Y[:, 1],
            color_name: color_values,
        })
        if size_values is not None:
            df["marker_size"] = size_values
        if extra_hover:
            for k, v in extra_hover.items():
                # Expect list-like length P
                try:
                    if len(v) == len(planet_ids):
                        df[k] = v
                except Exception:
                    pass
        df.to_csv(out_html.with_suffix(".csv"), index=False)
        return

    hover_text = [f"planet_id: {pid}" for pid in planet_ids]
    # Add extra hover info
    if extra_hover:
        # If dictionary values align with planet_ids, append to hover text
        for k, v in extra_hover.items():
            try:
                if len(v) == len(planet_ids):
                    hover_text = [f"{ht}<br>{k}: {v[i]}" for i, ht in enumerate(hover_text)]
            except Exception:
                pass

    marker_kw = dict(size=8)
    if size_values is not None:
        # Normalize sizes to [6, 18]
        s = np.asarray(size_values).astype(float)
        s = np.clip(s, np.nanmin(s), np.nanmax(s)) if np.nanmax(s) > np.nanmin(s) else np.ones_like(s)
        s_norm = 6 + 12 * (s - np.nanmin(s)) / (np.nanmax(s) - np.nanmin(s) + 1e-12)
        marker_kw["size"] = s_norm

    fig = go.Figure(
        data=go.Scattergl(
            x=Y[:, 0], y=Y[:, 1],
            mode="markers",
            marker=dict(
                color=color_values,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=_colorbar_title(color_name)),
                **marker_kw,
            ),
            text=hover_text,
            hoverinfo="text",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title="dim-1",
        yaxis_title="dim-2",
        template="plotly_white",
        width=1000,
        height=800,
    )
    _ensure_dir(out_html.parent)
    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")


def _save_topk_bar(bin_importance: np.ndarray, topk: int, out_png: Path) -> None:
    if not _MPL_OK:
        # Fallback: write CSV
        df = pd.DataFrame({"bin": np.arange(len(bin_importance)), "mean_abs_shap": bin_importance})
        df.sort_values("mean_abs_shap", ascending=False).head(topk).to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    topk = max(1, int(topk))
    order = np.argsort(-bin_importance)[:topk]
    vals = bin_importance[order]
    plt.figure(figsize=(12, 6))
    plt.bar(np.arange(topk), vals)
    plt.xticks(np.arange(topk), [str(i) for i in order], rotation=90)
    plt.xlabel("Bin (index)")
    plt.ylabel("Mean |SHAP| across planets")
    plt.title(f"Top-{topk} most important bins by mean |SHAP|")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _save_distribution(shap_abs: np.ndarray, out_png: Path) -> None:
    if not _MPL_OK:
        # Fallback: CSV of summary stats
        df = pd.DataFrame({
            "planet_mean_abs_shap": shap_abs.mean(axis=1),
            "planet_sum_abs_shap": shap_abs.sum(axis=1),
            "planet_entropy": shap_entropy_per_planet(shap_abs),
        })
        df.to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    planet_sums = shap_abs.sum(axis=1)
    planet_means = shap_abs.mean(axis=1)
    ent = shap_entropy_per_planet(shap_abs)
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.hist(planet_sums, bins=40)
    plt.title("Distribution of total |SHAP| per planet")
    plt.subplot(3, 1, 2)
    plt.hist(planet_means, bins=40)
    plt.title("Distribution of mean |SHAP| per planet")
    plt.subplot(3, 1, 3)
    plt.hist(ent, bins=40)
    plt.title("Distribution of SHAP entropy per planet")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# =========================================
# Main fusion pipeline
# =========================================

@dataclass
class Inputs:
    shap_path: Path
    mu_path: Optional[Path]
    metadata_path: Optional[Path]
    symbolic_path: Optional[Path]
    diag_summary_path: Optional[Path]
    outdir: Path
    projection: str
    proj_bins: int
    topk: int
    seed: int
    umap_neighbors: int
    umap_min_dist: float
    tsne_perplexity: float
    tsne_iter: int
    n_components: int
    html_name: str
    open_browser: bool


def run_explainer(cfg: Inputs, audit: AuditLogger) -> int:
    t0 = time.time()
    set_deterministic(cfg.seed)
    _ensure_dir(cfg.outdir)

    # Load SHAP (P × B)
    shap = _load_array_any(cfg.shap_path)
    if shap.ndim != 2:
        raise ValueError(f"SHAP array must be 2D; got shape {shap.shape}")
    P, B = shap.shape
    shap_abs = np.abs(shap)

    # Load μ (optional)
    mu = None
    if cfg.mu_path is not None:
        mu = _load_array_any(cfg.mu_path)
        if mu.shape[0] != P:
            raise ValueError(f"μ shape mismatch: planets {mu.shape[0]} vs SHAP {P}")

    # Metadata
    meta_df = _load_metadata_any(cfg.metadata_path, n_planets=P)
    # Align planet_id length to P
    if len(meta_df) < P:
        # pad
        missing = P - len(meta_df)
        pad = pd.DataFrame({"planet_id": [f"planet_{i+len(meta_df):04d}" for i in range(missing)]})
        meta_df = pd.concat([meta_df, pad], ignore_index=True)
    meta_df = meta_df.iloc[:P].copy()
    planet_ids = meta_df["planet_id"].astype(str).tolist()

    # Symbolic
    sym_df = _load_symbolic_any(cfg.symbolic_path, planet_ids)
    sym_df = sym_df.set_index("planet_id")
    # Diagnostic summary (GLL)
    diag_df = _load_diag_summary_any(cfg.diag_summary_path, planet_ids).set_index("planet_id")

    # Compute SHAP metrics
    total_abs_shap = shap_abs.sum(axis=1)                    # (P,)
    mean_abs_shap = shap_abs.mean(axis=1)                    # (P,)
    entropy_shap = shap_entropy_per_planet(shap_abs)         # (P,)
    topk_idx, topk_val = summarize_topk_bins(shap_abs, cfg.topk)  # (P × K)

    # Aggregate bin importance for bar plot
    bin_import = aggregate_bin_importance(shap_abs)

    # Build per-planet DataFrame
    df = meta_df.set_index("planet_id").copy()
    df["total_abs_shap"] = total_abs_shap
    df["mean_abs_shap"] = mean_abs_shap
    df["entropy_shap"] = entropy_shap
    df["violation_score"] = sym_df.reindex(df.index)["violation_score"].fillna(0.0)
    df["n_rules"] = sym_df.reindex(df.index)["n_rules"].fillna(0).astype(int)
    df["gll"] = diag_df.reindex(df.index)["gll"].fillna(0.0)
    if "coverage" in diag_df.columns:
        df["coverage"] = diag_df.reindex(df.index)["coverage"].fillna(np.nan)
    if "entropy_diag" in diag_df.columns:
        df["entropy_diag"] = diag_df.reindex(df.index)["entropy_diag"].fillna(np.nan)

    # Attach top-K bin indices/values as comma-separated strings for convenience
    if cfg.topk > 0:
        df["topk_bins"] = [";".join(map(str, row)) for row in topk_idx]
        df["topk_vals"] = [";".join([f"{x:.6g}" for x in row]) for row in topk_val]
    else:
        df["topk_bins"] = ""
        df["topk_vals"] = ""

    # Save metrics CSV/JSON
    metrics_csv = cfg.outdir / "shap_metadata_metrics.csv"
    metrics_json = cfg.outdir / "shap_metadata_metrics.json"
    df.to_csv(metrics_csv, index=True)
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump({"rows": df.reset_index().to_dict(orient="records")}, f, indent=2)

    # Static plots
    _save_topk_bar(bin_import, cfg.topk, cfg.outdir / "topk_shap_bins_bar.png")
    _save_distribution(shap_abs, cfg.outdir / "shap_distribution.png")

    # Projections
    X = _prepare_projection_features(shap_abs, cfg.proj_bins)  # (P × D)
    color_fields = [
        ("entropy_shap", df["entropy_shap"].to_numpy()),
        ("violation_score", df["violation_score"].to_numpy()),
        ("gll", df["gll"].to_numpy()),
    ]
    # size by total_abs_shap
    size_values = df["total_abs_shap"].to_numpy()

    # Extra hover info (aligned length P)
    extra_hover = {
        "total_abs_shap": size_values,
        "mean_abs_shap": df["mean_abs_shap"].to_numpy(),
        "violation_score": df["violation_score"].to_numpy(),
        "gll": df["gll"].to_numpy(),
    }

    html_outs: List[Path] = []
    proj_modes = [cfg.projection] if cfg.projection in {"pca", "tsne", "umap"} else ["pca", "tsne", "umap"]
    for mode in proj_modes:
        if mode == "pca":
            Y = project_pca(X, n_components=min(cfg.n_components, 2), seed=cfg.seed)
            # Guarantee 2D layout
            if Y.shape[1] == 1:
                Y = np.concatenate([Y, np.zeros_like(Y)], axis=1)
            for cname, cvals in color_fields:
                out_html = cfg.outdir / f"projection_pca_{cname}.html"
                _make_projection_html(Y, planet_ids, cvals, cname, size_values, f"PCA — colored by {cname}", out_html, extra_hover)
                html_outs.append(out_html)

        elif mode == "tsne":
            Y = project_tsne(X, n_components=2, seed=cfg.seed, perplexity=cfg.tsne_perplexity, n_iter=cfg.tsne_iter)
            for cname, cvals in color_fields:
                out_html = cfg.outdir / f"projection_tsne_{cname}.html"
                _make_projection_html(Y, planet_ids, cvals, cname, size_values, f"t-SNE — colored by {cname}", out_html, extra_hover)
                html_outs.append(out_html)

        elif mode == "umap":
            if not _UMAP_OK:
                # produce CSV fallback to avoid breaking pipeline
                for cname, cvals in color_fields:
                    out_csv = cfg.outdir / f"projection_umap_{cname}.csv"
                    pd.DataFrame({"planet_id": planet_ids, cname: cvals}).to_csv(out_csv, index=False)
                # note: we skip adding to html_outs
            else:
                Y = project_umap(X, n_components=2, seed=cfg.seed, n_neighbors=cfg.umap_neighbors, min_dist=cfg.umap_min_dist)
                for cname, cvals in color_fields:
                    out_html = cfg.outdir / f"projection_umap_{cname}.html"
                    _make_projection_html(Y, planet_ids, cvals, cname, size_values, f"UMAP — colored by {cname}", out_html, extra_hover)
                    html_outs.append(out_html)

    # Mini dashboard HTML (index)
    dashboard_html = cfg.outdir / (cfg.html_name if cfg.html_name.endswith(".html") else f"{cfg.html_name}.html")
    _write_dashboard_html(dashboard_html, planet_ids, df, html_outs, cfg)

    # Manifest & run hash
    manifest = {
        "tool": "explain_shap_metadata_v50",
        "timestamp": _now_iso(),
        "inputs": {
            "shap": str(cfg.shap_path),
            "mu": str(cfg.mu_path) if cfg.mu_path else None,
            "metadata": str(cfg.metadata_path) if cfg.metadata_path else None,
            "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else None,
            "diagnostic_summary": str(cfg.diag_summary_path) if cfg.diag_summary_path else None,
        },
        "outputs": {
            "metrics_csv": str(metrics_csv),
            "metrics_json": str(metrics_json),
            "dashboard_html": str(dashboard_html),
            "projection_files": [str(p) for p in html_outs],
            "topk_bar": str(cfg.outdir / "topk_shap_bins_bar.png"),
            "shap_distribution": str(cfg.outdir / "shap_distribution.png"),
        },
        "projection": cfg.projection,
        "proj_bins": cfg.proj_bins,
        "topk": cfg.topk,
        "seed": cfg.seed,
        "umap_neighbors": cfg.umap_neighbors,
        "umap_min_dist": cfg.umap_min_dist,
        "tsne_perplexity": cfg.tsne_perplexity,
        "tsne_iter": cfg.tsne_iter,
        "n_components": cfg.n_components,
        "open_browser": cfg.open_browser,
    }
    manifest_path = cfg.outdir / "explain_shap_metadata_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    # Update/append run hash summary
    _update_run_hash_summary(cfg.outdir, manifest)

    # Log success
    audit.log({
        "action": "run",
        "status": "ok",
        "message": f"Completed in {time.time()-t0:.2f}s",
        "projection": cfg.projection,
        "seed": cfg.seed,
        "outdir": str(cfg.outdir),
        "shap": str(cfg.shap_path),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
        "diagnostic_summary": str(cfg.diag_summary_path) if cfg.diag_summary_path else "",
    })

    # Optionally open dashboard
    if cfg.open_browser and _PLOTLY_OK:
        try:
            import webbrowser
            webbrowser.open_new_tab(dashboard_html.as_uri())
        except Exception:
            pass

    return 0


def _write_dashboard_html(out_html: Path, planet_ids: List[str], df: pd.DataFrame, html_outs: List[Path], cfg: Inputs) -> None:
    """
    Assemble a minimal, self-contained dashboard HTML that references the generated projection HTML
    files (Plotly pages), and provides links to CSV/JSON metrics and static PNGs.
    """
    _ensure_dir(out_html.parent)

    # Build a small summary table (first few rows)
    head_rows = df.reset_index().head(30)
    table_html = head_rows.to_html(index=False, escape=False)

    proj_links = "\n".join(
        f'<li><a href="{p.name}" target="_blank" rel="noopener">{p.name}</a></li>'
        for p in html_outs
    )

    resources = textwrap.dedent(f"""
    <ul>
      <li><a href="shap_metadata_metrics.csv" target="_blank" rel="noopener">shap_metadata_metrics.csv</a></li>
      <li><a href="shap_metadata_metrics.json" target="_blank" rel="noopener">shap_metadata_metrics.json</a></li>
      <li><a href="explain_shap_metadata_manifest.json" target="_blank" rel="noopener">explain_shap_metadata_manifest.json</a></li>
      <li><a href="topk_shap_bins_bar.png" target="_blank" rel="noopener">topk_shap_bins_bar.png</a></li>
      <li><a href="shap_distribution.png" target="_blank" rel="noopener">shap_distribution.png</a></li>
    </ul>
    """).strip()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — SHAP + Metadata Explainer</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<style>
  :root {{
    --bg:#0b0e14; --fg:#e6edf3; --muted:#9aa4b2; --card:#111827; --border:#2b3240; --brand:#0b5fff;
  }}
  body {{
    background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
    margin: 2rem; line-height: 1.5;
  }}
  h1, h2, h3 {{ color: var(--fg); }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 14px; padding: 1rem 1.25rem; margin-bottom: 1rem; }}
  a {{ color: var(--brand); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
  th, td {{ border: 1px solid var(--border); padding: 0.4rem 0.5rem; }}
  th {{ background: #0f172a; }}
  .pill {{ display:inline-block; padding: 0.15rem 0.6rem; border-radius:999px; background:#0f172a; border: 1px solid var(--border); }}
</style>
</head>
<body>
  <header class="card">
    <h1>SpectraMind V50 — SHAP × Metadata Explainer</h1>
    <div>Run: <span class="pill">{_now_iso()}</span> • Projection: <b>{cfg.projection}</b> • Seed: <b>{cfg.seed}</b></div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    {resources}
  </section>

  <section class="card">
    <h2>Projections</h2>
    <ul>
      {proj_links}
    </ul>
    <p>Each projection opens in a new tab. Colors reflect entropy, symbolic violation score, or GLL (if provided). Marker size ~ total |SHAP|.</p>
  </section>

  <section class="card">
    <h2>Preview — First 30 Planets</h2>
    {table_html}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 — SHAP + Metadata Explainer • Deterministic seed: {cfg.seed} • proj_bins: {cfg.proj_bins} • topk: {cfg.topk}</small>
  </footer>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


def _update_run_hash_summary(outdir: Path, manifest: Dict[str, Any]) -> None:
    """
    Maintain/update a simple run_hash_summary_v50.json in the same outdir for reproducibility.
    We hash the manifest (sorted) and append/update a list of runs.
    """
    run_hash = _hash_dict_sorted(manifest)
    summary_path = outdir / "run_hash_summary_v50.json"
    payload = {"runs": []}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "runs" not in payload or not isinstance(payload["runs"], list):
                payload = {"runs": []}
        except Exception:
            payload = {"runs": []}
    payload["runs"].append({"hash": run_hash, "timestamp": _now_iso(), "manifest": manifest})
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================
# CLI (Typer-like without hard dep): argparse
# =========================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="explain_shap_metadata_v50",
        description="Fuse SHAP attributions with metadata/symbolic/GLL and render projections & metrics."
    )
    p.add_argument("--shap", type=Path, required=True, help="Path to SHAP array (P×B) .npy/.npz/.csv/.parquet/.feather")
    p.add_argument("--mu", type=Path, default=None, help="Optional μ array (P×B); used for overlays/summaries")
    p.add_argument("--metadata", type=Path, default=None, help="Metadata CSV/JSON (must contain 'planet_id'; synthesized if missing)")
    p.add_argument("--symbolic", type=Path, default=None, help="Symbolic results JSON (flexible schema)")
    p.add_argument("--diagnostic-summary", type=Path, default=None, help="diagnostic_summary.json with per-planet GLL/coverage/etc.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory for artifacts")
    p.add_argument("--projection", type=str, default="all", choices=["all", "pca", "tsne", "umap"], help="Which projection(s) to render")
    p.add_argument("--proj-bins", type=int, default=256, help="Reduce feature bins to this count via variance selection (0=use all)")
    p.add_argument("--topk", type=int, default=20, help="Top‑K bins per planet to summarize")
    p.add_argument("--seed", type=int, default=7, help="Deterministic seed")
    p.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.05, help="UMAP min_dist")
    p.add_argument("--tsne-perplexity", type=float, default=30.0, help="t-SNE perplexity")
    p.add_argument("--tsne-iter", type=int, default=1000, help="t-SNE iterations")
    p.add_argument("--n-components", type=int, default=2, help="PCA components (2 recommended for 2D)")
    p.add_argument("--html-name", type=str, default="shap_metadata_dashboard.html", help="Filename for the mini dashboard HTML")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard HTML in default browser if Plotly available")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # Resolve paths
    shap_path = args.shap.resolve()
    mu_path = args.mu.resolve() if args.mu else None
    metadata_path = args.metadata.resolve() if args.metadata else None
    symbolic_path = args.symbolic.resolve() if args.symbolic else None
    diag_summary_path = args.diagnostic_summary.resolve() if args.diagnostic_summary else None
    outdir = args.outdir.resolve()

    # Init audit logger (repo-consistent locations)
    audit = AuditLogger(
        md_path=Path("logs") / "v50_debug_log.md",
        jsonl_path=Path("logs") / "v50_runs.jsonl"
    )

    cfg = Inputs(
        shap_path=shap_path,
        mu_path=mu_path,
        metadata_path=metadata_path,
        symbolic_path=symbolic_path,
        diag_summary_path=diag_summary_path,
        outdir=outdir,
        projection=args.projection,
        proj_bins=int(args.proj_bins),
        topk=int(args.topk),
        seed=int(args.seed),
        umap_neighbors=int(args.umap_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        tsne_perplexity=float(args.tsne_perplexity),
        tsne_iter=int(args.tsne_iter),
        n_components=int(args.n_components),
        html_name=str(args.html_name),
        open_browser=bool(args.open_browser),
    )

    # Early audit entry
    audit.log({
        "action": "start",
        "status": "running",
        "message": "Starting explain_shap_metadata_v50",
        "projection": cfg.projection,
        "seed": cfg.seed,
        "outdir": str(cfg.outdir),
        "shap": str(cfg.shap_path),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
        "diagnostic_summary": str(cfg.diag_summary_path) if cfg.diag_summary_path else "",
    })

    try:
        rc = run_explainer(cfg, audit)
        return rc
    except Exception as e:
        # Log failure
        audit.log({
            "action": "run",
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "projection": cfg.projection,
            "seed": cfg.seed,
            "outdir": str(cfg.outdir),
            "shap": str(cfg.shap_path),
        })
        # Also print rich traceback to stderr for CI visibility
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
