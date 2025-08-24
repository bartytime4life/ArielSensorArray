#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/generate_fft_symbolic_fusion.py

SpectraMind V50 — FFT × Symbolic Fingerprint Fusion (Ultimate, Challenge‑Grade)

Purpose
-------
Fuse frequency‑domain features of predicted spectra with symbolic fingerprints and (optionally)
SHAP/entropy overlays; then project to 2D/3D (UMAP / t‑SNE), cluster, and export diagnostics.

Inputs
------
• --mu                 : npy/npz/csv/tsv/parquet/feather, shape (P×B) or (B,)  [P=planets, B=bins]
• --wavelengths        : optional wavelengths vector (B,)
• --shap               : optional SHAP array, (P×B) or (P×I×B) or (B,)
• --symbolic-rules     : optional rules JSON with "rule_masks" (see symbolic_rule_table.py)
• --symbolic-results   : optional JSON with per‑planet violations/scores (flexible schema)
• --metadata           : optional CSV/JSON/Parquet with 'planet_id' column (synthesized if missing)

What it computes
----------------
1) FFT feature bank from μ(λ):
   - rFFT magnitudes up to N_FREQ (low‑frequency content), log‑scaled if requested
   - Optionally Hann window + mean detrend before FFT
   - Optional PCA to compact FFT features

2) Symbolic fingerprints (if rules are provided):
   - For each rule r: mean(μ over rule mask) and (if SHAP provided) sum(|SHAP| over rule mask)
   - These are appended to the fusion feature vector

3) Global overlays:
   - SHAP entropy per planet (if SHAP provided)
   - total |SHAP| per planet (if SHAP provided)
   - symbolic violation score per planet (if symbolic‑results provided)

4) Projection:
   - UMAP (if umap-learn available)
   - t‑SNE (sklearn)
   - Both produce HTML (Plotly) when available, plus PNG/CSV fallbacks

5) Clustering:
   - KMeans (k>1), with optional silhouette score
   - cluster_assignments.csv (planet_id, cluster), centroids.json

6) Dashboard:
   - Self‑contained HTML with quick links to CSV/PNG/HTML artifacts
   - Manifest JSON + append-only run-hash summary
   - Audit logs to logs/v50_debug_log.md and logs/v50_runs.jsonl

Determinism
-----------
All stochastic components seeded via --seed. No external network calls.

Examples
--------
poetry run python tools/generate_fft_symbolic_fusion.py \
  --mu outputs/predictions/mu.npy \
  --wavelengths data/wavelengths.npy \
  --shap outputs/shap/shap_values.npy \
  --symbolic-rules configs/symbolic_rules.json \
  --symbolic-results outputs/diagnostics/symbolic_results.json \
  --umap --tsne --kmeans 8 --silhouette \
  --nfft 512 --n-freq 120 --pca-k 64 --log-power \
  --outdir outputs/fft_symbolic_fusion --open-browser
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Tabular I/O
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for this tool. Please `pip install pandas`.") from e

# Optional viz: Plotly + Matplotlib (graceful fallbacks)
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False

# ML utilities
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
    _SK_OK = True
except Exception:
    _SK_OK = False

# Optional UMAP
try:
    import umap
    _UMAP_OK = True
except Exception:
    _UMAP_OK = False


# ==============================================================================
# Utilities: time, dirs, hashing, audit logging
# ==============================================================================

def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _hash_jsonable(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


@dataclass
class AuditLogger:
    md_path: Path
    jsonl_path: Path

    def log(self, event: Dict[str, Any]) -> None:
        _ensure_dir(self.md_path.parent); _ensure_dir(self.jsonl_path.parent)
        row = dict(event); row.setdefault("timestamp", _now_iso())
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        md = textwrap.dedent(f"""
        ---
        time: {row["timestamp"]}
        tool: generate_fft_symbolic_fusion
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        mu: {row.get("mu","")}
        shap: {row.get("shap","")}
        rules: {row.get("rules","")}
        symbolic_results: {row.get("symbolic_results","")}
        outdir: {row.get("outdir","")}
        message: {row.get("message","")}
        """).strip() + "\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(md)


def _update_run_hash_summary(outdir: Path, manifest: Dict[str, Any]) -> None:
    p = outdir / "run_hash_summary_v50.json"
    payload = {"runs": []}
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "runs" not in payload:
                payload = {"runs": []}
        except Exception:
            payload = {"runs": []}
    payload["runs"].append({"hash": _hash_jsonable(manifest), "timestamp": _now_iso(), "manifest": manifest})
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ==============================================================================
# Robust loaders
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if s == ".npz":
        z = np.load(path, allow_pickle=False)
        # pick first array‑like
        for k in z.files:
            return np.asarray(z[k])
        raise ValueError(f"No arrays found in {path}")
    if s in {".csv", ".tsv"}:
        df = pd.read_csv(path) if s == ".csv" else pd.read_csv(path, sep="\t")
        return df.to_numpy()
    if s == ".parquet":
        return pd.read_parquet(path).to_numpy()
    if s == ".feather":
        return pd.read_feather(path).to_numpy()
    raise ValueError(f"Unsupported array format: {path}")


def _load_metadata_any(path: Optional[Path], n_planets: int) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame({"planet_id": [f"planet_{i:04d}" for i in range(n_planets)]})
    s = path.suffix.lower()
    if s in {".csv", ".tsv"}:
        df = pd.read_csv(path) if s == ".csv" else pd.read_csv(path, sep="\t")
    elif s == ".json":
        df = pd.read_json(path)
    elif s == ".parquet":
        df = pd.read_parquet(path)
    elif s == ".feather":
        df = pd.read_feather(path)
    else:
        raise ValueError(f"Unsupported metadata format: {path}")
    if "planet_id" not in df.columns:
        df = df.copy()
        df["planet_id"] = [f"planet_{i:04d}" for i in range(len(df))]
    return df.iloc[:n_planets].copy()


# ==============================================================================
# SHAP helpers
# ==============================================================================

def _prepare_shap(shap: np.ndarray, P: int, B: int) -> np.ndarray:
    """
    Normalize to absolute SHAP (P×B):
      - (P,B)    → abs
      - (P,I,B)  → sum(abs, axis=1)
      - (B,)     → broadcast to P
    """
    a = np.asarray(shap, dtype=float)
    if a.ndim == 1:
        a = np.repeat(a[None, :], P, axis=0)
    elif a.ndim == 2:
        if a.shape[0] != P and a.shape[1] == P:
            a = a.T
        if a.shape[0] != P:
            if a.shape[0] == 1:
                a = np.repeat(a, P, axis=0)
            else:
                raise ValueError(f"SHAP planets mismatch; got {a.shape}, expected P={P}")
    elif a.ndim == 3:
        if a.shape[0] != P:
            raise ValueError(f"SHAP leading dim != P; got {a.shape}, P={P}")
        a = np.sum(np.abs(a), axis=1)
    else:
        raise ValueError(f"Unsupported SHAP shape: {a.shape}")
    if a.shape[1] != B:
        out = np.zeros((P, B), dtype=float)
        copyB = min(B, a.shape[1]); out[:, :copyB] = a[:, :copyB]
        a = out
    return np.abs(a)


def shap_entropy_per_planet(shap_abs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    P, B = shap_abs.shape
    p = shap_abs / (shap_abs.sum(axis=1, keepdims=True) + eps)
    return -np.sum(p * (np.log(p + eps)), axis=1)


# ==============================================================================
# Symbolic rules loader
# ==============================================================================

@dataclass
class RuleSet:
    rule_names: List[str]
    rule_masks: np.ndarray  # R×B (>=0)


def _as_mask_vector(spec: Any, B: int) -> np.ndarray:
    mask = np.zeros(B, dtype=float)
    if isinstance(spec, list):
        if len(spec) == B and all(isinstance(x, (int, float, bool, np.integer, np.floating)) for x in spec):
            arr = np.array(spec, dtype=float)
            mask = np.maximum(np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        else:
            for x in spec:
                if isinstance(x, (int, np.integer)) and 0 <= int(x) < B:
                    mask[int(x)] = 1.0
    elif isinstance(spec, dict):
        for k, v in spec.items():
            try:
                idx = int(k)
                if 0 <= idx < B:
                    mask[idx] = max(float(v), 0.0)
            except Exception:
                continue
    else:
        raise ValueError("Unsupported rule mask spec; must be list or dict.")
    return mask


def _load_rules_json(path: Optional[Path], B: int) -> Optional[RuleSet]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "rule_masks" not in obj or not isinstance(obj["rule_masks"], dict):
        raise ValueError("Rules JSON must contain a 'rule_masks' object mapping name→mask")
    names, mats = [], []
    for name, spec in obj["rule_masks"].items():
        names.append(str(name))
        mats.append(_as_mask_vector(spec, B))
    rule_masks = np.vstack(mats) if mats else np.zeros((0, B), dtype=float)
    return RuleSet(rule_names=names, rule_masks=rule_masks)


def _load_symbolic_results(path: Optional[Path], planet_ids: List[str]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "rows" in obj and isinstance(obj["rows"], list):
            for r in obj["rows"]:
                if isinstance(r, dict):
                    pid = r.get("planet_id"); val = r.get("violation", r.get("score", r.get("value", 0.0)))
                    if pid is not None:
                        rows.append({"planet_id": str(pid), "violation_score": float(val)})
        elif "planets" in obj and isinstance(obj["planets"], dict):
            for pid, payload in obj["planets"].items():
                if isinstance(payload, dict):
                    val = payload.get("violation", payload.get("score", 0.0))
                    rows.append({"planet_id": str(pid), "violation_score": float(val) if isinstance(val, (int, float)) else 0.0})
    df = pd.DataFrame(rows)
    if df.empty:
        return None
    # Align ordering to planet_ids
    return df.set_index("planet_id").reindex(planet_ids).reset_index().fillna(0.0)


# ==============================================================================
# FFT feature bank
# ==============================================================================

def _fft_features(mu: np.ndarray, nfft: int, n_freq: int, detrend: bool, hann: bool, log_power: bool) -> np.ndarray:
    """
    mu: P×B
    Return: P×n_freq (low‑freq magnitudes of rFFT)
    """
    P, B = mu.shape
    Z = mu.copy()
    if detrend:
        Z = Z - Z.mean(axis=1, keepdims=True)
    if hann:
        win = np.hanning(B).astype(Z.dtype)
        norm = np.sqrt((win**2).sum())
        Z = Z * win[None, :] / (norm + 1e-12)
    Y = np.fft.rfft(Z, n=nfft, axis=1)  # P×K
    mag = np.abs(Y)  # magnitude spectrum
    if log_power:
        mag = np.log1p(mag**2)
    K = mag.shape[1]
    k_keep = min(n_freq, K)
    return mag[:, 1:k_keep]  # drop DC


# ==============================================================================
# Viz helpers
# ==============================================================================

def _plot_scatter(coords: np.ndarray, color: Optional[np.ndarray], title: str, out_png: Path, out_html: Path,
                  color_label: str) -> None:
    _ensure_dir(out_png.parent)
    if _MPL_OK:
        plt.figure(figsize=(10, 6))
        if color is None:
            plt.scatter(coords[:, 0], coords[:, 1], s=10, alpha=0.85)
        else:
            sc = plt.scatter(coords[:, 0], coords[:, 1], s=10, c=color, cmap="viridis", alpha=0.9)
            cbar = plt.colorbar(sc, pad=0.01)
            cbar.set_label(color_label)
        plt.xlabel("dim-1"); plt.ylabel("dim-2"); plt.title(title); plt.tight_layout()
        plt.savefig(out_png, dpi=170); plt.close()
    if _PLOTLY_OK:
        fig = go.Figure()
        if color is None:
            fig.add_trace(go.Scattergl(x=coords[:, 0], y=coords[:, 1], mode="markers", marker=dict(size=5)))
        else:
            fig.add_trace(go.Scattergl(
                x=coords[:, 0], y=coords[:, 1], mode="markers",
                marker=dict(size=6, color=color, colorscale="Viridis", showscale=True, colorbar=dict(title=color_label))
            ))
        fig.update_layout(title=title, template="plotly_white", xaxis_title="dim-1", yaxis_title="dim-2")
        pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclass
class Config:
    mu_path: Path
    wavelengths_path: Optional[Path]
    shap_path: Optional[Path]
    rules_path: Optional[Path]
    symbolic_results_path: Optional[Path]
    metadata_path: Optional[Path]
    outdir: Path
    # FFT
    nfft: int
    n_freq: int
    detrend: bool
    hann: bool
    log_power: bool
    pca_k: int
    std: bool
    # Projections
    do_umap: bool
    do_tsne: bool
    umap_neighbors: int
    umap_min_dist: float
    umap_metric: str
    tsne_perplexity: float
    tsne_iter: int
    # Clustering
    kmeans_k: int
    silhouette: bool
    # Misc
    seed: int
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load μ
    MU = _load_array_any(cfg.mu_path)
    if MU.ndim == 1:
        MU = MU[None, :]
    if MU.ndim != 2:
        raise ValueError(f"--mu must be 2D (P×B) or 1D (B,). Got {MU.shape}")
    P, B = MU.shape

    # Metadata/planet ids
    meta_df = _load_metadata_any(cfg.metadata_path, n_planets=P)
    planet_ids = meta_df["planet_id"].astype(str).tolist()

    # Wavelengths (optional, only used in some summaries)
    WL = None
    if cfg.wavelengths_path:
        WL = _load_array_any(cfg.wavelengths_path).reshape(-1).astype(float)
        if WL.shape[0] != B:
            tmp = np.zeros(B, dtype=float)
            tmp[:min(B, WL.shape[0])] = WL[:min(B, WL.shape[0])]
            WL = tmp

    # SHAP (optional)
    SHAP_SUM = None        # per‑planet total |SHAP|
    SHAP_ENTROPY = None
    SABS = None
    if cfg.shap_path:
        raw = _load_array_any(cfg.shap_path)
        SABS = _prepare_shap(raw, P=P, B=B)
        SHAP_SUM = SABS.sum(axis=1)
        SHAP_ENTROPY = shap_entropy_per_planet(SABS)

    # FFT feature bank
    fft_feats = _fft_features(MU, nfft=cfg.nfft, n_freq=cfg.n_freq, detrend=cfg.detrend, hann=cfg.hann, log_power=cfg.log_power)
    features = [fft_feats]
    feat_names: List[str] = [f"fft_{i}" for i in range(fft_feats.shape[1])]

    # Symbolic fingerprints
    rules = _load_rules_json(cfg.rules_path, B=B) if cfg.rules_path else None
    if rules is not None and rules.rule_masks.shape[0] > 0:
        R = len(rules.rule_names)
        masks = np.maximum(rules.rule_masks, 0.0)  # R×B
        mu_rule_mean = (MU @ masks.T) / (np.count_nonzero(masks > 0.0, axis=1)[None, :] + 1e-9)  # P×R
        features.append(mu_rule_mean)
        feat_names += [f"mu_rule_mean::{n}" for n in rules.rule_names]
        if SABS is not None:
            shap_rule_sum = (SABS @ masks.T)  # P×R
            features.append(shap_rule_sum)
            feat_names += [f"shap_rule_sum::{n}" for n in rules.rule_names]

    # Global overlays as features (optional)
    if SHAP_SUM is not None:
        features.append(SHAP_SUM.reshape(-1, 1)); feat_names.append("shap_total")
    if SHAP_ENTROPY is not None:
        features.append(SHAP_ENTROPY.reshape(-1, 1)); feat_names.append("shap_entropy")

    # Symbolic result overlay (per‑planet violation score)
    sym_df = _load_symbolic_results(cfg.symbolic_results_path, planet_ids=planet_ids)
    if sym_df is not None and "violation_score" in sym_df.columns:
        vs = sym_df.set_index("planet_id").reindex(planet_ids)["violation_score"].fillna(0.0).to_numpy()
        features.append(vs.reshape(-1, 1)); feat_names.append("violation_score")

    # Assemble fusion matrix
    X = np.concatenate(features, axis=1).astype(np.float32)
    if cfg.std:
        X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    if cfg.pca_k and cfg.pca_k > 0 and _SK_OK:
        k = min(cfg.pca_k, X.shape[1])
        X = PCA(n_components=k, random_state=cfg.seed).fit_transform(X)

    # Persist fusion features
    feats_csv = cfg.outdir / "fusion_features.csv"
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    df.insert(0, "planet_id", planet_ids)
    # include overlays columns for convenience
    if SHAP_SUM is not None: df["shap_total"] = SHAP_SUM
    if SHAP_ENTROPY is not None: df["shap_entropy"] = SHAP_ENTROPY
    if sym_df is not None and "violation_score" in sym_df.columns:
        df["violation_score"] = df["planet_id"].map(sym_df.set_index("planet_id")["violation_score"].to_dict()).fillna(0.0)
    df.to_csv(feats_csv, index=False)

    # Projections
    html_links: List[str] = []
    coords_registry: Dict[str, str] = {}

    # UMAP
    if cfg.do_umap:
        if not _UMAP_OK or not _SK_OK:
            # Save CSV fallback (no projection)
            umap_csv = cfg.outdir / "umap_coords.csv"
            pd.DataFrame({"planet_id": planet_ids}).to_csv(umap_csv, index=False)
            coords_registry["umap_csv"] = str(umap_csv)
        else:
            reducer = umap.UMAP(
                n_components=2, n_neighbors=cfg.umap_neighbors, min_dist=cfg.umap_min_dist,
                metric=cfg.umap_metric, random_state=cfg.seed
            )
            U = reducer.fit_transform(X)
            umap_csv = cfg.outdir / "umap_coords.csv"
            pd.DataFrame({"planet_id": planet_ids, "x": U[:, 0], "y": U[:, 1]}).to_csv(umap_csv, index=False)
            col = None
            label = ""
            if "violation_score" in df.columns:
                col = df["violation_score"].to_numpy().astype(float); label = "violation_score"
            elif "shap_entropy" in df.columns:
                col = df["shap_entropy"].to_numpy().astype(float); label = "shap_entropy"
            _plot_scatter(U, col, "UMAP — FFT × Symbolic Fusion", cfg.outdir / "umap.png", cfg.outdir / "umap.html", color_label=label)
            coords_registry["umap_csv"] = str(umap_csv)
            html_links += ["umap.html", "umap.png"]

    # t‑SNE
    if cfg.do_tsne:
        if not _SK_OK:
            tsne_csv = cfg.outdir / "tsne_coords.csv"
            pd.DataFrame({"planet_id": planet_ids}).to_csv(tsne_csv, index=False)
            coords_registry["tsne_csv"] = str(tsne_csv)
        else:
            ts = TSNE(
                n_components=2, random_state=cfg.seed, perplexity=cfg.tsne_perplexity,
                n_iter=cfg.tsne_iter, learning_rate="auto", init="pca", metric="euclidean", verbose=0
            ).fit_transform(X)
            tsne_csv = cfg.outdir / "tsne_coords.csv"
            pd.DataFrame({"planet_id": planet_ids, "x": ts[:, 0], "y": ts[:, 1]}).to_csv(tsne_csv, index=False)
            col = None; label = ""
            if "violation_score" in df.columns:
                col = df["violation_score"].to_numpy().astype(float); label = "violation_score"
            elif "shap_entropy" in df.columns:
                col = df["shap_entropy"].to_numpy().astype(float); label = "shap_entropy"
            _plot_scatter(ts, col, "t‑SNE — FFT × Symbolic Fusion", cfg.outdir / "tsne.png", cfg.outdir / "tsne.html", color_label=label)
            coords_registry["tsne_csv"] = str(tsne_csv)
            html_links += ["tsne.html", "tsne.png"]

    # Clustering
    cluster_csv = cfg.outdir / "cluster_assignments.csv"
    centroids_json = cfg.outdir / "kmeans_centroids.json"
    cluster_links = []
    if cfg.kmeans_k and cfg.kmeans_k > 1 and _SK_OK:
        km = KMeans(n_clusters=cfg.kmeans_k, n_init="auto", random_state=cfg.seed)
        km_labels = km.fit_predict(X)
        out = pd.DataFrame({"planet_id": planet_ids, "cluster": km_labels})
        out.to_csv(cluster_csv, index=False)
        with open(centroids_json, "w", encoding="utf-8") as f:
            json.dump({"centroids": km.cluster_centers_.tolist()}, f, indent=2)
        cluster_links += [cluster_csv.name, centroids_json.name]
        if cfg.silhouette and X.shape[0] > cfg.kmeans_k:
            try:
                sil = float(silhouette_score(X, km_labels))
                with open(cfg.outdir / "silhouette.txt", "w", encoding="utf-8") as f:
                    f.write(f"{sil:.6f}\n")
                cluster_links.append("silhouette.txt")
            except Exception:
                pass
    else:
        # minimal stub to simplify downstream scripts
        pd.DataFrame({"planet_id": planet_ids}).to_csv(cluster_csv, index=False)

    # Dashboard
    html_name = cfg.html_name if cfg.html_name.endswith(".html") else "fft_symbolic_fusion.html"
    dash = cfg.outdir / html_name
    links = [
        "fusion_features.csv",
        *html_links,
        cluster_csv.name,
        *(cluster_links[1:] if cluster_links else []),
        "umap_coords.csv" if (cfg.outdir / "umap_coords.csv").exists() else "",
        "tsne_coords.csv" if (cfg.outdir / "tsne_coords.csv").exists() else "",
    ]
    links = [f'<li><a href="{x}" target="_blank" rel="noopener">{x}</a></li>' for x in links if x]
    dash.write_text(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — FFT × Symbolic Fusion</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<style>
  :root {{ --bg:#0b0e14; --fg:#e6edf3; --muted:#9aa4b2; --card:#111827; --border:#2b3240; --brand:#0b5fff; }}
  body {{ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:2rem; line-height:1.5; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.25rem; margin-bottom:1rem; }}
  a {{ color:var(--brand); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
</style>
</head>
<body>
  <header class="card">
    <h1>FFT × Symbolic Fusion — SpectraMind V50</h1>
    <div>Generated: {_now_iso()} • nfft={cfg.nfft} • n_freq={cfg.n_freq} • pca_k={cfg.pca_k} • std={cfg.std}</div>
  </header>
  <section class="card">
    <h2>Artifacts</h2>
    <ul>
      {''.join(links)}
    </ul>
  </section>
  <footer class="card">
    <small>© SpectraMind V50 • generate_fft_symbolic_fusion</small>
  </footer>
</body>
</html>
""", encoding="utf-8")

    # Manifest
    manifest = {
        "tool": "generate_fft_symbolic_fusion",
        "timestamp": _now_iso(),
        "inputs": {
            "mu": str(cfg.mu_path),
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else None,
            "shap": str(cfg.shap_path) if cfg.shap_path else None,
            "rules": str(cfg.rules_path) if cfg.rules_path else None,
            "symbolic_results": str(cfg.symbolic_results_path) if cfg.symbolic_results_path else None,
            "metadata": str(cfg.metadata_path) if cfg.metadata_path else None,
        },
        "params": {
            "nfft": cfg.nfft, "n_freq": cfg.n_freq, "detrend": cfg.detrend, "hann": cfg.hann, "log_power": cfg.log_power,
            "pca_k": cfg.pca_k, "std": cfg.std,
            "umap": cfg.do_umap, "tsne": cfg.do_tsne,
            "umap_neighbors": cfg.umap_neighbors, "umap_min_dist": cfg.umap_min_dist, "umap_metric": cfg.umap_metric,
            "tsne_perplexity": cfg.tsne_perplexity, "tsne_iter": cfg.tsne_iter,
            "kmeans_k": cfg.kmeans_k, "silhouette": cfg.silhouette,
            "seed": cfg.seed
        },
        "shapes": {"P": int(P), "B": int(B), "fusion_dim": int(X.shape[1])},
        "outputs": {
            "fusion_features_csv": str(feats_csv),
            "umap_html": str(cfg.outdir / "umap.html") if (cfg.outdir / "umap.html").exists() else None,
            "tsne_html": str(cfg.outdir / "tsne.html") if (cfg.outdir / "tsne.html").exists() else None,
            "cluster_csv": str(cluster_csv),
            "dashboard_html": str(dash)
        }
    }
    with open(cfg.outdir / "fft_symbolic_fusion_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "mu": str(cfg.mu_path),
        "shap": str(cfg.shap_path) if cfg.shap_path else "",
        "rules": str(cfg.rules_path) if cfg.rules_path else "",
        "symbolic_results": str(cfg.symbolic_results_path) if cfg.symbolic_results_path else "",
        "outdir": str(cfg.outdir),
        "message": f"Fusion complete; dashboard={dash.name}"
    })

    if cfg.open_browser and dash.exists() and _PLOTLY_OK:
        try:
            import webbrowser
            webbrowser.open_new_tab(dash.as_uri())
        except Exception:
            pass

    return 0


# ==============================================================================
# CLI
# ==============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="generate_fft_symbolic_fusion",
        description="Fuse FFT features of μ with symbolic fingerprints (+optional SHAP/violations), project & cluster."
    )
    p.add_argument("--mu", type=Path, required=True, help="μ array (P×B) or (B,) in npy/npz/csv/tsv/parquet/feather")
    p.add_argument("--wavelengths", type=Path, default=None, help="Optional wavelengths vector (B,)")
    p.add_argument("--shap", type=Path, default=None, help="Optional SHAP array: (P×B) or (P×I×B) or (B,)")
    p.add_argument("--symbolic-rules", type=Path, default=None, help="Optional rules JSON with 'rule_masks'")
    p.add_argument("--symbolic-results", type=Path, default=None, help="Optional per‑planet violations JSON (flexible schema)")
    p.add_argument("--metadata", type=Path, default=None, help="Optional metadata with 'planet_id'")

    # FFT feature params
    p.add_argument("--nfft", type=int, default=512, help="FFT size (bins axis)")
    p.add_argument("--n-freq", type=int, default=120, help="# of low‑frequency magnitudes to keep (excl. DC)")
    p.add_argument("--detrend", action="store_true", help="Subtract per‑planet mean before FFT")
    p.add_argument("--hann", action="store_true", help="Apply Hann window before FFT (unit‑energy normalized)")
    p.add_argument("--log-power", action="store_true", help="Use log(1+|FFT|^2) magnitudes")
    p.add_argument("--pca-k", type=int, default=0, help="Optional PCA dim after fusion (0=disabled)")
    p.add_argument("--std", action="store_true", help="Standardize fusion features before PCA")

    # Projection params
    p.add_argument("--umap", dest="do_umap", action="store_true", help="Run UMAP projection (if umap-learn installed)")
    p.add_argument("--tsne", dest="do_tsne", action="store_true", help="Run t‑SNE projection")
    p.add_argument("--umap-neighbors", type=int, default=30, help="UMAP n_neighbors")
    p.add_argument("--umap-min-dist", type=float, default=0.05, help="UMAP min_dist")
    p.add_argument("--umap-metric", type=str, default="euclidean", help="UMAP metric (euclidean, cosine, ...)")
    p.add_argument("--tsne-perplexity", type=float, default=30.0, help="t‑SNE perplexity")
    p.add_argument("--tsne-iter", type=int, default=1000, help="t‑SNE iterations")

    # Clustering
    p.add_argument("--kmeans", dest="kmeans_k", type=int, default=0, help="Run KMeans with k clusters (k>1)")
    p.add_argument("--silhouette", action="store_true", help="Compute silhouette score (if kmeans>1)")

    # General
    p.add_argument("--seed", type=int, default=7, help="Deterministic seed")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")
    p.add_argument("--html-name", type=str, default="fft_symbolic_fusion.html", help="Dashboard filename")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in browser (if Plotly available)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        mu_path=args.mu.resolve(),
        wavelengths_path=args.wavelengths.resolve() if args.wavelengths else None,
        shap_path=args.shap.resolve() if args.shap else None,
        rules_path=args.symbolic_rules.resolve() if args.symbolic_rules else None,
        symbolic_results_path=args.symbolic_results.resolve() if args.symbolic_results else None,
        metadata_path=args.metadata.resolve() if args.metadata else None,
        outdir=args.outdir.resolve(),
        nfft=int(args.nfft),
        n_freq=int(args.n_freq),
        detrend=bool(args.detrend),
        hann=bool(args.hann),
        log_power=bool(args.log_power),
        pca_k=int(args.pca_k),
        std=bool(args.std),
        do_umap=bool(args.do_umap),
        do_tsne=bool(args.do_tsne),
        umap_neighbors=int(args.umap_neighbors),
        umap_min_dist=float(args.umap_min_dist),
        umap_metric=str(args.umap_metric),
        tsne_perplexity=float(args.tsne_perplexity),
        tsne_iter=int(args.tsne_iter),
        kmeans_k=int(args.kmeans_k),
        silhouette=bool(args.silhouette),
        seed=int(args.seed),
        html_name=str(args.html_name),
        open_browser=bool(args.open_browser),
    )

    audit = AuditLogger(
        md_path=Path("logs") / "v50_debug_log.md",
        jsonl_path=Path("logs") / "v50_runs.jsonl"
    )
    audit.log({
        "action": "start",
        "status": "running",
        "mu": str(cfg.mu_path),
        "shap": str(cfg.shap_path) if cfg.shap_path else "",
        "rules": str(cfg.rules_path) if cfg.rules_path else "",
        "symbolic_results": str(cfg.symbolic_results_path) if cfg.symbolic_results_path else "",
        "outdir": str(cfg.outdir),
        "message": "Starting generate_fft_symbolic_fusion"
    })

    # Set seeds (numpy, sklearn uses numpy RNG)
    np.random.seed(cfg.seed)

    try:
        rc = run(cfg, audit)
        return rc
    except Exception as e:
        import traceback
        traceback.print_exc()
        audit.log({
            "action": "run",
            "status": "error",
            "mu": str(cfg.mu_path),
            "outdir": str(cfg.outdir),
            "message": f"{type(e).__name__}: {e}",
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
