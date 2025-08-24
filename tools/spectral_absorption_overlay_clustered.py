#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/spectral_absorption_overlay_clustered.py

SpectraMind V50 — Spectral Absorption Overlay (Clustered) • Ultimate, Challenge‑Grade

Purpose
-------
Cluster spectral bins by their cross‑planet absorption behavior and render an
overlayed visualization that fuses:
  • Mean/median μ(λ) curve(s)
  • Clustered bin regions (color‑coded, contiguous spans)
  • Molecular band overlays (H2O, CO2, CH4, CO, NH3, etc.; configurable)
  • Optional symbolic overlays (per‑bin or per‑planet; flexible schema)
  • Exported cluster tables, band overlaps, HTML dashboard, and PNG figures

Inputs
------
  --mu                : N×B array (n_planets × n_bins) for μ (transmission) OR flux
  --wavelengths       : length‑B array of wavelengths (e.g., microns). If absent, fall back to bin indices.
  --bands-json        : (optional) JSON with molecular band definitions; see format below.
  --symbolic-bins     : (optional) per‑bin JSON/CSV with row per bin (violation, weight, flag). Flexible schema.
  --symbolic-planets  : (optional) per‑planet JSON/CSV (not required; used for cluster stats).
  --units             : "micron" (default) or "index" — used in labels when wavelengths missing.
  --mu-mode           : "transmission" (μ already absorption‑like) or "flux" (convert to absorption as 1‑norm_flux)
  --norm              : "zscore" | "minmax" | "none"  (bin‑wise normalization across planets before clustering)
  --feature           : "raw" | "std" | "entropy" | "fft" | "pca"
  --k                 : number of clusters (k‑means)
  --cluster           : "kmeans" | "agglo"
  --seed              : RNG seed
  --outdir            : output directory
  --html-name         : dashboard HTML filename (default: spectral_absorption_overlay_clustered.html)
  --open-browser      : open dashboard after run (if environment allows)

Molecular bands JSON format (example)
------------------------------------
{
  "units": "micron",
  "bands": {
    "H2O":  [[1.3,1.5], [1.8,2.0]],
    "CO2":  [[2.0,2.1], [4.2,4.4]],
    "CH4":  [[3.2,3.4]],
    "CO":   [[4.5,4.8]],
    "NH3":  [[2.2,2.35]]
  }
}

If omitted, a default pragmatic set of bands is used for Ariel‑like ranges.

Outputs
-------
outdir/
  cluster_assignments.csv                 # bin → cluster_id (+ wavelength, flags, band overlaps)
  cluster_stats.csv                       # size, λ stats, mean μ, band coverage
  band_overlap_matrix.csv                 # clusters × molecules (% of cluster in band)
  mean_mu.png                             # mean μ(λ) with cluster spans + band overlays
  cluster_heatmap.png/.html               # bins × features heatmap (optional Plotly)
  bands_overlay_only.png                  # molecular bands overlay for quick review
  symbolic_bin_overlay.png                # if symbolic-bins provided (importance bar)
  spectral_absorption_overlay_manifest.json
  run_hash_summary_v50.json               # append‑only reproducibility trail
  dashboard.html                          # self-contained quick links + preview table

Design & Integration Notes
--------------------------
• Deterministic seeding, append‑only logs to logs/v50_debug_log.md and logs/v50_runs.jsonl
• No external network calls; optional Plotly/Matplotlib degrade gracefully to CSV
• Molecular overlays align to wavelength vector if present, else bin index domain
• Cluster features are derived from across‑planet statistics per bin (robust & fast)
• Clusters are colored consistently across plots; legend is exported
• Works standalone but designed to mesh with SpectraMind V50 diagnostics dashboard
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import hashlib
import datetime as _dt
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Required for tables/IO
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required. Please `pip install pandas`.") from e

# Optional viz (graceful fallbacks)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

# Optional ML utilities
try:
    from sklearn.cluster import KMeans, AgglomerativeClustering
    from sklearn.decomposition import PCA
except Exception:
    # We only strictly need KMeans/Agglo/PCA for certain options.
    KMeans = None
    AgglomerativeClustering = None
    PCA = None


# ==============================================================================
# Logging & reproducibility utilities
# ==============================================================================

def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditLogger:
    md_path: Path
    jsonl_path: Path

    def log(self, event: Dict[str, Any]) -> None:
        _ensure_dir(self.md_path.parent); _ensure_dir(self.jsonl_path.parent)
        row = dict(event); row.setdefault("timestamp", _now_iso())
        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Markdown
        md = textwrap.dedent(f"""
        ---
        time: {row["timestamp"]}
        tool: spectral_absorption_overlay_clustered
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        mu: {row.get("mu","")}
        wavelengths: {row.get("wavelengths","")}
        bands_json: {row.get("bands_json","")}
        symbolic_bins: {row.get("symbolic_bins","")}
        outdir: {row.get("outdir","")}
        k: {row.get("k","")}
        feature: {row.get("feature","raw")}
        cluster: {row.get("cluster","kmeans")}
        message: {row.get("message","")}
        """).strip() + "\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(md)


def _hash_jsonable(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(b).hexdigest()


def _update_run_hash_summary(outdir: Path, manifest: Dict[str, Any]) -> None:
    path = outdir / "run_hash_summary_v50.json"
    payload = {"runs": []}
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            if not isinstance(payload, dict) or "runs" not in payload:
                payload = {"runs": []}
        except Exception:
            payload = {"runs": []}
    payload["runs"].append({"hash": _hash_jsonable(manifest), "timestamp": _now_iso(), "manifest": manifest})
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ==============================================================================
# Loading helpers
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if s == ".npz":
        z = np.load(path, allow_pickle=False)
        # pick first
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


def _load_wavelengths(path: Optional[Path], B_hint: int) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = _load_array_any(path)
    if arr.ndim == 2:
        vec = np.asarray(arr[:, 0]).reshape(-1)
    else:
        vec = np.asarray(arr).reshape(-1)
    vec = vec.astype(float)
    # align length
    if vec.shape[0] != B_hint:
        out = np.zeros(B_hint, dtype=float)
        copy = min(B_hint, vec.shape[0])
        out[:copy] = vec[:copy]
        return out
    return vec


def _load_bands_json(path: Optional[Path]) -> Dict[str, List[Tuple[float, float]]]:
    """
    Return {"molecule": [(start,end), ...], ...} in wavelength units (e.g., microns).
    If path is None, return a default set of pragmatic bands.
    """
    if path is None:
        # Default Ariel‑like coverage (rough, adjustable)
        return {
            "H2O": [(1.30, 1.55), (1.75, 2.05)],
            "CO2": [(1.95, 2.10), (4.20, 4.45)],
            "CH4": [(3.20, 3.45)],
            "CO":  [(4.50, 4.85)],
            "NH3": [(2.20, 2.38)],
        }
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    bands = obj.get("bands", obj)
    result: Dict[str, List[Tuple[float, float]]] = {}
    for mol, spans in bands.items():
        spans_norm = []
        for pair in spans:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            a, b = float(pair[0]), float(pair[1])
            if b < a:
                a, b = b, a
            spans_norm.append((a, b))
        if spans_norm:
            result[str(mol)] = spans_norm
    return result


def _load_symbolic_bins(path: Optional[Path], B: int) -> Optional[pd.DataFrame]:
    """
    Load optional per‑bin symbolic overlay, flexible schema.
    Returns DataFrame with columns: 'bin', and optionally 'score' (float), 'flag' (str/int)
    """
    if path is None:
        return None
    s = path.suffix.lower()
    if s in {".csv", ".tsv"}:
        df = pd.read_csv(path) if s == ".csv" else pd.read_csv(path, sep="\t")
    elif s == ".json":
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        if isinstance(obj, dict) and "rows" in obj and isinstance(obj["rows"], list):
            df = pd.DataFrame(obj["rows"])
        else:
            df = pd.DataFrame(obj)
    else:
        # try generic table
        arr = _load_array_any(path)
        if arr.ndim == 2 and arr.shape[1] >= 1:
            df = pd.DataFrame(arr, columns=[f"c{i}" for i in range(arr.shape[1])])
            df["bin"] = np.arange(len(df))
        else:
            return None
    # normalize
    if "bin" not in df.columns:
        df = df.reset_index().rename(columns={"index": "bin"})
    df["bin"] = pd.to_numeric(df["bin"], errors="coerce").fillna(-1).astype(int)
    df = df[(df["bin"] >= 0) & (df["bin"] < B)].copy()
    # best‑effort 'score'
    if "score" not in df.columns:
        num = df.select_dtypes(include=[np.number]).copy()
        if "bin" in num.columns:
            num = num.drop(columns=["bin"])
        df["score"] = num.sum(axis=1) if not num.empty else 0.0
    return df[["bin", "score"] + [c for c in df.columns if c not in {"bin", "score"}]].drop_duplicates(subset=["bin"])


# ==============================================================================
# Feature engineering for clustering
# ==============================================================================

def _normalize_binwise(mu: np.ndarray, mode: str) -> np.ndarray:
    """
    Normalize each bin's across‑planet vector per chosen mode.
    mu: N×B
    Returns N×B
    """
    if mode == "none":
        return mu
    if mode == "zscore":
        m = mu.mean(axis=0, keepdims=True)
        s = mu.std(axis=0, keepdims=True) + 1e-12
        return (mu - m) / s
    if mode == "minmax":
        lo = mu.min(axis=0, keepdims=True)
        hi = mu.max(axis=0, keepdims=True)
        return (mu - lo) / (hi - lo + 1e-12)
    raise ValueError(f"Unknown norm mode: {mode}")


def _feature_per_bin(muN: np.ndarray, feature: str) -> np.ndarray:
    """
    Build a per‑bin feature vector from the across‑planet values μ[:, b].
    Return array of shape (B, D_features).
    """
    N, B = muN.shape
    X = np.zeros((B, 1), dtype=float)

    if feature == "raw":
        # mean across planets per bin
        X = muN.mean(axis=0, keepdims=False).reshape(B, 1)
        return X

    if feature == "std":
        X = muN.std(axis=0, keepdims=False).reshape(B, 1)
        return X

    if feature == "entropy":
        # treat normalized μ as probabilities after softmax along planets
        m = muN - muN.max(axis=0, keepdims=True)
        p = np.exp(m)
        p /= p.sum(axis=0, keepdims=True) + 1e-12
        ent = -(p * np.log(p + 1e-12)).sum(axis=0)  # length B
        return ent.reshape(B, 1)

    if feature == "fft":
        # magnitude of first K low‑frequency FFT coeffs of μ[:, b]; N timeseries length across planets
        # choose K adaptively (min(8, N//2))
        K = max(4, min(16, N // 2))
        X = np.zeros((B, K), dtype=float)
        for b in range(B):
            v = muN[:, b]
            f = np.fft.rfft(v)
            mag = np.abs(f)[1:K+1]  # skip DC
            if len(mag) < K:
                pad = np.zeros(K, dtype=float)
                pad[:len(mag)] = mag
                mag = pad
            X[b] = mag
        # log‑scale
        X = np.log1p(X)
        return X

    if feature == "pca":
        # Use principal components across planets per bin.
        # Build (B × N) matrix then project to few dims with PCA.
        if PCA is None:
            raise RuntimeError("scikit‑learn PCA not available. Install scikit‑learn or choose another feature.")
        Z = muN.T  # B×N
        pca = PCA(n_components=min(5, Z.shape[1]))
        X = pca.fit_transform(Z)  # B×d
        return X

    raise ValueError(f"Unknown feature mode: {feature}")


def _cluster_bins(X: np.ndarray, k: int, algo: str, seed: int) -> np.ndarray:
    """
    Cluster rows of X (B×D) into k clusters.
    Returns cluster labels length B in [0..k-1].
    """
    B = X.shape[0]
    if k <= 1:
        return np.zeros(B, dtype=int)
    if algo == "kmeans":
        if KMeans is None:
            raise RuntimeError("scikit‑learn KMeans not available. Install scikit‑learn or choose agglo.")
        km = KMeans(n_clusters=k, random_state=seed, n_init=10, max_iter=300)
        lab = km.fit_predict(X)
        return lab.astype(int)
    if algo == "agglo":
        if AgglomerativeClustering is None:
            raise RuntimeError("scikit‑learn AgglomerativeClustering not available.")
        ac = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lab = ac.fit_predict(X)
        return lab.astype(int)
    raise ValueError(f"Unknown cluster algo: {algo}")


# ==============================================================================
# Visualization helpers
# ==============================================================================

def _pick_cluster_colors(k: int) -> List[str]:
    """
    Return at most k color hex strings. If Plotly available, use a qualitative palette,
    else generate simple HSL‑like hexes.
    """
    if k <= 1:
        return ["#4f46e5"]
    if _PLOTLY_OK:
        # plotly qualitative palette fallback
        base = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3","#FF6692","#B6E880",
                "#FF97FF","#FECB52","#1F77B4","#FF7F0E","#2CA02C","#D62728","#9467BD","#8C564B"]
        if k <= len(base):
            return base[:k]
        # extend by cycling + darkening
        out = []
        for i in range(k):
            out.append(base[i % len(base)])
        return out
    # generate HSL hex
    cols = []
    for i in range(k):
        h = (i / k) % 1.0
        s = 0.65
        l = 0.50
        # convert to RGB
        import colorsys
        r,g,b = colorsys.hls_to_rgb(h, l, s)
        cols.append("#%02x%02x%02x" % (int(255*r), int(255*g), int(255*b)))
    return cols


def _plot_mean_mu_with_overlays(
    wl: np.ndarray,
    mean_mu: np.ndarray,
    clusters: np.ndarray,
    colors: List[str],
    bands: Dict[str, List[Tuple[float,float]]],
    units_label: str,
    out_png: Path
) -> None:
    """
    Line plot of mean μ(λ) with translucent cluster spans and band overlays.
    Fallback to CSV if MPL unavailable.
    """
    if not _MPL_OK:
        # CSV fallback
        pd.DataFrame({"wavelength": wl, "mean_mu": mean_mu, "cluster": clusters}).to_csv(out_png.with_suffix(".csv"), index=False)
        return

    _ensure_dir(out_png.parent)
    plt.figure(figsize=(14, 6))

    # Draw cluster spans (contiguous regions by cluster id)
    B = len(wl)
    for c in np.unique(clusters):
        mask = (clusters == c)
        # find segments
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        s = idx[0]; prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                # segment [s, prev]
                a = wl[s]; b = wl[prev]
                if b < a: a, b = b, a
                plt.axvspan(a, b, color=colors[int(c)], alpha=0.12, lw=0)
                s = i; prev = i; continue
            prev = i
        # last segment
        a = wl[s]; b = wl[prev]
        if b < a: a, b = b, a
        plt.axvspan(a, b, color=colors[int(c)], alpha=0.12, lw=0)

    # Plot mean μ(λ)
    plt.plot(wl, mean_mu, lw=2.0, color="#111827")

    # Band overlays (draw as thicker translucent spans on top)
    for mol, spans in bands.items():
        for (a, b) in spans:
            plt.axvspan(a, b, color="#0b5fff", alpha=0.08, lw=0)

    plt.title("Mean μ with Clustered Absorption Overlays + Molecular Bands")
    plt.xlabel(f"Wavelength ({units_label})")
    plt.ylabel("μ (mean across planets)")
    # Make a legend proxy for clusters
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=colors[i], edgecolor='none', alpha=0.25, label=f"Cluster {i}") for i in np.unique(clusters)]
    handles.append(Patch(facecolor="#0b5fff", edgecolor='none', alpha=0.10, label="Molecular bands"))
    plt.legend(handles=handles, loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_bands_only(wl: np.ndarray, bands: Dict[str,List[Tuple[float,float]]], units_label: str, out_png: Path) -> None:
    if not _MPL_OK:
        # write simple CSV of bands
        rows=[]
        for mol, spans in bands.items():
            for (a,b) in spans:
                rows.append({"molecule":mol,"start":a,"end":b})
        pd.DataFrame(rows).to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    plt.figure(figsize=(14, 2.5))
    ymin, ymax = 0, 1
    for mol, spans in bands.items():
        for (a, b) in spans:
            plt.axvspan(a, b, alpha=0.20, label=mol)
    # build unique legend
    handles = []
    labels_seen = set()
    for mol in bands.keys():
        if mol not in labels_seen:
            labels_seen.add(mol)
            handles.append(plt.Rectangle((0,0),1,1,alpha=0.20,label=mol))
    plt.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5,1.4), ncol=min(6, len(handles)))
    plt.xlim(wl.min(), wl.max())
    plt.yticks([])
    plt.xlabel(f"Wavelength ({units_label})")
    plt.title("Molecular Bands Overlay (reference)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_symbolic_bin_overlay(wl: np.ndarray, sym_bin_df: pd.DataFrame, units_label: str, out_png: Path) -> None:
    if not _MPL_OK:
        sym_bin_df.to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    plt.figure(figsize=(14, 3.5))
    plt.bar(wl, sym_bin_df["score"].to_numpy(), width=(wl.max()-wl.min())/len(wl)*0.9 if len(wl) > 1 else 0.9)
    plt.xlabel(f"Wavelength ({units_label})")
    plt.ylabel("Symbolic Bin Score")
    plt.title("Symbolic Per‑Bin Overlay")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _plot_cluster_heatmap(X: np.ndarray, clusters: np.ndarray, out_png: Path, out_html: Path) -> None:
    """
    Simple bins × features heatmap sorted by cluster id.
    """
    order = np.argsort(clusters)
    Xs = X[order]
    if _PLOTLY_OK:
        _ensure_dir(out_html.parent)
        fig = go.Figure(data=go.Heatmap(z=Xs, colorscale="Viridis", showscale=True))
        fig.update_layout(title="Cluster Feature Heatmap (bins sorted by cluster)", xaxis_title="feature", yaxis_title="bin (sorted)")
        pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")
    if _MPL_OK:
        _ensure_dir(out_png.parent)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 6))
        vmax = np.percentile(Xs, 99) if np.any(np.isfinite(Xs)) else 1.0
        plt.imshow(Xs, aspect="auto", interpolation="nearest", cmap="viridis", vmin=np.nanmin(Xs), vmax=vmax)
        plt.colorbar(label="feature value")
        plt.xlabel("feature")
        plt.ylabel("bin (sorted by cluster)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
    if not _PLOTLY_OK and not _MPL_OK:
        # CSV fallback
        pd.DataFrame(Xs).to_csv(out_png.with_suffix(".csv"), index=False)


# ==============================================================================
# Core pipeline
# ==============================================================================

@dataclass
class Config:
    mu_path: Path
    wavelengths_path: Optional[Path]
    bands_json_path: Optional[Path]
    symbolic_bins_path: Optional[Path]
    units: str
    mu_mode: str
    norm: str
    feature: str
    k: int
    cluster_algo: str
    seed: int
    outdir: Path
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load μ (N×B)
    mu = _load_array_any(cfg.mu_path)
    if mu.ndim == 1:
        mu = mu.reshape(1, -1)
    if mu.ndim != 2:
        raise ValueError(f"--mu must be 2D (N×B). Got {mu.shape}")

    N, B = mu.shape

    # Wavelengths (optional)
    wl = _load_wavelengths(cfg.wavelengths_path, B)
    if wl is None:
        wl = np.arange(B, dtype=float)
        units_label = "bin"
    else:
        units_label = "μm" if cfg.units.lower().startswith("micr") else cfg.units

    # Convert μ to absorption if needed
    mu_work = mu.copy()
    if cfg.mu_mode == "flux":
        # Normalize each planet to [0,1] then absorption = 1 - norm_flux
        lo = mu_work.min(axis=1, keepdims=True)
        hi = mu_work.max(axis=1, keepdims=True)
        norm_flux = (mu_work - lo) / (hi - lo + 1e-12)
        mu_work = 1.0 - norm_flux
    elif cfg.mu_mode == "transmission":
        # μ already "absorption‑like"
        pass
    else:
        raise ValueError("--mu-mode must be 'transmission' or 'flux'")

    # Normalize across planets per bin (optional)
    muN = _normalize_binwise(mu_work, cfg.norm)  # N×B

    # Features per bin
    X = _feature_per_bin(muN, cfg.feature)  # B×D

    # Cluster bins
    clusters = _cluster_bins(X, cfg.k, cfg.cluster_algo, cfg.seed)  # length B
    k_eff = int(np.max(clusters) + 1) if clusters.size else 1
    colors = _pick_cluster_colors(max(k_eff, cfg.k))

    # Stats per cluster
    mean_mu = mu_work.mean(axis=0)          # length B
    median_mu = np.median(mu_work, axis=0)  # optional, not currently plotted
    assignments = pd.DataFrame({
        "bin": np.arange(B, dtype=int),
        "wavelength": wl,
        "cluster": clusters.astype(int),
        "mean_mu": mean_mu
    })

    # Bands
    bands = _load_bands_json(cfg.bands_json_path)

    # Overlap fractions: for each cluster, % of its bins inside each molecule band
    def _in_any_band(lam: float, spans: List[Tuple[float,float]]) -> bool:
        for (a,b) in spans:
            if a <= lam <= b:
                return True
        return False

    # band flags per bin
    band_cols = {}
    for mol, spans in bands.items():
        band_cols[f"in_{mol}"] = np.array([1 if _in_any_band(wl[i], spans) else 0 for i in range(B)], dtype=int)
    for name, col in band_cols.items():
        assignments[name] = col

    # cluster stats
    rows=[]
    for c in range(k_eff):
        sub = assignments[assignments["cluster"] == c]
        if sub.empty:
            rows.append({"cluster": c, "size": 0})
            continue
        row = {
            "cluster": c,
            "size": int(len(sub)),
            "wavelength_min": float(sub["wavelength"].min()),
            "wavelength_max": float(sub["wavelength"].max()),
            "mean_mu_mean": float(sub["mean_mu"].mean()),
            "mean_mu_std": float(sub["mean_mu"].std()),
        }
        for mol in bands.keys():
            row[f"{mol}_frac"] = float(sub[f"in_{mol}"].mean())
        rows.append(row)
    cluster_stats = pd.DataFrame(rows)

    # cluster × molecule overlap matrix
    mol_names = list(bands.keys())
    overlap_mat = np.zeros((k_eff, len(mol_names)), dtype=float)
    for ci in range(k_eff):
        sub = assignments[assignments["cluster"] == ci]
        if len(sub) == 0:
            continue
        for j, mol in enumerate(mol_names):
            overlap_mat[ci, j] = float(sub[f"in_{mol}"].mean())
    overlap_df = pd.DataFrame(overlap_mat, index=[f"cluster_{i}" for i in range(k_eff)], columns=mol_names)

    # Optional symbolic per‑bin overlay
    sym_bin_df = _load_symbolic_bins(cfg.symbolic_bins_path, B)
    if sym_bin_df is not None and not sym_bin_df.empty:
        assignments = assignments.merge(sym_bin_df, on="bin", how="left")
        assignments["score"] = assignments["score"].fillna(0.0)

    # Write tables
    out_assign = cfg.outdir / "cluster_assignments.csv"
    out_stats = cfg.outdir / "cluster_stats.csv"
    out_overlap = cfg.outdir / "band_overlap_matrix.csv"
    assignments.to_csv(out_assign, index=False)
    cluster_stats.to_csv(out_stats, index=False)
    overlap_df.to_csv(out_overlap)

    # Figures
    mean_mu_png = cfg.outdir / "mean_mu.png"
    _plot_mean_mu_with_overlays(wl, mean_mu, clusters, colors, bands, units_label, mean_mu_png)

    bands_png = cfg.outdir / "bands_overlay_only.png"
    _plot_bands_only(wl, bands, units_label, bands_png)

    if sym_bin_df is not None and not sym_bin_df.empty:
        # align symbolic scores to wavelength bins (fill missing as 0)
        sb = pd.DataFrame({"bin": np.arange(B), "score": 0.0})
        sb.loc[sym_bin_df["bin"].values, "score"] = sym_bin_df["score"].values
        sym_overlay_png = cfg.outdir / "symbolic_bin_overlay.png"
        _plot_symbolic_bin_overlay(wl, sb, units_label, sym_overlay_png)

    heatmap_png = cfg.outdir / "cluster_heatmap.png"
    heatmap_html = cfg.outdir / "cluster_heatmap.html"
    _plot_cluster_heatmap(X, clusters, heatmap_png, heatmap_html)

    # Dashboard HTML
    dashboard_html = cfg.outdir / (cfg.html_name if cfg.html_name.endswith(".html") else "spectral_absorption_overlay_clustered.html")
    preview = assignments.head(40).to_html(index=False)
    quick_links = textwrap.dedent(f"""
    <ul>
      <li><a href="{out_assign.name}" target="_blank" rel="noopener">{out_assign.name}</a></li>
      <li><a href="{out_stats.name}" target="_blank" rel="noopener">{out_stats.name}</a></li>
      <li><a href="{out_overlap.name}" target="_blank" rel="noopener">{out_overlap.name}</a></li>
      <li><a href="mean_mu.png" target="_blank" rel="noopener">mean_mu.png</a></li>
      <li><a href="bands_overlay_only.png" target="_blank" rel="noopener">bands_overlay_only.png</a></li>
      <li><a href="{heatmap_html.name if _PLOTLY_OK else heatmap_png.name}" target="_blank" rel="noopener">{heatmap_html.name if _PLOTLY_OK else heatmap_png.name}</a></li>
    </ul>
    """).strip()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — Spectral Absorption Overlay (Clustered)</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<style>
  :root {{ --bg:#0b0e14; --fg:#e6edf3; --muted:#9aa4b2; --card:#111827; --border:#2b3240; --brand:#0b5fff; }}
  body {{ background:var(--bg); color:var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; margin:2rem; line-height:1.5; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.25rem; margin-bottom:1rem; }}
  a {{ color:var(--brand); text-decoration:none; }} a:hover {{ text-decoration:underline; }}
  table {{ border-collapse: collapse; width: 100%; font-size: .95rem; }}
  th, td {{ border:1px solid var(--border); padding:.4rem .5rem; }} th {{ background:#0f172a; }}
  .pill {{ display:inline-block; padding:.15rem .6rem; border-radius:999px; background:#0f172a; border:1px solid var(--border); }}
</style>
</head>
<body>
  <header class="card">
    <h1>Spectral Absorption Overlay — Clustered</h1>
    <div>Generated: <span class="pill">{_now_iso()}</span> • k={k_eff} • feature={cfg.feature} • cluster={cfg.cluster_algo}</div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    {quick_links}
  </section>

  <section class="card">
    <h2>Preview — First 40 Bins</h2>
    {preview}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 • Deterministic seed: {cfg.seed} • norm={cfg.norm} • μ‑mode={cfg.mu_mode}</small>
  </footer>
</body>
</html>
"""
    dashboard_html.write_text(html, encoding="utf-8")

    # Manifest
    manifest = {
        "tool": "spectral_absorption_overlay_clustered",
        "timestamp": _now_iso(),
        "inputs": {
            "mu": str(cfg.mu_path),
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else None,
            "bands_json": str(cfg.bands_json_path) if cfg.bands_json_path else None,
            "symbolic_bins": str(cfg.symbolic_bins_path) if cfg.symbolic_bins_path else None,
        },
        "params": {
            "units": cfg.units,
            "mu_mode": cfg.mu_mode,
            "norm": cfg.norm,
            "feature": cfg.feature,
            "k": cfg.k,
            "cluster_algo": cfg.cluster_algo,
            "seed": cfg.seed,
        },
        "shapes": {"N": int(N), "B": int(B), "features": int(X.shape[1])},
        "outputs": {
            "cluster_assignments_csv": str(out_assign),
            "cluster_stats_csv": str(out_stats),
            "band_overlap_matrix_csv": str(out_overlap),
            "mean_mu_png": str(mean_mu_png if mean_mu_png.exists() else mean_mu_png.with_suffix(".csv")),
            "bands_overlay_only_png": str(bands_png if bands_png.exists() else bands_png.with_suffix(".csv")),
            "heatmap_png": str(heatmap_png if heatmap_png.exists() else heatmap_png.with_suffix(".csv")),
            "heatmap_html": str(heatmap_html) if _PLOTLY_OK else None,
            "dashboard_html": str(dashboard_html),
        }
    }
    with open(cfg.outdir / "spectral_absorption_overlay_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "mu": str(cfg.mu_path),
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "bands_json": str(cfg.bands_json_path) if cfg.bands_json_path else "",
        "symbolic_bins": str(cfg.symbolic_bins_path) if cfg.symbolic_bins_path else "",
        "outdir": str(cfg.outdir),
        "k": cfg.k,
        "feature": cfg.feature,
        "cluster": cfg.cluster_algo,
        "message": f"Clustered B={B} bins into k={k_eff} clusters; dashboard={dashboard_html.name}",
    })

    # Optionally open dashboard
    if cfg.open_browser:
        try:
            import webbrowser
            webbrowser.open_new_tab(dashboard_html.as_uri())
        except Exception:
            pass

    return 0


# ==============================================================================
# CLI
# ==============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spectral_absorption_overlay_clustered",
        description="Cluster spectral bins by absorption behavior and overlay with molecular bands."
    )
    p.add_argument("--mu", type=Path, required=True, help="N×B array (n_planets × n_bins) for μ (transmission) or flux.")
    p.add_argument("--wavelengths", type=Path, default=None, help="Optional wavelengths vector (length B).")
    p.add_argument("--bands-json", type=Path, default=None, help="Optional molecular bands JSON (see tool docs).")
    p.add_argument("--symbolic-bins", type=Path, default=None, help="Optional per‑bin symbolic overlay (JSON/CSV; flexible schema).")
    p.add_argument("--units", type=str, default="micron", choices=["micron","index","nm","um","μm"], help="Display units for wavelength axis.")
    p.add_argument("--mu-mode", type=str, default="transmission", choices=["transmission","flux"], help="Convert μ if given as flux.")
    p.add_argument("--norm", type=str, default="zscore", choices=["zscore","minmax","none"], help="Across‑planet normalization per bin.")
    p.add_argument("--feature", type=str, default="fft", choices=["raw","std","entropy","fft","pca"], help="Per‑bin features for clustering.")
    p.add_argument("--k", type=int, default=8, help="Number of clusters.")
    p.add_argument("--cluster", type=str, default="kmeans", choices=["kmeans","agglo"], help="Clustering algorithm.")
    p.add_argument("--seed", type=int, default=7, help="RNG seed.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    p.add_argument("--html-name", type=str, default="spectral_absorption_overlay_clustered.html", help="Dashboard HTML filename.")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in default browser.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        mu_path=args.mu.resolve(),
        wavelengths_path=args.wavelengths.resolve() if args.wavelengths else None,
        bands_json_path=args.bands_json.resolve() if args.bands_json else None,
        symbolic_bins_path=args.symbolic_bins.resolve() if args.symbolic_bins else None,
        units=str(args.units),
        mu_mode=str(args.mu_mode),
        norm=str(args.norm),
        feature=str(args.feature),
        k=int(args.k),
        cluster_algo=str(args.cluster),
        seed=int(args.seed),
        outdir=args.outdir.resolve(),
        html_name=str(args.html_name),
        open_browser=bool(args.open_browser),
    )

    # Audit logger
    audit = AuditLogger(
        md_path=Path("logs") / "v50_debug_log.md",
        jsonl_path=Path("logs") / "v50_runs.jsonl",
    )
    audit.log({
        "action": "start",
        "status": "running",
        "mu": str(cfg.mu_path),
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "bands_json": str(cfg.bands_json_path) if cfg.bands_json_path else "",
        "symbolic_bins": str(cfg.symbolic_bins_path) if cfg.symbolic_bins_path else "",
        "outdir": str(cfg.outdir),
        "k": cfg.k,
        "feature": cfg.feature,
        "cluster": cfg.cluster_algo,
        "message": "Starting spectral_absorption_overlay_clustered",
    })

    try:
        rc = run(cfg, audit)
        return rc
    except Exception as e:
        # Log error + traceback to stderr for CI visibility
        import traceback
        traceback.print_exc()
        audit.log({
            "action": "run",
            "status": "error",
            "mu": str(cfg.mu_path),
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
            "bands_json": str(cfg.bands_json_path) if cfg.bands_json_path else "",
            "symbolic_bins": str(cfg.symbolic_bins_path) if cfg.symbolic_bins_path else "",
            "outdir": str(cfg.outdir),
            "k": cfg.k,
            "feature": cfg.feature,
            "cluster": cfg.cluster_algo,
            "message": f"{type(e).__name__}: {e}",
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
