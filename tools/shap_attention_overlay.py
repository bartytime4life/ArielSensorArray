#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/shap_attention_overlay.py

SpectraMind V50 — SHAP × Attention Overlay (Ultimate, Challenge‑Grade)

Purpose
-------
Fuse per‑bin SHAP attributions with neural attention weights to localize and
explain spectral regions that most strongly influence model predictions.

This tool ingests SHAP and attention tensors in flexible shapes, reduces them
to aligned per‑planet × per‑bin matrices, applies configurable normalizations,
computes fusion overlays (product, weighted‑sum, geometric mean, rank‑average),
and exports:
  • Per‑planet Top‑K bin tables (with wavelengths),
  • Planet×Bin heatmaps for SHAP, attention, and fusion,
  • Per‑planet μ(λ) line plots with fusion band overlays,
  • A self‑contained HTML mini‑dashboard with quick links,
  • CSV/JSON manifests and a run hash trail for reproducibility.

Input shapes (flexible)
-----------------------
• SHAP:
  - (P, B)                 → per‑planet per‑bin |SHAP|
  - (P, I, B)              → per‑planet per‑input per‑bin → reduced via sum(|.|, axis=1)
  - (B,)                   → a global per‑bin vector, broadcast to P
• Attention:
  - (P, B)                 → per‑planet per‑bin attention
  - (P, H, B)              → per‑planet per‑head per‑bin → reduced via mean(axis=1)
  - (B,)                   → a global per‑bin vector, broadcast to P
• μ (optional):
  - (P, B)                 → predicted spectrum for overlay plots (per planet)
• Wavelengths (optional):
  - (B,)                   → wavelength grid (μm/nm/index)

Notation: P=planets, B=bins, I=input features, H=attention heads.

Outputs
-------
outdir/
  shap_attention_fusion.csv                 # long table (planet_id, bin, λ, shap, attn, fusion)
  topk_bins_per_planet.csv                  # per‑planet top‑K rows
  heatmap_fusion.png/.html                 # planet×bin heatmap (fusion)
  heatmap_shap.png/.html                   # planet×bin heatmap (|SHAP|)
  heatmap_attention.png/.html              # planet×bin heatmap (attention)
  overlay_planet_<id>.png                  # μ(λ) with fusion band (first N planets)
  overlay_distribution.png                  # distributions of SHAP/ATTN/FUSION
  shap_attention_overlay_manifest.json
  run_hash_summary_v50.json                 # append‑only reproducibility log
  dashboard.html                            # quick links + preview

CLI Example
-----------
poetry run python tools/shap_attention_overlay.py \
  --shap outputs/shap/shap_values.npy \
  --attention outputs/attn/decoder_attn.npy \
  --mu outputs/predictions/mu.npy \
  --wavelengths data/wavelengths.npy \
  --metadata data/planet_metadata.csv \
  --fusion product --alpha 1.0 --beta 1.0 \
  --norm-shap zscore --norm-attn minmax \
  --topk 20 --first-n 24 \
  --outdir outputs/shap_attention_overlay --open-browser

Design Notes
------------
• Deterministic: no RNG used; ordering is stable.
• No external network calls. Plotly/Matplotlib degrade gracefully to CSV.
• Robust loaders & shape normalization ensure drop‑in integration.
• Append‑only audit logging to logs/v50_debug_log.md and logs/v50_runs.jsonl.

"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
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
    raise RuntimeError("pandas is required. Please `pip install pandas`.") from e

# Optional visualizations
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
        _ensure_dir(self.md_path.parent)
        _ensure_dir(self.jsonl_path.parent)
        row = dict(event)
        row.setdefault("timestamp", _now_iso())
        # JSONL (machine‑readable)
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Markdown (human‑readable)
        md = textwrap.dedent(f"""
        ---
        time: {row["timestamp"]}
        tool: shap_attention_overlay
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        shap: {row.get("shap","")}
        attention: {row.get("attention","")}
        mu: {row.get("mu","")}
        wavelengths: {row.get("wavelengths","")}
        metadata: {row.get("metadata","")}
        fusion: {row.get("fusion","product")}
        norm_shap: {row.get("norm_shap","none")}
        norm_attn: {row.get("norm_attn","none")}
        outdir: {row.get("outdir","")}
        message: {row.get("message","")}
        """).strip() + "\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(md)


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
# Robust loaders
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
    """
    Load array from .npy/.npz/.csv/.tsv/.parquet/.feather → np.ndarray
    """
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if s == ".npz":
        z = np.load(path, allow_pickle=False)
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
# Shape alignment & reduction
# ==============================================================================

def _align_bins(arr: np.ndarray, B: int, name: str) -> np.ndarray:
    """
    Truncate/pad last dimension to B bins.
    """
    if arr.shape[-1] == B:
        return arr
    out = np.zeros(arr.shape[:-1] + (B,), dtype=float)
    copy = min(B, arr.shape[-1])
    out[..., :copy] = arr[..., :copy]
    return out


def _prepare_shap(shap: np.ndarray, P: int, B: int) -> np.ndarray:
    """
    Normalize SHAP to (P, B) non‑negative (absolute) matrix.
    Accepted shapes: (P,B), (P,I,B), (B,)
    Reduction: sum of abs across inputs if (P,I,B).
    Broadcast per‑bin vector to P if (B,).
    """
    a = np.asarray(shap, dtype=float)
    if a.ndim == 1:
        a = a[None, :]              # 1 × B
        a = np.repeat(a, P, axis=0) # P × B
    elif a.ndim == 2:
        # (P,B) expected
        if a.shape[0] != P and a.shape[1] == P:
            # possibly (B,P) → transpose
            a = a.T
        if a.shape[0] != P:
            # broadcast if can
            if a.shape[0] == 1:
                a = np.repeat(a, P, axis=0)
            else:
                raise ValueError(f"SHAP planets P mismatch; got {a.shape}, expected P={P}")
    elif a.ndim == 3:
        # (P,I,B) → reduce abs sum on I
        if a.shape[0] != P:
            raise ValueError(f"SHAP leading dim != P; got {a.shape}, P={P}")
        a = np.sum(np.abs(a), axis=1)  # P × B
    else:
        raise ValueError(f"Unsupported SHAP shape: {a.shape}")
    a = _align_bins(a, B, "shap")
    a = np.abs(a)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _prepare_attention(attn: np.ndarray, P: int, B: int) -> np.ndarray:
    """
    Normalize attention to (P, B) non‑negative matrix.
    Accepted shapes: (P,B), (P,H,B), (B,)
    Reduction: mean across heads if (P,H,B).
    Broadcast per‑bin vector to P if (B,).
    """
    a = np.asarray(attn, dtype=float)
    if a.ndim == 1:
        a = a[None, :]
        a = np.repeat(a, P, axis=0)
    elif a.ndim == 2:
        if a.shape[0] != P and a.shape[1] == P:
            a = a.T
        if a.shape[0] != P:
            if a.shape[0] == 1:
                a = np.repeat(a, P, axis=0)
            else:
                raise ValueError(f"Attention planets P mismatch; got {a.shape}, expected P={P}")
    elif a.ndim == 3:
        if a.shape[0] != P:
            raise ValueError(f"Attention leading dim != P; got {a.shape}, P={P}")
        a = a.mean(axis=1)  # P × B
    else:
        raise ValueError(f"Unsupported attention shape: {a.shape}")
    a = _align_bins(a, B, "attention")
    a = np.maximum(a, 0.0)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _normalize_rows(X: np.ndarray, mode: str, eps: float = 1e-12) -> np.ndarray:
    """
    Normalize row‑wise (per planet) according to mode: none|zscore|minmax|l1|l2
    """
    if mode == "none":
        return X
    X = X.astype(float)
    if mode == "zscore":
        m = X.mean(axis=1, keepdims=True)
        s = X.std(axis=1, keepdims=True) + eps
        return (X - m) / s
    if mode == "minmax":
        lo = X.min(axis=1, keepdims=True)
        hi = X.max(axis=1, keepdims=True)
        return (X - lo) / (hi - lo + eps)
    if mode == "l1":
        d = np.sum(np.abs(X), axis=1, keepdims=True) + eps
        return X / d
    if mode == "l2":
        d = np.sqrt(np.sum(X * X, axis=1, keepdims=True)) + eps
        return X / d
    raise ValueError(f"Unknown normalization mode: {mode}")


def _rank_normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert each row to ranks in [0,1] (0=lowest,1=highest), ties handled by average rank.
    """
    P, B = X.shape
    out = np.zeros_like(X, dtype=float)
    for p in range(P):
        order = np.argsort(X[p])
        ranks = np.empty(B, dtype=float)
        ranks[order] = np.arange(B, dtype=float)
        out[p] = ranks / max(1.0, B - 1.0)
    return out


# ==============================================================================
# Fusion strategies
# ==============================================================================

def fuse_scores(
    shapM: np.ndarray, attnM: np.ndarray,
    method: str,
    alpha: float, beta: float,
    w_shap: float, w_attn: float,
    use_rank: bool = False
) -> np.ndarray:
    """
    Compute fusion per row (planet) across bins.

    method:
      • product   : (shap^alpha) * (attn^beta)
      • wsum      : w_shap * shap + w_attn * attn
      • geom      : sqrt(shap * attn)
      • rank-avg  : average of rank‑normalized (or rank if use_rank=True)
    """
    S = np.maximum(shapM, 0.0)
    A = np.maximum(attnM, 0.0)

    if use_rank:
        S_r = _rank_normalize_rows(S)
        A_r = _rank_normalize_rows(A)
        S = S_r
        A = A_r

    if method == "product":
        return np.power(S, alpha) * np.power(A, beta)
    if method == "wsum":
        return w_shap * S + w_attn * A
    if method == "geom":
        return np.sqrt(S * A)
    if method == "rank-avg":
        S_r = _rank_normalize_rows(S) if not use_rank else S
        A_r = _rank_normalize_rows(A) if not use_rank else A
        return 0.5 * (S_r + A_r)
    raise ValueError(f"Unknown fusion method: {method}")


# ==============================================================================
# Visualization
# ==============================================================================

def _save_heatmap(Z: np.ndarray, title: str, out_png: Path, out_html: Path) -> None:
    """
    Planet×Bin heatmap. Plotly HTML if available (first‑class), Matplotlib PNG if available, CSV fallback.
    """
    _ensure_dir(out_png.parent)
    if _PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(z=Z, colorscale="Viridis", colorbar=dict(title="value")))
        fig.update_layout(title=title, xaxis_title="bin", yaxis_title="planet index", template="plotly_white")
        pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")
    if _MPL_OK:
        plt.figure(figsize=(12, 6))
        vmax = np.percentile(Z, 99.0) if np.any(np.isfinite(Z)) else 1.0
        plt.imshow(Z, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-12))
        plt.colorbar(label="value")
        plt.xlabel("bin")
        plt.ylabel("planet index")
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
    if not _PLOTLY_OK and not _MPL_OK:
        pd.DataFrame(Z).to_csv(out_png.with_suffix(".csv"), index=False)


def _save_overlay_mu_fusion(
    wl: np.ndarray, mu_row: Optional[np.ndarray], shap_row: np.ndarray, attn_row: np.ndarray,
    fusion_row: np.ndarray, planet_label: str, out_png: Path
) -> None:
    """
    Per‑planet line plot of μ(λ) with shaded fusion band and lines for SHAP/Attention (normalized).
    Fallback to CSV if Matplotlib unavailable or μ missing.
    """
    _ensure_dir(out_png.parent)
    if not _MPL_OK or mu_row is None:
        df = pd.DataFrame({
            "wavelength": wl,
            "mu": mu_row if mu_row is not None else np.zeros_like(wl),
            "shap": shap_row, "attention": attn_row, "fusion": fusion_row
        })
        df.to_csv(out_png.with_suffix(".csv"), index=False)
        return

    # Normalize shap/attn for visualization to [0,1]
    def mm(v):
        v = np.asarray(v, dtype=float)
        lo, hi = np.min(v), np.max(v)
        return (v - lo) / (hi - lo + 1e-12) if hi > lo else np.zeros_like(v)

    S = mm(shap_row)
    A = mm(attn_row)
    F = mm(fusion_row)

    plt.figure(figsize=(12, 5))
    plt.plot(wl, mu_row, lw=2.0, color="#0b5fff", label="μ(λ)")
    plt.fill_between(wl, mu_row, mu_row + 0.15 * (F - 0.5), color="#22c55e", alpha=0.2, label="Fusion band (scaled)")
    plt.plot(wl, mu_row + 0.15 * (S - 0.5), lw=1.2, color="#f59e0b", alpha=0.9, label="SHAP (scaled)")
    plt.plot(wl, mu_row + 0.15 * (A - 0.5), lw=1.2, color="#ef4444", alpha=0.9, label="Attention (scaled)")
    plt.title(f"μ × Fusion Overlay — {planet_label}")
    plt.xlabel("wavelength (index or μm)")
    plt.ylabel("μ (arbitrary scale)")
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _save_distributions(shapM: np.ndarray, attnM: np.ndarray, fusionM: np.ndarray, out_png: Path) -> None:
    if not _MPL_OK:
        pd.DataFrame({
            "shap": shapM.flatten(), "attention": attnM.flatten(), "fusion": fusionM.flatten()
        }).to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1); plt.hist(shapM.flatten(), bins=50); plt.title("|SHAP|"); plt.grid(alpha=0.2)
    plt.subplot(1, 3, 2); plt.hist(attnM.flatten(), bins=50); plt.title("Attention"); plt.grid(alpha=0.2)
    plt.subplot(1, 3, 3); plt.hist(fusionM.flatten(), bins=50); plt.title("Fusion"); plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclass
class Config:
    shap_path: Path
    attention_path: Path
    mu_path: Optional[Path]
    wavelengths_path: Optional[Path]
    metadata_path: Optional[Path]
    outdir: Path
    fusion: str
    alpha: float
    beta: float
    w_shap: float
    w_attn: float
    use_rank: bool
    norm_shap: str
    norm_attn: str
    topk: int
    first_n: int
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load SHAP/Attention
    shap_raw = _load_array_any(cfg.shap_path)
    attn_raw = _load_array_any(cfg.attention_path)

    # μ optional
    mu = _load_array_any(cfg.mu_path) if cfg.mu_path else None
    if mu is not None and mu.ndim == 1:
        mu = mu[None, :]
    # Determine P, B robustly
    P = None
    B = None
    for arr in (mu, shap_raw, attn_raw):
        if arr is None:
            continue
        if arr.ndim == 3:
            P = arr.shape[0] if P is None else P
            B = arr.shape[-1] if B is None else B
        elif arr.ndim == 2:
            P = arr.shape[0] if P is None else P
            B = arr.shape[1] if B is None else B
        elif arr.ndim == 1:
            B = arr.shape[0] if B is None else B
    if P is None or B is None:
        raise ValueError("Unable to infer (P,B). Provide at least one of SHAP/ATTN/μ with unambiguous shape.")

    # Align μ to P×B
    if mu is not None:
        if mu.shape != (P, B):
            # truncate/pad
            out = np.zeros((P, B), dtype=float)
            copyP = min(P, mu.shape[0])
            copyB = min(B, mu.shape[1])
            out[:copyP, :copyB] = mu[:copyP, :copyB]
            mu = out

    # Wavelengths
    wl = None
    if cfg.wavelengths_path:
        arr = _load_array_any(cfg.wavelengths_path)
        wl = arr.reshape(-1).astype(float)
        if wl.shape[0] != B:
            tmp = np.zeros(B, dtype=float)
            tmp[:min(B, wl.shape[0])] = wl[:min(B, wl.shape[0])]
            wl = tmp
    if wl is None:
        wl = np.arange(B, dtype=float)

    # Metadata
    meta_df = _load_metadata_any(cfg.metadata_path, n_planets=P)
    planet_ids = meta_df["planet_id"].astype(str).tolist()

    # Prepare matrices
    shapM = _prepare_shap(shap_raw, P, B)          # P×B
    attnM = _prepare_attention(attn_raw, P, B)     # P×B

    # Row‑wise normalization (optional)
    shapN = _normalize_rows(shapM, cfg.norm_shap)
    attnN = _normalize_rows(attnM, cfg.norm_attn)

    # Fusion
    fusionM = fuse_scores(
        shapN, attnN,
        method=cfg.fusion,
        alpha=cfg.alpha, beta=cfg.beta,
        w_shap=cfg.w_shap, w_attn=cfg.w_attn,
        use_rank=cfg.use_rank
    )
    fusionM = np.nan_to_num(fusionM, nan=0.0, posinf=0.0, neginf=0.0)

    # Long table export
    rows = []
    for p in range(P):
        pid = planet_ids[p] if p < len(planet_ids) else f"planet_{p:04d}"
        for b in range(B):
            rows.append({
                "planet_id": pid,
                "bin": b,
                "wavelength": float(wl[b]),
                "shap": float(shapM[p, b]),
                "attention": float(attnM[p, b]),
                "fusion": float(fusionM[p, b]),
            })
    long_df = pd.DataFrame(rows)
    long_csv = cfg.outdir / "shap_attention_fusion.csv"
    long_df.to_csv(long_csv, index=False)

    # Per‑planet Top‑K
    topk_rows = []
    K = max(1, int(cfg.topk))
    for p in range(P):
        pid = planet_ids[p] if p < len(planet_ids) else f"planet_{p:04d}"
        idx = np.argpartition(-fusionM[p], kth=min(K-1, B-1))[:K]
        # sort for readability
        idx = idx[np.argsort(-fusionM[p, idx])]
        for rank, b in enumerate(idx, 1):
            topk_rows.append({
                "planet_id": pid, "rank": rank, "bin": int(b), "wavelength": float(wl[b]),
                "fusion": float(fusionM[p, b]),
                "shap": float(shapM[p, b]),
                "attention": float(attnM[p, b]),
            })
    topk_df = pd.DataFrame(topk_rows)
    topk_csv = cfg.outdir / "topk_bins_per_planet.csv"
    topk_df.to_csv(topk_csv, index=False)

    # Heatmaps
    _save_heatmap(fusionM, "Fusion (SHAP×Attention)", cfg.outdir / "heatmap_fusion.png", cfg.outdir / "heatmap_fusion.html")
    _save_heatmap(shapM,   "|SHAP|",                 cfg.outdir / "heatmap_shap.png",   cfg.outdir / "heatmap_shap.html")
    _save_heatmap(attnM,   "Attention",              cfg.outdir / "heatmap_attention.png", cfg.outdir / "heatmap_attention.html")

    # Distributions
    _save_distributions(shapM, attnM, fusionM, cfg.outdir / "overlay_distribution.png")

    # Per‑planet overlays (first N)
    Nshow = max(0, int(cfg.first_n))
    for p in range(min(P, Nshow)):
        pid = planet_ids[p] if p < len(planet_ids) else f"planet_{p:04d}"
        mu_row = mu[p] if mu is not None else None
        _save_overlay_mu_fusion(wl, mu_row, shapM[p], attnM[p], fusionM[p], pid, cfg.outdir / f"overlay_planet_{p:04d}.png")

    # Dashboard
    dashboard_html = cfg.outdir / (cfg.html_name if cfg.html_name.endswith(".html") else "shap_attention_overlay.html")
    preview = topk_df.head(40).to_html(index=False)
    quick_links = textwrap.dedent(f"""
    <ul>
      <li><a href="{long_csv.name}" target="_blank" rel="noopener">{long_csv.name}</a></li>
      <li><a href="{topk_csv.name}" target="_blank" rel="noopener">{topk_csv.name}</a></li>
      <li><a href="heatmap_fusion.html" target="_blank" rel="noopener">heatmap_fusion.html</a> / <a href="heatmap_fusion.png" target="_blank" rel="noopener">PNG</a></li>
      <li><a href="heatmap_shap.html" target="_blank" rel="noopener">heatmap_shap.html</a> / <a href="heatmap_shap.png" target="_blank" rel="noopener">PNG</a></li>
      <li><a href="heatmap_attention.html" target="_blank" rel="noopener">heatmap_attention.html</a> / <a href="heatmap_attention.png" target="_blank" rel="noopener">PNG</a></li>
      <li><a href="overlay_distribution.png" target="_blank" rel="noopener">overlay_distribution.png</a></li>
    </ul>
    """).strip()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — SHAP × Attention Overlay</title>
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
    <h1>SHAP × Attention Overlay — SpectraMind V50</h1>
    <div>Generated: <span class="pill">{_now_iso()}</span> • fusion={cfg.fusion} • α={cfg.alpha} β={cfg.beta} • w=({cfg.w_shap},{cfg.w_attn})</div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    {quick_links}
  </section>

  <section class="card">
    <h2>Preview — Top‑K Bins (first 40 rows)</h2>
    {preview}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 • norm(|SHAP|)={cfg.norm_shap} • norm(attn)={cfg.norm_attn} • rank_mode={"on" if cfg.use_rank else "off"}</small>
  </footer>
</body>
</html>
"""
    dashboard_html.write_text(html, encoding="utf-8")

    # Manifest
    manifest = {
        "tool": "shap_attention_overlay",
        "timestamp": _now_iso(),
        "inputs": {
            "shap": str(cfg.shap_path),
            "attention": str(cfg.attention_path),
            "mu": str(cfg.mu_path) if cfg.mu_path else None,
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else None,
            "metadata": str(cfg.metadata_path) if cfg.metadata_path else None,
        },
        "params": {
            "fusion": cfg.fusion, "alpha": cfg.alpha, "beta": cfg.beta,
            "w_shap": cfg.w_shap, "w_attn": cfg.w_attn, "use_rank": cfg.use_rank,
            "norm_shap": cfg.norm_shap, "norm_attn": cfg.norm_attn,
            "topk": cfg.topk, "first_n": cfg.first_n,
        },
        "shapes": {
            "P": int(P), "B": int(B)
        },
        "outputs": {
            "long_csv": str(long_csv),
            "topk_csv": str(topk_csv),
            "heatmap_fusion_png": str(cfg.outdir / "heatmap_fusion.png"),
            "heatmap_fusion_html": str(cfg.outdir / "heatmap_fusion.html"),
            "heatmap_shap_png": str(cfg.outdir / "heatmap_shap.png"),
            "heatmap_shap_html": str(cfg.outdir / "heatmap_shap.html"),
            "heatmap_attention_png": str(cfg.outdir / "heatmap_attention.png"),
            "heatmap_attention_html": str(cfg.outdir / "heatmap_attention.html"),
            "overlay_distribution_png": str(cfg.outdir / "overlay_distribution.png"),
            "dashboard_html": str(dashboard_html),
        }
    }
    with open(cfg.outdir / "shap_attention_overlay_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "shap": str(cfg.shap_path),
        "attention": str(cfg.attention_path),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "fusion": cfg.fusion,
        "norm_shap": cfg.norm_shap,
        "norm_attn": cfg.norm_attn,
        "outdir": str(cfg.outdir),
        "message": f"Computed fusion overlays for P={P}, B={B}; dashboard={dashboard_html.name}",
    })

    # Optionally open dashboard
    if cfg.open_browser and dashboard_html.exists():
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
        prog="shap_attention_overlay",
        description="Fuse SHAP and attention to localize influential spectral bins; export heatmaps & overlays."
    )
    p.add_argument("--shap", type=Path, required=True, help="SHAP array: (P,B) or (P,I,B) or (B,).")
    p.add_argument("--attention", type=Path, required=True, help="Attention array: (P,B) or (P,H,B) or (B,).")
    p.add_argument("--mu", type=Path, default=None, help="Optional μ array (P,B) for line overlays.")
    p.add_argument("--wavelengths", type=Path, default=None, help="Optional wavelengths vector (B,).")
    p.add_argument("--metadata", type=Path, default=None, help="Optional metadata with 'planet_id' (CSV/JSON/Parquet).")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")

    p.add_argument("--fusion", type=str, default="product", choices=["product", "wsum", "geom", "rank-avg"],
                   help="Fusion method.")
    p.add_argument("--alpha", type=float, default=1.0, help="Exponent for SHAP in 'product' fusion.")
    p.add_argument("--beta", type=float, default=1.0, help="Exponent for Attention in 'product' fusion.")
    p.add_argument("--w-shap", type=float, default=0.5, help="Weight for SHAP in 'wsum' fusion.")
    p.add_argument("--w-attn", type=float, default=0.5, help="Weight for Attention in 'wsum' fusion.")
    p.add_argument("--use-rank", action="store_true", help="Rank‑normalize SHAP/ATTN before fusion.")

    p.add_argument("--norm-shap", type=str, default="none", choices=["none", "zscore", "minmax", "l1", "l2"],
                   help="Row‑wise normalization for |SHAP|.")
    p.add_argument("--norm-attn", type=str, default="none", choices=["none", "zscore", "minmax", "l1", "l2"],
                   help="Row‑wise normalization for attention.")

    p.add_argument("--topk", type=int, default=20, help="Top‑K bins per planet to export.")
    p.add_argument("--first-n", type=int, default=24, help="Render μ overlays for first N planets (0=skip).")

    p.add_argument("--html-name", type=str, default="shap_attention_overlay.html", help="Dashboard HTML filename.")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in default browser.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        shap_path=args.shap.resolve(),
        attention_path=args.attention.resolve(),
        mu_path=args.mu.resolve() if args.mu else None,
        wavelengths_path=args.wavelengths.resolve() if args.wavelengths else None,
        metadata_path=args.metadata.resolve() if args.metadata else None,
        outdir=args.outdir.resolve(),
        fusion=str(args.fusion),
        alpha=float(args.alpha),
        beta=float(args.beta),
        w_shap=float(args.w_shap),
        w_attn=float(args.w_attn),
        use_rank=bool(args.use_rank),
        norm_shap=str(args.norm_shap),
        norm_attn=str(args.norm_attn),
        topk=int(args.topk),
        first_n=int(args.first_n),
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
        "shap": str(cfg.shap_path),
        "attention": str(cfg.attention_path),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "fusion": cfg.fusion,
        "norm_shap": cfg.norm_shap,
        "norm_attn": cfg.norm_attn,
        "outdir": str(cfg.outdir),
        "message": "Starting shap_attention_overlay",
    })

    try:
        rc = run(cfg, audit)
        return rc
    except Exception as e:
        import traceback
        traceback.print_exc()
        audit.log({
            "action": "run",
            "status": "error",
            "shap": str(cfg.shap_path),
            "attention": str(cfg.attention_path),
            "outdir": str(cfg.outdir),
            "message": f"{type(e).__name__}: {e}",
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
