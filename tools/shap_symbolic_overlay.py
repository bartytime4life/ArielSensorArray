#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/shap_symbolic_overlay.py

SpectraMind V50 — SHAP × Symbolic Overlay (Ultimate, Challenge‑Grade)

Purpose
-------
Fuse per‑bin SHAP attributions with physics‑informed symbolic rule masks to:
  • quantify per‑planet × per‑rule importance (rule‑weighted |SHAP|, fractions),
  • rank rules per planet and compare against symbolic violation scores (if available),
  • render planet×rule heatmaps and per‑planet μ(λ) overlays shaded by top rules,
  • export authoritative CSV/JSON tables + a compact HTML diagnostics dashboard,
  • maintain append‑only, reproducible run manifests and logs.

Inputs (flexible)
-----------------
• --shap
    |SHAP| per bin with robust shapes:
      - (P, B)       per‑planet per‑bin absolute SHAP  (recommended)
      - (P, I, B)    per‑planet per‑input per‑bin → reduces via sum(|.|, axis=1) → (P,B)
      - (B,)         global per‑bin |SHAP| → broadcast to P planets

• --rules
    JSON with "rule_masks" mapping names → bin selections or weights (length B):
      {
        "rule_masks": {
          "H2O_band_1": [idx, idx, ...] | [0/1/weight, ... len B] | {"bin": weight, ...},
          "CO2_4.3um" : ...
        },
        "rule_groups": { "water": ["H2O_band_1", "H2O_band_2"], ... }   # optional
      }

• --symbolic (optional)
    JSON of symbolic diagnostics; if per‑rule or per‑planet violation scores exist,
    they will be attached for correlation and ranking. Flexible schema supported.

• --mu (optional)
    Predicted μ spectra, shape (P, B) — used for line overlays under shaded rule spans.

• --wavelengths (optional)
    Length‑B wavelength vector (μm/nm/index). Defaults to 0..B‑1.

• --metadata (optional)
    CSV/JSON with 'planet_id'. Synthesized if missing.

Key Outputs
-----------
outdir/
  rule_importance_per_planet.csv          # P×R matrix (long table, with fractions and optional violations)
  rule_leaderboard.csv                    # total rule importance across planets (sum, mean, fraction)
  topk_rules_per_planet.csv               # top‑K rules per planet (with spans & fractions)
  planet_rule_heatmap.png/.html           # heatmap (P×R) of rule importance
  overlay_planet_<id>.png                 # μ(λ) with shaded spans of top rules (first N planets)
  rule_table.json                         # normalized rule stats (coverage, segments, weights)
  shap_symbolic_overlay_manifest.json     # manifest
  run_hash_summary_v50.json               # reproducibility trail (append‑only)
  dashboard.html                          # quick links + preview

Design & Integration
--------------------
• Deterministic (no RNG). No network calls. Plotly/Matplotlib degrade to CSV if unavailable.
• Rules loader is identical in spirit to `symbolic_rule_table.py` for consistency.
• All paths/arrays are validated and padded/truncated safely.
• Append‑only audit logging to logs/v50_debug_log.md and logs/v50_runs.jsonl.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Tabular
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required. Please `pip install pandas`.") from e

# Visualization (optional)
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
# Utilities — time, dirs, hashing, logging
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
        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        # Markdown block
        md = textwrap.dedent(f"""
        ---
        time: {row["timestamp"]}
        tool: shap_symbolic_overlay
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        shap: {row.get("shap","")}
        rules: {row.get("rules","")}
        mu: {row.get("mu","")}
        wavelengths: {row.get("wavelengths","")}
        metadata: {row.get("metadata","")}
        topk: {row.get("topk","")}
        first_n: {row.get("first_n","")}
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
# Generic loaders
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
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


def _load_symbolic_any(path: Optional[Path]) -> Dict[str, Any]:
    """
    Load optional symbolic results JSON. Flexible schema; return raw dict or {}.
    """
    if path is None:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except Exception:
            return {}


# ==============================================================================
# Rules loader & helpers (aligned with symbolic_rule_table.py)
# ==============================================================================

@dataclass
class RuleSet:
    rule_names: List[str]          # length R
    rule_masks: np.ndarray         # R×B (float >=0)
    groups: Dict[str, List[str]]   # optional group name -> list of rules


def _as_mask_vector(spec: Any, B: int) -> np.ndarray:
    """
    Convert a rule mask specification to a length‑B float vector.
      • list of bin indices -> 1.0 at those indices
      • list of 0/1/weight (len B) -> float array
      • dict {bin_idx: weight} -> weight at indices
    """
    mask = np.zeros(B, dtype=float)
    if isinstance(spec, list):
        if len(spec) == B and all(isinstance(x, (int, float, bool, np.floating, np.integer)) for x in spec):
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


def _load_rules_json(path: Path, B: int) -> RuleSet:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "rule_masks" not in obj or not isinstance(obj["rule_masks"], dict):
        raise ValueError("Rules JSON must contain a 'rule_masks' object mapping name→mask")
    names: List[str] = []
    mats: List[np.ndarray] = []
    for name, spec in obj["rule_masks"].items():
        names.append(str(name))
        mats.append(_as_mask_vector(spec, B))
    rule_masks = np.vstack(mats) if mats else np.zeros((0, B), dtype=float)
    groups: Dict[str, List[str]] = {}
    if "rule_groups" in obj and isinstance(obj["rule_groups"], dict):
        for gname, lst in obj["rule_groups"].items():
            groups[str(gname)] = [str(x) for x in lst if str(x) in names]
    return RuleSet(rule_names=names, rule_masks=rule_masks, groups=groups)


def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return [start,end] (inclusive) segments for contiguous mask>0.
    """
    idx = np.where(mask > 0.0)[0]
    if idx.size == 0:
        return []
    segs: List[Tuple[int, int]] = []
    s = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        segs.append((s, prev))
        s, prev = i, i
    segs.append((s, prev))
    return segs


# ==============================================================================
# SHAP alignment & entropy
# ==============================================================================

def _align_bins(arr: np.ndarray, B: int) -> np.ndarray:
    if arr.shape[-1] == B:
        return arr
    out = np.zeros(arr.shape[:-1] + (B,), dtype=float)
    copy = min(B, arr.shape[-1])
    out[..., :copy] = arr[..., :copy]
    return out


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
    a = _align_bins(a, B)
    a = np.abs(a)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def shap_entropy_per_planet(shap_abs: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Shannon entropy of normalized |SHAP| over bins, per planet.
    """
    P, B = shap_abs.shape
    p = shap_abs / (shap_abs.sum(axis=1, keepdims=True) + eps)
    return -np.sum(p * (np.log(p + eps)), axis=1)


# ==============================================================================
# Visualization helpers
# ==============================================================================

def _cluster_colors(k: int) -> List[str]:
    if k <= 1:
        return ["#636EFA"]
    if _PLOTLY_OK:
        base = ["#636EFA","#EF553B","#00CC96","#AB63FA","#FFA15A","#19D3F3",
                "#FF6692","#B6E880","#FF97FF","#FECB52","#1F77B4","#FF7F0E",
                "#2CA02C","#D62728","#9467BD","#8C564B"]
        if k <= len(base):
            return base[:k]
        return [base[i % len(base)] for i in range(k)]
    # simple HSL fallback
    cols = []
    for i in range(k):
        import colorsys
        h = (i / k) % 1.0
        r,g,b = colorsys.hls_to_rgb(h, 0.5, 0.6)
        cols.append("#%02x%02x%02x" % (int(255*r), int(255*g), int(255*b)))
    return cols


def _save_heatmap(Z: np.ndarray, xnames: List[str], ynames: List[str], title: str, out_png: Path, out_html: Path) -> None:
    _ensure_dir(out_png.parent)
    if _PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(z=Z, x=xnames, y=ynames, colorscale="Viridis",
                                        colorbar=dict(title="importance")))
        fig.update_layout(title=title, template="plotly_white",
                          width=max(900, min(1800, 120 + 24*len(xnames))),
                          height=max(600, min(1800, 120 + 18*len(ynames))))
        pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")
    if _MPL_OK:
        plt.figure(figsize=(max(10, 0.25*len(xnames)), max(6, 0.25*len(ynames))))
        vmax = np.percentile(Z, 99.0) if np.any(np.isfinite(Z)) else 1.0
        plt.imshow(Z, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=max(vmax, 1e-12))
        plt.colorbar(label="importance")
        plt.xticks(np.arange(len(xnames)), xnames, rotation=75, ha="right", fontsize=8)
        plt.yticks(np.arange(len(ynames)), ynames, fontsize=8)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
    if not _PLOTLY_OK and not _MPL_OK:
        pd.DataFrame(Z, index=ynames, columns=xnames).to_csv(out_png.with_suffix(".csv"))


def _save_overlay_mu_rules(
    wl: np.ndarray,
    mu_row: Optional[np.ndarray],
    rule_names: List[str],
    rule_masks: np.ndarray,          # R×B
    rule_order: List[int],           # indices of rules to draw, ordered (top→bottom)
    colors: List[str],
    planet_label: str,
    out_png: Path
) -> None:
    """
    Per‑planet μ(λ) line plot with shaded spans for top rules (using mask segments).
    Fallback to CSV if Matplotlib unavailable or μ missing.
    """
    _ensure_dir(out_png.parent)
    if not _MPL_OK or mu_row is None:
        # Save which spans would be drawn
        rows=[]
        for rank, r in enumerate(rule_order, 1):
            mask = rule_masks[r] > 0.0
            idx = np.where(mask)[0]
            if idx.size == 0:
                continue
            # segments
            s = idx[0]; prev = idx[0]
            for i in idx[1:]:
                if i != prev+1:
                    rows.append({"rank":rank,"rule":rule_names[r],"start_bin":int(s),"end_bin":int(prev)})
                    s=i; prev=i; continue
                prev=i
            rows.append({"rank":rank,"rule":rule_names[r],"start_bin":int(s),"end_bin":int(prev)})
        pd.DataFrame(rows).to_csv(out_png.with_suffix(".csv"), index=False)
        return

    plt.figure(figsize=(12, 5))
    plt.plot(wl, mu_row, lw=2.0, color="#0b5fff", label="μ(λ)")

    # Draw shaded spans for each rule in order
    for rank, r in enumerate(rule_order, 1):
        mask = rule_masks[r] > 0.0
        idx = np.where(mask)[0]
        if idx.size == 0:
            continue
        s = idx[0]; prev = idx[0]
        for i in idx[1:]:
            if i != prev + 1:
                a, b = wl[s], wl[prev]
                if b < a: a, b = b, a
                plt.axvspan(a, b, color=colors[rank-1], alpha=0.16, lw=0,
                            label=f"{rule_names[r]}" if rank == 1 else None)
                s, prev = i, i
                continue
            prev = i
        a, b = wl[s], wl[prev]
        if b < a: a, b = b, a
        plt.axvspan(a, b, color=colors[rank-1], alpha=0.16, lw=0)

    plt.title(f"μ(λ) with Top Rule Spans — {planet_label}")
    plt.xlabel("wavelength (index or μm)")
    plt.ylabel("μ (relative)")
    plt.legend(loc="best", fontsize=9, ncol=2)
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# ==============================================================================
# Overlay core
# ==============================================================================

def _rule_importance_from_shap(shap_abs: np.ndarray, rule_masks: np.ndarray, power: float = 1.0, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute per‑planet × per‑rule importance by integrating |SHAP| over rule masks.

    Returns:
      imp:  (P×R) absolute importance = sum_b shap[p,b] * (mask[r,b]^power)
      frac: (P×R) fraction of planet's total |SHAP| mass that lies within rule r
    """
    P, B = shap_abs.shape
    R, B2 = rule_masks.shape
    assert B == B2
    W = np.power(np.maximum(rule_masks, 0.0), power)  # R×B
    # imp[p,r] = sum_b shap[p,b] * W[r,b]
    imp = shap_abs @ W.T  # (P×B) @ (B×R) = P×R
    total = shap_abs.sum(axis=1, keepdims=True) + eps
    frac = imp / total
    imp = np.nan_to_num(imp, nan=0.0, posinf=0.0, neginf=0.0)
    frac = np.nan_to_num(frac, nan=0.0, posinf=0.0, neginf=0.0)
    return imp, frac


def _extract_rule_scores_from_symbolic(symbolic_obj: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """
    Try to extract per‑rule scores from a flexible symbolic JSON.
    Accepts patterns:
      • {"rule_scores": {"RuleA":score,...}}
      • {"rows":[{"rule":"RuleA","score":...}, ...]}
    Returns DataFrame with columns ['rule','score'] or None.
    """
    if not symbolic_obj:
        return None
    if isinstance(symbolic_obj, dict):
        if "rule_scores" in symbolic_obj and isinstance(symbolic_obj["rule_scores"], dict):
            rows = [{"rule": k, "score": float(v)} for k,v in symbolic_obj["rule_scores"].items()
                    if isinstance(v, (int,float))]
            return pd.DataFrame(rows)
        if "rows" in symbolic_obj and isinstance(symbolic_obj["rows"], list):
            rows=[]
            for r in symbolic_obj["rows"]:
                if isinstance(r, dict) and "rule" in r:
                    val = r.get("score", r.get("value", r.get("violation", 0.0)))
                    try:
                        rows.append({"rule": str(r["rule"]), "score": float(val)})
                    except Exception:
                        pass
            return pd.DataFrame(rows) if rows else None
    return None


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclass
class Config:
    shap_path: Path
    rules_path: Path
    symbolic_path: Optional[Path]
    mu_path: Optional[Path]
    wavelengths_path: Optional[Path]
    metadata_path: Optional[Path]
    outdir: Path
    topk: int
    first_n: int
    mask_power: float
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load SHAP (robust shapes) and infer P,B
    shap_raw = _load_array_any(cfg.shap_path)
    P, B = None, None
    if shap_raw.ndim == 1:
        B = shap_raw.shape[0]
        P = P or 1  # will broadcast later after metadata
    elif shap_raw.ndim == 2:
        P, B = shap_raw.shape
    elif shap_raw.ndim == 3:
        P, _, B = shap_raw.shape
    else:
        raise ValueError(f"Unsupported SHAP shape: {shap_raw.shape}")

    # μ optional
    mu = None
    if cfg.mu_path:
        mu = _load_array_any(cfg.mu_path)
        if mu.ndim == 1:
            mu = mu[None, :]
        if P is None and mu is not None:
            P = mu.shape[0]
        if B is None and mu is not None:
            B = mu.shape[1]

    if P is None or B is None:
        raise ValueError("Could not infer (P,B). Provide SHAP and/or μ with clear shape.")

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

    # Metadata & planet IDs
    meta_df = _load_metadata_any(cfg.metadata_path, n_planets=P)
    planet_ids = meta_df["planet_id"].astype(str).tolist()

    # Finalize SHAP matrix
    shap_abs = _prepare_shap(shap_raw, len(planet_ids), B)  # P×B

    # Rules
    rules = _load_rules_json(cfg.rules_path, B)
    rule_names = rules.rule_names
    rule_masks = rules.rule_masks  # R×B
    R = len(rule_names)

    # Compute rule importance from SHAP
    imp, frac = _rule_importance_from_shap(shap_abs, rule_masks, power=float(cfg.mask_power))  # P×R
    # Summaries
    ent = shap_entropy_per_planet(shap_abs)  # P

    # Long table export
    rows=[]
    for p in range(len(planet_ids)):
        for r in range(R):
            rows.append({
                "planet_id": planet_ids[p],
                "rule": rule_names[r],
                "importance": float(imp[p, r]),
                "fraction": float(frac[p, r]),
                "entropy_shap": float(ent[p]),
            })
    long_df = pd.DataFrame(rows)

    # Optional symbolic per‑rule scores (attach for correlation)
    symbolic_obj = _load_symbolic_any(cfg.symbolic_path)
    rule_score_df = _extract_rule_scores_from_symbolic(symbolic_obj)
    if rule_score_df is not None and not rule_score_df.empty:
        # merge on rule (broadcast to all planets)
        long_df = long_df.merge(rule_score_df, on="rule", how="left")
        long_df["score"] = long_df["score"].fillna(0.0)

    long_csv = cfg.outdir / "rule_importance_per_planet.csv"
    long_df.to_csv(long_csv, index=False)

    # Rule leaderboard (total across planets)
    leaderboard = long_df.groupby("rule", as_index=False).agg(
        total_importance=("importance","sum"),
        mean_importance=("importance","mean"),
        mean_fraction=("fraction","mean")
    ).sort_values("total_importance", ascending=False)
    leaderboard_csv = cfg.outdir / "rule_leaderboard.csv"
    leaderboard.to_csv(leaderboard_csv, index=False)

    # Top‑K rules per planet (by importance)
    topk_rows=[]
    K = max(1, int(cfg.topk))
    name_to_idx = {n:i for i,n in enumerate(rule_names)}
    for p, pid in enumerate(planet_ids):
        vals = imp[p]  # length R
        idx = np.argpartition(-vals, kth=min(K-1, R-1))[:K]
        idx = idx[np.argsort(-vals[idx])]
        for rank, r in enumerate(idx, 1):
            # build span string (bin segments)
            segs = _segments(rule_masks[r])
            spans = ";".join([f"{a}-{b}" for a,b in segs]) if segs else ""
            topk_rows.append({
                "planet_id": pid,
                "rank": rank,
                "rule": rule_names[r],
                "importance": float(vals[r]),
                "fraction": float(frac[p, r]),
                "segments_bins": spans
            })
    topk_df = pd.DataFrame(topk_rows)
    topk_csv = cfg.outdir / "topk_rules_per_planet.csv"
    topk_df.to_csv(topk_csv, index=False)

    # Heatmap (P×R)
    heat = imp  # numeric matrix
    heat_png = cfg.outdir / "planet_rule_heatmap.png"
    heat_html = cfg.outdir / "planet_rule_heatmap.html"
    _save_heatmap(heat, xnames=rule_names, ynames=planet_ids, title="Rule Importance (sum of |SHAP| within rule mask)",
                  out_png=heat_png, out_html=heat_html)

    # Per‑planet overlays (first N planets)
    Nshow = max(0, int(cfg.first_n))
    colors = _cluster_colors(max(1, min(K, 12)))
    for p in range(min(len(planet_ids), Nshow)):
        pid = planet_ids[p]
        mu_row = mu[p] if (mu is not None and mu.shape == (len(planet_ids), B)) else None
        # choose top rules by importance
        vals = imp[p]
        idx = np.argsort(-vals)[:min(K, R)]
        _save_overlay_mu_rules(
            wl=wl,
            mu_row=mu_row,
            rule_names=rule_names,
            rule_masks=rule_masks,
            rule_order=list(idx),
            colors=colors,
            planet_label=pid,
            out_png=cfg.outdir / f"overlay_planet_{p:04d}.png"
        )

    # Rule table JSON (coverage, segments, weight stats)
    rule_rows=[]
    for r, name in enumerate(rule_names):
        m = np.maximum(rule_masks[r], 0.0)
        nz = m[m>0.0]
        segs = _segments(m)
        rule_rows.append({
            "rule": name,
            "coverage_count": int((m>0.0).sum()),
            "coverage_fraction": float((m>0.0).mean()),
            "weight_sum": float(m.sum()),
            "weight_min": float(nz.min()) if nz.size else 0.0,
            "weight_mean": float(nz.mean()) if nz.size else 0.0,
            "weight_max": float(nz.max()) if nz.size else 0.0,
            "n_segments": len(segs),
            "segments_bins": ";".join([f"{a}-{b}" for a,b in segs]) if segs else "",
        })
    rule_table_json = cfg.outdir / "rule_table.json"
    with open(rule_table_json, "w", encoding="utf-8") as f:
        json.dump({"rows": rule_rows}, f, indent=2)

    # Dashboard
    dashboard_html = cfg.outdir / (cfg.html_name if cfg.html_name.endswith(".html") else "shap_symbolic_overlay.html")
    preview = topk_df.head(40).to_html(index=False)
    quick_links = textwrap.dedent(f"""
    <ul>
      <li><a href="{long_csv.name}" target="_blank" rel="noopener">{long_csv.name}</a></li>
      <li><a href="{leaderboard_csv.name}" target="_blank" rel="noopener">{leaderboard_csv.name}</a></li>
      <li><a href="{topk_csv.name}" target="_blank" rel="noopener">{topk_csv.name}</a></li>
      <li><a href="planet_rule_heatmap.html" target="_blank" rel="noopener">planet_rule_heatmap.html</a> / <a href="planet_rule_heatmap.png" target="_blank" rel="noopener">PNG</a></li>
      <li><a href="{rule_table_json.name}" target="_blank" rel="noopener">{rule_table_json.name}</a></li>
    </ul>
    """).strip()

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — SHAP × Symbolic Overlay</title>
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
    <h1>SHAP × Symbolic Overlay — SpectraMind V50</h1>
    <div>Generated: <span class="pill">{_now_iso()}</span> • Rules: {R} • Planets: {len(planet_ids)} • Bins: {B}</div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    {quick_links}
  </section>

  <section class="card">
    <h2>Preview — Top‑K Rules per Planet</h2>
    {preview}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 • mask_power={cfg.mask_power} • topk={cfg.topk} • first_n={cfg.first_n}</small>
  </footer>
</body>
</html>
"""
    dashboard_html.write_text(html, encoding="utf-8")

    # Manifest
    manifest = {
        "tool": "shap_symbolic_overlay",
        "timestamp": _now_iso(),
        "inputs": {
            "shap": str(cfg.shap_path),
            "rules": str(cfg.rules_path),
            "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else None,
            "mu": str(cfg.mu_path) if cfg.mu_path else None,
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else None,
            "metadata": str(cfg.metadata_path) if cfg.metadata_path else None,
        },
        "params": {
            "topk": cfg.topk,
            "first_n": cfg.first_n,
            "mask_power": cfg.mask_power,
        },
        "shapes": {
            "P": int(len(planet_ids)),
            "B": int(B),
            "R": int(R),
        },
        "outputs": {
            "long_csv": str(long_csv),
            "leaderboard_csv": str(leaderboard_csv),
            "topk_csv": str(topk_csv),
            "heatmap_png": str(heat_png),
            "heatmap_html": str(heat_html),
            "rule_table_json": str(rule_table_json),
            "dashboard_html": str(dashboard_html),
        }
    }
    with open(cfg.outdir / "shap_symbolic_overlay_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "shap": str(cfg.shap_path),
        "rules": str(cfg.rules_path),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "topk": cfg.topk,
        "first_n": cfg.first_n,
        "outdir": str(cfg.outdir),
        "message": f"Computed rule importance (P={len(planet_ids)}, R={R}, B={B}); dashboard={dashboard_html.name}",
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
        prog="shap_symbolic_overlay",
        description="Fuse |SHAP| with symbolic rule masks; export per‑planet × per‑rule importance and overlays."
    )
    p.add_argument("--shap", type=Path, required=True, help="SHAP array: (P,B) or (P,I,B) or (B,).")
    p.add_argument("--rules", type=Path, required=True, help="Rules JSON with 'rule_masks' and optional 'rule_groups'.")
    p.add_argument("--symbolic", type=Path, default=None, help="Optional symbolic JSON (per‑rule/planet scores).")
    p.add_argument("--mu", type=Path, default=None, help="Optional μ array (P,B) for line overlays.")
    p.add_argument("--wavelengths", type=Path, default=None, help="Optional wavelength vector (B,).")
    p.add_argument("--metadata", type=Path, default=None, help="Optional metadata with 'planet_id' (CSV/JSON/Parquet).")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory for artifacts.")

    p.add_argument("--topk", type=int, default=8, help="Top‑K rules per planet for ranking/overlays.")
    p.add_argument("--first-n", type=int, default=24, help="Render μ overlays for first N planets (0=disable).")
    p.add_argument("--mask-power", type=float, default=1.0, help="Exponent applied to rule mask weights when integrating |SHAP|.")
    p.add_argument("--html-name", type=str, default="shap_symbolic_overlay.html", help="Dashboard HTML filename.")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in default browser.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        shap_path=args.shap.resolve(),
        rules_path=args.rules.resolve(),
        symbolic_path=args.symbolic.resolve() if args.symbolic else None,
        mu_path=args.mu.resolve() if args.mu else None,
        wavelengths_path=args.wavelengths.resolve() if args.wavelengths else None,
        metadata_path=args.metadata.resolve() if args.metadata else None,
        outdir=args.outdir.resolve(),
        topk=int(args.topk),
        first_n=int(args.first_n),
        mask_power=float(args.mask_power),
        html_name=str(args.html_name),
        open_browser=bool(args.open_browser),
    )

    audit = AuditLogger(
        md_path=Path("logs") / "v50_debug_log.md",
        jsonl_path=Path("logs") / "v50_runs.jsonl",
    )
    audit.log({
        "action": "start",
        "status": "running",
        "shap": str(cfg.shap_path),
        "rules": str(cfg.rules_path),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "topk": cfg.topk,
        "first_n": cfg.first_n,
        "outdir": str(cfg.outdir),
        "message": "Starting shap_symbolic_overlay",
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
            "rules": str(cfg.rules_path),
            "outdir": str(cfg.outdir),
            "message": f"{type(e).__name__}: {e}",
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
