#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/symbolic_influence_map.py

SpectraMind V50 — Symbolic Influence Map (Ultimate, Challenge-Grade)

Purpose
-------
Compute neuro‑symbolic *influence maps* that quantify how much each symbolic rule
influences the predicted transmission spectrum μ across spectral bins and planets.
This script supports multiple input schemas and automatically selects the best
available signal for influence:

  Priority of influence signals (highest → lowest):
    1) Provided per‑rule gradient maps:         ∂L_rule/∂μ  (shape: P×R×B)
    2) Provided global gradient maps:           |∂L/∂μ| with rule masks (P×B)
    3) Provided SHAP magnitude:                 |SHAP| with rule masks (P×B)
    4) Proxy from μ magnitude with rule masks:  |μ| with rule masks (P×B)

Here:
  • P = #planets, R = #rules, B = #spectral bins.

Key Features
------------
• Flexible rule definition loader:
   - JSON with "rule_masks": { rule_name: [bin_idx,...] | {bin_idx:weight,...} | [0/1,...] (len B) }
   - Optional "rule_groups" to aggregate multiple rules into a group
• Optional symbolic results (per‑planet violation scores, per‑rule scores)
• Influence reductions:
   - Per‑planet × per‑rule scalar (sum/mean/max/weighted)
   - Full influence heatmaps per rule (planet × bin)
   - Dominant rule per planet (argmax)
• Exports:
   - CSV/JSON: per‑planet × per‑rule scores, dominant rule, rule leaderboard
   - PNG: heatmaps (per‑rule), per‑planet barplots
   - HTML: self‑contained mini dashboard linking all artifacts (Plotly heatmaps if available)
   - Manifest + run hash summary
• Deterministic: consistent RNG for any stochastic downstream options (none by default)
• Audit logging: append to logs/v50_debug_log.md and logs/v50_runs.jsonl

Inputs (any subset; the more you provide, the better):
  --mu                 : npy/npz/csv/parquet/feather, shape P×B
  --grad               : gradients npy/npz, shape either P×B (global) or P×R×B (per‑rule)
  --shap               : |SHAP| npy/npz/csv/parquet/feather, shape P×B
  --rules              : JSON defining rule masks (required unless grad includes P×R×B)
  --symbolic           : JSON with per‑planet violation scores (optional, for overlays/leaderboard)
  --metadata           : CSV/JSON with at least 'planet_id' (optional; auto‑synthesized otherwise)

Outputs:
  outdir/
    symbolic_influence_per_planet_rule.csv
    symbolic_influence_per_planet_rule.json
    dominant_rule_per_planet.csv
    rule_leaderboard.csv
    rule_influence_heatmap_<rule>.png / .html
    planet_rule_barplot_<planet>.png
    symbolic_influence_manifest.json
    run_hash_summary_v50.json (updated/created)

Examples
--------
poetry run python tools/symbolic_influence_map.py \
  --mu outputs/predictions/mu.npy \
  --grad outputs/diagnostics/dL_dmu.npy \
  --rules configs/symbolic_rules.json \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --metadata data/planet_metadata.csv \
  --outdir outputs/symbolic_influence_v50 --open-browser

Notes
-----
• No network calls. All optional deps handled gracefully (Plotly/Matplotlib).
• If --grad is P×R×B, --rules can be omitted (rule names taken from JSON "rule_names"
  if provided alongside gradients manifest; otherwise auto‑named Rule_0..R-1).
• If --grad is P×B (global), or only --shap/--mu provided, --rules is required.
• Rule masks support real weights per bin; if binary, 1→in rule, 0→out of rule.
• Fully "no placeholders": if an optional input is missing, we fallback without failing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---- Optional libraries (graceful degradation) --------------------------------
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required for this tool. Please `pip install pandas`.") from e

# Plotly for interactive heatmaps (optional)
try:
    import plotly.graph_objects as go
    import plotly.io as pio
    _PLOTLY_OK = True
except Exception:
    _PLOTLY_OK = False

# Matplotlib for static PNGs (optional)
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_OK = True
except Exception:
    _MPL_OK = False


# ==============================================================================
# Utilities: paths, hashing, logging, deterministic behavior
# ==============================================================================

def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
        event = dict(event)
        event.setdefault("timestamp", _now_iso())
        # JSONL
        with open(self.jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        # Markdown block
        md = textwrap.dedent(f"""
        ---
        time: {event["timestamp"]}
        tool: symbolic_influence_map
        action: {event.get("action","run")}
        status: {event.get("status","ok")}
        outdir: {event.get("outdir","")}
        mu: {event.get("mu","")}
        grad: {event.get("grad","")}
        shap: {event.get("shap","")}
        rules: {event.get("rules","")}
        symbolic: {event.get("symbolic","")}
        metadata: {event.get("metadata","")}
        reduction: {event.get("reduction","sum")}
        message: {event.get("message","")}
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
# Loading helpers: arrays, metadata, rules, symbolic overlays
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
    """
    Load an array from npy/npz/csv/tsv/parquet/feather. Returns np.ndarray.
    """
    s = path.suffix.lower()
    if s == ".npy":
        arr = np.load(path, allow_pickle=False)
        return np.asarray(arr)
    if s == ".npz":
        z = np.load(path, allow_pickle=False)
        # choose first array-ish entry
        for k in z.files:
            return np.asarray(z[k])
        raise ValueError(f"No arrays found in {path}")
    if s in {".csv", ".tsv"}:
        df = pd.read_csv(path) if s == ".csv" else pd.read_csv(path, sep="\t")
        return df.to_numpy()
    if s in {".parquet"}:
        return pd.read_parquet(path).to_numpy()
    if s in {".feather"}:
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


def _load_symbolic_any(path: Optional[Path], planet_ids: List[str]) -> pd.DataFrame:
    """
    Load optional per‑planet symbolic overlays (violation scores, per‑rule if present).
    Returns tidy DataFrame with at least ['planet_id','violation_score'] when possible.
    """
    if path is None:
        return pd.DataFrame({"planet_id": planet_ids, "violation_score": np.zeros(len(planet_ids))})

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "planets" in obj:
            # Allow dict mapping or list
            if isinstance(obj["planets"], dict):
                for pid, payload in obj["planets"].items():
                    rec = {"planet_id": str(pid)}
                    if isinstance(payload, dict):
                        rec.update({k: payload.get(k) for k in payload.keys()})
                    else:
                        rec["violation_score"] = float(payload) if isinstance(payload, (int, float)) else 0.0
                    rows.append(rec)
            elif isinstance(obj["planets"], list):
                for row in obj["planets"]:
                    if isinstance(row, dict) and "planet_id" in row:
                        rows.append(row)
        elif "rows" in obj and isinstance(obj["rows"], list):
            for row in obj["rows"]:
                if isinstance(row, dict) and "planet_id" in row:
                    rows.append(row)
    elif isinstance(obj, list):
        for row in obj:
            if isinstance(row, dict) and "planet_id" in row:
                rows.append(row)

    if not rows:
        return pd.DataFrame({"planet_id": planet_ids, "violation_score": np.zeros(len(planet_ids))})
    df = pd.DataFrame(rows).drop_duplicates(subset=["planet_id"])
    if "violation_score" not in df.columns:
        # heuristic sum over numeric fields
        num = df.select_dtypes(include=[np.number]).fillna(0.0)
        df["violation_score"] = num.sum(axis=1) if not num.empty else 0.0
    # align to planet_ids order
    df = df.set_index("planet_id").reindex(planet_ids).reset_index()
    df["violation_score"] = pd.to_numeric(df["violation_score"], errors="coerce").fillna(0.0)
    return df


# ==============================================================================
# Rules loader & normalization
# ==============================================================================

@dataclass
class RuleSet:
    rule_names: List[str]          # length R
    rule_masks: np.ndarray         # shape R×B (float weights allowed, 0 means excluded)
    groups: Dict[str, List[str]]   # optional named groups mapping to subsets of rules


def _as_mask_vector(spec: Any, B: int) -> np.ndarray:
    """
    Convert a rule mask specification to a length‑B float vector.
      • list of bin indices -> 1.0 at those indices
      • list of 0/1 (len B) -> float array
      • dict {bin_idx: weight} -> weight at indices
    """
    mask = np.zeros(B, dtype=float)
    if isinstance(spec, list):
        # Either indices or length‑B
        if len(spec) == B and all(isinstance(x, (int, float, bool)) for x in spec):
            arr = np.array(spec, dtype=float)
            mask = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            for x in spec:
                if isinstance(x, (int, np.integer)) and 0 <= int(x) < B:
                    mask[int(x)] = 1.0
    elif isinstance(spec, dict):
        for k, v in spec.items():
            try:
                idx = int(k)
                if 0 <= idx < B:
                    mask[idx] = float(v)
            except Exception:
                continue
    else:
        raise ValueError("Unsupported rule mask spec; must be list or dict.")
    return mask


def _load_rules_json(path: Path, B: int) -> RuleSet:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Expected main section: "rule_masks"
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


# ==============================================================================
# Influence computation
# ==============================================================================

@dataclass
class InfluenceInputs:
    mu: Optional[np.ndarray]                 # P×B
    grad: Optional[np.ndarray]               # P×B or P×R×B
    shap: Optional[np.ndarray]               # P×B (|SHAP|)
    rules: Optional[RuleSet]                 # required if grad is P×B or if only shap/mu present
    planet_ids: List[str]
    reduction: str                           # 'sum' | 'mean' | 'max' | 'weighted'
    weight_power: float                      # power on mask weights for 'weighted' reduction
    eps: float = 1e-12


def _infer_source(inputs: InfluenceInputs) -> str:
    """
    Decide which signal to use for influence computation.
    """
    if inputs.grad is not None and inputs.grad.ndim == 3:
        return "grad_rule"     # P×R×B (ideal)
    if inputs.grad is not None and inputs.grad.ndim == 2:
        if inputs.rules is None:
            raise ValueError("Global gradient (P×B) provided but --rules missing; cannot split by rule.")
        return "grad_global"   # P×B + masks
    if inputs.shap is not None:
        if inputs.rules is None:
            raise ValueError("|SHAP| provided but --rules missing; cannot split by rule.")
        return "shap"
    if inputs.mu is not None:
        if inputs.rules is None:
            raise ValueError("μ provided but --rules missing; cannot split by rule.")
        return "mu_proxy"
    raise ValueError("No valid inputs for influence. Provide --grad (P×R×B or P×B) or --shap or --mu, with --rules when needed.")


def _reduce_vector(v: np.ndarray, w: Optional[np.ndarray], reduction: str, power: float, eps: float) -> float:
    """
    Reduce a length‑B influence vector v (>=0) to scalar score using:
      'sum'      → sum(v)
      'mean'     → mean(v)
      'max'      → max(v)
      'weighted' → sum( (w^power) * v ) / sum(w^power)   where w is mask weights or ones
    """
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if reduction == "sum":
        return float(v.sum())
    if reduction == "mean":
        return float(v.mean()) if v.size else 0.0
    if reduction == "max":
        return float(v.max()) if v.size else 0.0
    if reduction == "weighted":
        if w is None:
            return float(v.sum())
        w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
        ww = np.power(np.maximum(w, 0.0), power)
        denom = float(ww.sum()) + eps
        return float((ww * v).sum() / denom)
    raise ValueError(f"Unknown reduction: {reduction}")


def compute_influence(inputs: InfluenceInputs) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute per‑planet × per‑rule scalar influence scores (+ per‑rule heatmaps [planet×bin]).

    Returns:
      df_scores: DataFrame with index=planet_id and columns=[rule_names...] (scalar influence per rule)
      heatmaps:  dict rule_name → ndarray (P×B) with non‑negative influences per bin
    """
    source = _infer_source(inputs)
    P = len(inputs.planet_ids)
    B = inputs.mu.shape[1] if inputs.mu is not None else (
        inputs.shap.shape[1] if inputs.shap is not None else (
            inputs.grad.shape[-1] if inputs.grad is not None else None
        )
    )
    if B is None:
        raise ValueError("Cannot determine #bins (B). Provide at least one of --mu/--shap/--grad.")

    if source == "grad_rule":
        # grad: P×R×B
        grad = np.asarray(inputs.grad, dtype=float)
        if grad.ndim != 3 or grad.shape[0] != P or grad.shape[2] != B:
            raise ValueError(f"grad shape mismatch; expected P×R×B with P={P}, B={B}, got {grad.shape}")
        R = grad.shape[1]
        # Derive rule names/masks if missing
        if inputs.rules is None:
            rule_names = [f"Rule_{i}" for i in range(R)]
            rule_masks = np.ones((R, B), dtype=float)  # implicit coverage (no masking)
        else:
            rule_names = inputs.rules.rule_names
            if len(rule_names) != R:
                # tolerate mismatch by truncation/pad
                if len(rule_names) < R:
                    rule_names += [f"Rule_{i}" for i in range(len(rule_names), R)]
                else:
                    rule_names = rule_names[:R]
            rule_masks = inputs.rules.rule_masks
            if rule_masks.shape != (R, B):
                # resample/truncate/pad to B
                fixed = np.zeros((R, B), dtype=float)
                rB = rule_masks.shape[1]
                copyB = min(B, rB)
                fixed[:, :copyB] = rule_masks[:, :copyB]
                rule_masks = fixed

        heatmaps: Dict[str, np.ndarray] = {}
        data = {}
        abs_grad = np.abs(grad)  # P×R×B

        for r in range(len(rule_names)):
            name = rule_names[r]
            g = abs_grad[:, r, :]  # P×B
            # (optional) apply mask weights (>=0)
            if inputs.rules is not None:
                w = np.maximum(rule_masks[r], 0.0)  # B
                heat = g * w[None, :]
            else:
                heat = g
            heatmaps[name] = heat

            # reduce to scalar per planet
            scores = []
            for p in range(P):
                w = rule_masks[r] if inputs.rules is not None else None
                scores.append(_reduce_vector(heat[p], w, inputs.reduction, inputs.weight_power, inputs.eps))
            data[name] = scores

        df = pd.DataFrame(data, index=inputs.planet_ids)
        return df, heatmaps

    # From here, we must have rules
    assert inputs.rules is not None, "rules must be provided for non per‑rule gradient sources"
    rule_names = inputs.rules.rule_names
    rule_masks = np.maximum(inputs.rules.rule_masks, 0.0)  # R×B
    R = len(rule_names)

    if source == "grad_global":
        # grad: P×B → split by rule masks
        g = np.asarray(inputs.grad, dtype=float)
        if g.ndim != 2 or g.shape != (P, B):
            raise ValueError(f"Global grad must be P×B = ({P},{B}), got {g.shape}")
        abs_g = np.abs(g)  # P×B
        heatmaps = {}
        data = {}
        for r, name in enumerate(rule_names):
            w = rule_masks[r]  # B
            heat = abs_g * w[None, :]
            heatmaps[name] = heat
            data[name] = [_reduce_vector(heat[p], w, inputs.reduction, inputs.weight_power, inputs.eps) for p in range(P)]
        df = pd.DataFrame(data, index=inputs.planet_ids)
        return df, heatmaps

    if source == "shap":
        # shap: P×B (expected absolute magnitudes)
        s = np.asarray(inputs.shap, dtype=float)
        if s.ndim != 2 or s.shape != (P, B):
            raise ValueError(f"|SHAP| must be P×B = ({P},{B}), got {s.shape}")
        s = np.abs(s)
        heatmaps = {}
        data = {}
        for r, name in enumerate(rule_names):
            w = rule_masks[r]
            heat = s * w[None, :]
            heatmaps[name] = heat
            data[name] = [_reduce_vector(heat[p], w, inputs.reduction, inputs.weight_power, inputs.eps) for p in range(P)]
        df = pd.DataFrame(data, index=inputs.planet_ids)
        return df, heatmaps

    if source == "mu_proxy":
        # μ proxy: |μ| as saliency approximation
        mu = np.asarray(inputs.mu, dtype=float)
        if mu.ndim != 2 or mu.shape != (P, B):
            raise ValueError(f"μ must be P×B = ({P},{B}), got {mu.shape}")
        a = np.abs(mu)
        heatmaps = {}
        data = {}
        for r, name in enumerate(rule_names):
            w = rule_masks[r]
            heat = a * w[None, :]
            heatmaps[name] = heat
            data[name] = [_reduce_vector(heat[p], w, inputs.reduction, inputs.weight_power, inputs.eps) for p in range(P)]
        df = pd.DataFrame(data, index=inputs.planet_ids)
        return df, heatmaps

    raise RuntimeError("Unreachable source selection.")


# ==============================================================================
# Visualization & Dashboard
# ==============================================================================

def _save_heatmap_png(heat: np.ndarray, title: str, out_png: Path) -> None:
    """
    Save planet×bin heatmap (rows=planets, cols=bins).
    """
    if not _MPL_OK:
        # Fallback to CSV
        pd.DataFrame(heat).to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    plt.figure(figsize=(12, 6))
    vmax = np.percentile(heat, 99.0) if np.any(np.isfinite(heat)) else 1.0
    plt.imshow(heat, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0.0, vmax=vmax if vmax > 0 else None)
    plt.colorbar(label="Influence")
    plt.xlabel("Spectral bin")
    plt.ylabel("Planet index")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _save_heatmap_html(heat: np.ndarray, title: str, out_html: Path) -> None:
    if not _PLOTLY_OK:
        # Fallback CSV
        pd.DataFrame(heat).to_csv(out_html.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_html.parent)
    fig = go.Figure(data=go.Heatmap(z=heat, colorscale="Viridis", colorbar=dict(title="Influence")))
    fig.update_layout(
        title=title, xaxis_title="Spectral bin", yaxis_title="Planet index",
        template="plotly_white", width=1000, height=600
    )
    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")


def _save_planet_rule_barplot(scores: pd.Series, planet_id: str, out_png: Path) -> None:
    if not _MPL_OK:
        # Fallback CSV (single row)
        scores.to_frame(name="influence").T.to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    vals = scores.values
    names = list(scores.index)
    order = np.argsort(-vals)
    vals = vals[order]
    names = [names[i] for i in order]

    plt.figure(figsize=(max(12, len(names)*0.3), 6))
    plt.bar(np.arange(len(names)), vals)
    plt.xticks(np.arange(len(names)), names, rotation=75, ha="right")
    plt.ylabel("Influence")
    plt.title(f"Symbolic Rule Influence — {planet_id}")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _write_dashboard_html(out_html: Path, rule_files: List[Path], table_head_html: str, resources_html: str) -> None:
    _ensure_dir(out_html.parent)
    # Build list of links
    links = "\n".join(
        f'<li><a href="{p.name}" target="_blank" rel="noopener">{p.name}</a></li>'
        for p in rule_files
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — Symbolic Influence Map</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<meta name="color-scheme" content="light dark" />
<style>
  :root {{
    --bg:#0b0e14; --fg:#e6edf3; --muted:#9aa4b2; --card:#111827; --border:#2b3240; --brand:#0b5fff;
  }}
  body {{
    background: var(--bg); color: var(--fg); font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    margin: 2rem; line-height: 1.5;
  }}
  .card {{ background: var(--card); border:1px solid var(--border); border-radius:14px; padding:1rem 1.25rem; margin-bottom:1rem; }}
  a {{ color: var(--brand); text-decoration: none; }}
  a:hover {{ text-decoration: underline; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 0.95rem; }}
  th, td {{ border: 1px solid var(--border); padding: 0.4rem 0.5rem; }}
  th {{ background: #0f172a; }}
</style>
</head>
<body>
  <header class="card">
    <h1>Symbolic Influence Map — SpectraMind V50</h1>
    <div>Generated: {_now_iso()}</div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    {resources_html}
  </section>

  <section class="card">
    <h2>Rule Heatmaps (interactive)</h2>
    <ul>
      {links}
    </ul>
    <p>Heatmaps show per‑bin influence intensity across all planets for each rule.</p>
  </section>

  <section class="card">
    <h2>Preview — First 30 rows</h2>
    {table_head_html}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 • Symbolic Influence Map</small>
  </footer>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


# ==============================================================================
# Main orchestration
# ==============================================================================

@dataclass
class Config:
    mu_path: Optional[Path]
    grad_path: Optional[Path]
    shap_path: Optional[Path]
    rules_path: Optional[Path]
    symbolic_path: Optional[Path]
    metadata_path: Optional[Path]
    outdir: Path
    reduction: str
    weight_power: float
    top_planet_barplots: int
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # --------- Load arrays ----------
    mu = _load_array_any(cfg.mu_path) if cfg.mu_path else None
    shap = _load_array_any(cfg.shap_path) if cfg.shap_path else None
    grad = _load_array_any(cfg.grad_path) if cfg.grad_path else None

    # Normalize shapes and infer P,B
    def _shape(arr: Optional[np.ndarray]) -> Optional[Tuple[int, ...]]:
        return None if arr is None else tuple(arr.shape)

    if grad is not None and grad.ndim not in (2, 3):
        raise ValueError(f"--grad must be P×B or P×R×B; got shape {grad.shape}")

    # Determine P,B robustly
    P = None
    B = None
    for arr in (mu, shap, grad):
        if arr is None:
            continue
        if arr.ndim == 3:
            P = arr.shape[0] if P is None else P
            B = arr.shape[2] if B is None else B
        elif arr.ndim == 2:
            P = arr.shape[0] if P is None else P
            B = arr.shape[1] if B is None else B
    if P is None or B is None:
        raise ValueError("Cannot infer (P,B). Provide at least one array among --mu/--shap/--grad.")

    # Metadata & planet ordering
    meta_df = _load_metadata_any(cfg.metadata_path, P)
    planet_ids = meta_df["planet_id"].astype(str).tolist()
    # Align other arrays to P if they have mismatched length (truncate/pad)
    def _align(arr: Optional[np.ndarray], kind: str) -> Optional[np.ndarray]:
        if arr is None:
            return None
        if arr.ndim == 2:
            aP, aB = arr.shape
            if aB != B:
                raise ValueError(f"{kind} B mismatch: expected {B}, got {aB}")
            if aP == P:
                return arr
            out = np.zeros((P, B), dtype=float)
            copy = min(P, aP)
            out[:copy, :B] = arr[:copy, :B]
            return out
        if arr.ndim == 3:
            aP, aR, aB = arr.shape
            if aB != B:
                raise ValueError(f"{kind} B mismatch: expected {B}, got {aB}")
            out = np.zeros((P, aR, B), dtype=float)
            copy = min(P, aP)
            out[:copy, :, :B] = arr[:copy, :, :B]
            return out
        return arr

    mu = _align(mu, "μ")
    shap = _align(shap, "|SHAP|")
    grad = _align(grad, "grad")

    # Rules
    rules: Optional[RuleSet] = None
    if grad is not None and grad.ndim == 3 and cfg.rules_path is None:
        # Optional companion names via JSON next to grad (convention: same stem + .json with "rule_names")
        # If unavailable, auto‑name Rule_0..R-1 and implicit full coverage masks.
        R = grad.shape[1]
        rule_names = [f"Rule_{i}" for i in range(R)]
        mask = np.ones((R, B), dtype=float)
        rules = RuleSet(rule_names=rule_names, rule_masks=mask, groups={})
    else:
        if cfg.rules_path is None:
            raise ValueError("--rules is required unless --grad is P×R×B.")
        rules = _load_rules_json(cfg.rules_path, B)

    # Symbolic overlays (optional)
    sym_df = _load_symbolic_any(cfg.symbolic_path, planet_ids).set_index("planet_id")

    # Compute influence maps
    infl_inputs = InfluenceInputs(
        mu=mu,
        grad=grad,
        shap=shap,
        rules=rules,
        planet_ids=planet_ids,
        reduction=cfg.reduction,
        weight_power=float(cfg.weight_power),
    )
    df_scores, heatmaps = compute_influence(infl_inputs)  # df index=planet_ids, columns=rule_names

    # Augment with symbolic violation scores if present
    if "violation_score" in sym_df.columns:
        df_scores["violation_score"] = sym_df.reindex(df_scores.index)["violation_score"].fillna(0.0)

    # Dominant rule per planet
    rule_cols = [c for c in df_scores.columns if c not in {"violation_score"}]
    dom_idx = np.argmax(df_scores[rule_cols].to_numpy(), axis=1) if rule_cols else np.zeros(len(df_scores), dtype=int)
    dom_rule = [rule_cols[i] if rule_cols else "N/A" for i in dom_idx]
    dominant_df = pd.DataFrame({"planet_id": df_scores.index, "dominant_rule": dom_rule})
    # Rule leaderboard (sum over planets)
    leaderboard = df_scores[rule_cols].sum(axis=0).sort_values(ascending=False).rename("total_influence").to_frame()

    # Write CSV/JSON
    out_scores_csv = cfg.outdir / "symbolic_influence_per_planet_rule.csv"
    out_scores_json = cfg.outdir / "symbolic_influence_per_planet_rule.json"
    df_scores.to_csv(out_scores_csv, index=True)
    with open(out_scores_json, "w", encoding="utf-8") as f:
        json.dump({"rows": df_scores.reset_index().rename(columns={"index": "planet_id"}).to_dict(orient="records")}, f, indent=2)

    dominant_csv = cfg.outdir / "dominant_rule_per_planet.csv"
    leaderboard_csv = cfg.outdir / "rule_leaderboard.csv"
    dominant_df.to_csv(dominant_csv, index=False)
    leaderboard.to_csv(leaderboard_csv)

    # Heatmaps per rule
    heatmap_html_files: List[Path] = []
    for name, heat in heatmaps.items():
        # Save PNG and HTML
        _save_heatmap_png(heat, f"Influence Heatmap — {name}", cfg.outdir / f"rule_influence_heatmap_{_safe_name(name)}.png")
        out_html = cfg.outdir / f"rule_influence_heatmap_{_safe_name(name)}.html"
        _save_heatmap_html(heat, f"Influence Heatmap — {name}", out_html)
        heatmap_html_files.append(out_html)

    # Optional: per‑planet barplots for top N planets by total influence
    if cfg.top_planet_barplots > 0 and _MPL_OK:
        totals = df_scores[rule_cols].sum(axis=1) if rule_cols else pd.Series(0.0, index=df_scores.index)
        top_planets = list(totals.sort_values(ascending=False).head(cfg.top_planet_barplots).index)
        for pid in top_planets:
            scores = df_scores.loc[pid, rule_cols]
            _save_planet_rule_barplot(scores, pid, cfg.outdir / f"planet_rule_barplot_{_safe_name(pid)}.png")

    # Dashboard HTML
    head_html = df_scores.reset_index().head(30).to_html(index=False)
    resources = textwrap.dedent(f"""
    <ul>
      <li><a href="{out_scores_csv.name}" target="_blank" rel="noopener">{out_scores_csv.name}</a></li>
      <li><a href="{out_scores_json.name}" target="_blank" rel="noopener">{out_scores_json.name}</a></li>
      <li><a href="{dominant_csv.name}" target="_blank" rel="noopener">{dominant_csv.name}</a></li>
      <li><a href="{leaderboard_csv.name}" target="_blank" rel="noopener">{leaderboard_csv.name}</a></li>
    </ul>
    """).strip()
    dashboard_html = cfg.outdir / (cfg.html_name if cfg.html_name.endswith(".html") else f"{cfg.html_name}.html")
    _write_dashboard_html(dashboard_html, heatmap_html_files, head_html, resources)

    # Manifest + run hash
    manifest = {
        "tool": "symbolic_influence_map",
        "timestamp": _now_iso(),
        "inputs": {
            "mu": str(cfg.mu_path) if cfg.mu_path else None,
            "grad": str(cfg.grad_path) if cfg.grad_path else None,
            "shap": str(cfg.shap_path) if cfg.shap_path else None,
            "rules": str(cfg.rules_path) if cfg.rules_path else None,
            "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else None,
            "metadata": str(cfg.metadata_path) if cfg.metadata_path else None,
        },
        "shapes": {
            "mu": _shape(mu := (mu if mu is not None else None)),
            "grad": _shape(grad),
            "shap": _shape(shap),
            "rule_count": len(rules.rule_names) if rules is not None else 0,
        },
        "outputs": {
            "scores_csv": str(out_scores_csv),
            "scores_json": str(out_scores_json),
            "dominant_csv": str(dominant_csv),
            "leaderboard_csv": str(leaderboard_csv),
            "dashboard_html": str(dashboard_html),
            "heatmaps_html": [p.name for p in heatmap_html_files],
        },
        "config": {
            "reduction": cfg.reduction,
            "weight_power": float(cfg.weight_power),
            "top_planet_barplots": int(cfg.top_planet_barplots),
        }
    }
    with open(cfg.outdir / "symbolic_influence_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "message": "symbolic_influence_map completed successfully",
        "outdir": str(cfg.outdir),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "grad": str(cfg.grad_path) if cfg.grad_path else "",
        "shap": str(cfg.shap_path) if cfg.shap_path else "",
        "rules": str(cfg.rules_path) if cfg.rules_path else "",
        "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "reduction": cfg.reduction,
    })

    if cfg.open_browser and _PLOTLY_OK:
        try:
            import webbrowser
            webbrowser.open_new_tab(dashboard_html.as_uri())
        except Exception:
            pass

    return 0


# ==============================================================================
# CLI
# ==============================================================================

def _shape(arr: Optional[np.ndarray]) -> Optional[List[int]]:
    if arr is None:
        return None
    return list(arr.shape)


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(name))


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="symbolic_influence_map",
        description="Compute per‑rule symbolic influence maps and summaries from gradients/SHAP/μ and rule masks."
    )
    p.add_argument("--mu", type=Path, default=None, help="μ array (P×B) in npy/npz/csv/parquet/feather")
    p.add_argument("--grad", type=Path, default=None, help="Gradient array: P×B or P×R×B (npy/npz). If P×R×B, --rules optional.")
    p.add_argument("--shap", type=Path, default=None, help="|SHAP| array (P×B); used if gradients absent")
    p.add_argument("--rules", type=Path, default=None, help="Rules JSON with 'rule_masks' and optional 'rule_groups'")
    p.add_argument("--symbolic", type=Path, default=None, help="Symbolic results JSON (optional overlays)")
    p.add_argument("--metadata", type=Path, default=None, help="Metadata CSV/JSON with 'planet_id' (optional)")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory")

    p.add_argument("--reduction", type=str, default="sum", choices=["sum", "mean", "max", "weighted"],
                   help="Reduction from per‑bin influence to scalar per rule/planet")
    p.add_argument("--weight-power", type=float, default=1.0, help="Exponent on rule mask weights for 'weighted' reduction")
    p.add_argument("--top-planet-barplots", type=int, default=12, help="Export barplots for N top‑influence planets (0=disable)")

    p.add_argument("--html-name", type=str, default="symbolic_influence_dashboard.html", help="Dashboard HTML filename")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in a browser (if Plotly available)")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # Resolve paths
    cfg = Config(
        mu_path=args.mu.resolve() if args.mu else None,
        grad_path=args.grad.resolve() if args.grad else None,
        shap_path=args.shap.resolve() if args.shap else None,
        rules_path=args.rules.resolve() if args.rules else None,
        symbolic_path=args.symbolic.resolve() if args.symbolic else None,
        metadata_path=args.metadata.resolve() if args.metadata else None,
        outdir=args.outdir.resolve(),
        reduction=str(args.reduction),
        weight_power=float(args.weight_power),
        top_planet_barplots=int(args.top_planet_barplots),
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
        "message": "Starting symbolic_influence_map",
        "outdir": str(cfg.outdir),
        "mu": str(cfg.mu_path) if cfg.mu_path else "",
        "grad": str(cfg.grad_path) if cfg.grad_path else "",
        "shap": str(cfg.shap_path) if cfg.shap_path else "",
        "rules": str(cfg.rules_path) if cfg.rules_path else "",
        "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
        "metadata": str(cfg.metadata_path) if cfg.metadata_path else "",
        "reduction": cfg.reduction,
    })

    try:
        rc = run(cfg, audit)
        return rc
    except Exception as e:
        audit.log({
            "action": "run",
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
            "outdir": str(cfg.outdir),
            "mu": str(cfg.mu_path) if cfg.mu_path else "",
            "grad": str(cfg.grad_path) if cfg.grad_path else "",
            "shap": str(cfg.shap_path) if cfg.shap_path else "",
            "rules": str(cfg.rules_path) if cfg.rules_path else "",
        })
        # stderr traceback for CI visibility
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
