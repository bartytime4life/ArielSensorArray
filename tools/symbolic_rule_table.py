#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/symbolic_rule_table.py

SpectraMind V50 — Symbolic Rule Table Generator (Ultimate, Challenge‑Grade)

Purpose
-------
Generate an authoritative, reproducible *symbolic rule table* and diagnostics from a
rules JSON describing spectral-bin masks for symbolic constraints. This tool validates,
summarizes, and visualizes the rule set, producing CSV/JSON tables and a small HTML
dashboard. It integrates cleanly with the rest of the V50 diagnostics ecosystem.

Key capabilities
----------------
1) Rules loader (robust, flexible):
   • Expected JSON layout:
       {
         "rule_masks": {
           "H2O_band_1": [idx, idx, ...] | [0/1/weight, ... length B] | {"bin": weight, ...}
           "CO2_band_43": {...}
         },
         "rule_groups": { "water": ["H2O_band_1", "H2O_band_2"], ... }  # optional
         "meta": { ... }                                                # optional
       }
   • Supports integer indices, dense 0/1/weight vectors, or dict {bin_idx: weight}.
   • Auto‑normalizes to a float mask per rule of length B.

2) Optional wavelength grid:
   • Pass --wavelengths as .npy/.npz/.csv (column 'wavelength')/.parquet/.feather
   • Computes wavelength spans per rule (min/max, segment spans), and overlap stats in λ‑space.

3) Rule statistics & validation:
   • coverage_count, coverage_weight_sum, coverage_fraction
   • n_segments (count contiguous stretches)
   • segment_start_end indices (and wavelengths if provided)
   • per‑rule mask density and summary weight stats (min/mean/max, L1/L2)
   • overlap_count/weight with other rules; Jaccard index matrix
   • group summaries if rule_groups provided
   • optional symbolic overlay ingestion (per‑rule scores if found) to attach global means

4) Exports:
   outdir/
     rule_table.csv / .json                 # per‑rule stats
     rule_overlap_counts.csv                # pairwise overlap counts (bin count)
     rule_overlap_jaccard.csv               # pairwise Jaccard indices
     rule_groups.csv                        # optional group rollups
     coverage_bar.png (or .csv fallback)    # coverage fraction per rule
     overlap_heatmap.png/.html (Plotly if available; CSV fallback)
     symbolic_rule_table_manifest.json      # manifest of inputs/outputs/options
     run_hash_summary_v50.json              # reproducibility trail (append‑only)
     (small HTML dashboard with quick links)

5) Reproducibility & logging:
   • Deterministic computation (no RNG)
   • Append‑only audit entries to logs/v50_debug_log.md and logs/v50_runs.jsonl

CLI examples
------------
poetry run python tools/symbolic_rule_table.py \
  --rules configs/symbolic_rules.json \
  --wavelengths data/wavelengths.npy \
  --outdir outputs/symbolic_rule_table

poetry run python tools/symbolic_rule_table.py \
  --rules configs/symbolic_rules.json \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --outdir outputs/symbolic_rule_table --open-browser

Design notes
------------
• No external network calls; optional deps (Plotly/Matplotlib) degrade gracefully.
• HTML dashboard is self‑contained (links to sibling artifacts).
• "No placeholders": all outputs are meaningful even if optional inputs are missing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

# --------------------------
# Required tabular dependency
# --------------------------
try:
    import pandas as pd
except Exception as e:
    raise RuntimeError("pandas is required. Please `pip install pandas`.") from e

# --------------------------
# Optional viz dependencies
# --------------------------
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
# Utilities: I/O, logging, hashing
# ==============================================================================

def _now_iso() -> str:
    """Return timezone‑aware ISO8601 timestamp (seconds precision)."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _ensure_dir(p: Path) -> None:
    """Create directory recursively if missing."""
    p.mkdir(parents=True, exist_ok=True)


@dataclass
class AuditLogger:
    """Append‑only audit logger to Markdown + JSONL."""
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
        tool: symbolic_rule_table
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        rules: {row.get("rules","")}
        wavelengths: {row.get("wavelengths","")}
        symbolic: {row.get("symbolic","")}
        outdir: {row.get("outdir","")}
        message: {row.get("message","")}
        """).strip() + "\n"
        with open(self.md_path, "a", encoding="utf-8") as f:
            f.write(md)


def _hash_jsonable(obj: Any) -> str:
    """Stable SHA‑256 over canonical JSON encoding."""
    b = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _update_run_hash_summary(outdir: Path, manifest: Dict[str, Any]) -> None:
    """Append run manifest to outdir/run_hash_summary_v50.json."""
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
# Loaders: arrays, wavelengths, rules, symbolic overlays
# ==============================================================================

def _load_array_any(path: Path) -> np.ndarray:
    """
    Load 1D/2D array from .npy/.npz or via pandas for CSV/TSV/Parquet/Feather.
    """
    s = path.suffix.lower()
    if s == ".npy":
        return np.asarray(np.load(path, allow_pickle=False))
    if s == ".npz":
        z = np.load(path, allow_pickle=False)
        # choose first
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


def _load_wavelengths(path: Optional[Path], B_hint: Optional[int]) -> Optional[np.ndarray]:
    """
    Load wavelengths array as float64, length B. Accepts:
      • .npy/.npz → 1D or 2D (take 1st column)
      • .csv/.tsv with a 'wavelength' column (preferred) or first numeric column
      • .parquet/.feather: use 'wavelength' column if exists, else first column
    """
    if path is None:
        return None
    arr = _load_array_any(path)
    # If 2D, attempt to select a single vector
    if arr.ndim == 2:
        # If a DataFrame was converted, it will be numeric values only
        # Prefer first column
        vec = np.asarray(arr[:, 0]).reshape(-1)
    else:
        vec = arr.reshape(-1)
    vec = vec.astype(float)
    if B_hint is not None and vec.shape[0] != B_hint:
        # tolerate mismatch by truncation/pad
        B = B_hint
        out = np.zeros(B, dtype=float)
        copy = min(B, vec.shape[0])
        out[:copy] = vec[:copy]
        return out
    return vec


@dataclass
class RuleSet:
    rule_names: List[str]          # length R
    rule_masks: np.ndarray         # shape R×B (float weights allowed, >=0)
    groups: Dict[str, List[str]]   # optional named groups mapping


def _as_mask_vector(spec: Any, B: int) -> np.ndarray:
    """
    Convert a rule mask specification to a length‑B float vector.
      • list of bin indices -> 1.0 at those indices
      • list of 0/1/weights (len B) -> float array
      • dict {bin_idx: weight} -> weight at indices
    """
    mask = np.zeros(B, dtype=float)
    if isinstance(spec, list):
        if len(spec) == B and all(isinstance(x, (int, float, bool, np.floating, np.integer)) for x in spec):
            arr = np.array(spec, dtype=float)
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
            mask = np.maximum(arr, 0.0)
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


def _load_rules_json(path: Path, B_hint: Optional[int]) -> RuleSet:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "rule_masks" not in obj or not isinstance(obj["rule_masks"], dict):
        raise ValueError("Rules JSON must contain a 'rule_masks' object mapping name→mask")

    # Determine B (bins). Use B_hint if present, else infer from first rule mask.
    B: Optional[int] = B_hint
    if B is None:
        # Try to infer from first entry
        for _, spec in obj["rule_masks"].items():
            if isinstance(spec, list) and not spec:
                continue
            if isinstance(spec, list) and all(isinstance(x, (int, float, bool, np.number)) for x in spec):
                # Could be dense vector len B OR list of indices
                # If length is large and not sparse, treat as dense vector
                if len(spec) >= 8 and not all(isinstance(x, (int, np.integer)) for x in spec):
                    B = len(spec)
                    break
            if isinstance(spec, dict):
                # At least get a sense of max index
                try:
                    idxs = [int(k) for k in spec.keys()]
                    B = max(idxs) + 1 if idxs else B
                    if B is not None:
                        break
                except Exception:
                    pass
        if B is None:
            raise ValueError("Unable to infer #bins (B). Provide --bins or a dense vector mask for at least one rule.")

    # Build matrices
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


def _load_symbolic_any(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Load optional symbolic overlays. We only attempt to extract per‑rule aggregate scores
    if an obvious schema is present. Otherwise, return None.
    Accepted hints:
      • {"rule_scores": {"RuleA": val, "RuleB": val, ...}, ...}
      • {"rows":[{"rule":"RuleA","score":...}, ...]}
    """
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    # Heuristic parsing
    if isinstance(obj, dict):
        if "rule_scores" in obj and isinstance(obj["rule_scores"], dict):
            rows = [{"rule": k, "score": float(v)} for k, v in obj["rule_scores"].items() if isinstance(v, (int, float))]
            return pd.DataFrame(rows)
        if "rows" in obj and isinstance(obj["rows"], list):
            rows = []
            for r in obj["rows"]:
                if isinstance(r, dict) and "rule" in r and any(k in r for k in ("score", "violation", "value")):
                    val = r.get("score", r.get("violation", r.get("value", 0.0)))
                    try:
                        rows.append({"rule": str(r["rule"]), "score": float(val)})
                    except Exception:
                        pass
            if rows:
                return pd.DataFrame(rows)
    return None


# ==============================================================================
# Core computations: segments, coverage, overlaps, groups
# ==============================================================================

def _segments(mask: np.ndarray) -> List[Tuple[int, int]]:
    """
    Return list of contiguous [start,end] (inclusive) bin index segments where mask>0.
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
        s = i
        prev = i
    segs.append((s, prev))
    return segs


def _segment_stats(mask: np.ndarray, wavelengths: Optional[np.ndarray]) -> Tuple[int, List[Tuple[int, int]], Optional[List[Tuple[float, float]]]]:
    segs = _segments(mask)
    if wavelengths is None:
        return len(segs), segs, None
    lam_segs: List[Tuple[float, float]] = []
    for a, b in segs:
        a = int(a); b = int(b)
        a = max(a, 0); b = min(b, len(wavelengths) - 1)
        lam_segs.append((float(wavelengths[a]), float(wavelengths[b])))
    return len(segs), segs, lam_segs


def _overlap_matrices(rule_masks: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise overlap count and Jaccard index between rule masks (threshold >0).
      • overlap_counts[r1, r2] = |bins where both >0|
      • jaccard[r1, r2] = |A∩B| / |A∪B|
    """
    R, B = rule_masks.shape
    bin_bool = rule_masks > 0.0
    counts = np.zeros((R, R), dtype=int)
    jacc = np.zeros((R, R), dtype=float)
    sums = bin_bool.sum(axis=1)  # |A|
    for i in range(R):
        Ai = bin_bool[i]
        for j in range(i, R):
            Aj = bin_bool[j]
            inter = int(np.logical_and(Ai, Aj).sum())
            uni = int(np.logical_or(Ai, Aj).sum())
            counts[i, j] = counts[j, i] = inter
            jacc[i, j] = jacc[j, i] = (inter / uni) if uni > 0 else 0.0
    return counts, jacc


def _group_rollup(groups: Dict[str, List[str]], rule_names: List[str], table: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per‑rule table into groups by summing coverage metrics and averaging densities.
    """
    if not groups:
        return pd.DataFrame(columns=["group", "n_rules", "coverage_count", "coverage_fraction_mean", "coverage_weight_sum"])
    name_to_idx = {n: i for i, n in enumerate(rule_names)}
    rows = []
    for g, members in groups.items():
        idxs = [name_to_idx[m] for m in members if m in name_to_idx]
        if not idxs:
            continue
        sub = table.iloc[idxs]
        rows.append({
            "group": g,
            "n_rules": len(idxs),
            "coverage_count": int(sub["coverage_count"].sum()),
            "coverage_fraction_mean": float(sub["coverage_fraction"].mean()),
            "coverage_weight_sum": float(sub["coverage_weight_sum"].sum()),
            "segments_total": int(sub["n_segments"].sum()),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# Visualization helpers
# ==============================================================================

def _save_coverage_bar(table: pd.DataFrame, out_png: Path) -> None:
    """
    Save bar chart of coverage_fraction per rule. Fallback to CSV if Matplotlib unavailable.
    """
    if not _MPL_OK:
        table[["rule", "coverage_fraction"]].to_csv(out_png.with_suffix(".csv"), index=False)
        return
    _ensure_dir(out_png.parent)
    # Sort by coverage_fraction
    t = table.sort_values("coverage_fraction", ascending=False)
    names = t["rule"].tolist()
    vals = t["coverage_fraction"].to_numpy()
    plt.figure(figsize=(max(12, len(names) * 0.25), 6))
    plt.bar(np.arange(len(names)), vals)
    plt.xticks(np.arange(len(names)), names, rotation=75, ha="right")
    plt.ylabel("Coverage fraction (count/B)")
    plt.title("Symbolic Rule Coverage Fraction")
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def _save_overlap_heatmap(counts: np.ndarray, rule_names: List[str], out_png: Path, out_html: Path) -> None:
    """
    Save overlap heatmap (counts). Plotly HTML if available; PNG via Matplotlib; CSV fallback.
    """
    _ensure_dir(out_png.parent)
    if _PLOTLY_OK:
        fig = go.Figure(data=go.Heatmap(
            z=counts,
            x=rule_names, y=rule_names,
            colorscale="Viridis",
            colorbar=dict(title="Overlap (bins)")
        ))
        fig.update_layout(
            title="Rule Overlap Heatmap (counts)",
            xaxis_nticks=len(rule_names),
            yaxis_nticks=len(rule_names),
            template="plotly_white",
            width=max(800, min(1800, 90 + 30 * len(rule_names))),
            height=max(800, min(1800, 90 + 30 * len(rule_names))),
        )
        pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")
    if _MPL_OK:
        plt.figure(figsize=(max(8, len(rule_names)*0.35), max(8, len(rule_names)*0.35)))
        vmax = np.percentile(counts, 99.0) if counts.size else 1.0
        plt.imshow(counts, cmap="viridis", vmin=0.0, vmax=vmax if vmax > 0 else None)
        plt.colorbar(label="Overlap (bins)")
        plt.xticks(np.arange(len(rule_names)), rule_names, rotation=75, ha="right", fontsize=8)
        plt.yticks(np.arange(len(rule_names)), rule_names, fontsize=8)
        plt.title("Rule Overlap Heatmap (counts)")
        plt.tight_layout()
        plt.savefig(out_png, dpi=160)
        plt.close()
    if not _PLOTLY_OK and not _MPL_OK:
        # CSV fallback
        pd.DataFrame(counts, index=rule_names, columns=rule_names).to_csv(out_png.with_suffix(".csv"))


def _write_dashboard_html(out_html: Path, quick_links: List[Tuple[str, str]], preview_table_html: str) -> None:
    """
    Small, self‑contained HTML with quick links to core artifacts and a head preview.
    """
    _ensure_dir(out_html.parent)
    links = "\n".join(f'<li><a href="{href}" target="_blank" rel="noopener">{label}</a></li>' for label, href in quick_links)
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — Symbolic Rule Table</title>
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
    <h1>Symbolic Rule Table — SpectraMind V50</h1>
    <div>Generated: {_now_iso()}</div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    <ul>
      {links}
    </ul>
  </section>

  <section class="card">
    <h2>Preview — First 30 Rules</h2>
    {preview_table_html}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 • Rule Table Diagnostics</small>
  </footer>
</body>
</html>
"""
    out_html.write_text(html, encoding="utf-8")


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclass
class Config:
    rules_path: Path
    outdir: Path
    bins: Optional[int]
    wavelengths_path: Optional[Path]
    symbolic_path: Optional[Path]
    html_name: str
    open_browser: bool


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load rules (normalizes masks to R×B)
    rules = _load_rules_json(cfg.rules_path, cfg.bins)
    rule_names = rules.rule_names
    rule_masks = np.maximum(rules.rule_masks.astype(float), 0.0)
    R, B = rule_masks.shape

    # Load wavelengths (optional)
    wavelengths = _load_wavelengths(cfg.wavelengths_path, B_hint=B)

    # Per‑rule stats
    rows: List[Dict[str, Any]] = []
    for r, name in enumerate(rule_names):
        mask = rule_masks[r]
        # coverage & weights
        coverage_count = int(np.count_nonzero(mask > 0.0))
        coverage_weight_sum = float(mask.sum())
        coverage_fraction = float(coverage_count / B) if B > 0 else 0.0
        # weight stats
        nz = mask[mask > 0.0]
        w_min = float(nz.min()) if nz.size else 0.0
        w_mean = float(nz.mean()) if nz.size else 0.0
        w_max = float(nz.max()) if nz.size else 0.0
        l1 = float(np.sum(np.abs(mask)))
        l2 = float(np.sqrt(np.sum(mask ** 2)))
        # segments
        n_segments, segs, lam_segs = _segment_stats(mask, wavelengths)
        segs_str = ";".join([f"{a}-{b}" for a, b in segs]) if segs else ""
        lam_segs_str = ";".join([f"{float(a):.6g}-{float(b):.6g}" for a, b in lam_segs]) if lam_segs else ""
        rows.append({
            "rule": name,
            "coverage_count": coverage_count,
            "coverage_weight_sum": coverage_weight_sum,
            "coverage_fraction": coverage_fraction,
            "n_segments": n_segments,
            "segments_bins": segs_str,
            "segments_wavelengths": lam_segs_str,
            "weight_min": w_min,
            "weight_mean": w_mean,
            "weight_max": w_max,
            "l1": l1,
            "l2": l2,
        })
    table = pd.DataFrame(rows)

    # Overlap matrices
    overlap_counts, jaccard = _overlap_matrices(rule_masks)
    overlap_counts_df = pd.DataFrame(overlap_counts, index=rule_names, columns=rule_names)
    jaccard_df = pd.DataFrame(jaccard, index=rule_names, columns=rule_names)

    # Group rollups (optional)
    groups_df = _group_rollup(rules.groups, rule_names, table) if rules.groups else pd.DataFrame(
        columns=["group", "n_rules", "coverage_count", "coverage_fraction_mean", "coverage_weight_sum", "segments_total"]
    )

    # Optional symbolic overlay: attach per‑rule score if provided
    sym_df = _load_symbolic_any(cfg.symbolic_path)
    if sym_df is not None and not sym_df.empty:
        sym_df = sym_df.groupby("rule", as_index=False)["score"].mean()
        table = table.merge(sym_df, left_on="rule", right_on="rule", how="left")
        table["score"] = table["score"].fillna(0.0)

    # Write core tables
    rule_table_csv = cfg.outdir / "rule_table.csv"
    rule_table_json = cfg.outdir / "rule_table.json"
    overlap_counts_csv = cfg.outdir / "rule_overlap_counts.csv"
    jaccard_csv = cfg.outdir / "rule_overlap_jaccard.csv"
    groups_csv = cfg.outdir / "rule_groups.csv"

    table.to_csv(rule_table_csv, index=False)
    with open(rule_table_json, "w", encoding="utf-8") as f:
        json.dump({"rows": table.to_dict(orient="records")}, f, indent=2)
    overlap_counts_df.to_csv(overlap_counts_csv)
    jaccard_df.to_csv(jaccard_csv)
    if not groups_df.empty:
        groups_df.to_csv(groups_csv, index=False)

    # Visualizations
    coverage_bar_png = cfg.outdir / "coverage_bar.png"
    _save_coverage_bar(table, coverage_bar_png)

    overlap_png = cfg.outdir / "overlap_heatmap.png"
    overlap_html = cfg.outdir / "overlap_heatmap.html"
    _save_overlap_heatmap(overlap_counts, rule_names, overlap_png, overlap_html)

    # Dashboard
    html_name = cfg.html_name if cfg.html_name.endswith(".html") else f"{cfg.html_name}.html"
    dashboard_html = cfg.outdir / html_name
    preview = table.head(30).to_html(index=False)
    quick_links: List[Tuple[str, str]] = [
        ("rule_table.csv", rule_table_csv.name),
        ("rule_table.json", rule_table_json.name),
        ("rule_overlap_counts.csv", overlap_counts_csv.name),
        ("rule_overlap_jaccard.csv", jaccard_csv.name),
        ("coverage_bar.png", coverage_bar_png.name if coverage_bar_png.exists() else coverage_bar_png.with_suffix(".csv").name),
        ("overlap_heatmap.html" if _PLOTLY_OK else "overlap_heatmap.png", (overlap_html.name if _PLOTLY_OK else overlap_png.name)),
    ]
    if not groups_df.empty:
        quick_links.append(("rule_groups.csv", groups_csv.name))
    _write_dashboard_html(dashboard_html, quick_links, preview)

    # Manifest + run hash
    manifest = {
        "tool": "symbolic_rule_table",
        "timestamp": _now_iso(),
        "inputs": {
            "rules": str(cfg.rules_path),
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else None,
            "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else None,
            "bins": int(cfg.bins) if cfg.bins is not None else None,
        },
        "shapes": {"R": int(R), "B": int(B)},
        "outputs": {
            "rule_table_csv": str(rule_table_csv),
            "rule_table_json": str(rule_table_json),
            "overlap_counts_csv": str(overlap_counts_csv),
            "overlap_jaccard_csv": str(jaccard_csv),
            "groups_csv": str(groups_csv) if not groups_df.empty else None,
            "coverage_bar": str(coverage_bar_png if coverage_bar_png.exists() else coverage_bar_png.with_suffix(".csv")),
            "overlap_heatmap_png": str(overlap_png if overlap_png.exists() else overlap_png.with_suffix(".csv")),
            "overlap_heatmap_html": str(overlap_html) if _PLOTLY_OK else None,
            "dashboard_html": str(dashboard_html),
        }
    }
    with open(cfg.outdir / "symbolic_rule_table_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "rules": str(cfg.rules_path),
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
        "outdir": str(cfg.outdir),
        "message": f"Generated R={R} rules, B={B} bins; dashboard: {dashboard_html.name}",
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
        prog="symbolic_rule_table",
        description="Generate a symbolic rule table (coverage, segments, overlaps, groups) with CSV/JSON/HTML exports."
    )
    p.add_argument("--rules", type=Path, required=True, help="Rules JSON with 'rule_masks' and optional 'rule_groups'.")
    p.add_argument("--outdir", type=Path, required=True, help="Output directory for tables/figures/dashboard.")
    p.add_argument("--bins", type=int, default=None, help="Optional explicit #bins (B). If omitted, inferred from rules.")
    p.add_argument("--wavelengths", type=Path, default=None, help="Optional wavelengths array (.npy/.npz/.csv/.parquet/.feather).")
    p.add_argument("--symbolic", type=Path, default=None, help="Optional symbolic overlays with per‑rule scores (flexible schema).")
    p.add_argument("--html-name", type=str, default="symbolic_rule_table.html", help="Dashboard HTML filename.")
    p.add_argument("--open-browser", action="store_true", help="Open dashboard in default browser.")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        rules_path=args.rules.resolve(),
        outdir=args.outdir.resolve(),
        bins=int(args.bins) if args.bins is not None else None,
        wavelengths_path=args.wavelengths.resolve() if args.wavelengths else None,
        symbolic_path=args.symbolic.resolve() if args.symbolic else None,
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
        "rules": str(cfg.rules_path),
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
        "outdir": str(cfg.outdir),
        "message": "Starting symbolic_rule_table",
    })

    try:
        rc = run(cfg, audit)
        return rc
    except Exception as e:
        # Print traceback for CI visibility and record error
        import traceback
        traceback.print_exc()
        audit.log({
            "action": "run",
            "status": "error",
            "rules": str(cfg.rules_path),
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
            "symbolic": str(cfg.symbolic_path) if cfg.symbolic_path else "",
            "outdir": str(cfg.outdir),
            "message": f"{type(e).__name__}: {e}",
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
