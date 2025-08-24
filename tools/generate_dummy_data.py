#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/neural_logic_graph.py

SpectraMind V50 — Neural Logic Graph (Ultimate, Challenge‑Grade)

Purpose
-------
Render, analyze, and export an interactive **neuro‑symbolic logic graph** that connects:
  • Molecule/group nodes  →  Rule nodes  →  (optional) spectral bin nodes,
  • Weighted by per‑planet **influence** (e.g., |∂L/∂μ| masks, SHAP×rules, or precomputed tables),
  • Annotated with **symbolic violations** if available,
  • Laid out with spring / Kamada‑Kawai / circular coordinates,
  • Exported as HTML (Plotly), PNG (if Matplotlib), and CSV (nodes/edges/positions).

The tool ingests flexible inputs:
  1) A symbolic **rules JSON** (masks per rule, optional rule groups),
  2) An optional **program JSON/YAML** describing node relationships,
  3) An **influence** table (e.g., from `symbolic_influence_map.py` or `shap_symbolic_overlay.py`),
  4) Optional **violations JSON** (per‑planet/per‑rule scores),
  5) Optional **SHAP** + **rules** to compute influence if no table is provided.

Design Notes
------------
• Deterministic (seeded), no network calls, graceful dependency fallbacks (Plotly/Matplotlib optional).
• Append‑only audit logs to `logs/v50_debug_log.md` and `logs/v50_runs.jsonl`.
• Manifest + run hash to ensure reproducibility (`run_hash_summary_v50.json`).
• Outputs integrate with V50 diagnostics dashboard (HTML embeddable).

Typical Usage
-------------
poetry run python tools/neural_logic_graph.py \
  --rules configs/symbolic_rules.json \
  --influence outputs/symbolic_influence_v50/symbolic_influence_per_planet_rule.csv \
  --violations outputs/diagnostics/symbolic_results.json \
  --planet-id planet_0007 \
  --wavelengths data/wavelengths.npy \
  --layout spring --show-bins --bin-step 8 \
  --outdir outputs/logic_graph --open-browser

If you don’t have an influence table:
  add --shap outputs/shap/shap_values.npy and we’ll integrate |SHAP| over rule masks.

Outputs
-------
outdir/
  logic_nodes.csv                 # node_id, label, type, score, extra...
  logic_edges.csv                 # src, dst, weight, kind
  logic_positions.csv             # node_id, x, y
  logic_graph.html                # interactive Plotly network (if Plotly available)
  logic_graph.png                 # static PNG (if Matplotlib available)
  neural_logic_graph_manifest.json
  run_hash_summary_v50.json       # append‑only reproducibility log
  dashboard.html                  # quick links + preview

Node Types
----------
• "group"     : symbolic rule group (e.g., water, carbon, ...), optional
• "molecule"  : molecule/node alias for grouping (if present in program or groups)
• "rule"      : symbolic rule from rules JSON
• "bin"       : spectral bin node (optional; downsampled by --bin-step)
• "root"      : synthetic root (optional, when helpful for disconnected components)

Edge Types
----------
• "group→rule"     : membership of rule in group
• "molecule→rule"  : from program/mapping if provided (or same as group)
• "rule→bin"       : rule mask coverage for bin (weight ~ mask weight)
• "root→*"         : synthetic, to ensure connectivity when needed

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

# --------------------------
# Optional layout dependency
# --------------------------
try:
    import networkx as nx
    _NX_OK = True
except Exception:
    _NX_OK = False


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
        tool: neural_logic_graph
        action: {row.get("action","run")}
        status: {row.get("status","ok")}
        rules: {row.get("rules","")}
        program: {row.get("program","")}
        influence: {row.get("influence","")}
        violations: {row.get("violations","")}
        shap: {row.get("shap","")}
        wavelengths: {row.get("wavelengths","")}
        planet: {row.get("planet_id","")}#{row.get("planet_index","")}
        layout: {row.get("layout","spring")}
        show_bins: {row.get("show_bins","False")}
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
# Loaders: arrays, wavelengths, rules, program, influence, violations, metadata
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


def _load_wavelengths(path: Optional[Path], B_hint: Optional[int]) -> Optional[np.ndarray]:
    if path is None:
        return None
    arr = _load_array_any(path)
    vec = arr.reshape(-1).astype(float)
    if B_hint is not None and vec.shape[0] != B_hint:
        out = np.zeros(B_hint, dtype=float)
        copy = min(B_hint, vec.shape[0])
        out[:copy] = vec[:copy]
        return out
    return vec


@dataclass
class RuleSet:
    rule_names: List[str]
    rule_masks: np.ndarray   # R×B (>=0)
    groups: Dict[str, List[str]]


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


def _load_rules_json(path: Path, B_hint: Optional[int]) -> RuleSet:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if "rule_masks" not in obj or not isinstance(obj["rule_masks"], dict):
        raise ValueError("Rules JSON must contain a 'rule_masks' mapping name→mask")
    # Determine B
    B = B_hint
    if B is None:
        for _, spec in obj["rule_masks"].items():
            if isinstance(spec, list) and len(spec) > 0:
                if all(isinstance(x, (int, float, bool, np.number)) for x in spec):
                    B = len(spec)
                    break
            if isinstance(spec, dict) and spec:
                B = max(int(k) for k in spec.keys() if str(k).isdigit()) + 1
                break
        if B is None:
            raise ValueError("Unable to infer #bins (B); provide dense vector for at least one rule or pass SHAP/μ.")
    names, mats = [], []
    for name, spec in obj["rule_masks"].items():
        names.append(str(name))
        mats.append(_as_mask_vector(spec, B))
    rule_masks = np.vstack(mats) if mats else np.zeros((0, B), dtype=float)
    groups: Dict[str, List[str]] = {}
    if "rule_groups" in obj and isinstance(obj["rule_groups"], dict):
        for gname, lst in obj["rule_groups"].items():
            groups[str(gname)] = [str(x) for x in lst if str(x) in names]
    return RuleSet(rule_names=names, rule_masks=rule_masks, groups=groups)


def _load_program(path: Optional[Path]) -> Dict[str, Any]:
    """
    Program schema is flexible. We accept either:
      • JSON/YAML with {"nodes":[{"id","type","label",...},...], "edges":[{"src","dst","kind","weight"},...]}
      • Or {"molecules":{"H2O":["ruleA","ruleB"], ...}} (we will generate edges)
    """
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # lazy
        except Exception as e:
            raise RuntimeError("YAML program given but PyYAML not installed. `pip install pyyaml`.") from e
        return yaml.safe_load(text)
    return json.loads(text)


def _load_influence_table(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Accepts:
      • CSV with columns: planet_id, rule, importance[, fraction, ...]
      • JSON {rows:[{planet_id, rule, importance, ...}, ...]}
    """
    if path is None:
        return None
    s = path.suffix.lower()
    if s in {".csv", ".tsv"}:
        df = pd.read_csv(path) if s == ".csv" else pd.read_csv(path, sep="\t")
        return df
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict) and "rows" in obj and isinstance(obj["rows"], list):
        return pd.DataFrame(obj["rows"])
    return pd.DataFrame(obj)


def _load_violations(path: Optional[Path]) -> Optional[pd.DataFrame]:
    """
    Try to coerce flexible symbolic JSON to table with (planet_id, rule, violation or score).
    Accept patterns:
      • {"planets": {"pid": {"rule_scores":{"r1":val,...}}}}
      • {"rows":[{"planet_id","rule","violation" or "score"}, ...]}
    """
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    rows: List[Dict[str, Any]] = []
    if isinstance(obj, dict):
        if "rows" in obj and isinstance(obj["rows"], list):
            for r in obj["rows"]:
                if isinstance(r, dict) and "planet_id" in r and "rule" in r:
                    val = r.get("violation", r.get("score", r.get("value", 0.0)))
                    rows.append({"planet_id": str(r["planet_id"]), "rule": str(r["rule"]), "violation": float(val)})
        elif "planets" in obj and isinstance(obj["planets"], dict):
            for pid, payload in obj["planets"].items():
                if isinstance(payload, dict) and "rule_scores" in payload and isinstance(payload["rule_scores"], dict):
                    for rname, val in payload["rule_scores"].items():
                        if isinstance(val, (int, float)):
                            rows.append({"planet_id": str(pid), "rule": str(rname), "violation": float(val)})
    return pd.DataFrame(rows) if rows else None


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
# Influence computation (fallback via SHAP×rules)
# ==============================================================================

def _prepare_shap(shap: np.ndarray, P: int, B: int) -> np.ndarray:
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
        copyB = min(B, a.shape[1])
        out[:, :copyB] = a[:, :copyB]
        a = out
    a = np.abs(a)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _compute_influence_from_shap(shap_abs: np.ndarray, rule_masks: np.ndarray, power: float = 1.0, eps: float = 1e-12) -> np.ndarray:
    """
    Influence[p, r] = sum_b |SHAP|[p,b] * mask[r,b]^power
    """
    W = np.power(np.maximum(rule_masks, 0.0), power)  # R×B
    return shap_abs @ W.T  # (P×B)@(B×R) → P×R


# ==============================================================================
# Graph construction
# ==============================================================================

@dataclass
class GraphConfig:
    show_bins: bool
    bin_step: int
    min_edge_weight: float
    attach_root: bool


def _downsample_bins(B: int, step: int) -> np.ndarray:
    if step <= 1:
        return np.arange(B, dtype=int)
    return np.arange(0, B, step, dtype=int)


def _build_nodes_edges(
    rules: RuleSet,
    wl: Optional[np.ndarray],
    influence_row: Optional[pd.Series],
    violation_row: Optional[pd.Series],
    cfg: GraphConfig,
    program: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build node/edge tables.

    Node columns:
      node_id, label, type, score (influence/violation composite), influence, violation, extra...

    Edge columns:
      src, dst, weight, kind
    """
    rule_names = rules.rule_names
    rule_masks = np.maximum(rules.rule_masks, 0.0)
    R, B = rule_masks.shape
    # scores per rule (influence & violation)
    infl = pd.Series(0.0, index=rule_names)
    viol = pd.Series(0.0, index=rule_names)
    if influence_row is not None:
        for r in rule_names:
            if r in influence_row.index:
                infl[r] = float(influence_row[r])
    if violation_row is not None:
        for r in rule_names:
            if r in violation_row.index:
                viol[r] = float(violation_row[r])

    # base nodes (rules)
    nodes: List[Dict[str, Any]] = []
    for r, name in enumerate(rule_names):
        nodes.append({
            "node_id": f"rule::{name}",
            "label": name,
            "type": "rule",
            "influence": float(infl[name]),
            "violation": float(viol[name]),
            "score": float(infl[name]) if infl[name] > 0 else float(viol[name]),
            "coverage": int(np.count_nonzero(rule_masks[r] > 0.0)),
        })

    # group/molecule nodes from rules.groups or program
    edges: List[Dict[str, Any]] = []
    group_to_rules: Dict[str, List[str]] = {}

    # From rules.groups
    for g, members in (rules.groups or {}).items():
        group_to_rules.setdefault(g, [])
        for m in members:
            if m in rule_names:
                group_to_rules[g].append(m)

    # From program shortcut: {"molecules":{"H2O":["ruleA","ruleB"], ...}}
    if "molecules" in program and isinstance(program["molecules"], dict):
        for mol, members in program["molecules"].items():
            group_to_rules.setdefault(str(mol), [])
            for m in members:
                if m in rule_names:
                    group_to_rules[str(mol)].append(m)

    # Emit group nodes and edges
    for g, members in group_to_rules.items():
        nodes.append({
            "node_id": f"group::{g}",
            "label": g,
            "type": "group",
            "influence": float(sum(infl.get(m, 0.0) for m in members)),
            "violation": float(sum(viol.get(m, 0.0) for m in members)),
            "score": float(sum(infl.get(m, 0.0) for m in members)),
        })
        for m in members:
            edges.append({
                "src": f"group::{g}",
                "dst": f"rule::{m}",
                "weight": float(max(infl.get(m, 0.0), 1e-12)),
                "kind": "group→rule",
            })

    # Bin nodes (optional; downsample)
    bin_index = _downsample_bins(B, cfg.bin_step) if cfg.show_bins else np.array([], dtype=int)
    if cfg.show_bins:
        for b in bin_index:
            label = f"{wl[b]:.6g} μm" if wl is not None else f"bin {b}"
            nodes.append({
                "node_id": f"bin::{b}",
                "label": label,
                "type": "bin",
                "influence": 0.0,
                "violation": 0.0,
                "score": 0.0,
            })
        # rule→bin edges (weight = mask[r,b])
        for r, name in enumerate(rule_names):
            mask = rule_masks[r]
            for b in bin_index:
                w = float(mask[b])
                if w <= 0:
                    continue
                if w < cfg.min_edge_weight:
                    continue
                edges.append({
                    "src": f"rule::{name}",
                    "dst": f"bin::{b}",
                    "weight": w,
                    "kind": "rule→bin",
                })

    # Synthetic root if graph disconnected and no groups defined
    if cfg.attach_root and not group_to_rules:
        nodes.append({
            "node_id": "root::logic",
            "label": "ROOT",
            "type": "root",
            "influence": float(infl.sum()),
            "violation": float(viol.sum()),
            "score": float(infl.sum()),
        })
        for name in rule_names:
            edges.append({
                "src": "root::logic",
                "dst": f"rule::{name}",
                "weight": float(max(infl.get(name, 0.0), 1e-12)),
                "kind": "root→rule",
            })

    nodes_df = pd.DataFrame(nodes)
    edges_df = pd.DataFrame(edges) if edges else pd.DataFrame(columns=["src", "dst", "weight", "kind"])
    return nodes_df, edges_df


# ==============================================================================
# Layout & Rendering
# ==============================================================================

def _layout_positions(nodes_df: pd.DataFrame, edges_df: pd.DataFrame, layout: str, seed: int) -> pd.DataFrame:
    """
    Compute 2D positions for nodes. Use NetworkX if available; otherwise fallback to simple circular layout.
    """
    ids = nodes_df["node_id"].tolist()
    if _NX_OK:
        G = nx.Graph()
        for nid in ids:
            G.add_node(nid)
        for _, e in edges_df.iterrows():
            G.add_edge(e["src"], e["dst"], weight=float(e.get("weight", 1.0)))
        if layout == "spring":
            pos = nx.spring_layout(G, seed=seed, k=None, weight="weight", iterations=200)
        elif layout == "kk":
            pos = nx.kamada_kawai_layout(G, weight="weight")
        elif layout == "circular":
            pos = nx.circular_layout(G)
        else:
            pos = nx.spring_layout(G, seed=seed, weight="weight", iterations=200)
        rows = [{"node_id": nid, "x": float(p[0]), "y": float(p[1])} for nid, p in pos.items()]
        return pd.DataFrame(rows)
    # Fallback: circular
    n = len(ids)
    theta = np.linspace(0, 2*np.pi, num=n, endpoint=False)
    rows = [{"node_id": ids[i], "x": float(np.cos(theta[i])), "y": float(np.sin(theta[i]))} for i in range(n)]
    return pd.DataFrame(rows)


def _render_plotly(nodes: pd.DataFrame, edges: pd.DataFrame, pos: pd.DataFrame, title: str, out_html: Path) -> None:
    if not _PLOTLY_OK:
        # Fallback: write CSV only
        return
    # Merge positions
    nodes = nodes.merge(pos, on="node_id", how="left")
    id2xy = {r["node_id"]: (r["x"], r["y"]) for _, r in nodes.iterrows()}

    # Edge segments
    edge_traces = []
    for _, e in edges.iterrows():
        u, v = e["src"], e["dst"]
        if u not in id2xy or v not in id2xy:
            continue
        x0, y0 = id2xy[u]
        x1, y1 = id2xy[v]
        edge_traces.append(go.Scatter(
            x=[x0, x1], y=[y0, y1],
            mode="lines",
            line=dict(width=1.0, color="rgba(120,120,120,0.45)"),
            hoverinfo="skip",
            showlegend=False
        ))

    # Node scatter by type
    def _mk_trace(df: pd.DataFrame, name: str, size_scale: float, symbol: str):
        return go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=np.clip(6 + size_scale * (df["score"].fillna(0.0).to_numpy()), 6, 32),
                color=df["score"].fillna(0.0),
                colorscale="Viridis",
                showscale=True if name == "rule" else False,
                line=dict(width=1, color="rgba(20,20,20,0.6)"),
                symbol=symbol,
            ),
            text=[
                f"{r['label']}<br>type: {r['type']}<br>influence: {r.get('influence',0):.4g}<br>violation: {r.get('violation',0):.4g}"
                for _, r in df.iterrows()
            ],
            hoverinfo="text",
            name=name
        )

    layers = []
    for t, symbol in [("group", "square"), ("rule", "circle"), ("bin", "diamond"), ("root", "cross")]:
        sub = nodes[nodes["type"] == t]
        if not sub.empty:
            layers.append(_mk_trace(sub, t, size_scale=12.0 if t == "rule" else 8.0, symbol=symbol))

    fig = go.Figure(data=[*edge_traces, *layers])
    fig.update_layout(
        title=title,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_white",
        width=1080, height=720,
        margin=dict(l=10, r=10, t=50, b=10)
    )
    pio.write_html(fig, file=str(out_html), auto_open=False, include_plotlyjs="cdn")


def _render_matplotlib(nodes: pd.DataFrame, edges: pd.DataFrame, pos: pd.DataFrame, title: str, out_png: Path) -> None:
    if not _MPL_OK:
        return
    nodes = nodes.merge(pos, on="node_id", how="left")
    id2xy = {r["node_id"]: (r["x"], r["y"]) for _, r in nodes.iterrows()}
    plt.figure(figsize=(12, 8))
    # Edges
    for _, e in edges.iterrows():
        u, v = e["src"], e["dst"]
        if u not in id2xy or v not in id2xy:
            continue
        x0, y0 = id2xy[u]; x1, y1 = id2xy[v]
        plt.plot([x0, x1], [y0, y1], color="0.6", alpha=0.45, lw=0.8, zorder=1)
    # Nodes by type
    for t, marker, z in [("group", "s", 3), ("rule", "o", 4), ("bin", "D", 2), ("root", "x", 5)]:
        sub = nodes[nodes["type"] == t]
        if sub.empty:
            continue
        sizes = np.clip(20 + 80 * sub["score"].fillna(0.0).to_numpy(), 20, 240)
        plt.scatter(sub["x"], sub["y"], s=sizes, marker=marker, label=t, zorder=z, alpha=0.9, edgecolor="k", linewidths=0.5)
    plt.legend(loc="upper right")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=170)
    plt.close()


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclass
class Config:
    rules_path: Path
    program_path: Optional[Path]
    influence_path: Optional[Path]
    violations_path: Optional[Path]
    shap_path: Optional[Path]
    wavelengths_path: Optional[Path]
    metadata_path: Optional[Path]
    planet_id: Optional[str]
    planet_index: Optional[int]
    mask_power: float
    layout: str
    show_bins: bool
    bin_step: int
    min_edge_weight: float
    attach_root: bool
    outdir: Path
    html_name: str
    open_browser: bool
    seed: int


def run(cfg: Config, audit: AuditLogger) -> int:
    _ensure_dir(cfg.outdir)

    # Load rules first to know B
    rules = _load_rules_json(cfg.rules_path, B_hint=None)
    R, B = rules.rule_masks.shape

    # Wavelengths & metadata
    wl = _load_wavelengths(cfg.wavelengths_path, B_hint=B)
    # Influence table & violations
    infl_df = _load_influence_table(cfg.influence_path)
    viol_df = _load_violations(cfg.violations_path)

    # Planet identification
    P_guess = 0
    if infl_df is not None and "planet_id" in infl_df.columns:
        P_guess = max(P_guess, infl_df["planet_id"].nunique())
    meta_df = _load_metadata_any(cfg.metadata_path, n_planets=max(P_guess, 1))
    planet_ids = meta_df["planet_id"].astype(str).tolist()
    # Choose planet
    focus_pid: str
    if cfg.planet_id:
        focus_pid = cfg.planet_id
    elif cfg.planet_index is not None and 0 <= cfg.planet_index < len(planet_ids):
        focus_pid = planet_ids[cfg.planet_index]
    else:
        focus_pid = planet_ids[0]

    # Select per-planet influence row: wide pivot (columns=rule)
    influence_row = None
    if infl_df is not None and not infl_df.empty:
        if {"planet_id", "rule"}.issubset(infl_df.columns):
            if "importance" not in infl_df.columns:
                # Try alternative column names
                alt = [c for c in infl_df.columns if c not in {"planet_id", "rule"}]
                if alt:
                    infl_df = infl_df.rename(columns={alt[0]: "importance"})
                else:
                    infl_df["importance"] = 0.0
            wide = infl_df.pivot_table(index="planet_id", columns="rule", values="importance", aggfunc="mean").fillna(0.0)
            if focus_pid in wide.index:
                influence_row = wide.loc[focus_pid]
    # If missing, compute from SHAP if provided
    if influence_row is None and cfg.shap_path is not None:
        shap_raw = _load_array_any(cfg.shap_path)
        # Guess P from metadata (>=1)
        P = len(planet_ids)
        shap_abs = _prepare_shap(shap_raw, P=P, B=B)
        infl = _compute_influence_from_shap(shap_abs, rules.rule_masks, power=cfg.mask_power)  # P×R
        w = pd.DataFrame(infl, index=planet_ids, columns=rules.rule_names)
        influence_row = w.loc[focus_pid]

    # Violations per rule per planet (optional)
    violation_row = None
    if viol_df is not None and not viol_df.empty:
        if {"planet_id", "rule"}.issubset(viol_df.columns):
            val_col = "violation" if "violation" in viol_df.columns else ("score" if "score" in viol_df.columns else None)
            if val_col:
                wide = viol_df.pivot_table(index="planet_id", columns="rule", values=val_col, aggfunc="mean").fillna(0.0)
                if focus_pid in wide.index:
                    violation_row = wide.loc[focus_pid]

    # Program (optional)
    program = _load_program(cfg.program_path)

    # Build graph
    gcfg = GraphConfig(
        show_bins=bool(cfg.show_bins),
        bin_step=int(cfg.bin_step),
        min_edge_weight=float(cfg.min_edge_weight),
        attach_root=bool(cfg.attach_root),
    )
    nodes_df, edges_df = _build_nodes_edges(
        rules=rules,
        wl=wl,
        influence_row=influence_row,
        violation_row=violation_row,
        cfg=gcfg,
        program=program
    )

    # Layout
    pos_df = _layout_positions(nodes_df, edges_df, layout=cfg.layout, seed=cfg.seed)

    # Save core tables
    nodes_csv = cfg.outdir / "logic_nodes.csv"
    edges_csv = cfg.outdir / "logic_edges.csv"
    pos_csv = cfg.outdir / "logic_positions.csv"
    nodes_df.to_csv(nodes_csv, index=False)
    edges_df.to_csv(edges_csv, index=False)
    pos_df.to_csv(pos_csv, index=False)

    # Render
    title = f"Neural Logic Graph — planet {focus_pid}"
    html_path = cfg.outdir / (cfg.html_name if cfg.html_name.endswith(".html") else "logic_graph.html")
    _render_plotly(nodes_df, edges_df, pos_df, title=title, out_html=html_path)
    png_path = cfg.outdir / "logic_graph.png"
    _render_matplotlib(nodes_df, edges_df, pos_df, title=title, out_png=png_path)

    # Dashboard
    dash_html = cfg.outdir / "dashboard.html"
    preview = nodes_df.head(30).to_html(index=False)
    quick_links = textwrap.dedent(f"""
    <ul>
      <li><a href="{nodes_csv.name}" target="_blank" rel="noopener">{nodes_csv.name}</a></li>
      <li><a href="{edges_csv.name}" target="_blank" rel="noopener">{edges_csv.name}</a></li>
      <li><a href="{pos_csv.name}" target="_blank" rel="noopener">{pos_csv.name}</a></li>
      <li><a href="{html_path.name}" target="_blank" rel="noopener">{html_path.name}</a></li>
      <li><a href="{png_path.name}" target="_blank" rel="noopener">{png_path.name}</a></li>
    </ul>
    """).strip()
    dash = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>SpectraMind V50 — Neural Logic Graph</title>
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
    <h1>Neural Logic Graph — SpectraMind V50</h1>
    <div>Generated: <span class="pill">{_now_iso()}</span> • Planet: <b>{focus_pid}</b> • Layout: <b>{cfg.layout}</b> • Bins shown: <b>{cfg.show_bins}</b></div>
  </header>

  <section class="card">
    <h2>Quick Links</h2>
    {quick_links}
  </section>

  <section class="card">
    <h2>Preview — First 30 Nodes</h2>
    {preview}
  </section>

  <footer class="card">
    <small>© SpectraMind V50 • mask_power={cfg.mask_power} • min_edge_weight={cfg.min_edge_weight} • bin_step={cfg.bin_step}</small>
  </footer>
</body>
</html>
"""
    dash_html.write_text(dash, encoding="utf-8")

    # Manifest + hash
    manifest = {
        "tool": "neural_logic_graph",
        "timestamp": _now_iso(),
        "inputs": {
            "rules": str(cfg.rules_path),
            "program": str(cfg.program_path) if cfg.program_path else None,
            "influence": str(cfg.influence_path) if cfg.influence_path else None,
            "violations": str(cfg.violations_path) if cfg.violations_path else None,
            "shap": str(cfg.shap_path) if cfg.shap_path else None,
            "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else None,
            "metadata": str(cfg.metadata_path) if cfg.metadata_path else None,
        },
        "params": {
            "planet_id": focus_pid,
            "mask_power": cfg.mask_power,
            "layout": cfg.layout,
            "show_bins": cfg.show_bins,
            "bin_step": cfg.bin_step,
            "min_edge_weight": cfg.min_edge_weight,
            "attach_root": cfg.attach_root,
            "seed": cfg.seed,
        },
        "shapes": {
            "R": int(R),
            "B": int(B),
            "nodes": int(len(nodes_df)),
            "edges": int(len(edges_df)),
        },
        "outputs": {
            "nodes_csv": str(nodes_csv),
            "edges_csv": str(edges_csv),
            "positions_csv": str(pos_csv),
            "graph_html": str(html_path),
            "graph_png": str(png_path),
            "dashboard_html": str(dash_html),
        }
    }
    with open(cfg.outdir / "neural_logic_graph_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    _update_run_hash_summary(cfg.outdir, manifest)

    # Audit success
    audit.log({
        "action": "run",
        "status": "ok",
        "rules": str(cfg.rules_path),
        "program": str(cfg.program_path) if cfg.program_path else "",
        "influence": str(cfg.influence_path) if cfg.influence_path else "",
        "violations": str(cfg.violations_path) if cfg.violations_path else "",
        "shap": str(cfg.shap_path) if cfg.shap_path else "",
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "planet_id": focus_pid,
        "layout": cfg.layout,
        "show_bins": cfg.show_bins,
        "outdir": str(cfg.outdir),
        "message": f"Built logic graph with {len(nodes_df)} nodes / {len(edges_df)} edges; HTML={html_path.name}",
    })

    if cfg.open_browser and html_path.exists() and _PLOTLY_OK:
        try:
            import webbrowser
            webbrowser.open_new_tab(html_path.as_uri())
        except Exception:
            pass

    return 0


# ==============================================================================
# CLI
# ==============================================================================

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="neural_logic_graph",
        description="Build and render a neuro‑symbolic logic graph (groups/molecules → rules → bins) weighted by influence."
    )
    p.add_argument("--rules", type=Path, required=True, help="Rules JSON with 'rule_masks' and optional 'rule_groups'.")
    p.add_argument("--program", type=Path, default=None, help="Optional program JSON/YAML (nodes/edges or molecules mapping).")
    p.add_argument("--influence", type=Path, default=None, help="Optional influence table (CSV/JSON) with columns planet_id, rule, importance.")
    p.add_argument("--violations", type=Path, default=None, help="Optional symbolic violations JSON with per‑planet/per‑rule scores.")
    p.add_argument("--shap", type=Path, default=None, help="Optional SHAP array to compute influence if table missing.")
    p.add_argument("--wavelengths", type=Path, default=None, help="Optional wavelength vector (B,).")
    p.add_argument("--metadata", type=Path, default=None, help="Optional metadata with 'planet_id'.")
    p.add_argument("--planet-id", type=str, default=None, help="Planet ID focus for per‑rule scores.")
    p.add_argument("--planet-index", type=int, default=None, help="Planet index focus (if ID not provided).")

    p.add_argument("--mask-power", type=float, default=1.0, help="Exponent for rule mask weights when integrating SHAP.")
    p.add_argument("--layout", type=str, default="spring", choices=["spring", "kk", "circular"], help="Graph layout algorithm.")
    p.add_argument("--show-bins", action="store_true", help="Include spectral bin nodes.")
    p.add_argument("--bin-step", type=int, default=8, help="Downsample bins by this step when --show-bins is used (>=1).")
    p.add_argument("--min-edge-weight", type=float, default=0.0, help="Drop rule→bin edges with weight below this threshold.")
    p.add_argument("--attach-root", action="store_true", help="Attach a synthetic root to all rules if no groups present.")

    p.add_argument("--outdir", type=Path, required=True, help="Output directory.")
    p.add_argument("--html-name", type=str, default="logic_graph.html", help="HTML filename.")
    p.add_argument("--open-browser", action="store_true", help="Open the HTML graph in your default browser.")
    p.add_argument("--seed", type=int, default=7, help="Layout seed (deterministic).")
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    cfg = Config(
        rules_path=args.rules.resolve(),
        program_path=args.program.resolve() if args.program else None,
        influence_path=args.influence.resolve() if args.influence else None,
        violations_path=args.violations.resolve() if args.violations else None,
        shap_path=args.shap.resolve() if args.shap else None,
        wavelengths_path=args.wavelengths.resolve() if args.wavelengths else None,
        metadata_path=args.metadata.resolve() if args.metadata else None,
        planet_id=str(args.planet_id) if args.planet_id else None,
        planet_index=int(args.planet_index) if args.planet_index is not None else None,
        mask_power=float(args.mask_power),
        layout=str(args.layout),
        show_bins=bool(args.show_bins),
        bin_step=max(1, int(args.bin_step)),
        min_edge_weight=float(args.min_edge_weight),
        attach_root=bool(args.attach_root),
        outdir=args.outdir.resolve(),
        html_name=str(args.html_name),
        open_browser=bool(args.open_browser),
        seed=int(args.seed),
    )

    audit = AuditLogger(
        md_path=Path("logs") / "v50_debug_log.md",
        jsonl_path=Path("logs") / "v50_runs.jsonl",
    )
    audit.log({
        "action": "start",
        "status": "running",
        "rules": str(cfg.rules_path),
        "program": str(cfg.program_path) if cfg.program_path else "",
        "influence": str(cfg.influence_path) if cfg.influence_path else "",
        "violations": str(cfg.violations_path) if cfg.violations_path else "",
        "shap": str(cfg.shap_path) if cfg.shap_path else "",
        "wavelengths": str(cfg.wavelengths_path) if cfg.wavelengths_path else "",
        "planet_id": cfg.planet_id if cfg.planet_id else "",
        "planet_index": cfg.planet_index if cfg.planet_index is not None else "",
        "layout": cfg.layout,
        "show_bins": cfg.show_bins,
        "outdir": str(cfg.outdir),
        "message": "Starting neural_logic_graph",
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
            "rules": str(cfg.rules_path),
            "program": str(cfg.program_path) if cfg.program_path else "",
            "influence": str(cfg.influence_path) if cfg.influence_path else "",
            "outdir": str(cfg.outdir),
            "message": f"{type(e).__name__}: {e}",
        })
        return 2


if __name__ == "__main__":
    sys.exit(main())
