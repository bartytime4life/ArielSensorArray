#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/visualization/plot_umap_v50.py

SpectraMind V50 — UMAP Latent Visualizer (Upgraded, Challenge-Grade)

Purpose
-------
Project high-dimensional latent vectors to 2D or 3D with UMAP and render a rich,
interactive Plotly figure (HTML) plus optional static PNG. Supports:
• Symbolic & SHAP overlays in hover and coloring
• Confidence shading (marker opacity/size)
• Planet-level hyperlinks
• De-duplication of repeated latents/planet_ids
• 2D/3D modes, clustering labels, and rich legend
• CLI-first (Typer) with Hydra-safe defaults and audit logging to logs/v50_debug_log.md

Typical usage
-------------
python -m src.visualization.plot_umap_v50 \
  --latents outputs/latents.npy \
  --planet-ids data/metadata/planet_ids.csv \
  --labels-csv outputs/labels.csv \
  --symbolic-json outputs/diagnostics/symbolic_results.json \
  --shap-json outputs/diagnostics/shap_overlay.json \
  --confidence-csv outputs/diagnostics/confidence.csv \
  --dim 2 \
  --color-by label \
  --size-by confidence \
  --opacity-by confidence \
  --link-template "planets/{planet_id}.html" \
  --title "UMAP — SpectraMind V50" \
  --html-out outputs/umap/umap_v50.html \
  --png-out outputs/umap/umap_v50.png \
  --dedupe \
  --open-html
"""

from __future__ import annotations

import json
import math
import os
import sys
import csv
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Optional imports guarded to provide friendly errors
try:
    import umap
except Exception as _e:
    umap = None

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except Exception as _e:
    go = None
    make_subplots = None

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# -----------------------------
# I/O utilities
# -----------------------------

def _read_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing .npy file: {path}")
    return np.load(str(path))

def _read_planet_ids(path: Path) -> List[str]:
    if path.suffix.lower() in {".txt"}:
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # CSV: assume first column is planet_id
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row:
                continue
            out.append(str(row[0]).strip())
    return out

def _read_labels_csv(path: Optional[Path]) -> Dict[str, str]:
    if not path:
        return {}
    if not path.exists():
        console.print(f"[yellow]WARN[/] labels csv not found: {path}")
        return {}
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # must include planet_id, label
        for row in reader:
            pid = str(row.get("planet_id","")).strip()
            lab = str(row.get("label","")).strip()
            if pid:
                out[pid] = lab
    return out

def _read_confidence_csv(path: Optional[Path]) -> Dict[str, float]:
    if not path:
        return {}
    if not path.exists():
        console.print(f"[yellow]WARN[/] confidence csv not found: {path}")
        return {}
    out: Dict[str, float] = {}
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # must include planet_id, confidence
        for row in reader:
            pid = str(row.get("planet_id","")).strip()
            try:
                val = float(row.get("confidence", ""))
            except Exception:
                val = float("nan")
            if pid and np.isfinite(val):
                out[pid] = float(np.clip(val, 0.0, 1.0))
    return out

def _read_json(path: Optional[Path]) -> Dict[str, Any]:
    if not path:
        return {}
    if not path.exists():
        console.print(f"[yellow]WARN[/] JSON not found: {path}")
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        console.print(f"[yellow]WARN[/] Failed reading {path}: {e}")
        return {}

# -----------------------------
# Overlays & feature engineering
# -----------------------------

def _compute_symbolic_dominant_rule(symbolic_per_planet: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    symbolic_per_planet: {planet_id: {rule_name: score, ...}}
    Returns dominant rule (max by score) label per planet.
    """
    out: Dict[str, str] = {}
    for pid, rules in symbolic_per_planet.items():
        if not isinstance(rules, dict) or not rules:
            continue
        # Larger score => more violation/influence (convention-agnostic: use max magnitude)
        k = max(rules, key=lambda r: abs(rules[r]) if rules[r] is not None else -np.inf)
        out[pid] = str(k)
    return out

def _compute_shap_strength(shap_per_planet: Dict[str, Any]) -> Dict[str, float]:
    """
    shap_per_planet: {planet_id: {...}}; Attempt to summarize to a scalar magnitude for coloring/size.
    If per-bin arrays exist, compute mean |value|.
    """
    out: Dict[str, float] = {}
    for pid, obj in shap_per_planet.items():
        if isinstance(obj, dict):
            # try common keys
            for key in ("mean_abs", "magnitude", "avg_abs", "avg_abs_shap"):
                if key in obj and np.isfinite(obj[key]):
                    out[pid] = float(obj[key])
                    break
            else:
                # fallback: if a list/array stored under "values" or "bins"
                arr = None
                for key in ("values", "bins", "per_bin", "per_bin_abs"):
                    if key in obj and isinstance(obj[key], (list, tuple)):
                        arr = np.asarray(obj[key], dtype=float)
                        break
                if arr is not None and arr.size > 0:
                    out[pid] = float(np.nanmean(np.abs(arr)))
        elif isinstance(obj, (list, tuple, np.ndarray)):
            arr = np.asarray(obj, dtype=float)
            if arr.size > 0:
                out[pid] = float(np.nanmean(np.abs(arr)))
    return out

def _norm_or_default(vals: List[float], a: float = 0.35, b: float = 1.0) -> List[float]:
    """
    Normalize a list to [a,b]. If degenerate or empty, return 0.8s.
    """
    if not vals:
        return []
    arr = np.asarray(vals, dtype=float)
    m, M = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return [0.8 for _ in vals]
    x = (arr - m) / (M - m)
    return list(a + (b - a) * x)

# -----------------------------
# Deduplication
# -----------------------------

def _dedupe(latents: np.ndarray, planet_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Remove duplicate planet_ids (keep first) and duplicate latent rows.
    """
    # First dedupe by planet_id (keep first occurrence)
    seen = set()
    keep_idx: List[int] = []
    for i, pid in enumerate(planet_ids):
        if pid in seen:
            continue
        seen.add(pid)
        keep_idx.append(i)
    latents1 = latents[keep_idx]
    pids1 = [planet_ids[i] for i in keep_idx]

    # Then dedupe identical latent rows
    # Use structured array trick for uniqueness
    lat_flat = np.ascontiguousarray(latents1).view(
        np.dtype((np.void, latents1.dtype.itemsize * latents1.shape[1]))
    )
    _, uniq_indices = np.unique(lat_flat, return_index=True)
    uniq_indices = sorted(list(uniq_indices))
    return latents1[uniq_indices], [pids1[i] for i in uniq_indices]

# -----------------------------
# UMAP projection
# -----------------------------

def _umap_project(
    latents: np.ndarray,
    dim: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 33,
) -> np.ndarray:
    if umap is None:
        raise RuntimeError("umap-learn is not installed. Please `pip install umap-learn`.")
    reducer = umap.UMAP(
        n_components=dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=False,
    )
    return reducer.fit_transform(latents)

# -----------------------------
# Plotly figure
# -----------------------------

def _build_hover(pid: str, label: Optional[str], dom_rule: Optional[str],
                 shap_mag: Optional[float], conf: Optional[float]) -> str:
    lines = [f"<b>{pid}</b>"]
    if label:
        lines.append(f"label: {label}")
    if dom_rule:
        lines.append(f"symbolic: {dom_rule}")
    if shap_mag is not None and np.isfinite(shap_mag):
        lines.append(f"SHAP: {shap_mag:.4f}")
    if conf is not None and np.isfinite(conf):
        lines.append(f"confidence: {conf:.3f}")
    return "<br>".join(lines)

def _figure_umap(
    emb2: np.ndarray,
    planet_ids: List[str],
    labels: Dict[str, str],
    dom_rules: Dict[str, str],
    shap_strength: Dict[str, float],
    confidence: Dict[str, float],
    link_template: Optional[str],
    title: str,
    dim: int,
    color_by: str,
    size_by: Optional[str],
    opacity_by: Optional[str],
) -> go.Figure:
    if go is None:
        raise RuntimeError("plotly is not installed. Please `pip install plotly`.")
    # Build per-point attributes
    xs = emb2[:,0]
    ys = emb2[:,1]
    zs = emb2[:,2] if dim == 3 else None

    # Color logic
    color_vals = []
    color_text = []
    if color_by == "label":
        # map label strings to categorical colors
        # Store for legend using separate traces per label to get categorical legend
        unique_labels = sorted({labels.get(pid, "unknown") for pid in planet_ids})
        traces = []
        for lab in unique_labels:
            idx = [i for i, pid in enumerate(planet_ids) if labels.get(pid, "unknown") == lab]
            if not idx:
                continue
            hover = [
                _build_hover(
                    planet_ids[i],
                    lab,
                    dom_rules.get(planet_ids[i]),
                    shap_strength.get(planet_ids[i]),
                    confidence.get(planet_ids[i]),
                )
                for i in idx
            ]
            url = None
            if link_template:
                url = [link_template.format(planet_id=planet_ids[i]) for i in idx]
            marker_size, marker_opacity = _marker_style(
                idx, planet_ids, size_by, opacity_by, shap_strength, confidence
            )
            if dim == 3:
                tr = go.Scatter3d(
                    x=xs[idx], y=ys[idx], z=zs[idx],
                    mode="markers",
                    name=str(lab),
                    text=hover,
                    hoverinfo="text",
                    marker=dict(
                        size=marker_size, opacity=marker_opacity, symbol="circle"
                    ),
                    customdata=url if link_template else None,
                )
            else:
                tr = go.Scattergl(
                    x=xs[idx], y=ys[idx],
                    mode="markers",
                    name=str(lab),
                    text=hover,
                    hoverinfo="text",
                    marker=dict(
                        size=marker_size, opacity=marker_opacity, symbol="circle"
                    ),
                    customdata=url if link_template else None,
                )
            traces.append(tr)
        fig = go.Figure(data=traces)
    elif color_by == "symbolic":
        cats = sorted({dom_rules.get(pid, "none") for pid in planet_ids})
        traces = []
        for cat in cats:
            idx = [i for i,pid in enumerate(planet_ids) if dom_rules.get(pid,"none")==cat]
            if not idx: continue
            hover = [
                _build_hover(
                    planet_ids[i],
                    labels.get(planet_ids[i]),
                    dom_rules.get(planet_ids[i]),
                    shap_strength.get(planet_ids[i]),
                    confidence.get(planet_ids[i]),
                )
                for i in idx
            ]
            url = None
            if link_template:
                url = [link_template.format(planet_id=planet_ids[i]) for i in idx]
            marker_size, marker_opacity = _marker_style(
                idx, planet_ids, size_by, opacity_by, shap_strength, confidence
            )
            name = f"sym:{cat}"
            tr = (go.Scatter3d if dim==3 else go.Scattergl)(
                x=xs[idx], y=ys[idx], **({"z": zs[idx]} if dim==3 else {}),
                mode="markers",
                name=name,
                text=hover,
                hoverinfo="text",
                marker=dict(size=marker_size, opacity=marker_opacity, symbol="circle"),
                customdata=url if link_template else None,
            )
            traces.append(tr)
        fig = go.Figure(data=traces)
    elif color_by == "shap":
        # continuous color using shap strength
        c = np.array([shap_strength.get(pid, np.nan) for pid in planet_ids], dtype=float)
        c = np.where(np.isfinite(c), c, np.nan)
        # Normalize for color scale
        cmin = np.nanmin(c) if np.any(np.isfinite(c)) else 0.0
        cmax = np.nanmax(c) if np.any(np.isfinite(c)) else 1.0
        c = np.where(np.isfinite(c), c, (cmin+cmax)/2.0)
        hover = [
            _build_hover(pid, labels.get(pid), dom_rules.get(pid), shap_strength.get(pid), confidence.get(pid))
            for pid in planet_ids
        ]
        url = None
        if link_template:
            url = [link_template.format(planet_id=pid) for pid in planet_ids]
        marker_size, marker_opacity = _marker_style(
            list(range(len(planet_ids))), planet_ids, size_by, opacity_by, shap_strength, confidence
        )
        common = dict(
            mode="markers",
            text=hover,
            hoverinfo="text",
            marker=dict(
                size=marker_size,
                opacity=marker_opacity,
                color=c, colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="SHAP")
            ),
            customdata=url if link_template else None,
        )
        if dim==3:
            tr = go.Scatter3d(x=xs, y=ys, z=zs, **common)
        else:
            tr = go.Scattergl(x=xs, y=ys, **common)
        fig = go.Figure(data=[tr])
    elif color_by == "confidence":
        c = np.array([confidence.get(pid, np.nan) for pid in planet_ids], dtype=float)
        c = np.where(np.isfinite(c), c, np.nan)
        cmin = np.nanmin(c) if np.any(np.isfinite(c)) else 0.0
        cmax = np.nanmax(c) if np.any(np.isfinite(c)) else 1.0
        c = np.where(np.isfinite(c), c, (cmin+cmax)/2.0)
        hover = [
            _build_hover(pid, labels.get(pid), dom_rules.get(pid), shap_strength.get(pid), confidence.get(pid))
            for pid in planet_ids
        ]
        url = None
        if link_template:
            url = [link_template.format(planet_id=pid) for pid in planet_ids]
        marker_size, marker_opacity = _marker_style(
            list(range(len(planet_ids))), planet_ids, size_by, opacity_by, shap_strength, confidence
        )
        common = dict(
            mode="markers",
            text=hover,
            hoverinfo="text",
            marker=dict(
                size=marker_size,
                opacity=marker_opacity,
                color=c, colorscale="Blues",
                showscale=True,
                colorbar=dict(title="Confidence")
            ),
            customdata=url if link_template else None,
        )
        if dim==3:
            tr = go.Scatter3d(x=xs, y=ys, z=zs, **common)
        else:
            tr = go.Scattergl(x=xs, y=ys, **common)
        fig = go.Figure(data=[tr])
    else:
        # default single series, uniform color
        hover = [
            _build_hover(pid, labels.get(pid), dom_rules.get(pid), shap_strength.get(pid), confidence.get(pid))
            for pid in planet_ids
        ]
        url = None
        if link_template:
            url = [link_template.format(planet_id=pid) for pid in planet_ids]
        marker_size, marker_opacity = _marker_style(
            list(range(len(planet_ids))), planet_ids, size_by, opacity_by, shap_strength, confidence
        )
        if dim==3:
            tr = go.Scatter3d(
                x=xs, y=ys, z=zs, mode="markers",
                text=hover, hoverinfo="text",
                marker=dict(size=marker_size, opacity=marker_opacity, symbol="circle"),
                customdata=url if link_template else None,
                name="UMAP"
            )
        else:
            tr = go.Scattergl(
                x=xs, y=ys, mode="markers",
                text=hover, hoverinfo="text",
                marker=dict(size=marker_size, opacity=marker_opacity, symbol="circle"),
                customdata=url if link_template else None,
                name="UMAP"
            )
        fig = go.Figure(data=[tr])

    # Title & Layout
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_layout(
        scene=dict(
            xaxis_title="UMAP-1", yaxis_title="UMAP-2", zaxis_title=("UMAP-3" if dim==3 else None)
        ) if dim==3 else {},
        xaxis_title="UMAP-1", yaxis_title="UMAP-2",
    )

    # Click behavior: embed url in customdata, add hovertemplate with hint
    # (Plotly can't natively open link on click in pure HTML offline; but customdata enables JS hooks in dashboards.)
    return fig

def _marker_style(
    indices: List[int],
    planet_ids: List[str],
    size_by: Optional[str],
    opacity_by: Optional[str],
    shap_strength: Dict[str, float],
    confidence: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    # Size logic
    if size_by == "shap":
        vals = [shap_strength.get(planet_ids[i], np.nan) for i in indices]
        vals = [v if (v is not None and np.isfinite(v)) else np.nan for v in vals]
        sizes = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=6, b=14)
    elif size_by == "confidence":
        vals = [confidence.get(planet_ids[i], np.nan) for i in indices]
        vals = [v if (v is not None and np.isfinite(v)) else np.nan for v in vals]
        sizes = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=6, b=16)
    else:
        sizes = [8 for _ in indices]

    # Opacity logic
    if opacity_by == "confidence":
        vals = [confidence.get(planet_ids[i], np.nan) for i in indices]
        vals = [v if (v is not None and np.isfinite(v)) else np.nan for v in vals]
        opac = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=0.35, b=0.95)
    elif opacity_by == "shap":
        vals = [shap_strength.get(planet_ids[i], np.nan) for i in indices]
        vals = [v if (v is not None and np.isfinite(v)) else np.nan for v in vals]
        opac = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=0.35, b=0.95)
    else:
        opac = [0.85 for _ in indices]
    return sizes, opac

# -----------------------------
# Save outputs
# -----------------------------

def _save_html(fig: go.Figure, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)

def _save_png(fig: go.Figure, out: Optional[Path]):
    if not out:
        return
    try:
        import kaleido  # noqa: F401
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out), scale=2.0, height=900, width=1200)
    except Exception as e:
        console.print(f"[yellow]WARN[/] Could not write PNG via kaleido: {e}")

def _save_manifest(outdir: Path, meta: Dict[str, Any]):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "umap_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def _append_audit(log_path: Path, message: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- [{dt.datetime.now().isoformat(timespec='seconds')}] plot_umap_v50: {message}\n")

# -----------------------------
# CLI
# -----------------------------

@app.command("run")
def cli_run(
    latents: Path = typer.Option(..., help="Path to latents .npy (N,D)"),
    planet_ids: Path = typer.Option(..., help="Path to planet IDs (.txt or .csv with first column planet_id)"),
    labels_csv: Optional[Path] = typer.Option(None, "--labels-csv", help="Optional CSV with columns: planet_id,label"),
    symbolic_json: Optional[Path] = typer.Option(None, "--symbolic-json", help="Optional symbolic results JSON (per-planet rule scores)."),
    shap_json: Optional[Path] = typer.Option(None, "--shap-json", help="Optional SHAP overlay JSON (per-planet magnitudes or arrays)."),
    confidence_csv: Optional[Path] = typer.Option(None, "--confidence-csv", help="Optional CSV with columns: planet_id,confidence in [0,1]."),
    dim: int = typer.Option(2, help="UMAP output dimension: 2 or 3.", min=2, max=3),
    n_neighbors: int = typer.Option(15, help="UMAP n_neighbors."),
    min_dist: float = typer.Option(0.1, help="UMAP min_dist."),
    metric: str = typer.Option("euclidean", help="UMAP metric."),
    random_state: int = typer.Option(33, help="UMAP random_state."),
    color_by: str = typer.Option("label", help="Color by: label|symbolic|shap|confidence|none"),
    size_by: Optional[str] = typer.Option(None, help="Size by: shap|confidence"),
    opacity_by: Optional[str] = typer.Option(None, help="Opacity by: shap|confidence"),
    link_template: Optional[str] = typer.Option(None, "--link-template", help="Hyperlink template, e.g. 'planets/{planet_id}.html'"),
    dedupe: bool = typer.Option(False, "--dedupe/--no-dedupe", help="Remove duplicate planet_ids and identical latent rows."),
    title: str = typer.Option("SpectraMind V50 — UMAP Latents", help="Plot title"),
    html_out: Path = typer.Option(Path("outputs/umap/umap_v50.html"), help="Output HTML path."),
    png_out: Optional[Path] = typer.Option(None, help="Optional static PNG path (requires kaleido)."),
    outdir: Path = typer.Option(Path("outputs/umap"), help="Directory for manifest, defaults next to HTML."),
    open_html: bool = typer.Option(False, "--open-html/--no-open-html", help="Open HTML in browser on completion."),
    log_path: Path = typer.Option(Path("logs/v50_debug_log.md"), help="Append audit line to this log."),
):
    """
    Generate UMAP projection + interactive Plotly figure with overlays and hyperlinks.
    """
    try:
        # Load core inputs
        lat = _read_npy(latents)
        pids = _read_planet_ids(planet_ids)
        if lat.shape[0] != len(pids):
            raise ValueError(f"Row mismatch: latents N={lat.shape[0]} vs planet_ids={len(pids)}")

        if dedupe:
            lat, pids = _dedupe(lat, pids)
            console.print(f"[green]INFO[/] After de-duplication: N={lat.shape[0]}")

        labels = _read_labels_csv(labels_csv)
        sym_raw = _read_json(symbolic_json)
        shap_raw = _read_json(shap_json)
        conf = _read_confidence_csv(confidence_csv)

        # Prepare symbolic & SHAP summaries
        # Expect symbolic format: {planet_id: {rule: score, ...}}
        dom_rules = _compute_symbolic_dominant_rule(sym_raw if isinstance(sym_raw, dict) else {})
        # Expect shap format: {planet_id: ...}
        shap_strength = _compute_shap_strength(shap_raw if isinstance(shap_raw, dict) else {})

        console.print(Panel.fit("Fitting UMAP...", style="cyan"))
        emb = _umap_project(
            latents=lat, dim=dim, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state
        )

        console.print(Panel.fit("Building Plotly figure...", style="cyan"))
        fig = _figure_umap(
            emb2=emb,
            planet_ids=pids,
            labels=labels,
            dom_rules=dom_rules,
            shap_strength=shap_strength,
            confidence=conf,
            link_template=link_template,
            title=title,
            dim=dim,
            color_by=color_by.lower().strip(),
            size_by=(size_by.lower().strip() if size_by else None),
            opacity_by=(opacity_by.lower().strip() if opacity_by else None),
        )

        _save_html(fig, html_out)
        _save_png(fig, png_out)

        meta = dict(
            timestamp=dt.datetime.utcnow().isoformat(),
            latents=str(latents),
            planet_ids=str(planet_ids),
            labels_csv=str(labels_csv) if labels_csv else None,
            symbolic_json=str(symbolic_json) if symbolic_json else None,
            shap_json=str(shap_json) if shap_json else None,
            confidence_csv=str(confidence_csv) if confidence_csv else None,
            dim=dim, n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=random_state,
            color_by=color_by, size_by=size_by, opacity_by=opacity_by, link_template=link_template,
            html_out=str(html_out), png_out=(str(png_out) if png_out else None),
            N=lat.shape[0], D=lat.shape[1],
        )
        _save_manifest(outdir, meta)
        _append_audit(log_path, f"html={html_out.as_posix()} N={lat.shape[0]} dim={dim} color_by={color_by}")

        console.print(Panel.fit(f"UMAP report written to: {html_out}", style="green"))
        if open_html:
            import webbrowser
            webbrowser.open(f"file://{html_out.resolve().as_posix()}")

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted[/]")
        raise typer.Exit(code=130)
    except Exception as e:
        console.print(Panel.fit(f"ERROR: {e}", style="red"))
        raise typer.Exit(code=1)

@app.callback()
def _cb():
    """
    SpectraMind V50 — UMAP Latent Visualizer. De-duplication, overlays, hyperlinks, 2D/3D.
    """
    pass

if __name__ == "__main__":
    app()
