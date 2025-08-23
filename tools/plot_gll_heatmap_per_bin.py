#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/visualization/plot_tsne_interactive.py

SpectraMind V50 — t‑SNE Latent Visualizer (Upgraded, Challenge‑Grade)
=====================================================================

Purpose
-------
Project high‑dimensional latent vectors to 2D or 3D with t‑SNE and render a rich,
interactive Plotly figure (HTML) plus optional static PNG. Supports:
• Symbolic & SHAP overlays (hover + coloring)
• Confidence shading (marker size/opacity)
• Planet‑level hyperlinks via template
• De‑duplication (repeated planet_ids / duplicate latents)
• Optional PCA pre‑reduction + standardization
• 2D/3D modes, fast Barnes‑Hut (2D) or exact solver
• CLI‑first (Typer) with audit logging to logs/v50_debug_log.md
• Manifest JSON for reproducibility / dashboard pickup

Typical usage
-------------
python -m src.visualization.plot_tsne_interactive run \
  --latents outputs/latents.npy \
  --planet-ids data/metadata/planet_ids.csv \
  --labels-csv outputs/labels.csv \
  --symbolic-json outputs/diagnostics/symbolic_results.json \
  --shap-json outputs/diagnostics/shap_overlay.json \
  --confidence-csv outputs/diagnostics/confidence.csv \
  --dim 2 \
  --perplexity 30 \
  --learning-rate 200 \
  --n-iter 1000 \
  --pca-components 64 \
  --standardize \
  --color-by label \
  --size-by confidence \
  --opacity-by confidence \
  --link-template "planets/{planet_id}.html" \
  --title "t‑SNE — SpectraMind V50" \
  --html-out outputs/tsne/tsne_v50.html \
  --png-out outputs/tsne/tsne_v50.png \
  --dedupe \
  --open-html
"""

from __future__ import annotations

import csv
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# Optional dependencies guarded for graceful error messages
try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception:
    TSNE = None  # type: ignore
    PCA = None   # type: ignore
    StandardScaler = None  # type: ignore

try:
    import plotly.graph_objects as go
except Exception:
    go = None  # type: ignore

# =====================================================================================
# I/O helpers
# =====================================================================================

def _read_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return np.load(str(path))

def _read_planet_ids(path: Path) -> List[str]:
    """
    Supports .txt (one id per line) or .csv (first column planet_id).
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing planet ids file: {path}")
    if path.suffix.lower() == ".txt":
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    # CSV
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
        for row in reader:
            pid = str(row.get("planet_id", "")).strip()
            lab = str(row.get("label", "")).strip()
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
        for row in reader:
            pid = str(row.get("planet_id", "")).strip()
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

# =====================================================================================
# Overlays & feature engineering
# =====================================================================================

def _compute_symbolic_dominant_rule(symbolic_per_planet: Dict[str, Dict[str, float]]) -> Dict[str, str]:
    """
    symbolic_per_planet: {planet_id: {rule_name: score, ...}}
    Returns dominant rule (max |score|) label per planet.
    """
    out: Dict[str, str] = {}
    for pid, rules in symbolic_per_planet.items():
        if not isinstance(rules, dict) or not rules:
            continue
        k = max(rules, key=lambda r: abs(rules[r]) if rules[r] is not None else -np.inf)
        out[pid] = str(k)
    return out

def _compute_shap_strength(shap_per_planet: Dict[str, Any]) -> Dict[str, float]:
    """
    shap_per_planet: {planet_id: {...}}; Summarize to a scalar magnitude.
    Priority keys: mean_abs|magnitude|avg_abs|avg_abs_shap, else mean(|values|).
    """
    out: Dict[str, float] = {}
    for pid, obj in shap_per_planet.items():
        if isinstance(obj, dict):
            for key in ("mean_abs", "magnitude", "avg_abs", "avg_abs_shap"):
                if key in obj and np.isfinite(obj[key]):
                    out[pid] = float(obj[key])
                    break
            else:
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

def _norm_or_default(vals: List[float], a: float, b: float) -> List[float]:
    """
    Normalize list to [a,b]; if degenerate/empty returns midpoints.
    """
    if not vals:
        return []
    arr = np.asarray(vals, dtype=float)
    m, M = np.nanmin(arr), np.nanmax(arr)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m:
        return [0.5 * (a + b) for _ in vals]
    x = (arr - m) / (M - m)
    return list(a + (b - a) * x)

# =====================================================================================
# De‑duplication
# =====================================================================================

def _dedupe(latents: np.ndarray, planet_ids: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Remove duplicate planet_ids (keep first occurrence) and identical latent rows.
    """
    # De‑dupe by id
    seen = set()
    keep_idx: List[int] = []
    for i, pid in enumerate(planet_ids):
        if pid in seen:
            continue
        seen.add(pid)
        keep_idx.append(i)
    lat1 = latents[keep_idx]
    pids1 = [planet_ids[i] for i in keep_idx]

    # De‑dupe by latent row equality
    lat_flat = np.ascontiguousarray(lat1).view(
        np.dtype((np.void, lat1.dtype.itemsize * lat1.shape[1]))
    )
    _, uniq_indices = np.unique(lat_flat, return_index=True)
    uniq_indices = sorted(list(uniq_indices))
    return lat1[uniq_indices], [pids1[i] for i in uniq_indices]

# =====================================================================================
# t‑SNE projection
# =====================================================================================

def _tsne_project(
    latents: np.ndarray,
    dim: int,
    perplexity: float,
    learning_rate: float,
    n_iter: int,
    metric: str,
    angle: float,
    method: str,
    random_state: int,
    pca_components: Optional[int],
    standardize: bool,
) -> np.ndarray:
    """
    Compute t‑SNE embedding with optional PCA pre‑reduction + standardization.
    Uses sklearn.manifold.TSNE; Barnes‑Hut is only valid for 2D (n_components=2).
    """
    if TSNE is None or PCA is None:
        raise RuntimeError("scikit‑learn is required. Please `pip install scikit-learn`.")

    X = latents
    if standardize:
        if StandardScaler is None:
            raise RuntimeError("scikit‑learn StandardScaler not found.")
        X = StandardScaler().fit_transform(X)

    if pca_components and pca_components > 0 and pca_components < X.shape[1]:
        X = PCA(n_components=pca_components, random_state=random_state).fit_transform(X)

    # Validate method for dimension
    _method = method.lower().strip()
    if dim != 2 and _method == "barnes_hut":
        console.print("[yellow]WARN[/] Barnes‑Hut is only supported for 2D; switching to exact.")
        _method = "exact"

    tsne = TSNE(
        n_components=dim,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        init="pca",
        random_state=random_state,
        metric=metric,
        angle=angle if _method == "barnes_hut" else 0.0,  # angle only for barnes_hut
        method=_method,
        verbose=1,
    )
    emb = tsne.fit_transform(X)
    return emb

# =====================================================================================
# Plotly utilities
# =====================================================================================

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

def _marker_style(
    indices: List[int],
    planet_ids: List[str],
    size_by: Optional[str],
    opacity_by: Optional[str],
    shap_strength: Dict[str, float],
    confidence: Dict[str, float],
) -> Tuple[List[float], List[float]]:
    # Size mapping
    if size_by == "shap":
        vals = [shap_strength.get(planet_ids[i], np.nan) for i in indices]
        sizes = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=6.0, b=14.0)
    elif size_by == "confidence":
        vals = [confidence.get(planet_ids[i], np.nan) for i in indices]
        sizes = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=6.0, b=16.0)
    else:
        sizes = [8.0 for _ in indices]

    # Opacity mapping
    if opacity_by == "confidence":
        vals = [confidence.get(planet_ids[i], np.nan) for i in indices]
        opac = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=0.35, b=0.95)
    elif opacity_by == "shap":
        vals = [shap_strength.get(planet_ids[i], np.nan) for i in indices]
        opac = _norm_or_default([0 if not np.isfinite(v) else v for v in vals], a=0.35, b=0.95)
    else:
        opac = [0.85 for _ in indices]
    return sizes, opac

def _figure_tsne(
    emb: np.ndarray,
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
) -> "go.Figure":
    if go is None:
        raise RuntimeError("plotly is required. Please `pip install plotly`.")
    xs = emb[:, 0]
    ys = emb[:, 1]
    zs = emb[:, 2] if dim == 3 else None

    # Coloring strategies
    if color_by == "label":
        cats = sorted({labels.get(pid, "unknown") for pid in planet_ids})
        traces = []
        for cat in cats:
            idx = [i for i, pid in enumerate(planet_ids) if labels.get(pid, "unknown") == cat]
            if not idx:
                continue
            hover = [
                _build_hover(
                    planet_ids[i],
                    labels.get(planet_ids[i]),
                    dom_rules.get(planet_ids[i]),
                    shap_strength.get(planet_ids[i]),
                    confidence.get(planet_ids[i]),
                ) for i in idx
            ]
            url = None
            if link_template:
                url = [link_template.format(planet_id=planet_ids[i]) for i in idx]
            size, opac = _marker_style(idx, planet_ids, size_by, opacity_by, shap_strength, confidence)
            common = dict(
                mode="markers", name=str(cat), text=hover, hoverinfo="text",
                marker=dict(size=size, opacity=opac, symbol="circle"),
                customdata=url if link_template else None
            )
            tr = (go.Scatter3d if dim == 3 else go.Scattergl)(
                x=xs[idx], y=ys[idx], **({"z": zs[idx]} if dim == 3 else {}), **common
            )
            traces.append(tr)
        fig = go.Figure(data=traces)
    elif color_by == "symbolic":
        cats = sorted({dom_rules.get(pid, "none") for pid in planet_ids})
        traces = []
        for cat in cats:
            idx = [i for i, pid in enumerate(planet_ids) if dom_rules.get(pid, "none") == cat]
            if not idx:
                continue
            hover = [
                _build_hover(
                    planet_ids[i],
                    labels.get(planet_ids[i]),
                    dom_rules.get(planet_ids[i]),
                    shap_strength.get(planet_ids[i]),
                    confidence.get(planet_ids[i]),
                ) for i in idx
            ]
            url = None
            if link_template:
                url = [link_template.format(planet_id=planet_ids[i]) for i in idx]
            size, opac = _marker_style(idx, planet_ids, size_by, opacity_by, shap_strength, confidence)
            common = dict(
                mode="markers", name=f"sym:{cat}", text=hover, hoverinfo="text",
                marker=dict(size=size, opacity=opac, symbol="circle"),
                customdata=url if link_template else None
            )
            tr = (go.Scatter3d if dim == 3 else go.Scattergl)(
                x=xs[idx], y=ys[idx], **({"z": zs[idx]} if dim == 3 else {}), **common
            )
            traces.append(tr)
        fig = go.Figure(data=traces)
    elif color_by == "shap":
        c = np.array([shap_strength.get(pid, np.nan) for pid in planet_ids], dtype=float)
        c = np.where(np.isfinite(c), c, np.nan)
        cmin = np.nanmin(c) if np.any(np.isfinite(c)) else 0.0
        cmax = np.nanmax(c) if np.any(np.isfinite(c)) else 1.0
        c = np.where(np.isfinite(c), c, 0.5 * (cmin + cmax))
        hover = [
            _build_hover(pid, labels.get(pid), dom_rules.get(pid), shap_strength.get(pid), confidence.get(pid))
            for pid in planet_ids
        ]
        url = None
        if link_template:
            url = [link_template.format(planet_id=pid) for pid in planet_ids]
        size, opac = _marker_style(list(range(len(planet_ids))), planet_ids, size_by, opacity_by, shap_strength, confidence)
        common = dict(
            mode="markers",
            text=hover, hoverinfo="text",
            marker=dict(size=size, opacity=opac, color=c, colorscale="Viridis", showscale=True, colorbar=dict(title="SHAP")),
            customdata=url if link_template else None,
            name="t‑SNE"
        )
        tr = (go.Scatter3d if dim == 3 else go.Scattergl)(
            x=xs, y=ys, **({"z": zs} if dim == 3 else {}), **common
        )
        fig = go.Figure(data=[tr])
    elif color_by == "confidence":
        c = np.array([confidence.get(pid, np.nan) for pid in planet_ids], dtype=float)
        c = np.where(np.isfinite(c), c, np.nan)
        cmin = np.nanmin(c) if np.any(np.isfinite(c)) else 0.0
        cmax = np.nanmax(c) if np.any(np.isfinite(c)) else 1.0
        c = np.where(np.isfinite(c), c, 0.5 * (cmin + cmax))
        hover = [
            _build_hover(pid, labels.get(pid), dom_rules.get(pid), shap_strength.get(pid), confidence.get(pid))
            for pid in planet_ids
        ]
        url = None
        if link_template:
            url = [link_template.format(planet_id=pid) for pid in planet_ids]
        size, opac = _marker_style(list(range(len(planet_ids))), planet_ids, size_by, opacity_by, shap_strength, confidence)
        common = dict(
            mode="markers",
            text=hover, hoverinfo="text",
            marker=dict(size=size, opacity=opac, color=c, colorscale="Blues", showscale=True, colorbar=dict(title="Confidence")),
            customdata=url if link_template else None,
            name="t‑SNE"
        )
        tr = (go.Scatter3d if dim == 3 else go.Scattergl)(
            x=xs, y=ys, **({"z": zs} if dim == 3 else {}), **common
        )
        fig = go.Figure(data=[tr])
    else:
        # Single color
        hover = [
            _build_hover(pid, labels.get(pid), dom_rules.get(pid), shap_strength.get(pid), confidence.get(pid))
            for pid in planet_ids
        ]
        url = None
        if link_template:
            url = [link_template.format(planet_id=pid) for pid in planet_ids]
        size, opac = _marker_style(list(range(len(planet_ids))), planet_ids, size_by, opacity_by, shap_strength, confidence)
        common = dict(
            mode="markers",
            text=hover, hoverinfo="text",
            marker=dict(size=size, opacity=opac, symbol="circle"),
            customdata=url if link_template else None,
            name="t‑SNE"
        )
        tr = (go.Scatter3d if dim == 3 else go.Scattergl)(
            x=xs, y=ys, **({"z": zs} if dim == 3 else {}), **common
        )
        fig = go.Figure(data=[tr])

    # Layout
    fig.update_layout(
        title=title,
        template="plotly_dark",
        margin=dict(l=10, r=10, t=60, b=10),
    )
    if dim == 3:
        fig.update_layout(scene=dict(xaxis_title="t‑SNE‑1", yaxis_title="t‑SNE‑2", zaxis_title="t‑SNE‑3"))
    else:
        fig.update_xaxes(title="t‑SNE‑1")
        fig.update_yaxes(title="t‑SNE‑2")

    return fig

# =====================================================================================
# Save utilities
# =====================================================================================

def _save_html(fig: "go.Figure", out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out), include_plotlyjs="cdn", full_html=True)

def _save_png(fig: "go.Figure", out: Optional[Path]):
    if not out:
        return
    try:
        import kaleido  # noqa: F401
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.write_image(str(out), scale=2.0, height=900, width=1200)
    except Exception as e:
        console.print(f"[yellow]WARN[/] Could not export PNG via kaleido: {e}")

def _save_manifest(outdir: Path, meta: Dict[str, Any]):
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "tsne_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

def _append_audit(log_path: Path, message: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- [{dt.datetime.now().isoformat(timespec='seconds')}] plot_tsne_interactive: {message}\n")

# =====================================================================================
# CLI
# =====================================================================================

@app.command("run")
def cli_run(
    latents: Path = typer.Option(..., help="Path to latents .npy (N,D)"),
    planet_ids: Path = typer.Option(..., help="Path to planet IDs (.txt or .csv first column planet_id)"),
    labels_csv: Optional[Path] = typer.Option(None, help="Optional CSV with columns: planet_id,label"),
    symbolic_json: Optional[Path] = typer.Option(None, help="Optional symbolic results JSON (per-planet rule scores)"),
    shap_json: Optional[Path] = typer.Option(None, help="Optional SHAP overlay JSON (per-planet arrays or scalars)"),
    confidence_csv: Optional[Path] = typer.Option(None, help="Optional CSV with columns: planet_id,confidence in [0,1]"),
    # t‑SNE params
    dim: int = typer.Option(2, help="Output dimension: 2 or 3", min=2, max=3),
    perplexity: float = typer.Option(30.0, help="Perplexity (roughly neighborhood size)"),
    learning_rate: float = typer.Option(200.0, help="Learning rate"),
    n_iter: int = typer.Option(1000, help="Number of optimization iterations"),
    metric: str = typer.Option("euclidean", help="Distance metric"),
    angle: float = typer.Option(0.5, help="Barnes‑Hut tradeoff angle (0.2‑0.8 typical)"),
    method: str = typer.Option("barnes_hut", help="Method: barnes_hut|exact"),
    random_state: int = typer.Option(33, help="Random seed"),
    pca_components: Optional[int] = typer.Option(None, help="Optional PCA pre‑reduction components (e.g., 64)"),
    standardize: bool = typer.Option(False, "--standardize/--no-standardize", help="Apply StandardScaler before PCA/t‑SNE"),
    # Visualization params
    color_by: str = typer.Option("label", help="Color by: label|symbolic|shap|confidence|none"),
    size_by: Optional[str] = typer.Option(None, help="Size by: shap|confidence"),
    opacity_by: Optional[str] = typer.Option(None, help="Opacity by: shap|confidence"),
    link_template: Optional[str] = typer.Option(None, help="Hyperlink template, e.g. 'planets/{planet_id}.html'"),
    dedupe: bool = typer.Option(False, "--dedupe/--no-dedupe", help="Remove duplicate ids and identical latent rows"),
    # Output
    title: str = typer.Option("SpectraMind V50 — t‑SNE Latents", help="Plot title"),
    html_out: Path = typer.Option(Path("outputs/tsne/tsne_v50.html"), help="Output HTML path"),
    png_out: Optional[Path] = typer.Option(None, help="Optional static PNG path (requires kaleido)"),
    outdir: Path = typer.Option(Path("outputs/tsne"), help="Directory for manifest JSON"),
    open_html: bool = typer.Option(False, "--open-html/--no-open-html", help="Open HTML in browser when done"),
    log_path: Path = typer.Option(Path("logs/v50_debug_log.md"), help="Append audit line here"),
):
    """
    Generate t‑SNE projection + interactive Plotly figure with overlays and hyperlinks.
    """
    try:
        if TSNE is None or go is None:
            raise RuntimeError("Required packages missing: scikit‑learn and plotly are needed.")

        # Load inputs
        lat = _read_npy(latents)
        pids = _read_planet_ids(planet_ids)
        if lat.shape[0] != len(pids):
            raise ValueError(f"Row mismatch: latents N={lat.shape[0]} vs planet_ids={len(pids)}")

        if dedupe:
            lat, pids = _dedupe(lat, pids)
            console.print(f"[green]INFO[/] After de‑duplication: N={lat.shape[0]}")

        labels = _read_labels_csv(labels_csv)
        sym_raw = _read_json(symbolic_json)
        shap_raw = _read_json(shap_json)
        conf = _read_confidence_csv(confidence_csv)

        dom_rules = _compute_symbolic_dominant_rule(sym_raw if isinstance(sym_raw, dict) else {})
        shap_strength = _compute_shap_strength(shap_raw if isinstance(shap_raw, dict) else {})

        console.print(Panel.fit("Running t‑SNE (sklearn)...", style="cyan"))
        emb = _tsne_project(
            latents=lat,
            dim=dim,
            perplexity=perplexity,
            learning_rate=learning_rate,
            n_iter=n_iter,
            metric=metric,
            angle=angle,
            method=method,
            random_state=random_state,
            pca_components=pca_components,
            standardize=standardize,
        )

        console.print(Panel.fit("Building Plotly figure...", style="cyan"))
        fig = _figure_tsne(
            emb=emb,
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
            dim=dim, perplexity=perplexity, learning_rate=learning_rate, n_iter=n_iter,
            metric=metric, angle=angle, method=method, random_state=random_state,
            pca_components=pca_components, standardize=standardize,
            color_by=color_by, size_by=size_by, opacity_by=opacity_by, link_template=link_template,
            html_out=str(html_out), png_out=(str(png_out) if png_out else None),
            N=lat.shape[0], D=lat.shape[1],
        )
        _save_manifest(outdir, meta)
        _append_audit(log_path, f"html={html_out.as_posix()} N={lat.shape[0]} dim={dim} color_by={color_by} method={method}")

        console.print(Panel.fit(f"t‑SNE report written to: {html_out}", style="green"))
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
    SpectraMind V50 — t‑SNE Interactive Visualizer.
    2D/3D, overlays (label/symbolic/SHAP/confidence), hyperlinks, de‑duplication,
    PCA pre‑reduction, audit logging, and HTML/PNG outputs.
    """
    pass

if __name__ == "__main__":
    app()
