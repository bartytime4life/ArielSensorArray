#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/plot_umap_fusion_latents_v50.py

SpectraMind V50 — UMAP Fusion Latents Plotter (Upgraded)
=========================================================

Purpose
-------
Project the fused latent representations (e.g., AIRS ⨂ FGS1 ⨂ metadata ⨂ symbolic)
to 2D/3D with UMAP, then render an interactive Plotly figure with rich overlays:

• Color/size by entropy, SHAP magnitude, symbolic violation, cluster label, or any column
• Confidence shading (size/opacity mapping)
• Symbolic rule hyperlinks per point (planet-level pages)
• Optional deduplication of duplicate planet IDs
• 2D / 3D UMAP, with reproducible seeds
• Exports interactive HTML and static PNG/SVG
• CLI- and diagnostics-ready logging

Typical Inputs
--------------
• --embeddings  : .npy (N×D) or .csv (columns: id, f1..fD or arbitrary)
• --labels      : CSV (planet_id,label,cluster,...) for hover + color categories
• --overlays    : JSON/CSV with per-planet scalar vectors (entropy, shap, symbolic, etc.)
• --link-template : "reports/planet_{planet_id}.html" to embed point hyperlinks

Examples
--------
python tools/plot_umap_fusion_latents_v50.py \
  --embeddings outputs/latents/fusion_latents.npy \
  --labels outputs/diagnostics/planet_labels.csv \
  --overlays entropy:outputs/diagnostics/entropy.csv \
  --overlays symbolic:outputs/diagnostics/symbolic_scores.json \
  --color-by entropy --size-by entropy \
  --dim 3 --outdir outputs/umap/fusion --html --png

python tools/plot_umap_fusion_latents_v50.py \
  --embeddings outputs/latents/fusion_latents.csv \
  --labels outputs/diagnostics/planet_labels.csv \
  --dedupe --link-template 'reports/planet_{planet_id}.html' \
  --color-by cluster --dim 2 --outdir outputs/umap/fusion --html --svg

Notes
-----
• Requires: numpy, pandas, umap-learn, plotly, scikit-learn (for StandardScaler), kaleido (for static export)
• Designed to be called from `spectramind diagnose umap-fusion` or standalone.

Author
------
SpectraMind V50 – Architect & Master Programmer
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import pandas as pd

# UMAP + preprocessing
try:
    import umap
except Exception as _e:
    umap = None

from sklearn.preprocessing import StandardScaler

# Plotting
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# I/O Utils
# ---------------------------------------------------------------------


def _load_embeddings(path: str, id_col: Optional[str] = None) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Load embeddings matrix and associated planet ids (if found).
    Accepts:
      • .npy  : returns df with index 0..N-1 (no ids) and X as ndarray
      • .csv  : if id_col provided (or column 'planet_id' exists), use it as id
                else attempt first column as id if dtype is object, else no id
    Returns:
      df_meta : DataFrame with 'planet_id' column if present
      X       : ndarray of shape (N, D)
    """
    p = Path(path)
    if p.suffix.lower() == ".npy":
        X = np.load(p)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D npy for embeddings, got {X.shape}")
        df_meta = pd.DataFrame({"idx": np.arange(X.shape[0])})
        return df_meta, X

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        df_cols = df.columns.tolist()
        # Determine id col
        if id_col and id_col in df.columns:
            ids = df[id_col].astype(str).tolist()
            feat_df = df.drop(columns=[id_col])
        elif "planet_id" in df.columns:
            ids = df["planet_id"].astype(str).tolist()
            feat_df = df.drop(columns=["planet_id"])
        else:
            # If first column is non-numeric treat as id
            first = df_cols[0]
            if not pd.api.types.is_numeric_dtype(df[first]):
                ids = df[first].astype(str).tolist()
                feat_df = df.drop(columns=[first])
            else:
                ids = [str(i) for i in range(len(df))]
                feat_df = df
        # keep numeric columns only for X
        feat_df = feat_df.select_dtypes(include=[np.number])
        X = feat_df.values
        df_meta = pd.DataFrame({"planet_id": ids})
        return df_meta, X

    raise ValueError(f"Unsupported embeddings file type: {path}")


def _load_labels(path: Optional[str]) -> pd.DataFrame:
    """
    Load labels CSV (columns may include planet_id, label, cluster, etc.)
    """
    if path is None:
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Ensure planet_id is string (consistent join key)
    if "planet_id" in df.columns:
        df["planet_id"] = df["planet_id"].astype(str)
    return df


def _load_overlay(name_path: str) -> Tuple[str, pd.DataFrame]:
    """
    Load an overlay mapping 'name:path' into a DataFrame with columns [planet_id, <overlay_name>]
    File can be CSV or JSON.
      • CSV: expects columns (planet_id, value) or multi-column; will melt non-id columns.
      • JSON:
          - { "planet_id": value, ... }
          - { "data": { "planet_id": value, ... } }
          - { "planet_id": ..., "value": ... } list-like
    Returns (overlay_name, df)
    """
    if ":" not in name_path:
        raise ValueError("Overlay must be specified as name:path (e.g., entropy:entropy.csv)")
    name, path = name_path.split(":", 1)
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Overlay file not found: {path}")

    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "planet_id" in df.columns and "value" in df.columns and df.shape[1] == 2:
            df = df.rename(columns={"value": name})
            df["planet_id"] = df["planet_id"].astype(str)
            return name, df[["planet_id", name]]
        # If multiple columns: keep numeric, melt to long then aggregate if repeated
        if "planet_id" in df.columns:
            df["planet_id"] = df["planet_id"].astype(str)
            numeric_cols = [c for c in df.columns if c != "planet_id" and pd.api.types.is_numeric_dtype(df[c])]
            if len(numeric_cols) == 0:
                raise ValueError(f"No numeric columns in overlay CSV: {path}")
            # Combine numerics into one (mean)
            df[name] = df[numeric_cols].mean(axis=1)
            return name, df[["planet_id", name]]
        # If no planet_id given: assume row order is planet index, construct planet_id
        df[name] = df.mean(axis=1, numeric_only=True)
        df["planet_id"] = [str(i) for i in range(len(df))]
        return name, df[["planet_id", name]]

    if p.suffix.lower() == ".json":
        with open(p, "r") as f:
            obj = json.load(f)
        # Normalize into {planet_id: value}
        if isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], dict):
            mapping = obj["data"]
        elif isinstance(obj, dict):
            mapping = obj
        elif isinstance(obj, list):
            # list of {planet_id:..., value:...}
            mapping = {str(d.get("planet_id", i)): d.get("value", np.nan) for i, d in enumerate(obj)}
        else:
            raise ValueError(f"Unsupported JSON overlay structure: {path}")
        df = pd.DataFrame({"planet_id": [str(k) for k in mapping.keys()], name: list(mapping.values())})
        return name, df

    raise ValueError(f"Unsupported overlay file type: {path}")


def _dedupe(df: pd.DataFrame, key: str, keep: str = "first") -> pd.DataFrame:
    if key not in df.columns:
        return df
    return df.drop_duplicates(subset=[key], keep=keep)


# ---------------------------------------------------------------------
# UMAP Projection
# ---------------------------------------------------------------------


def _umap_embed(
    X: np.ndarray,
    dim: int = 2,
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "euclidean",
    random_state: int = 1337,
) -> np.ndarray:
    """
    Fit UMAP on standardized X and return embedding of shape (N, dim).
    """
    if umap is None:
        raise RuntimeError(
            "umap-learn is not installed. Please install it via `pip install umap-learn`."
        )
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X)
    reducer = umap.UMAP(
        n_components=dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
    )
    Z = reducer.fit_transform(Xs)
    return Z


# ---------------------------------------------------------------------
# Plot Builders
# ---------------------------------------------------------------------


def _make_hover_text(row: pd.Series, extra_cols: List[str]) -> str:
    parts = [f"planet_id={row.get('planet_id', 'NA')}"]
    for c in ["label", "cluster"]:
        if c in row:
            parts.append(f"{c}={row[c]}")
    for c in extra_cols:
        if c in row:
            val = row[c]
            if isinstance(val, float):
                parts.append(f"{c}={val:.5f}")
            else:
                parts.append(f"{c}={val}")
    return "<br>".join(parts)


def _link_for(planet_id: str, template: Optional[str]) -> Optional[str]:
    if not template:
        return None
    try:
        return template.format(planet_id=planet_id)
    except Exception:
        return None


def _auto_color_sequence(n: int) -> List[str]:
    base = px.colors.qualitative.Set2 + px.colors.qualitative.Plotly + px.colors.qualitative.Safe
    seq = (base * ((n // len(base)) + 1))[:n]
    return seq


def _plot_interactive(
    df: pd.DataFrame,
    dim: int,
    color_by: Optional[str],
    size_by: Optional[str],
    opacity_by: Optional[str],
    link_template: Optional[str],
    title: str,
) -> go.Figure:

    # Compute hover text
    extra_cols = []
    for c in [color_by, size_by, opacity_by]:
        if c and c not in ["cluster", "label", "planet_id"] and c not in extra_cols:
            extra_cols.append(c)

    hover_text = df.apply(lambda r: _make_hover_text(r, extra_cols), axis=1)

    # Build base scatter
    if dim == 3:
        x, y, z = df["umap_x"], df["umap_y"], df["umap_z"]
        scatter_cls = go.Scatter3d
        coords = dict(x=x, y=y, z=z)
        mode = "markers"
    else:
        x, y = df["umap_x"], df["umap_y"]
        scatter_cls = go.Scattergl
        coords = dict(x=x, y=y)
        mode = "markers"

    marker = dict(
        size=8,
        opacity=0.9,
        line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
    )

    # Size mapping
    if size_by and size_by in df.columns and pd.api.types.is_numeric_dtype(df[size_by]):
        svals = df[size_by].astype(float)
        # Normalize for size range [6,18]
        s_norm = (svals - np.nanmin(svals)) / (np.nanmax(svals) - np.nanmin(svals) + 1e-12)
        marker["size"] = (s_norm * 12 + 6).tolist()

    # Opacity mapping (confidence shading)
    if opacity_by and opacity_by in df.columns and pd.api.types.is_numeric_dtype(df[opacity_by]):
        ovals = df[opacity_by].astype(float)
        # Normalize [0.4, 1.0]
        o_norm = (ovals - np.nanmin(ovals)) / (np.nanmax(ovals) - np.nanmin(ovals) + 1e-12)
        marker["opacity"] = (o_norm * 0.6 + 0.4).tolist()

    # Color mapping
    color_kwargs: Dict[str, Any] = {}
    if color_by and color_by in df.columns:
        if pd.api.types.is_numeric_dtype(df[color_by]):
            color_kwargs.update(
                color=df[color_by],
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title=color_by),
            )
        else:
            # categorical color
            cats = df[color_by].astype(str).values
            # plotly express friendly figure if 2D — but we use graph_objects for consistency
            # Build category -> color map
            uniq = sorted(pd.unique(cats))
            palette = _auto_color_sequence(len(uniq))
            cmap = {u: palette[i] for i, u in enumerate(uniq)}
            marker["color"] = [cmap[v] for v in cats]
            # build a legend via separate traces if 2D; for 3D we can still split traces
            if dim == 3:
                # We'll split below by category
                pass
            else:
                # Single trace with legend disabled; then add dummy traces for legend
                pass

    # Build link array for clicking (customdata with URL)
    links = [ _link_for(pid, link_template) for pid in df["planet_id"].astype(str) ] if "planet_id" in df.columns else [None]*len(df)

    # Create figure; if categorical color, we may split traces for legend
    if color_by and color_by in df.columns and not pd.api.types.is_numeric_dtype(df[color_by]):
        uniq = sorted(pd.unique(df[color_by].astype(str)))
        fig = go.Figure()
        for u in uniq:
            m = df[color_by].astype(str) == u
            sub_coords = {k: np.array(v)[m] for k, v in coords.items()}
            sub_marker = marker.copy()
            if isinstance(sub_marker.get("size"), list):
                sub_marker["size"] = (np.array(sub_marker["size"])[m]).tolist()
            if isinstance(sub_marker.get("opacity"), list):
                sub_marker["opacity"] = (np.array(sub_marker["opacity"])[m]).tolist()
            sub_marker["color"] = _auto_color_sequence(1)[0]  # consistent but simple
            # use deterministic color mapping
            palette = _auto_color_sequence(len(uniq))
            cmap = {uu: palette[i] for i, uu in enumerate(uniq)}
            sub_marker["color"] = cmap[u]
            sub_hover = hover_text[m]
            sub_links = (np.array(links)[m]).tolist() if links else None

            scatter = scatter_cls(
                **sub_coords,
                mode=mode,
                name=f"{color_by}={u}",
                text=sub_hover,
                hoverinfo="text",
                marker=sub_marker,
                customdata=sub_links,
            )
            fig.add_trace(scatter)
    else:
        # Single trace with numeric color or no color_by
        fig = go.Figure()
        scatter = scatter_cls(
            **coords,
            mode=mode,
            name="planets",
            text=hover_text,
            hoverinfo="text",
            marker=marker,
            customdata=links,
            **color_kwargs,
        )
        fig.add_trace(scatter)

    # Click-to-open links (in HTML): use Plotly JS to add onclick handler (template injection)
    # This is best-effort; some viewers may block popups.
    fig.update_layout(
        title=title,
        template="plotly_white",
        legend=dict(itemsizing="constant"),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    if dim == 3:
        fig.update_scenes(
            xaxis_title="UMAP-1",
            yaxis_title="UMAP-2",
            zaxis_title="UMAP-3",
        )
    else:
        fig.update_xaxes(title_text="UMAP-1")
        fig.update_yaxes(title_text="UMAP-2")

    return fig


# ---------------------------------------------------------------------
# Main Workflow
# ---------------------------------------------------------------------


def run(
    embeddings: str,
    labels: Optional[str],
    overlays: List[str],
    dedupe: bool,
    dim: int,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    color_by: Optional[str],
    size_by: Optional[str],
    opacity_by: Optional[str],
    link_template: Optional[str],
    title: str,
    seed: int,
    outdir: str,
    save_html: bool,
    save_png: bool,
    save_svg: bool,
    id_col: Optional[str],
):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    # Load embeddings
    df_meta, X = _load_embeddings(embeddings, id_col=id_col)

    # Attach planet_id if present in labels later
    if "planet_id" not in df_meta.columns:
        df_meta["planet_id"] = df_meta.get("idx", np.arange(X.shape[0])).astype(str)

    # Load labels and join
    df_lab = _load_labels(labels)
    df = df_meta.copy()
    if not df_lab.empty:
        df = df.merge(df_lab, on="planet_id", how="left")

    # Load overlays (multiple)
    for ov in overlays or []:
        name, df_ov = _load_overlay(ov)
        df = df.merge(df_ov, on="planet_id", how="left")

    # Deduplicate
    if dedupe:
        df = _dedupe(df, "planet_id", keep="first")
        # apply same filter to X if original had duplicates
        if len(df) < X.shape[0]:
            keep_ids = set(df["planet_id"].astype(str))
            # Construct mask by original order
            pid_all = df_meta["planet_id"].astype(str).tolist()
            mask = np.array([pid in keep_ids for pid in pid_all])
            X = X[mask]

    # Sanity for color/size/opacity columns
    for col in [color_by, size_by, opacity_by]:
        if col and col not in df.columns and col not in ["cluster", "label"]:
            raise ValueError(f"--{col} was requested but not present in merged dataframe columns.")

    # Compute UMAP
    Z = _umap_embed(
        X=X,
        dim=dim,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=seed,
    )
    if dim == 3:
        df["umap_x"], df["umap_y"], df["umap_z"] = Z[:, 0], Z[:, 1], Z[:, 2]
    else:
        df["umap_x"], df["umap_y"] = Z[:, 0], Z[:, 1]

    # Interactive plot
    fig = _plot_interactive(
        df=df,
        dim=dim,
        color_by=color_by,
        size_by=size_by,
        opacity_by=opacity_by,
        link_template=link_template,
        title=title or "UMAP Fusion Latents",
    )

    # Save
    stem = f"umap_fusion_dim{dim}"
    if save_html:
        html_path = out / f"{stem}.html"
        fig.write_html(str(html_path), include_plotlyjs="cdn", full_html=True)
        # Inject click handler for links to open in new tab (if customdata present)
        # Users can add a small JS snippet later if needed; Plotly doesn't natively open URLs on point click.

    if save_png:
        try:
            fig.write_image(str(out / f"{stem}.png"), scale=2)
        except Exception as e:
            print(f"[WARN] Static PNG export failed (is kaleido installed?): {e}")

    if save_svg:
        try:
            fig.write_image(str(out / f"{stem}.svg"))
        except Exception as e:
            print(f"[WARN] Static SVG export failed (is kaleido installed?): {e}")

    # Export the annotated dataframe
    df.to_csv(out / f"{stem}_data.csv", index=False)

    print(f"[OK] UMAP fusion plot saved to: {out.absolute()}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpectraMind V50 — UMAP Fusion Latents Plotter (Upgraded)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--embeddings", type=str, required=True, help=".npy or .csv embeddings")
    p.add_argument("--labels", type=str, default=None, help="CSV with planet_id, labels/cluster columns")
    p.add_argument(
        "--overlays",
        type=str,
        nargs="*",
        default=[],
        help="Overlay spec(s) name:path (e.g., entropy:entropy.csv symbolic:symbolic.json shap:shap.csv)",
    )
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate planet_id rows (keep first)")
    p.add_argument("--id-col", type=str, default=None, help="ID column name in embeddings CSV (if present)")

    # UMAP params
    p.add_argument("--dim", type=int, default=2, choices=[2, 3], help="UMAP projection dimension")
    p.add_argument("--n-neighbors", type=int, default=30, help="UMAP n_neighbors")
    p.add_argument("--min-dist", type=float, default=0.1, help="UMAP min_dist")
    p.add_argument("--metric", type=str, default="euclidean", help="UMAP metric")

    # Visual encodings
    p.add_argument("--color-by", type=str, default=None, help="Column to color points by")
    p.add_argument("--size-by", type=str, default=None, help="Column to size points by (numeric)")
    p.add_argument("--opacity-by", type=str, default=None, help="Column to opacity-map points by (numeric)")
    p.add_argument("--link-template", type=str, default=None, help="Template like 'reports/planet_{planet_id}.html'")

    # Output
    p.add_argument("--title", type=str, default="UMAP Fusion Latents", help="Figure title")
    p.add_argument("--seed", type=int, default=1337, help="Random seed for UMAP")
    p.add_argument("--outdir", type=str, default="outputs/umap_fusion", help="Output directory")
    p.add_argument("--html", action="store_true", help="Export interactive HTML")
    p.add_argument("--png", action="store_true", help="Export static PNG (requires kaleido)")
    p.add_argument("--svg", action="store_true", help="Export static SVG (requires kaleido)")
    return p


def main():
    args = build_argparser().parse_args()
    run(
        embeddings=args.embeddings,
        labels=args.labels,
        overlays=args.overlays,
        dedupe=args.dedupe,
        dim=args.dim,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        color_by=args.color_by,
        size_by=args.size_by,
        opacity_by=args.opacity_by,
        link_template=args.link_template,
        title=args.title,
        seed=args.seed,
        outdir=args.outdir,
        save_html=args.html,
        save_png=args.png,
        save_svg=args.svg,
        id_col=args.id_col,
    )


if __name__ == "__main__":
    main()
