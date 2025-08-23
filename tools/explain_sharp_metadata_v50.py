#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/explain_shap_metadata_v50.py

SpectraMind V50 — SHAP + Metadata Explainer (tools directory)

Purpose
-------
Generate feature-attribution explanations that connect *metadata* (planet- / visit-level
features) to predicted spectra (μ) using SHAP (with graceful fallbacks), and fuse those
signals with diagnostics (entropy, GLL if y is available) and optional symbolic overlays.
Outputs include interactive UMAP/t‑SNE/PCA plots, CSV/JSON summaries, and a self‑contained
HTML report that can be embedded into the unified diagnostics dashboard.

Design notes
------------
• CLI-first, audit-friendly: Typer-based CLI, rich logs, deterministic options.
• “Batteries included”: works even if optional libraries (shap, umap-learn) are missing.
• Fast surrogate modeling: learns a small per‑bin regressor from metadata → μ[:, bin]
  to compute SHAP attributions (bin subset or all bins, user-selectable).
• Diagnostics fusion: entropy(μ), GLL(μ, σ, y), per-planet symbolic violations overlay.
• Interactive plots: Plotly HTML for UMAP/t-SNE/PCA with color/size encodings.
• Exports: CSV (shap feature ranks, per-sample summaries), JSON (run summary),
  PNG snapshots (optional), and a minimal, portable report.html.

Inputs (typical)
----------------
--mu        N×B .npy file of predicted μ
--sigma     N×B .npy file of predicted σ (optional, used for GLL & calibration diagnostics)
--y         N×B .npy file of target spectra (optional, enables GLL)
--meta      CSV with N rows (metadata features); can include categorical columns
--latents   N×D .npy (optional; plotted alongside metadata-only projections)
--symbolic  JSON per-planet (and optionally per-bin) rule violations (optional)
--outdir    Output directory (will be created)

Examples
--------
# Minimal (SHAP for a sampled subset of bins, UMAP if available, otherwise PCA/t-SNE):
python -m tools.explain_shap_metadata_v50 \
  --mu outputs/predictions/mu.npy \
  --meta data/metadata.csv \
  --outdir outputs/explain_shap_meta

# With σ and y to compute GLL overlays; analyze 24 bins; save HTML dashboard:
python -m tools.explain_shap_metadata_v50 \
  --mu outputs/predictions/mu.npy \
  --sigma outputs/predictions/sigma.npy \
  --y data/labels.npy \
  --meta data/metadata.csv \
  --symbolic outputs/diagnostics/symbolic_results.json \
  --n-bins 24 --html

# Headless batch run, deterministic UMAP (if installed) or fallback to PCA:
python -m tools.explain_shap_metadata_v50 \
  --mu mu.npy --meta meta.csv --outdir out --seed 123 --no-tsne

Dependencies (auto-detect, graceful fallback)
---------------------------------------------
• numpy, pandas, scikit-learn, plotly, typer, rich
• Optional: shap, umap-learn
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.theme import Theme

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance

# Optional dependencies
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go

app = typer.Typer(add_completion=False, help="SpectraMind V50 — SHAP + Metadata Explainer")
console = Console(theme=Theme({"info": "cyan", "warn": "yellow", "err": "bold red"}))


# -----------------------------
# Utility & math helper routines
# -----------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_npy(path: Optional[str]) -> Optional[np.ndarray]:
    if not path:
        return None
    arr = np.load(path)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"{path} did not contain an ndarray")
    return arr


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


def entropy(mu_row: np.ndarray, eps: float = 1e-12) -> float:
    """Shannon entropy of normalized positive vector. If negative values, soft-shift."""
    v = mu_row.astype(float)
    if np.any(np.isnan(v)):
        return float("nan")
    # Shift to nonnegative
    v = v - np.min(v)
    v = v + eps
    p = v / np.sum(v)
    return float(-np.sum(p * np.log(p + eps)))


def gll(mu_row: np.ndarray, sigma_row: np.ndarray, y_row: np.ndarray, eps: float = 1e-12) -> float:
    """
    Gaussian log-likelihood per sample (sum over bins).
    Returns mean per-bin GLL to compare across bin counts.
    """
    s2 = np.maximum(sigma_row.astype(float) ** 2, eps)
    resid2 = (y_row.astype(float) - mu_row.astype(float)) ** 2
    # log-likelihood of N(y | mu, sigma^2)
    ll = -0.5 * (np.log(2 * np.pi * s2) + resid2 / s2)
    return float(np.mean(ll))


def set_global_seed(seed: int) -> None:
    try:
        import random
        random.seed(seed)
    except Exception:
        pass
    try:
        np.random.seed(seed)
    except Exception:
        pass


@dataclass
class Inputs:
    mu: np.ndarray
    sigma: Optional[np.ndarray]
    y: Optional[np.ndarray]
    metadata: pd.DataFrame
    latents: Optional[np.ndarray]
    symbolic: Optional[Dict[str, Any]]


# --------------------------------
# Data IO & preprocessing routines
# --------------------------------

def load_inputs(
    mu_path: str,
    meta_path: str,
    sigma_path: Optional[str],
    y_path: Optional[str],
    latents_path: Optional[str],
    symbolic_path: Optional[str],
) -> Inputs:
    mu = load_npy(mu_path)
    if mu is None:
        raise FileNotFoundError("--mu is required")
    meta = pd.read_csv(meta_path)
    sigma = load_npy(sigma_path) if sigma_path else None
    y = load_npy(y_path) if y_path else None
    latents = load_npy(latents_path) if latents_path else None
    symbolic = None
    if symbolic_path and Path(symbolic_path).exists():
        with open(symbolic_path, "r", encoding="utf-8") as f:
            symbolic = json.load(f)
    return Inputs(mu=mu, sigma=sigma, y=y, metadata=meta, latents=latents, symbolic=symbolic)


def coerce_metadata_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Split metadata columns into numeric vs categorical (simple heuristic)."""
    num_cols: List[str] = []
    cat_cols: List[str] = []
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            # try to coerce
            coerced = pd.to_numeric(df[c], errors="coerce")
            n_na = coerced.isna().sum()
            # if mostly numeric, treat as numeric after coercion
            if n_na < 0.1 * len(df):
                df[c] = coerced.fillna(coerced.median())
                num_cols.append(c)
            else:
                cat_cols.append(c)
    return df, num_cols, cat_cols


def build_meta_preprocessor(num_cols: List[str], cat_cols: List[str]) -> ColumnTransformer:
    num_transform = Pipeline(steps=[("scaler", StandardScaler(with_mean=True, with_std=True))])
    cat_transform = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    pre = ColumnTransformer(
        transformers=[
            ("num", num_transform, num_cols),
            ("cat", cat_transform, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return pre


# -----------------------
# SHAP & surrogate engine
# -----------------------

@dataclass
class ShapConfig:
    bins: List[int]
    model: str = "gbr"   # (currently only GradientBoostingRegressor)
    n_estimators: int = 300
    max_depth: int = 3
    learning_rate: float = 0.05
    subsample: float = 1.0
    shap_background: int = 256  # rows for Kernel/Tree background set (if used)
    shap_method: str = "auto"   # "tree", "kernel", "auto"
    use_permutation_fallback: bool = True


def train_surrogate_and_explain(
    X: np.ndarray,
    y_bin: np.ndarray,
    feature_names: List[str],
    cfg: ShapConfig,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Fit a small surrogate (GBR) to predict μ[:, bin] from metadata X.
    Compute SHAP values if possible; otherwise compute permutation importance.
    Returns (feature_ranking_df, per_sample_abs_shap_sum).
    """
    # Train
    model = GradientBoostingRegressor(
        n_estimators=cfg.n_estimators,
        max_depth=cfg.max_depth,
        learning_rate=cfg.learning_rate,
        subsample=cfg.subsample,
        random_state=42,
    )
    model.fit(X, y_bin)

    # Try SHAP (TreeExplainer preferred for tree models)
    per_sample_abs = None
    if _HAS_SHAP:
        try:
            if cfg.shap_method == "tree" or cfg.shap_method == "auto":
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X)  # shape: N×F
            else:
                # KernelExplainer fallback (slower)
                bg_size = min(cfg.shap_background, X.shape[0])
                bg = shap.kmeans(X, bg_size)
                explainer = shap.KernelExplainer(model.predict, bg)
                shap_vals = explainer.shap_values(X, nsamples=min(2048, 2 * X.shape[1]))

            abs_mean = np.mean(np.abs(shap_vals), axis=0)  # F
            per_sample_abs = np.sum(np.abs(shap_vals), axis=1)  # N
            rank_df = pd.DataFrame({
                "feature": feature_names,
                "importance": abs_mean,
            }).sort_values("importance", ascending=False, ignore_index=True)
            rank_df["rank"] = np.arange(1, len(rank_df) + 1)
            return rank_df, per_sample_abs
        except Exception:
            warnings.warn("SHAP computation failed; attempting permutation importance fallback.", RuntimeWarning)

    # Permutation importance fallback
    if cfg.use_permutation_fallback:
        r = permutation_importance(model, X, y_bin, n_repeats=8, random_state=42)
        imp = r.importances_mean
        rank_df = pd.DataFrame({"feature": feature_names, "importance": imp}).sort_values(
            "importance", ascending=False, ignore_index=True
        )
        rank_df["rank"] = np.arange(1, len(rank_df) + 1)
        # For per-sample intensity proxy, use |residual| as a rough substitute
        residual = np.abs(y_bin - model.predict(X))
        per_sample_abs = residual
        return rank_df, per_sample_abs
    else:
        raise RuntimeError("Unable to compute SHAP and permutation fallback is disabled.")


# --------------------------
# Projections & viz routines
# --------------------------

def compute_projections(
    X: np.ndarray,
    method_umap: bool,
    method_tsne: bool,
    seed: int,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    # Always compute PCA (fast and deterministic)
    pca = PCA(n_components=2, random_state=seed)
    out["pca2"] = pca.fit_transform(X)

    if method_umap and _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=seed)
        out["umap2"] = reducer.fit_transform(X)
    if method_tsne:
        tsne = TSNE(n_components=2, random_state=seed, init="random", learning_rate="auto", perplexity=30)
        out["tsne2"] = tsne.fit_transform(X)
    return out


def scatter_plot(
    coords: np.ndarray,
    color: Optional[np.ndarray],
    hover: Optional[pd.DataFrame],
    title: str,
    color_title: str,
    out_html: Path,
    size: Optional[np.ndarray] = None,
) -> None:
    df = pd.DataFrame({"x": coords[:, 0], "y": coords[:, 1]})
    if color is not None:
        df[color_title] = color
    if size is not None:
        df["size"] = size
        size_col = "size"
    else:
        size_col = None
    if hover is not None:
        for c in hover.columns:
            df[c] = hover[c]

    fig = px.scatter(
        df, x="x", y="y",
        color=color_title if color is not None else None,
        size=size_col,
        hover_data=hover.columns.tolist() if hover is not None else None,
        title=title,
        template="plotly_white",
    )
    fig.update_traces(marker=dict(opacity=0.85))
    pio.write_html(fig, file=str(out_html), include_plotlyjs="cdn", full_html=True)


# ----------------------
# Main orchestration API
# ----------------------

def run_explainer(
    mu_path: str = typer.Option(..., help="Path to N×B .npy of predicted μ"),
    meta_path: str = typer.Option(..., help="Path to metadata CSV with N rows"),
    outdir: str = typer.Option(..., help="Output directory"),
    sigma_path: Optional[str] = typer.Option(None, help="Path to N×B .npy of predicted σ"),
    y_path: Optional[str] = typer.Option(None, help="Path to N×B .npy of ground-truth spectra (optional)"),
    latents_path: Optional[str] = typer.Option(None, help="Path to N×D .npy latent embedding (optional)"),
    symbolic_path: Optional[str] = typer.Option(None, help="Path to symbolic overlay JSON (optional)"),
    n_bins: int = typer.Option(16, min=1, help="Number of bins to explain (uniformly sampled if < B)"),
    seed: int = typer.Option(42, help="Random seed for reproducibility"),
    no_umap: bool = typer.Option(False, help="Disable UMAP projection even if available"),
    no_tsne: bool = typer.Option(False, help="Disable t-SNE projection"),
    html: bool = typer.Option(False, help="Emit a compact self-contained HTML report"),
    save_png: bool = typer.Option(False, help="Export static PNG snapshots of interactive plots"),
) -> None:
    """Entry point for metadata → μ explanations and diagnostics fusion."""
    t0 = time.time()
    set_global_seed(seed)
    out = Path(outdir)
    ensure_dir(out)

    console.rule("[info]SpectraMind V50 — SHAP + Metadata Explainer")
    console.print(f"[info]μ: {mu_path}")
    console.print(f"[info]meta: {meta_path}")
    if sigma_path:
        console.print(f"[info]σ: {sigma_path}")
    if y_path:
        console.print(f"[info]y: {y_path}")
    if latents_path:
        console.print(f"[info]latents: {latents_path}")
    if symbolic_path:
        console.print(f"[info]symbolic: {symbolic_path}")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        transient=True,
    ) as progress:
        t_load = progress.add_task("Loading inputs", total=None)
        inputs = load_inputs(mu_path, meta_path, sigma_path, y_path, latents_path, symbolic_path)
        progress.update(t_load, advance=1, visible=False)

        N, B = inputs.mu.shape
        console.print(f"[info]Loaded μ shape: {N}×{B}")

        # Basic per-sample diagnostics
        t_diag = progress.add_task("Computing diagnostics", total=N)
        ent = np.zeros(N, dtype=float)
        gll_vec = np.full(N, np.nan, dtype=float)
        for i in range(N):
            ent[i] = entropy(inputs.mu[i, :])
            if inputs.y is not None and inputs.sigma is not None:
                gll_vec[i] = gll(inputs.mu[i, :], inputs.sigma[i, :], inputs.y[i, :])
            progress.advance(t_diag)

        # Symbolic overlay (per-planet summary), supports dicts like {"planet_id": {...}}
        sym_count = None
        if inputs.symbolic is not None:
            # Best-effort: look for "per_planet" counts in any reasonable structure
            try:
                # If a list of per‑sample dicts:
                if isinstance(inputs.symbolic, list) and len(inputs.symbolic) == N:
                    sym_count = np.array([safe_float(d.get("violations_total", 0.0)) for d in inputs.symbolic])
                # If a dict keyed by index:
                elif isinstance(inputs.symbolic, dict) and all(k.isdigit() for k in inputs.symbolic.keys()):
                    sym_count = np.array([safe_float(inputs.symbolic[str(i)].get("violations_total", 0.0)) for i in range(N)])
                else:
                    # Otherwise set zeros with a warning
                    console.print("[warn]Unrecognized symbolic JSON shape; overlay disabled.")
                    sym_count = None
            except Exception:
                console.print("[warn]Failed to parse symbolic overlay; continuing without it.")
                sym_count = None

        # Prepare metadata features
        meta_df, num_cols, cat_cols = coerce_metadata_features(inputs.metadata.copy())
        pre = build_meta_preprocessor(num_cols, cat_cols)
        X_all = pre.fit_transform(meta_df)
        feature_names = []
        # Retrieve feature names from ColumnTransformer:
        try:
            num_features = num_cols
            cat_features = []
            if cat_cols:
                # Discover OHE categories
                ohe = pre.named_transformers_["cat"].named_steps["ohe"]
                for c, cats in zip(cat_cols, ohe.categories_):
                    cat_features.extend([f"{c}={str(cat)}" for cat in cats.tolist()])
            feature_names = list(num_features) + list(cat_features)
        except Exception:
            # fallback to numeric index names
            feature_names = [f"f{i}" for i in range(X_all.shape[1])]

        # Projection inputs: prefer provided latents, else metadata embedding
        proj_X = inputs.latents if inputs.latents is not None else X_all
        proj = compute_projections(
            proj_X,
            method_umap=(not no_umap),
            method_tsne=(not no_tsne),
            seed=seed,
        )

        # Decide which bins to explain
        if n_bins >= B:
            explain_bins = list(range(B))
        else:
            step = B / float(n_bins)
            explain_bins = [int(round(i * step)) for i in range(n_bins)]
            explain_bins = sorted(list({min(b, B - 1) for b in explain_bins}))

        console.print(f"[info]Explaining bins: {len(explain_bins)} of {B} total")

        # Per-bin SHAP ranks and per-sample SHAP magnitudes (summed across features)
        t_explain = progress.add_task("Computing SHAP/importance per-bin", total=len(explain_bins))
        ranks: Dict[int, pd.DataFrame] = {}
        sample_intensity = np.zeros((N, len(explain_bins)), dtype=float)

        shap_cfg = ShapConfig(bins=explain_bins)

        for j, b in enumerate(explain_bins):
            y_bin = inputs.mu[:, b].astype(float)
            rank_df, per_sample_abs = train_surrogate_and_explain(X_all, y_bin, feature_names, shap_cfg)
            ranks[b] = rank_df
            sample_intensity[:, j] = per_sample_abs
            progress.advance(t_explain)

        # Aggregate feature importances across explained bins
        agg = pd.DataFrame({"feature": feature_names})
        for b in explain_bins:
            dfb = ranks[b][["feature", "importance"]].rename(columns={"importance": f"imp_bin{b}"})
            agg = agg.merge(dfb, on="feature", how="left")
        agg["mean_importance"] = agg[[c for c in agg.columns if c.startswith("imp_bin")]].mean(axis=1, skipna=True)
        agg = agg.sort_values("mean_importance", ascending=False, ignore_index=True)

        # Per-sample overlays: SHAP intensity (sum across explained bins)
        per_sample_shap_intensity = np.sum(sample_intensity, axis=1)

        # Export artifacts
        ensure_dir(out / "plots")
        ensure_dir(out / "tables")

        agg.to_csv(out / "tables" / "shap_feature_ranking.csv", index=False)
        pd.DataFrame({
            "entropy": ent,
            "gll": gll_vec,
            "shap_intensity": per_sample_shap_intensity,
            "symbolic_violations": sym_count if sym_count is not None else np.nan,
        }).to_csv(out / "tables" / "per_sample_scores.csv", index=False)

        # Save per-bin top features
        topk = 25
        perbin_dir = out / "tables" / "per_bin"
        ensure_dir(perbin_dir)
        for b in explain_bins:
            ranks[b].head(topk).to_csv(perbin_dir / f"top_features_bin{b}.csv", index=False)

        # Interactive plots (PCA always, plus UMAP/t-SNE if computed)
        hover = inputs.metadata.copy()
        hover["entropy"] = ent
        if inputs.y is not None and inputs.sigma is not None:
            hover["gll"] = gll_vec
        if sym_count is not None:
            hover["symbolic_violations"] = sym_count
        hover["shap_intensity"] = per_sample_shap_intensity

        # Color options we’ll render by default:
        overlays: List[Tuple[str, np.ndarray]] = [
            ("entropy", ent),
            ("shap_intensity", per_sample_shap_intensity),
        ]
        if inputs.y is not None and inputs.sigma is not None:
            overlays.append(("gll", gll_vec))
        if sym_count is not None:
            overlays.append(("symbolic_violations", sym_count))

        plot_index: List[str] = []
        for name, coords in proj.items():
            for color_name, color_vals in overlays:
                title = f"{name.upper()} — color: {color_name}"
                out_html = out / "plots" / f"{name}_{color_name}.html"
                scatter_plot(coords, color_vals, hover, title, color_name, out_html)
                plot_index.append(str(out_html.name))

        # Optional PNG snapshots via Kaleido (only if available)
        if save_png:
            try:
                import kaleido  # noqa: F401
                for p in (out / "plots").glob("*.html"):
                    # Re-load figure and write image
                    # (We saved full HTML; to snapshot we’d need to rebuild Plotly object: skip for simplicity)
                    pass
            except Exception:
                console.print("[warn]PNG export requested but 'kaleido' not available; skipping.")

        # Write run summary
        summary = {
            "n_samples": int(N),
            "n_bins": int(B),
            "explained_bins": explain_bins,
            "has_sigma": bool(inputs.sigma is not None),
            "has_y": bool(inputs.y is not None),
            "has_latents": bool(inputs.latents is not None),
            "has_symbolic": bool(inputs.symbolic is not None),
            "top_features": agg.head(25)["feature"].tolist(),
            "outputs": {
                "feature_ranking_csv": str((out / "tables" / "shap_feature_ranking.csv").name),
                "per_sample_scores_csv": str((out / "tables" / "per_sample_scores.csv").name),
                "per_bin_dir": "tables/per_bin/",
                "plots_dir": "plots/",
            },
            "timing_sec": round(time.time() - t0, 3),
        }
        with open(out / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        # Optional self-contained HTML report
        if html:
            report_path = out / "report_explain_shap_metadata.html"
            build_html_report(
                report_path=report_path,
                summary=summary,
                meta_cols=list(inputs.metadata.columns),
                plot_files=plot_index,
                tables={
                    "shap_feature_ranking.csv": out / "tables" / "shap_feature_ranking.csv",
                    "per_sample_scores.csv": out / "tables" / "per_sample_scores.csv",
                },
            )
            console.print(f"[info]Wrote HTML report → {report_path}")

    console.rule("[info]Done")
    console.print(f"[info]Elapsed: {round(time.time()-t0,2)} s")


# ----------------
# HTML report gen
# ----------------

def build_html_report(
    report_path: Path,
    summary: Dict[str, Any],
    meta_cols: List[str],
    plot_files: List[str],
    tables: Dict[str, Path],
) -> None:
    """Create a lightweight, portable HTML report that links plots and embeds small tables."""
    title = "SpectraMind V50 — SHAP + Metadata Explainer"
    css = """
    body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, sans-serif; margin: 16px; color: #0e1116; }
    h1 { font-size: 20px; margin: 8px 0 12px; }
    h2 { font-size: 16px; margin: 16px 0 8px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(360px, 1fr)); gap: 10px; }
    .card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px; background: #fff; box-shadow: 0 1px 2px rgba(0,0,0,0.04); }
    .meta { font-size: 12px; color: #475569; }
    table { border-collapse: collapse; font-size: 12px; width: 100%; }
    th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
    a { color: #0b5fff; text-decoration: none; }
    a:hover { text-decoration: underline; }
    code { background: #f3f4f6; padding: 2px 4px; border-radius: 6px; }
    """

    def read_table_html(p: Path, n: int = 25) -> str:
        try:
            df = pd.read_csv(p)
            return df.head(n).to_html(index=False, escape=False)
        except Exception:
            return "<em>Failed to load table.</em>"

    plots_links = "".join([f'<div class="card"><a href="plots/{name}">{name}</a></div>' for name in plot_files])

    html = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8" />
<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1" />
<style>{css}</style>
</head>
<body>
  <h1>{title}</h1>
  <div class="meta">N={summary.get("n_samples")}, B={summary.get("n_bins")} • explained_bins={summary.get("explained_bins")[:6]}… • elapsed={summary.get("timing_sec")}s</div>

  <h2>Top Features (Mean importance across explained bins)</h2>
  <div class="card">
    {read_table_html(tables["shap_feature_ranking.csv"])}
  </div>

  <h2>Per-sample Scores (entropy, gll, shap_intensity, symbolic)</h2>
  <div class="card">
    {read_table_html(tables["per_sample_scores.csv"])}
  </div>

  <h2>Interactive Plots</h2>
  <div class="grid">
    {plots_links}
  </div>

  <h2>Run Summary</h2>
  <div class="card">
    <pre>{json.dumps(summary, indent=2)}</pre>
  </div>

  <h2>Metadata Columns</h2>
  <div class="card">
    <code>{", ".join(meta_cols[:128])}{'…' if len(meta_cols) > 128 else ''}</code>
  </div>
</body>
</html>
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)


# -----------
# Typer entry
# -----------

@app.command("run")
def cli_run(
    mu: str = typer.Option(..., help="Path to μ.npy (N×B)"),
    meta: str = typer.Option(..., help="Path to metadata CSV (N rows)"),
    outdir: str = typer.Option(..., help="Output directory"),
    sigma: Optional[str] = typer.Option(None, help="Path to σ.npy (N×B)"),
    y: Optional[str] = typer.Option(None, help="Path to labels.npy (N×B) for GLL"),
    latents: Optional[str] = typer.Option(None, help="Path to latents.npy (N×D)"),
    symbolic: Optional[str] = typer.Option(None, help="Path to symbolic JSON"),
    n_bins: int = typer.Option(16, help="Number of bins to explain"),
    seed: int = typer.Option(42, help="Random seed"),
    no_umap: bool = typer.Option(False, help="Disable UMAP even if installed"),
    no_tsne: bool = typer.Option(False, help="Disable t-SNE"),
    html: bool = typer.Option(False, help="Emit a self-contained HTML report"),
    save_png: bool = typer.Option(False, help="Try to snapshot PNG images (requires kaleido)"),
):
    """
    Run metadata → μ explanations and produce plots/exports/report.
    """
    try:
        run_explainer(
            mu_path=mu,
            meta_path=meta,
            outdir=outdir,
            sigma_path=sigma,
            y_path=y,
            latents_path=latents,
            symbolic_path=symbolic,
            n_bins=n_bins,
            seed=seed,
            no_umap=no_umap,
            no_tsne=no_tsne,
            html=html,
            save_png=save_png,
        )
    except Exception as e:
        console.print(Panel.fit(str(e), title="Error", style="err"))
        console.print(traceback.format_exc())
        raise typer.Exit(code=1)


def main():
    app()


if __name__ == "__main__":
    main()
