# src/diagnostics/plot/umap/v50.py

# =============================================================================

# 🌌 SpectraMind V50 — UMAP Latent Plotter (Interactive HTML + PNG)

# -----------------------------------------------------------------------------

# Purpose

# Produce interactive UMAP plots for SpectraMind V50 latents with full

# diagnostics overlays (symbolic labels, confidences, SHAP/entropy/GLL), and

# optional hyperlinks to planet pages. Designed for CLI-first pipelines with

# NASA-grade reproducibility guarantees (Hydra configs, run hashing, logs).

#

# Key Features

# • Input: .npy/.npz/.csv latents (rows = planets, cols = features)

# • Optional labels/metrics CSV (planet\_id,label,confidence,entropy,shap,gll,…)

# • Optional symbolic overlays JSON (rule scores/masks per planet)

# • Deterministic UMAP with seed, PCA fallback if umap-learn not installed

# • Dedupe repeated planet\_id rows (--dedupe)

# • Rich Plotly hover (μ/GLL/σ summaries if provided)

# • Color/size/opacity mapping from provided columns

# • Hyperlinks per-point (--url-template "planet.html?id={planet\_id}")

# • HTML output (always), PNG output if kaleido present

# • Logs to v50\_debug\_log.md with config hash (if present) and CLI argv

# • CLI via Typer; import-safe API functions for programmatic use

#

# Example

# spectramind diagnose umap \\

# --latents artifacts/latents\_v50.npy \\

# --labels artifacts/latents\_meta.csv \\

# --symbolic-overlays artifacts/symbolic\_violation\_summary.json \\

# --color-by label --size-by confidence --opacity-by entropy \\

# --url-template "/planets/{planet\_id}.html" \\

# --out-html artifacts/umap\_v50.html --out-png artifacts/umap\_v50.png

#

# Dependencies (soft)

# numpy, pandas, plotly, scikit-learn, umap-learn (optional), typer (CLI)

# kaleido (optional for PNG)

#

# Notes

# • This module NEVER computes scientific analytics; it only visualizes

# already-produced latents/overlays/metrics to preserve reproducibility.

# • If umap-learn is unavailable, a PCA (2D/3D) fallback is used.

# • Designed to integrate with generate\_html\_report.py and CLI dashboard.

# =============================================================================

from **future** import annotations

import json
import logging
import os
import sys
import time
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Plotly for interactive visualization (no seaborn by project policy)

import plotly.express as px
import plotly.io as pio

# Scikit-learn for scaling and PCA fallback

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Typer for CLI (optional dependency)

try:
import typer  # type: ignore
except Exception:
typer = None  # Allow import/use as a library even if Typer isn't installed

# UMAP is optional; we fall back to PCA if missing

try:
import umap  # type: ignore
\_UMAP\_AVAILABLE = True
except Exception:
\_UMAP\_AVAILABLE = False

# PNG export (optional)

try:
import kaleido  # noqa: F401
\_KALEIDO\_AVAILABLE = True
except Exception:
\_KALEIDO\_AVAILABLE = False

# -----------------------------------------------------------------------------

# Constants and global defaults

# -----------------------------------------------------------------------------

DEFAULT\_LOG\_PATH = Path("v50\_debug\_log.md")
DEFAULT\_HTML\_OUT = Path("artifacts") / "umap\_v50.html"
DEFAULT\_PNG\_OUT = Path("artifacts") / "umap\_v50.png"

# Standard columns we try to recognize/merge

PLANET\_ID\_COL = "planet\_id"
LABEL\_COL = "label"
CONFIDENCE\_COL = "confidence"
ENTROPY\_COL = "entropy"
SHAP\_MAG\_COL = "shap"
GLL\_COL = "gll"

# Deterministic seed default

DEFAULT\_SEED = 1337

# -----------------------------------------------------------------------------

# Dataclasses

# -----------------------------------------------------------------------------

@dataclass
class UmapParams:
"""Parameters for UMAP / fallback embedding."""
n\_neighbors: int = 15
min\_dist: float = 0.1
metric: str = "euclidean"
n\_components: int = 2
seed: int = DEFAULT\_SEED
\# Fallback PCA components if UMAP not available (follow n\_components)
pca\_whiten: bool = False

@dataclass
class PlotMap:
"""Visual mapping configuration for Plotly scatter plots."""
color\_by: Optional\[str] = None
size\_by: Optional\[str] = None
opacity\_by: Optional\[str] = None
symbol\_by: Optional\[str] = None
hover\_cols: List\[str] = field(default\_factory=list)

@dataclass
class OverlayConfig:
"""Optional overlays loaded from symbolic/diagnostic JSON."""
symbolic\_overlays\_path: Optional\[Path] = None
symbolic\_score\_key: str = "violation\_score"  # key in JSON to map to color
symbolic\_label\_key: str = "top\_rule"         # key in JSON to map to label
\# If provided, map to these standardized columns on merge
map\_score\_to: Optional\[str] = "symbolic\_score"
map\_label\_to: Optional\[str] = "symbolic\_label"

@dataclass
class HyperlinkConfig:
"""Optional hyperlink generation per point."""
url\_template: Optional\[str] = None   # e.g., "/planet/{planet\_id}.html"
url\_col\_name: str = "url"            # column name where URL is stored

@dataclass
class OutputConfig:
"""Output configuration."""
out\_html: Path = DEFAULT\_HTML\_OUT
out\_png: Optional\[Path] = None
open\_browser: bool = False
title: str = "SpectraMind V50 — UMAP Latents"

@dataclass
class PipelineLogContext:
"""Context for run logging (appended to v50\_debug\_log.md)."""
log\_path: Path = DEFAULT\_LOG\_PATH
cli\_name: str = "spectramind diagnose umap"
config\_hash\_path: Optional\[Path] = Path("run\_hash\_summary\_v50.json")  # optional

# -----------------------------------------------------------------------------

# Utilities

# -----------------------------------------------------------------------------

def \_now\_iso() -> str:
"""Return current timestamp in ISO-like format (local time)."""
return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def \_hash\_file(path: Path) -> Optional\[str]:
"""Return SHA256 hash of a file, if exists; else None."""
if not path or not Path(path).exists():
return None
h = hashlib.sha256()
with open(path, "rb") as f:
for chunk in iter(lambda: f.read(1024 \* 1024), b""):
h.update(chunk)
return h.hexdigest()

def \_ensure\_parent\_dir(p: Path) -> None:
"""Ensure parent directory exists for a given path."""
p = Path(p)
if p.parent and not p.parent.exists():
p.parent.mkdir(parents=True, exist\_ok=True)

def setup\_logging(level: int = logging.INFO) -> None:
"""Configure logging for console (stdout)."""
logging.basicConfig(
level=level,
format="%(asctime)s | %(levelname)-7s | %(message)s",
datefmt="%H:%M:%S",
)

def append\_v50\_log(ctx: PipelineLogContext, metadata: Dict\[str, Any]) -> None:
"""
Append a structured log entry to v50\_debug\_log.md (Markdown table row).

```
This log is human-readable and part of the project's audit trail.
"""
try:
    _ensure_parent_dir(ctx.log_path)
    # Attempt to read run hash summary for config hash metadata if present
    config_hash = None
    if ctx.config_hash_path and Path(ctx.config_hash_path).exists():
        try:
            obj = json.loads(Path(ctx.config_hash_path).read_text())
            config_hash = obj.get("config_hash") or obj.get("hash") or obj.get("run_hash")
        except Exception:
            config_hash = None
    # Compose a compact row
    row = {
        "timestamp": _now_iso(),
        "cli": ctx.cli_name,
        "config_hash": config_hash or "",
        **metadata,
    }
    line = (
        f"| {row['timestamp']} "
        f"| {row['cli']} "
        f"| {row['config_hash']} "
        f"| {row.get('latents','')} "
        f"| {row.get('labels','')} "
        f"| {row.get('symbolic','')} "
        f"| {row.get('out_html','')} "
        f"| {row.get('out_png','')} |\n"
    )

    # If file empty or missing, write header
    if not ctx.log_path.exists() or ctx.log_path.stat().st_size == 0:
        header = (
            "# SpectraMind V50 — Debug Log (UMAP)\n\n"
            "| timestamp | cli | config_hash | latents | labels | symbolic | out_html | out_png |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        ctx.log_path.write_text(header + line, encoding="utf-8")
    else:
        with open(ctx.log_path, "a", encoding="utf-8") as f:
            f.write(line)
except Exception as e:
    logging.warning(f"Failed to append to {ctx.log_path}: {e}")
```

def \_coerce\_to\_str\_index(s: pd.Series) -> pd.Series:
"""Coerce identifiers to str, robustly."""
return s.astype(str).str.strip()

def \_safe\_read\_csv(path: Path) -> pd.DataFrame:
"""Read CSV with reasonable defaults."""
return pd.read\_csv(path)

def \_safe\_read\_json(path: Path) -> Any:
"""Read JSON from a file path safely."""
return json.loads(Path(path).read\_text())

# -----------------------------------------------------------------------------

# Data loading and merging

# -----------------------------------------------------------------------------

def load\_latents(latents\_path: Path, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""
Load latent vectors from .npy/.npz/.csv into a DataFrame.

```
Returns a DataFrame with:
  - A column `planet_id_col` (string id)
  - Feature columns named f"f{i}" for i in [0, D) if needed
"""
path = Path(latents_path)
if not path.exists():
    raise FileNotFoundError(f"Latents file not found: {path}")

ext = path.suffix.lower()
if ext in [".npy"]:
    mat = np.load(path)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D array in {path}, got shape={mat.shape}")
    df = pd.DataFrame(mat, columns=[f"f{i}" for i in range(mat.shape[1])])
    # If planet ids are not provided elsewhere, create synthetic incremental ids
    df[planet_id_col] = [str(i) for i in range(len(df))]
    df = df[[planet_id_col] + [c for c in df.columns if c != planet_id_col]]

elif ext in [".npz"]:
    data = np.load(path)
    if "X" not in data and "latents" not in data:
        raise KeyError(f"NPZ must contain 'X' or 'latents' array: keys={list(data.keys())}")
    X = data["X"] if "X" in data else data["latents"]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    # Try to read ids
    if "planet_id" in data:
        ids = data["planet_id"]
        ids = [str(x) for x in ids]
    else:
        ids = [str(i) for i in range(len(df))]
    df[planet_id_col] = ids
    df = df[[planet_id_col] + [c for c in df.columns if c != planet_id_col]]

elif ext in [".csv", ".txt"]:
    df = _safe_read_csv(path)
    # If features come unnamed or as generic columns, standardize
    if planet_id_col not in df.columns:
        # Try common alternatives
        alt_ids = [c for c in df.columns if c.lower() in ("planet_id", "planet", "id")]
        if alt_ids:
            df = df.rename(columns={alt_ids[0]: planet_id_col})
        else:
            # As a last resort, synthesize ids
            df[planet_id_col] = [str(i) for i in range(len(df))]
    # No hard rename for feature columns; accept any numeric col as feature
    # (Later we will filter non-numeric/known metadata columns.)

else:
    raise ValueError(f"Unsupported latents file extension: {ext}")

df[planet_id_col] = _coerce_to_str_index(df[planet_id_col])
return df
```

def dedupe\_by\_planet\_id(df: pd.DataFrame, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""Drop duplicate rows by planet\_id, keeping the first occurrence."""
if df\[planet\_id\_col].duplicated().any():
logging.info("Duplicates detected; applying dedupe by planet\_id (keep=first).")
return df.drop\_duplicates(subset=\[planet\_id\_col], keep="first", ignore\_index=True)
return df

def merge\_labels(df\_latents: pd.DataFrame, labels\_csv: Optional\[Path]) -> pd.DataFrame:
"""Merge optional labels CSV on planet\_id."""
if not labels\_csv:
return df\_latents
if not Path(labels\_csv).exists():
logging.warning(f"Labels CSV not found: {labels\_csv}; skipping.")
return df\_latents
meta = \_safe\_read\_csv(Path(labels\_csv))
\# Standardize id col
if PLANET\_ID\_COL not in meta.columns:
alt = \[c for c in meta.columns if c.lower() in ("planet\_id", "planet", "id")]
if alt:
meta = meta.rename(columns={alt\[0]: PLANET\_ID\_COL})
else:
raise KeyError(f"Labels CSV must include '{PLANET\_ID\_COL}' column")
meta\[PLANET\_ID\_COL] = \_coerce\_to\_str\_index(meta\[PLANET\_ID\_COL])
merged = df\_latents.merge(meta, on=PLANET\_ID\_COL, how="left", validate="one\_to\_one")
return merged

def merge\_symbolic\_overlays(
df: pd.DataFrame,
overlay\_cfg: OverlayConfig,
planet\_id\_col: str = PLANET\_ID\_COL,
) -> pd.DataFrame:
"""Merge symbolic overlays from JSON onto DataFrame."""
if not overlay\_cfg.symbolic\_overlays\_path:
return df
path = Path(overlay\_cfg.symbolic\_overlays\_path)
if not path.exists():
logging.warning(f"Symbolic overlays JSON not found: {path}; skipping.")
return df

```
obj = _safe_read_json(path)
# Expected formats:
#   1) { "<planet_id>": {"violation_score": x, "top_rule": "RULE"}, ... }
#   2) List[{"planet_id": "...", "violation_score": x, "top_rule": "..."}]
if isinstance(obj, dict) and all(isinstance(k, str) for k in obj.keys()):
    records = []
    for pid, payload in obj.items():
        if not isinstance(payload, dict):
            continue
        rec = {
            planet_id_col: str(pid),
            (overlay_cfg.map_score_to or "symbolic_score"): payload.get(overlay_cfg.symbolic_score_key),
            (overlay_cfg.map_label_to or "symbolic_label"): payload.get(overlay_cfg.symbolic_label_key),
        }
        records.append(rec)
    ov = pd.DataFrame.from_records(records)
elif isinstance(obj, list):
    ov = pd.DataFrame(obj)
    # Normalize columns
    if planet_id_col not in ov.columns:
        alt = [c for c in ov.columns if c.lower() in ("planet_id", "planet", "id")]
        if alt:
            ov = ov.rename(columns={alt[0]: planet_id_col})
        else:
            raise KeyError(f"Symbolic overlay list requires '{planet_id_col}' column.")
    # Map keys to desired names
    if overlay_cfg.symbolic_score_key in ov.columns and (overlay_cfg.map_score_to is not None):
        ov = ov.rename(columns={overlay_cfg.symbolic_score_key: overlay_cfg.map_score_to})
    if overlay_cfg.symbolic_label_key in ov.columns and (overlay_cfg.map_label_to is not None):
        ov = ov.rename(columns={overlay_cfg.symbolic_label_key: overlay_cfg.map_label_to})
else:
    raise ValueError("Unsupported symbolic overlay JSON structure.")

ov[planet_id_col] = _coerce_to_str_index(ov[planet_id_col])
merged = df.merge(ov, on=planet_id_col, how="left", validate="one_to_one")
return merged
```

def add\_hyperlinks(df: pd.DataFrame, link\_cfg: HyperlinkConfig, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""Add hyperlink column if url\_template provided."""
if not link\_cfg.url\_template:
return df

```
def _fmt(pid: str) -> str:
    try:
        return link_cfg.url_template.format(planet_id=pid)
    except Exception:
        # Keep robust if template missing braces
        return f"{link_cfg.url_template}{pid}"

df[link_cfg.url_col_name] = df[planet_id_col].map(_fmt)
return df
```

# -----------------------------------------------------------------------------

# Embedding

# -----------------------------------------------------------------------------

def compute\_embedding(
X: np.ndarray,
params: UmapParams,
) -> np.ndarray:
"""Compute UMAP embedding if available; otherwise PCA fallback."""
\# Standardize for stability/determinism (especially PCA fallback)
Xs = StandardScaler(with\_mean=True, with\_std=True).fit\_transform(X)

```
if _UMAP_AVAILABLE:
    reducer = umap.UMAP(
        n_neighbors=params.n_neighbors,
        min_dist=params.min_dist,
        metric=params.metric,
        n_components=params.n_components,
        random_state=params.seed,
        n_epochs=None,
        verbose=False,
    )
    Y = reducer.fit_transform(Xs)
    return Y

# PCA fallback (deterministic)
logging.warning("umap-learn not installed; using PCA fallback.")
pca = PCA(n_components=params.n_components, whiten=params.pca_whiten, random_state=params.seed)
Y = pca.fit_transform(Xs)
return Y
```

# -----------------------------------------------------------------------------

# Plotting

# -----------------------------------------------------------------------------

def \_build\_plot\_title(base: str, params: UmapParams) -> str:
"""Compose a plot title that includes algorithm and key params."""
algo = "UMAP" if \_UMAP\_AVAILABLE else "PCA (fallback)"
return f"{base} — {algo} (n={params.n\_components}, seed={params.seed})"

def build\_plot(
df: pd.DataFrame,
emb: np.ndarray,
plot\_map: PlotMap,
link\_cfg: HyperlinkConfig,
out\_cfg: OutputConfig,
planet\_id\_col: str = PLANET\_ID\_COL,
) -> "plotly.graph\_objs.\_figure.Figure":
"""Build a Plotly scatter plot for 2D/3D embeddings with rich hover."""
n\_components = emb.shape\[1]
coords: Dict\[str, np.ndarray] = {}
if n\_components >= 1:
coords\["x"] = emb\[:, 0]
if n\_components >= 2:
coords\["y"] = emb\[:, 1]
if n\_components >= 3:
coords\["z"] = emb\[:, 2]

```
# Build plotting DataFrame
pdf = df.copy()
for k, v in coords.items():
    pdf[k] = v

# Default hover columns
hover_cols = set(plot_map.hover_cols or [])
for c in (planet_id_col, LABEL_COL, CONFIDENCE_COL, ENTROPY_COL, SHAP_MAG_COL, GLL_COL, link_cfg.url_col_name):
    if c in pdf.columns:
        hover_cols.add(c)
# Ensure type is str for planet_id so hover is readable
if planet_id_col in pdf.columns:
    pdf[planet_id_col] = _coerce_to_str_index(pdf[planet_id_col])

# Common kwargs
common_kwargs = dict(
    data_frame=pdf,
    hover_data=sorted(list(hover_cols)) if hover_cols else None,
    title=out_cfg.title,
)

# Choose 2D vs 3D
if n_components >= 3:
    fig = px.scatter_3d(
        **common_kwargs,
        x="x",
        y="y",
        z="z",
        color=plot_map.color_by if (plot_map.color_by and plot_map.color_by in pdf.columns) else None,
        size=plot_map.size_by if (plot_map.size_by and plot_map.size_by in pdf.columns) else None,
        symbol=plot_map.symbol_by if (plot_map.symbol_by and plot_map.symbol_by in pdf.columns) else None,
    )
else:
    fig = px.scatter(
        **common_kwargs,
        x="x",
        y="y",
        color=plot_map.color_by if (plot_map.color_by and plot_map.color_by in pdf.columns) else None,
        size=plot_map.size_by if (plot_map.size_by and plot_map.size_by in pdf.columns) else None,
        symbol=plot_map.symbol_by if (plot_map.symbol_by and plot_map.symbol_by in pdf.columns) else None,
    )

# Opacity mapping if requested: normalize a numeric column into [0.25, 1.0]
if plot_map.opacity_by and plot_map.opacity_by in pdf.columns:
    col = pdf[plot_map.opacity_by]
    if pd.api.types.is_numeric_dtype(col):
        v = col.to_numpy(dtype=float)
        # Replace NaNs with the mean if finite, else 0.0
        mean_val = np.nanmean(v)
        v = np.nan_to_num(v, nan=(mean_val if np.isfinite(mean_val) else 0.0))
        # Normalize robustly via percentiles to resist outliers
        v_min, v_max = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        if v_max - v_min <= 1e-12:
            alpha = np.full_like(v, 0.9)
        else:
            alpha = 0.25 + 0.75 * (np.clip(v, v_min, v_max) - v_min) / (v_max - v_min)
        # Assign per-point opacity (works with Plotly array-like marker.opacity)
        fig.update_traces(marker={"opacity": alpha})
    else:
        logging.warning(f"opacity-by column '{plot_map.opacity_by}' is not numeric; ignoring.")

# Hyperlink hint (URL is included in hover via hover_data)
if link_cfg.url_template and link_cfg.url_col_name in pdf.columns:
    fig.add_annotation(
        text="Tip: Ctrl/Cmd+Click the URL in hover to open planet page.",
        xref="paper", yref="paper", x=0, y=1.08, showarrow=False, align="left",
        font={"size": 12}
    )

# Title and layout
fig.update_layout(
    title={"text": _build_plot_title(out_cfg.title, UmapParams(n_components=emb.shape[1]))},
    legend_title_text=plot_map.color_by if plot_map.color_by else "legend",
    template="plotly_white",
    margin=dict(l=40, r=40, t=80, b=40),
    hovermode="closest",
)

# Improve marker aesthetics
fig.update_traces(
    marker=dict(
        line=dict(width=0.5),
        sizemin=3,
    ),
    selector=dict(mode="markers")
)

return fig
```

# -----------------------------------------------------------------------------

# Main orchestration (library API)

# -----------------------------------------------------------------------------

def run\_umap\_pipeline(
latents\_path: Path,
out\_cfg: OutputConfig,
umap\_params: UmapParams,
plot\_map: PlotMap,
overlay\_cfg: Optional\[OverlayConfig] = None,
link\_cfg: Optional\[HyperlinkConfig] = None,
labels\_csv: Optional\[Path] = None,
dedupe: bool = False,
planet\_id\_col: str = PLANET\_ID\_COL,
log\_ctx: Optional\[PipelineLogContext] = None,
) -> Dict\[str, Any]:
"""
End-to-end pipeline: load → merge → embed → plot → save (HTML/PNG).

```
Returns a dict with output paths, counts, dims, and runtime metadata.
"""
setup_logging()
t0 = time.time()

# Load latents
df = load_latents(latents_path, planet_id_col=planet_id_col)
if dedupe:
    df = dedupe_by_planet_id(df, planet_id_col=planet_id_col)

# Merge labels
df = merge_labels(df, labels_csv)

# Merge overlays
if overlay_cfg:
    df = merge_symbolic_overlays(df, overlay_cfg, planet_id_col=planet_id_col)

# Add hyperlinks
if link_cfg:
    df = add_hyperlinks(df, link_cfg, planet_id_col=planet_id_col)

# Extract feature matrix
feat_cols = [c for c in df.columns if c.startswith("f") and pd.api.types.is_numeric_dtype(df[c])]
if not feat_cols:
    # Fallback: all numeric except non-features
    non_feats = {planet_id_col, LABEL_COL, CONFIDENCE_COL, ENTROPY_COL, SHAP_MAG_COL, GLL_COL}
    if link_cfg:
        non_feats.add(link_cfg.url_col_name)
    if overlay_cfg:
        if overlay_cfg.map_score_to:
            non_feats.add(overlay_cfg.map_score_to)
        if overlay_cfg.map_label_to:
            non_feats.add(overlay_cfg.map_label_to)
    feat_cols = [c for c in df.columns if (c not in non_feats) and pd.api.types.is_numeric_dtype(df[c])]
if not feat_cols:
    raise ValueError("No numeric feature columns found for embedding.")

X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)

# Compute embedding
Y = compute_embedding(X, umap_params)

# Build and save figure
fig = build_plot(
    df=df,
    emb=Y,
    plot_map=plot_map,
    link_cfg=link_cfg or HyperlinkConfig(),
    out_cfg=out_cfg,
    planet_id_col=planet_id_col,
)

# Save outputs
_ensure_parent_dir(out_cfg.out_html)
pio.write_html(fig, file=str(out_cfg.out_html), auto_open=False, include_plotlyjs="cdn")
logging.info(f"Saved HTML: {out_cfg.out_html}")

png_path = None
if out_cfg.out_png:
    _ensure_parent_dir(out_cfg.out_png)
    if _KALEIDO_AVAILABLE:
        fig.write_image(str(out_cfg.out_png), format="png", scale=2, width=1280, height=800)
        logging.info(f"Saved PNG: {out_cfg.out_png}")
        png_path = str(out_cfg.out_png)
    else:
        logging.warning("kaleido not installed; PNG export skipped.")

# Optional: open in browser
if out_cfg.open_browser:
    try:
        import webbrowser  # stdlib
        webbrowser.open_new_tab(out_cfg.out_html.as_uri())
    except Exception as e:
        logging.warning(f"Failed to open browser: {e}")

dt = time.time() - t0

# Append run log
if log_ctx:
    append_v50_log(
        log_ctx,
        metadata={
            "latents": str(latents_path),
            "labels": str(labels_csv) if labels_csv else "",
            "symbolic": str(overlay_cfg.symbolic_overlays_path) if (overlay_cfg and overlay_cfg.symbolic_overlays_path) else "",
            "out_html": str(out_cfg.out_html),
            "out_png": str(out_cfg.out_png) if out_cfg.out_png else "",
            "duration_sec": f"{dt:.2f}",
            "umap": f"neighbors={umap_params.n_neighbors},min_dist={umap_params.min_dist},metric={umap_params.metric},dims={umap_params.n_components},seed={umap_params.seed}",
            "color_by": plot_map.color_by or "",
            "size_by": plot_map.size_by or "",
            "opacity_by": plot_map.opacity_by or "",
            "symbol_by": plot_map.symbol_by or "",
            "umap_available": str(_UMAP_AVAILABLE),
            "kaleido_available": str(_KALEIDO_AVAILABLE),
        },
    )

return {
    "html": str(out_cfg.out_html),
    "png": png_path,
    "n": len(df),
    "dims": Y.shape[1],
    "umap_available": _UMAP_AVAILABLE,
    "kaleido_available": _KALEIDO_AVAILABLE,
    "duration_sec": dt,
}
```

# -----------------------------------------------------------------------------

# CLI

# -----------------------------------------------------------------------------

def \_build\_typer\_app() -> "typer.Typer":
"""Construct the Typer CLI application for this module."""
if typer is None:
raise RuntimeError(
"Typer is not installed. Install with `pip install typer[all]` "
"or import and use run\_umap\_pipeline() programmatically."
)

```
app = typer.Typer(
    add_completion=True,
    help="SpectraMind V50 — UMAP Latent Plotter (interactive HTML + PNG)",
    no_args_is_help=True,
)

@app.command("run")
def cli_run(
    latents: Path = typer.Option(
        ...,
        exists=True,
        dir_okay=False,
        help="Latents file (.npy/.npz/.csv) with rows=planets, cols=features.",
    ),
    labels: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False, help="Optional labels/metrics CSV with planet_id + metadata."
    ),
    symbolic_overlays: Optional[Path] = typer.Option(
        None, exists=True, dir_okay=False, help="Optional symbolic overlay JSON (per-planet)."
    ),
    symbolic_score_key: str = typer.Option(
        "violation_score", help="Key in symbolic JSON to use as score column."
    ),
    symbolic_label_key: str = typer.Option(
        "top_rule", help="Key in symbolic JSON to use as label column."
    ),
    map_score_to: str = typer.Option(
        "symbolic_score", help="Target column name to store symbolic score in the merged frame."
    ),
    map_label_to: str = typer.Option(
        "symbolic_label", help="Target column name to store symbolic label in the merged frame."
    ),
    color_by: Optional[str] = typer.Option(
        None, help="Column name to map to color (e.g., 'label', 'symbolic_score')."
    ),
    size_by: Optional[str] = typer.Option(
        None, help="Column name to map to marker size (e.g., 'confidence', 'shap')."
    ),
    opacity_by: Optional[str] = typer.Option(
        None, help="Column name to map to marker opacity (numeric only, e.g., 'entropy')."
    ),
    symbol_by: Optional[str] = typer.Option(
        None, help="Column name to map to marker symbol (categorical)."
    ),
    hover_cols: List[str] = typer.Option(
        [], help="Additional columns to include in hover."
    ),
    url_template: Optional[str] = typer.Option(
        None, help="Hyperlink template per point, e.g. '/planets/{planet_id}.html'."
    ),
    url_col_name: str = typer.Option(
        "url", help="Column name to store generated URLs when using url_template."
    ),
    out_html: Path = typer.Option(
        DEFAULT_HTML_OUT, help="Output HTML path."
    ),
    out_png: Optional[Path] = typer.Option(
        None, help="Optional PNG path (requires kaleido)."
    ),
    open_browser: bool = typer.Option(
        False, help="Open the HTML in a browser after generation."
    ),
    title: str = typer.Option(
        "SpectraMind V50 — UMAP Latents", help="Plot title."
    ),
    n_neighbors: int = typer.Option(15, help="UMAP: number of neighbors."),
    min_dist: float = typer.Option(0.1, help="UMAP: min_dist."),
    metric: str = typer.Option("euclidean", help="UMAP: metric."),
    n_components: int = typer.Option(2, help="Embedding dims (2 or 3)."),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed."),
    pca_whiten: bool = typer.Option(
        False, help="If UMAP is unavailable, enable PCA whitening in fallback."
    ),
    dedupe: bool = typer.Option(False, help="Dedupe by planet_id (keep first)."),
    planet_id_col: str = typer.Option(
        PLANET_ID_COL, help="Column name for planet ID."
    ),
    log_path: Path = typer.Option(
        DEFAULT_LOG_PATH, help="Path to v50_debug_log.md (for appending run rows)."
    ),
    config_hash_path: Optional[Path] = typer.Option(
        Path("run_hash_summary_v50.json"), help="Optional path to run hash summary JSON."
    ),
):
    """Run the UMAP plotting pipeline."""
    overlay_cfg = OverlayConfig(
        symbolic_overlays_path=symbolic_overlays,
        symbolic_score_key=symbolic_score_key,
        symbolic_label_key=symbolic_label_key,
        map_score_to=map_score_to,
        map_label_to=map_label_to,
    )
    link_cfg = HyperlinkConfig(
        url_template=url_template,
        url_col_name=url_col_name,
    )
    out_cfg = OutputConfig(
        out_html=out_html,
        out_png=out_png,
        open_browser=open_browser,
        title=title,
    )
    umap_params = UmapParams(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        n_components=n_components,
        seed=seed,
        pca_whiten=pca_whiten,
    )
    plot_map = PlotMap(
        color_by=color_by,
        size_by=size_by,
        opacity_by=opacity_by,
        symbol_by=symbol_by,
        hover_cols=hover_cols,
    )
    log_ctx = PipelineLogContext(
        log_path=log_path,
        cli_name="spectramind diagnose umap",
        config_hash_path=config_hash_path,
    )

    result = run_umap_pipeline(
        latents_path=latents,
        out_cfg=out_cfg,
        umap_params=umap_params,
        plot_map=plot_map,
        overlay_cfg=overlay_cfg if symbolic_overlays else None,
        link_cfg=link_cfg if url_template else None,
        labels_csv=labels,
        dedupe=dedupe,
        planet_id_col=planet_id_col,
        log_ctx=log_ctx,
    )
    typer.echo(json.dumps(result, indent=2))

@app.command("selftest")
def cli_selftest(
    tmp_dir: Path = typer.Option(Path("./artifacts/_umap_selftest"), help="Temporary dir for selftest artifacts."),
    n: int = typer.Option(256, help="Number of synthetic points."),
    d: int = typer.Option(16, help="Latent dimensionality."),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed."),
):
    """Generate a synthetic dataset and verify the pipeline end-to-end."""
    rng = np.random.default_rng(seed)
    X = np.r_[rng.normal(0, 1, size=(n // 2, d)) + 2.0, rng.normal(0, 1, size=(n // 2, d)) - 2.0]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(d)])
    df[PLANET_ID_COL] = [f"P{i:04d}" for i in range(n)]
    df[LABEL_COL] = ["A"] * (n // 2) + ["B"] * (n - n // 2)
    df[CONFIDENCE_COL] = np.clip(rng.beta(2, 5, size=n) * 1.2, 0, 1)
    df[ENTROPY_COL] = rng.random(size=n)
    df[SHAP_MAG_COL] = np.abs(rng.normal(0.2, 0.1, size=n))
    df[GLL_COL] = np.abs(rng.normal(0.0, 1.0, size=n))

    tmp_dir.mkdir(parents=True, exist_ok=True)
    latents_csv = tmp_dir / "latents.csv"
    df.to_csv(latents_csv, index=False)

    out_html = tmp_dir / "umap_selftest.html"
    out_png = tmp_dir / "umap_selftest.png"

    result = run_umap_pipeline(
        latents_path=latents_csv,
        out_cfg=OutputConfig(out_html=out_html, out_png=out_png, open_browser=False, title="UMAP Selftest"),
        umap_params=UmapParams(n_components=2, seed=seed),
        plot_map=PlotMap(color_by=LABEL_COL, size_by=CONFIDENCE_COL, opacity_by=ENTROPY_COL, symbol_by=None),
        overlay_cfg=None,
        link_cfg=HyperlinkConfig(url_template="/planets/{planet_id}.html"),
        labels_csv=None,
        dedupe=True,
        planet_id_col=PLANET_ID_COL,
        log_ctx=PipelineLogContext(log_path=DEFAULT_LOG_PATH, cli_name="spectramind diagnose umap(selftest)"),
    )
    typer.echo(json.dumps(result, indent=2))

return app
```

# -----------------------------------------------------------------------------

# Module Entrypoint

# -----------------------------------------------------------------------------

if **name** == "**main**":
\# When executed directly, expose the Typer CLI (if available)
if typer is None:
print(
"Typer is not installed. Install with `pip install typer[all]` or "
"import run\_umap\_pipelin e() from this module.",
file=sys.stderr,
)
sys.exit(2)
\_app = \_build\_typer\_app()
\_app()
