# src/diagnostics/plot/tsne/interactive.py

# =============================================================================

# ✨ SpectraMind V50 — Interactive t-SNE Latent Plotter (HTML + optional PNG)

# -----------------------------------------------------------------------------

# Purpose

# Create an interactive Plotly visualization of latent vectors using t-SNE,

# with rich diagnostics overlays (symbolic labels/scores, entropy, SHAP, GLL),

# optional per-point hyperlinks, and reproducibility logging that aligns with

# the CLI-first, Hydra-safe philosophy of SpectraMind V50.

#

# Design & Guarantees

# • Input latents: .npy/.npz/.csv (rows=planets, cols=features)

# • Optional labels/metrics CSV joined by planet\_id

# • Optional symbolic overlay JSON (rule scores/labels per planet)

# • Deterministic-ish t-SNE via explicit seed, PCA init, standardized inputs

# • 2D/3D projections supported (n\_components ∈ {2,3})

# • Plotly HTML always produced; PNG if kaleido is installed

# • Append run metadata to v50\_debug\_log.md for auditability

# • Typer CLI + importable Python API

#

# Example (CLI):

# spectramind diagnose tsne \\

# --latents artifacts/latents\_v50.npy \\

# --labels artifacts/latents\_meta.csv \\

# --symbolic-overlays artifacts/symbolic\_violation\_summary.json \\

# --color-by label --size-by confidence --opacity-by entropy \\

# --url-template "/planets/{planet\_id}.html" \\

# --out-html artifacts/tsne\_v50.html --out-png artifacts/tsne\_v50.png

#

# Notes

# • This script performs visualization only. All analytics/derivations are

# assumed to be produced elsewhere by the CLI pipeline (NASA-grade ethos).

# • If scikit-learn isn't available for TSNE, we gracefully fall back to PCA.

# =============================================================================

from **future** import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

# Plotly for interactive visualization (no seaborn by project policy)

import plotly.express as px
import plotly.io as pio

# Optional PNG export

try:
import kaleido  # noqa: F401
\_KALEIDO\_AVAILABLE = True
except Exception:
\_KALEIDO\_AVAILABLE = False

# Typer for CLI

try:
import typer  # type: ignore
except Exception:
typer = None

# Scikit-learn (TSNE + helpers). If not present, we will fallback to PCA.

try:
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
\_SKLEARN\_AVAILABLE = True
except Exception:
\_SKLEARN\_AVAILABLE = False

# -----------------------------------------------------------------------------

# Constants

# -----------------------------------------------------------------------------

DEFAULT\_LOG\_PATH = Path("v50\_debug\_log.md")
DEFAULT\_HTML\_OUT = Path("artifacts") / "tsne\_v50.html"
DEFAULT\_PNG\_OUT = Path("artifacts") / "tsne\_v50.png"

PLANET\_ID\_COL = "planet\_id"
LABEL\_COL = "label"
CONFIDENCE\_COL = "confidence"
ENTROPY\_COL = "entropy"
SHAP\_MAG\_COL = "shap"
GLL\_COL = "gll"

DEFAULT\_SEED = 1337

# -----------------------------------------------------------------------------

# Dataclasses

# -----------------------------------------------------------------------------

@dataclass
class TsneParams:
"""Parameters that govern t-SNE behavior."""
n\_components: int = 2
perplexity: float = 30.0
learning\_rate: float = 200.0
n\_iter: int = 1000
early\_exaggeration: float = 12.0
metric: str = "euclidean"
angle: float = 0.5
init: str = "pca"  # "random" or "pca"
seed: int = DEFAULT\_SEED

@dataclass
class PlotMap:
"""Visual encodings for the scatter plot."""
color\_by: Optional\[str] = None
size\_by: Optional\[str] = None
opacity\_by: Optional\[str] = None
symbol\_by: Optional\[str] = None
hover\_cols: List\[str] = field(default\_factory=list)

@dataclass
class OverlayConfig:
"""Symbolic overlay config for merging rule/score metadata."""
symbolic\_overlays\_path: Optional\[Path] = None
symbolic\_score\_key: str = "violation\_score"
symbolic\_label\_key: str = "top\_rule"
map\_score\_to: Optional\[str] = "symbolic\_score"
map\_label\_to: Optional\[str] = "symbolic\_label"

@dataclass
class HyperlinkConfig:
"""Per-point hyperlink configuration."""
url\_template: Optional\[str] = None
url\_col\_name: str = "url"

@dataclass
class OutputConfig:
"""Output paths and figure presentation."""
out\_html: Path = DEFAULT\_HTML\_OUT
out\_png: Optional\[Path] = None
open\_browser: bool = False
title: str = "SpectraMind V50 — t-SNE Latents"

@dataclass
class PipelineLogContext:
"""Append a table row to v50\_debug\_log.md for auditability."""
log\_path: Path = DEFAULT\_LOG\_PATH
cli\_name: str = "spectramind diagnose tsne"
config\_hash\_path: Optional\[Path] = Path("run\_hash\_summary\_v50.json")

# -----------------------------------------------------------------------------

# Utilities

# -----------------------------------------------------------------------------

def \_now\_iso() -> str:
"""Return current local timestamp."""
return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def \_ensure\_parent\_dir(p: Path) -> None:
"""Ensure parent directory exists."""
p = Path(p)
if p.parent and not p.parent.exists():
p.parent.mkdir(parents=True, exist\_ok=True)

def setup\_logging(level: int = logging.INFO) -> None:
"""Configure console logging."""
logging.basicConfig(
level=level,
format="%(asctime)s | %(levelname)-7s | %(message)s",
datefmt="%H:%M:%S",
)

def \_coerce\_to\_str\_index(s: pd.Series) -> pd.Series:
"""Coerce identifiers to str, robustly."""
return s.astype(str).str.strip()

def \_safe\_read\_csv(path: Path) -> pd.DataFrame:
"""Read CSV with reasonable defaults."""
return pd.read\_csv(path)

def \_safe\_read\_json(path: Path) -> Any:
"""Read JSON safely."""
return json.loads(Path(path).read\_text(encoding="utf-8"))

def append\_v50\_log(ctx: PipelineLogContext, metadata: Dict\[str, Any]) -> None:
"""Append a Markdown table row with the most relevant run metadata."""
try:
\_ensure\_parent\_dir(ctx.log\_path)
\# Try to read config hash (if the run recorded it)
config\_hash = ""
if ctx.config\_hash\_path and Path(ctx.config\_hash\_path).exists():
try:
obj = json.loads(Path(ctx.config\_hash\_path).read\_text(encoding="utf-8"))
config\_hash = obj.get("config\_hash") or obj.get("hash") or obj.get("run\_hash") or ""
except Exception:
config\_hash = ""

```
    row = {
        "timestamp": _now_iso(),
        "cli": ctx.cli_name,
        "config_hash": config_hash,
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

    if not ctx.log_path.exists() or ctx.log_path.stat().st_size == 0:
        header = (
            "# SpectraMind V50 — Debug Log (t-SNE)\n\n"
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

# -----------------------------------------------------------------------------

# IO & Merge

# -----------------------------------------------------------------------------

def load\_latents(latents\_path: Path, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""
Load latents from .npy/.npz/.csv and standardize column names.

```
Returns a DataFrame with:
  - `planet_id_col` column
  - feature columns (named f0..fD-1 if loaded from npy/npz)
"""
path = Path(latents_path)
if not path.exists():
    raise FileNotFoundError(f"Latents file not found: {path}")
ext = path.suffix.lower()

if ext == ".npy":
    mat = np.load(path)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D latents in {path}, got shape={mat.shape}")
    df = pd.DataFrame(mat, columns=[f"f{i}" for i in range(mat.shape[1])])
    df[planet_id_col] = [str(i) for i in range(len(df))]
    df = df[[planet_id_col] + [c for c in df.columns if c != planet_id_col]]

elif ext == ".npz":
    data = np.load(path)
    if "X" not in data and "latents" not in data:
        raise KeyError(f"NPZ must contain 'X' or 'latents'. keys={list(data.keys())}")
    X = data["X"] if "X" in data else data["latents"]
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    if "planet_id" in data:
        ids = [str(x) for x in data["planet_id"]]
    else:
        ids = [str(i) for i in range(len(df))]
    df[planet_id_col] = ids
    df = df[[planet_id_col] + [c for c in df.columns if c != planet_id_col]]

elif ext in (".csv", ".txt"):
    df = _safe_read_csv(path)
    if planet_id_col not in df.columns:
        # Try common alternatives
        alt = [c for c in df.columns if c.lower() in ("planet_id", "planet", "id")]
        if alt:
            df = df.rename(columns={alt[0]: planet_id_col})
        else:
            df[planet_id_col] = [str(i) for i in range(len(df))]
    # Ensure there's at least one numeric column; names preserved

    numeric_cols = [c for c in df.columns if c != planet_id_col and pd.api.types.is_numeric_dtype(df[c])]
    if not numeric_cols:
        raise ValueError("No numeric feature columns found in CSV.")
else:
    raise ValueError(f"Unsupported file extension: {ext}")

df[planet_id_col] = _coerce_to_str_index(df[planet_id_col])
return df
```

def dedupe\_by\_planet\_id(df: pd.DataFrame, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""Drop duplicate planet\_id rows, keeping the first occurrence."""
if df\[planet\_id\_col].duplicated().any():
logging.info("Duplicate planet\_id rows detected; deduping (keep first).")
return df.drop\_duplicates(subset=\[planet\_id\_col], keep="first", ignore\_index=True)
return df

def merge\_labels(df\_latents: pd.DataFrame, labels\_csv: Optional\[Path]) -> pd.DataFrame:
"""Merge optional labels CSV on planet\_id."""
if not labels\_csv:
return df\_latents
if not Path(labels\_csv).exists():
logging.warning(f"Labels CSV not found: {labels\_csv}; skipping merge.")
return df\_latents
meta = \_safe\_read\_csv(Path(labels\_csv))
if PLANET\_ID\_COL not in meta.columns:
alt = \[c for c in meta.columns if c.lower() in ("planet\_id", "planet", "id")]
if not alt:
raise KeyError(f"Labels CSV must include '{PLANET\_ID\_COL}'.")
meta = meta.rename(columns={alt\[0]: PLANET\_ID\_COL})
meta\[PLANET\_ID\_COL] = \_coerce\_to\_str\_index(meta\[PLANET\_ID\_COL])
return df\_latents.merge(meta, on=PLANET\_ID\_COL, how="left", validate="one\_to\_one")

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
# Accept dict[str, dict] or list[dict]
if isinstance(obj, dict):
    records = []
    for pid, payload in obj.items():
        if not isinstance(payload, dict):
            continue
        records.append({
            planet_id_col: str(pid),
            (overlay_cfg.map_score_to or "symbolic_score"): payload.get(overlay_cfg.symbolic_score_key),
            (overlay_cfg.map_label_to or "symbolic_label"): payload.get(overlay_cfg.symbolic_label_key),
        })
    ov = pd.DataFrame.from_records(records)
elif isinstance(obj, list):
    ov = pd.DataFrame(obj)
    if planet_id_col not in ov.columns:
        alt = [c for c in ov.columns if c.lower() in ("planet_id", "planet", "id")]
        if not alt:
            raise KeyError(f"Symbolic overlay JSON missing '{planet_id_col}'.")
        ov = ov.rename(columns={alt[0]: planet_id_col})
    if overlay_cfg.symbolic_score_key in ov.columns and overlay_cfg.map_score_to:
        ov = ov.rename(columns={overlay_cfg.symbolic_score_key: overlay_cfg.map_score_to})
    if overlay_cfg.symbolic_label_key in ov.columns and overlay_cfg.map_label_to:
        ov = ov.rename(columns={overlay_cfg.symbolic_label_key: overlay_cfg.map_label_to})
else:
    raise ValueError("Unsupported symbolic overlay JSON format.")

ov[planet_id_col] = _coerce_to_str_index(ov[planet_id_col])
return df.merge(ov, on=planet_id_col, how="left", validate="one_to_one")
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
        return f"{link_cfg.url_template}{pid}"

df[link_cfg.url_col_name] = df[planet_id_col].map(_fmt)
return df
```

# -----------------------------------------------------------------------------

# Embedding

# -----------------------------------------------------------------------------

def compute\_embedding\_tsne(X: np.ndarray, params: TsneParams) -> np.ndarray:
"""Compute t-SNE (or PCA fallback if scikit-learn unavailable)."""
if not \_SKLEARN\_AVAILABLE:
logging.warning("scikit-learn not available; using PCA fallback for t-SNE.")
d = max(2, min(params.n\_components, 3))
\# Standardize for stability
Xs = (X - X.mean(0, keepdims=True)) / (X.std(0, keepdims=True) + 1e-8)
pca = PCA(n\_components=d, random\_state=params.seed)
return pca.fit\_transform(Xs)

```
# Standardize features to help TSNE & improve determinism
Xs = StandardScaler(with_mean=True, with_std=True).fit_transform(X)

# Guard perplexity: must be < n_samples
n = Xs.shape[0]
safe_perp = min(params.perplexity, max(5.0, (n - 1) / 3.0))  # safe upper bound heuristic

tsne = TSNE(
    n_components=max(2, min(params.n_components, 3)),
    perplexity=safe_perp,
    learning_rate=params.learning_rate,
    n_iter=params.n_iter,
    early_exaggeration=params.early_exaggeration,
    metric=params.metric,
    init=params.init,
    angle=params.angle,
    random_state=params.seed,
    n_jobs=None,       # default threads; control via env if needed
    verbose=0,
    square_distances=True,
)
Y = tsne.fit_transform(Xs)
return Y
```

# -----------------------------------------------------------------------------

# Plotting

# -----------------------------------------------------------------------------

def \_build\_plot\_title(base: str, params: TsneParams) -> str:
"""Build a descriptive title with key t-SNE parameters."""
return (
f"{base} — t-SNE (dims={max(2, min(params.n\_components, 3))}, "
f"perplexity={params.perplexity}, lr={params.learning\_rate}, seed={params.seed})"
)

def build\_plot(
df: pd.DataFrame,
emb: np.ndarray,
plot\_map: PlotMap,
link\_cfg: HyperlinkConfig,
out\_cfg: OutputConfig,
planet\_id\_col: str = PLANET\_ID\_COL,
) -> "plotly.graph\_objs.\_figure.Figure":
"""Build a 2D/3D Plotly scatter with rich hover and optional opacity mapping."""
n\_components = emb.shape\[1]
coords: Dict\[str, np.ndarray] = {}
if n\_components >= 1:
coords\["x"] = emb\[:, 0]
if n\_components >= 2:
coords\["y"] = emb\[:, 1]
if n\_components >= 3:
coords\["z"] = emb\[:, 2]

```
pdf = df.copy()
for k, v in coords.items():
    pdf[k] = v

# Hover metadata (include known diagnostics if present)
hover_cols = set(plot_map.hover_cols or [])
for c in (planet_id_col, LABEL_COL, CONFIDENCE_COL, ENTROPY_COL, SHAP_MAG_COL, GLL_COL, link_cfg.url_col_name):
    if c in pdf.columns:
        hover_cols.add(c)
if planet_id_col in pdf.columns:
    pdf[planet_id_col] = _coerce_to_str_index(pdf[planet_id_col])

common_kwargs = dict(
    data_frame=pdf,
    hover_data=sorted(list(hover_cols)) if hover_cols else None,
    title=_build_plot_title(out_cfg.title, TsneParams(
        n_components=n_components,
        perplexity=0,            # display-only; already in title above
        learning_rate=0,         # ditto (kept in title builder)
        n_iter=0,
        early_exaggeration=0,
        metric="euclidean",
        angle=0.5,
        init="pca",
        seed=DEFAULT_SEED,
    )),
)

if n_components >= 3:
    fig = px.scatter_3d(
        **common_kwargs,
        x="x",
        y="y",
        z="z",
        color=plot_map.color_by if plot_map.color_by in pdf.columns else None,
        size=plot_map.size_by if plot_map.size_by in pdf.columns else None,
        symbol=plot_map.symbol_by if plot_map.symbol_by in pdf.columns else None,
    )
else:
    fig = px.scatter(
        **common_kwargs,
        x="x",
        y="y",
        color=plot_map.color_by if plot_map.color_by in pdf.columns else None,
        size=plot_map.size_by if plot_map.size_by in pdf.columns else None,
        symbol=plot_map.symbol_by if plot_map.symbol_by in pdf.columns else None,
    )

# Opacity mapping: numeric column → [0.25, 1.0]
if plot_map.opacity_by and plot_map.opacity_by in pdf.columns:
    col = pdf[plot_map.opacity_by]
    if pd.api.types.is_numeric_dtype(col):
        v = col.to_numpy(dtype=float, copy=False)
        mean_val = np.nanmean(v)
        v = np.nan_to_num(v, nan=(mean_val if np.isfinite(mean_val) else 0.0))
        p2, p98 = np.nanpercentile(v, 2), np.nanpercentile(v, 98)
        if p98 - p2 <= 1e-12:
            alpha = np.full_like(v, 0.9)
        else:
            alpha = 0.25 + 0.75 * (np.clip(v, p2, p98) - p2) / (p98 - p2)
        fig.update_traces(marker={"opacity": alpha})
    else:
        logging.warning(f"opacity-by column '{plot_map.opacity_by}' is not numeric; ignoring.")

# Add small instruction if we included URLs
if link_cfg.url_template and link_cfg.url_col_name in pdf.columns:
    fig.add_annotation(
        text="Tip: Ctrl/Cmd+Click the URL in hover to open the planet page.",
        xref="paper", yref="paper", x=0, y=1.08, showarrow=False, align="left",
        font={"size": 12}
    )

fig.update_layout(
    legend_title_text=plot_map.color_by if plot_map.color_by else "legend",
    template="plotly_white",
    margin=dict(l=40, r=40, t=80, b=40),
    hovermode="closest",
)
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

# Orchestration

# -----------------------------------------------------------------------------

def run\_tsne\_pipeline(
latents\_path: Path,
out\_cfg: OutputConfig,
tsne\_params: TsneParams,
plot\_map: PlotMap,
overlay\_cfg: Optional\[OverlayConfig] = None,
link\_cfg: Optional\[HyperlinkConfig] = None,
labels\_csv: Optional\[Path] = None,
dedupe: bool = False,
planet\_id\_col: str = PLANET\_ID\_COL,
log\_ctx: Optional\[PipelineLogContext] = None,
) -> Dict\[str, Any]:
"""End-to-end: load → merge → embed (t-SNE) → plot → save → log."""
setup\_logging()
t0 = time.time()

```
# Load & prep
df = load_latents(latents_path, planet_id_col=planet_id_col)
if dedupe:
    df = dedupe_by_planet_id(df, planet_id_col=planet_id_col)
df = merge_labels(df, labels_csv)

if overlay_cfg:
    df = merge_symbolic_overlays(df, overlay_cfg, planet_id_col=planet_id_col)

if link_cfg:
    df = add_hyperlinks(df, link_cfg, planet_id_col=planet_id_col)

# Feature matrix
feat_cols = [c for c in df.columns if c.startswith("f") and pd.api.types.is_numeric_dtype(df[c])]
if not feat_cols:
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
    raise ValueError("No numeric feature columns found for t-SNE embedding.")

X = df[feat_cols].to_numpy(dtype=np.float32, copy=False)

# t-SNE
Y = compute_embedding_tsne(X, tsne_params)

# Plot
fig = build_plot(
    df=df,
    emb=Y,
    plot_map=plot_map,
    link_cfg=link_cfg or HyperlinkConfig(),
    out_cfg=out_cfg,
    planet_id_col=planet_id_col,
)

# Save artifacts
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

if out_cfg.open_browser:
    try:
        import webbrowser
        webbrowser.open_new_tab(out_cfg.out_html.as_uri())
    except Exception as e:
        logging.warning(f"Failed to open browser: {e}")

dt = time.time() - t0

# Log
if log_ctx:
    append_v50_log(
        log_ctx,
        metadata={
            "latents": str(latents_path),
            "labels": str(labels_csv) if labels_csv else "",
            "symbolic": str(overlay_cfg.symbolic_overlays_path) if overlay_cfg and overlay_cfg.symbolic_overlays_path else "",
            "out_html": str(out_cfg.out_html),
            "out_png": str(out_cfg.out_png) if out_cfg.out_png else "",
            "duration_sec": f"{dt:.2f}",
            "tsne": f"dims={tsne_params.n_components},perp={tsne_params.perplexity},lr={tsne_params.learning_rate},iter={tsne_params.n_iter},seed={tsne_params.seed}",
            "color_by": plot_map.color_by or "",
            "size_by": plot_map.size_by or "",
            "opacity_by": plot_map.opacity_by or "",
            "symbol_by": plot_map.symbol_by or "",
            "sklearn_available": str(_SKLEARN_AVAILABLE),
            "kaleido_available": str(_KALEIDO_AVAILABLE),
        },
    )

return {
    "html": str(out_cfg.out_html),
    "png": png_path,
    "n": len(df),
    "dims": Y.shape[1],
    "sklearn_available": _SKLEARN_AVAILABLE,
    "kaleido_available": _KALEIDO_AVAILABLE,
    "duration_sec": dt,
}
```

# -----------------------------------------------------------------------------

# CLI

# -----------------------------------------------------------------------------

def \_build\_typer\_app() -> "typer.Typer":
"""Build the Typer CLI application for this module."""
if typer is None:
raise RuntimeError(
"Typer is not installed. Install with `pip install typer[all]` "
"or import and use run\_tsne\_pipeline() programmatically."
)

```
app = typer.Typer(
    add_completion=True,
    help="SpectraMind V50 — Interactive t-SNE Latent Plotter (HTML + PNG)",
    no_args_is_help=True,
)

@app.command("run")
def cli_run(
    latents: Path = typer.Option(..., exists=True, dir_okay=False, help="Latents file (.npy/.npz/.csv). Rows=planets, cols=features."),
    labels: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional labels/metrics CSV with planet_id."),
    symbolic_overlays: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional symbolic overlay JSON (per-planet)."),
    symbolic_score_key: str = typer.Option("violation_score", help="Key in symbolic JSON to use as score column."),
    symbolic_label_key: str = typer.Option("top_rule", help="Key in symbolic JSON to use as label column."),
    map_score_to: str = typer.Option("symbolic_score", help="Column name to store symbolic score."),
    map_label_to: str = typer.Option("symbolic_label", help="Column name to store symbolic label."),
    color_by: Optional[str] = typer.Option(None, help="Column used for color mapping (e.g., 'label', 'symbolic_score')."),
    size_by: Optional[str] = typer.Option(None, help="Column used for marker size mapping (e.g., 'confidence')."),
    opacity_by: Optional[str] = typer.Option(None, help="Numeric column used for per-point opacity (e.g., 'entropy')."),
    symbol_by: Optional[str] = typer.Option(None, help="Categorical column used for marker symbol."),
    hover_cols: List[str] = typer.Option([], help="Additional columns to include in hover."),
    url_template: Optional[str] = typer.Option(None, help="Per-point hyperlink template, e.g. '/planets/{planet_id}.html'."),
    url_col_name: str = typer.Option("url", help="Column name for generated URLs."),
    out_html: Path = typer.Option(DEFAULT_HTML_OUT, help="Output HTML path."),
    out_png: Optional[Path] = typer.Option(None, help="Optional PNG path (requires kaleido)."),
    open_browser: bool = typer.Option(False, help="Open the HTML in a browser after generation."),
    title: str = typer.Option("SpectraMind V50 — t-SNE Latents", help="Plot title."),
    n_components: int = typer.Option(2, help="t-SNE output dimensions (2 or 3)."),
    perplexity: float = typer.Option(30.0, help="t-SNE perplexity (must be < n_samples)."),
    learning_rate: float = typer.Option(200.0, help="t-SNE learning rate."),
    n_iter: int = typer.Option(1000, help="t-SNE iterations."),
    early_exaggeration: float = typer.Option(12.0, help="t-SNE early exaggeration."),
    metric: str = typer.Option("euclidean", help="t-SNE distance metric."),
    angle: float = typer.Option(0.5, help="t-SNE Barnes-Hut angle (0.2–0.8)."),
    init: str = typer.Option("pca", help="t-SNE init: 'pca' or 'random'."),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed."),
    dedupe: bool = typer.Option(False, help="Dedupe rows by planet_id (keep first)."),
    planet_id_col: str = typer.Option(PLANET_ID_COL, help="Name of planet ID column."),
    log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="Path to v50_debug_log.md."),
    config_hash_path: Optional[Path] = typer.Option(Path("run_hash_summary_v50.json"), help="Optional path to run hash summary JSON."),
):
    """Run the t-SNE plotting pipeline."""
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
    tsne_params = TsneParams(
        n_components=n_components,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
        early_exaggeration=early_exaggeration,
        metric=metric,
        angle=angle,
        init=init,
        seed=seed,
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
        cli_name="spectramind diagnose tsne",
        config_hash_path=config_hash_path,
    )

    result = run_tsne_pipeline(
        latents_path=latents,
        out_cfg=out_cfg,
        tsne_params=tsne_params,
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
    tmp_dir: Path = typer.Option(Path("./artifacts/_tsne_selftest"), help="Directory for selftest artifacts."),
    n: int = typer.Option(256, help="Number of synthetic samples."),
    d: int = typer.Option(16, help="Latent dimensionality."),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed."),
):
    """Create a synthetic dataset and verify the pipeline end-to-end."""
    rng = np.random.default_rng(seed)
    X = np.r_[rng.normal(0, 1, size=(n // 2, d)) + 2.0, rng.normal(0, 1, size=(n - n // 2, d)) - 2.0]
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

    out_html = tmp_dir / "tsne_selftest.html"
    out_png = tmp_dir / "tsne_selftest.png"

    result = run_tsne_pipeline(
        latents_path=latents_csv,
        out_cfg=OutputConfig(out_html=out_html, out_png=out_png, open_browser=False, title="t-SNE Selftest"),
        tsne_params=TsneParams(n_components=2, seed=seed),
        plot_map=PlotMap(color_by=LABEL_COL, size_by=CONFIDENCE_COL, opacity_by=ENTROPY_COL, symbol_by=None),
        overlay_cfg=None,
        link_cfg=HyperlinkConfig(url_template="/planets/{planet_id}.html"),
        labels_csv=None,
        dedupe=True,
        planet_id_col=PLANET_ID_COL,
        log_ctx=PipelineLogContext(log_path=DEFAULT_LOG_PATH, cli_name="spectramind diagnose tsne(selftest)"),
    )
    typer.echo(json.dumps(result, indent=2))

return app
```

# -----------------------------------------------------------------------------

# Entrypoint

# -----------------------------------------------------------------------------

if **name** == "**main**":
if typer is None:
print(
"Typer is not installed. Install with `pip install typer[all]` or "
"import run\_tsne\_pipeline() from this module.",
file=sys.stderr,
)
sys.exit(2)
\_app = \_build\_typer\_app()
\_app()
