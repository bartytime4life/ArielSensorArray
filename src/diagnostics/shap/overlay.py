# src/diagnostics/shap/overlay.py

# =============================================================================

# 🛰️ SpectraMind V50 — SHAP × μ Spectrum Overlay Generator

# -----------------------------------------------------------------------------

# Purpose

# • Generate science-ready overlays of SHAP importance over predicted μ spectra.

# • Export static (PNG) and interactive (HTML) visualizations, plus CSV/JSON.

# • Provide CLI-first, Hydra-friendly, reproducible execution with run hashing.

#

# Design Tenets

# • NASA-grade reproducibility: config snapshot, run hash, and artifact manifest.

# • CLI-first: all operations are invocable via Typer CLI with explicit flags.

# • Safe & robust: strict I/O validation, clear error messages, rich logging.

#

# Inputs (any combination of CSV or NPY for arrays; wavelengths optional):

# • --mu            Path to μ spectrum (shape \[B] for a single planet).

# • --shap          Path to SHAP values aligned to μ bins (shape \[B]).

# • --wavelengths   Optional path to wavelength centers (shape \[B]) in μm or nm.

# • --meta          Optional JSON/CSV with planet metadata (id, name, labels...).

#

# Key Features

# • Top-K bin finder (by |SHAP|) with export to CSV and JSON summary.

# • Static Matplotlib overlay: μ curve with SHAP bands + markers for top-K bins.

# • Interactive Plotly overlay: hover, zoom, and linked μ/|SHAP| traces.

# • Optional Savitzky–Golay reference smoothing (for visual comparison only).

# • Normalization options for μ and SHAP (zero-mean/unit-var, min-max).

# • Deterministic styling that’s publication-ready.

# • Comprehensive logging to v50\_debug\_log.md with run hash + artifact manifest.

#

# CLI Examples

# spectramind diagnose shap-overlay \\

# --mu runs/predict/planet\_0042\_mu.npy \\

# --shap runs/explain/planet\_0042\_shap.npy \\

# --wavelengths data/wavelengths\_283.csv \\

# --out-dir artifacts/diagnostics/shap/0042 \\

# --top-k 25 --png --html --csv --json --open-html

#

# python -m src.diagnostics.shap.overlay \\

# --mu mu.csv --shap shap.csv --wavelengths wl.csv --out-dir ./artifacts --png --html

#

# Notes

# • This tool does not compute SHAP; it visualizes CLI-produced arrays.

# • Units: If wavelengths absent, the x-axis uses bin indices.

# • Conforms to SpectraMind V50 logging & artifact conventions.

# =============================================================================

from **future** import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import json
import math
import os
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party scientific stack (assumed available in SpectraMind V50 env)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Headless safety for CI/Kaggle/servers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Plotly is optional but recommended for interactive HTML

try:
import plotly.graph\_objs as go
from plotly.offline import plot as plotly\_plot
\_HAS\_PLOTLY = True
except Exception:
\_HAS\_PLOTLY = False

# Optional Savitzky–Golay smoothing

try:
from scipy.signal import savgol\_filter
\_HAS\_SCIPY = True
except Exception:
\_HAS\_SCIPY = False

# ------------------------------- Utilities ----------------------------------

def \_now\_iso() -> str:
"""Return current timestamp in ISO 8601 with seconds precision (UTC)."""
return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def \_sha256\_bytes(\*chunks: bytes) -> str:
"""Compute a SHA256 over multiple byte chunks."""
h = hashlib.sha256()
for c in chunks:
h.update(c)
return h.hexdigest()

def \_read\_vector(path: Optional\[Union\[str, Path]]) -> Optional\[np.ndarray]:
"""
Read a 1D numeric vector from .npy, .csv, or .txt file.
Returns None if path is None. Raises on malformed inputs.
"""
if path is None:
return None
p = Path(path)
if not p.exists():
raise FileNotFoundError(f"Input file not found: {p}")
suffix = p.suffix.lower()
if suffix == ".npy":
arr = np.load(p)
arr = np.asarray(arr).reshape(-1)
return arr
elif suffix in \[".csv", ".txt", ".tsv"]:
\# Try pandas for robust parsing; allow single column of values
df = pd.read\_csv(p, sep=None, engine="python", header=None, comment="#")
\# Flatten to 1D in reading order
arr = df.values.reshape(-1).astype(float)
return arr
else:
raise ValueError(f"Unsupported file extension for vector: {p.suffix}. Use .npy/.csv/.txt")

def \_read\_metadata(path: Optional\[Union\[str, Path]]) -> Optional\[pd.DataFrame]:
"""
Read optional metadata describing the planet/sample. Supports JSON or CSV.
Returns a DataFrame (possibly with a single row) or None.
"""
if path is None:
return None
p = Path(path)
if not p.exists():
raise FileNotFoundError(f"Metadata file not found: {p}")
suffix = p.suffix.lower()
if suffix == ".json":
with open(p, "r", encoding="utf-8") as f:
data = json.load(f)
if isinstance(data, dict):
return pd.DataFrame(\[data])
elif isinstance(data, list):
return pd.DataFrame(data)
else:
raise ValueError("Unsupported JSON structure for metadata (expected object or list).")
elif suffix in \[".csv", ".tsv"]:
df = pd.read\_csv(p)
return df
else:
raise ValueError(f"Unsupported metadata extension: {p.suffix}. Use .json/.csv/.tsv")

def \_ensure\_out\_dir(path: Union\[str, Path]) -> Path:
"""Make sure output directory exists and return Path."""
out = Path(path)
out.mkdir(parents=True, exist\_ok=True)
return out

def *safe\_filename(s: str) -> str:
"""Transform arbitrary label into filesystem-safe filename fragment."""
return "".join(c if c.isalnum() or c in "-*." else "\_" for c in s)\[:200]

def \_normalize(arr: np.ndarray, mode: Optional\[str]) -> np.ndarray:
"""
Normalize a vector according to mode:
• None / "none": no change
• "zscore": zero-mean, unit-variance (with epsilon guard)
• "minmax": scale to \[0, 1] if range > 0; else zeros
"""
if mode is None or mode.lower() == "none":
return arr
eps = 1e-8
if mode.lower() == "zscore":
mu = float(np.mean(arr))
sd = float(np.std(arr))
sd = sd if sd > eps else 1.0
return (arr - mu) / sd
if mode.lower() == "minmax":
lo, hi = float(np.min(arr)), float(np.max(arr))
rng = hi - lo
if rng <= eps:
return np.zeros\_like(arr)
return (arr - lo) / rng
raise ValueError(f"Unknown normalization mode: {mode}")

def \_apply\_savgol(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
"""Apply Savitzky–Golay filter if SciPy is available; else return original."""
if not \_HAS\_SCIPY or window <= 2 or window % 2 == 0 or poly >= window:
return arr
try:
return savgol\_filter(arr, window\_length=window, polyorder=poly, mode="interp")
except Exception:
return arr

def \_topk\_indices\_by\_abs(arr: np.ndarray, k: int) -> np.ndarray:
"""Return indices of the largest |arr| values (stable), clipped to range."""
k = max(1, min(int(k), arr.shape\[0]))
idx = np.argpartition(np.abs(arr), -k)\[-k:]
\# Sort by abs value descending, then by index ascending for stability
idx = idx\[np.argsort(-np.abs(arr\[idx]), kind="stable")]
return np.sort(idx)  # return sorted for plotting clarity

def \_format\_float(x: float) -> str:
"""Format floats nicely for metadata/CSV exports."""
\# Prefer 6 significant digits; switch to scientific if very small/large
if x == 0.0:
return "0"
ax = abs(x)
if ax < 1e-3 or ax >= 1e4:
return f"{x:.6e}"
return f"{x:.6f}".rstrip("0").rstrip(".")

def \_render\_matplotlib\_overlay(
wavelengths: Optional\[np.ndarray],
mu: np.ndarray,
shap: np.ndarray,
topk\_idx: np.ndarray,
cfg: "OverlayConfig",
out\_png: Path,
) -> Dict\[str, Any]:
"""
Render a static Matplotlib figure: μ curve + |SHAP| band + Top-K markers.
Saves to out\_png. Returns dict with figure metadata (for manifest).
"""
x = wavelengths if wavelengths is not None else np.arange(mu.shape\[0], dtype=float)
x\_label = cfg.axis\_label if wavelengths is not None else "Bin Index"
shap\_abs = np.abs(shap)

```
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)
ax1.plot(x, mu, linewidth=1.8, label="μ spectrum")
if cfg.savgol_window and cfg.savgol_window > 2 and cfg.savgol_poly is not None:
    mu_sg = _apply_savgol(mu, cfg.savgol_window, cfg.savgol_poly)
    if not np.allclose(mu_sg, mu):
        ax1.plot(x, mu_sg, linestyle="--", linewidth=1.2, label="μ (Savitzky–Golay)")

ax1.set_xlabel(x_label)
ax1.set_ylabel("μ (normalized)" if cfg.mu_norm and cfg.mu_norm != "none" else "μ")
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_locator(MaxNLocator(nbins=8))

# Second y-axis for |SHAP|
ax2 = ax1.twinx()
ax2.fill_between(x, 0, shap_abs, alpha=0.25, step="pre", label="|SHAP| area")
ax2.set_ylabel("|SHAP|" if not cfg.shap_norm or cfg.shap_norm == "none" else f"|SHAP| ({cfg.shap_norm})")

# Top-K markers
x_top = x[topk_idx]
mu_top = mu[topk_idx]
shap_top = shap[topk_idx]
ax1.scatter(x_top, mu_top, s=36, marker="o", edgecolors="black", linewidths=0.5, label=f"Top-{cfg.top_k} by |SHAP|")
# Vertical lines for top-K
for xi in x_top:
    ax1.axvline(x=xi, ymin=0, ymax=1, alpha=0.10, linestyle=":", linewidth=0.8)

# Legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True)

title = cfg.title or "SHAP × μ Overlay"
ax1.set_title(title)

plt.tight_layout()
fig.savefig(out_png)
plt.close(fig)

return {
    "figure": "matplotlib",
    "path": str(out_png),
    "width_px": 12 * 150,
    "height_px": 6 * 150,
}
```

def \_render\_plotly\_overlay(
wavelengths: Optional\[np.ndarray],
mu: np.ndarray,
shap: np.ndarray,
topk\_idx: np.ndarray,
cfg: "OverlayConfig",
out\_html: Path,
) -> Dict\[str, Any]:
"""
Render an interactive Plotly overlay with μ and |SHAP| traces and Top-K markers.
Saves to out\_html. Returns dict with figure metadata (for manifest).
"""
if not \_HAS\_PLOTLY:
raise RuntimeError("Plotly is not installed. Re-run with --no-html or install plotly.")

```
x = wavelengths if wavelengths is not None else np.arange(mu.shape[0], dtype=float)
x_label = cfg.axis_label if wavelengths is not None else "Bin Index"
shap_abs = np.abs(shap)

# Base μ line
traces = [
    go.Scatter(
        x=x, y=mu,
        mode="lines",
        name="μ spectrum",
        hovertemplate=f"{x_label}: %{{x}}<br>μ: %{{y:.6f}}<extra></extra>",
    )
]

# Optional Savitzky–Golay
if cfg.savgol_window and cfg.savgol_window > 2 and cfg.savgol_poly is not None:
    mu_sg = _apply_savgol(mu, cfg.savgol_window, cfg.savgol_poly)
    if not np.allclose(mu_sg, mu):
        traces.append(
            go.Scatter(
                x=x, y=mu_sg,
                mode="lines",
                name="μ (Savitzky–Golay)",
                line=dict(dash="dash"),
                hovertemplate=f"{x_label}: %{{x}}<br>μ_sg: %{{y:.6f}}<extra></extra>",
            )
        )

# |SHAP| area (secondary axis)
traces.append(
    go.Scatter(
        x=x, y=shap_abs,
        name="|SHAP|",
        mode="lines",
        fill="tozeroy",
        yaxis="y2",
        hovertemplate=f"{x_label}: %{{x}}<br>|SHAP|: %{{y:.6f}}<extra></extra>",
    )
)

# Top-K points
traces.append(
    go.Scatter(
        x=x[topk_idx],
        y=mu[topk_idx],
        mode="markers",
        name=f"Top-{cfg.top_k} by |SHAP|",
        marker=dict(size=8, line=dict(width=0.5, color="black")),
        hovertemplate=(
            f"{x_label}: %{{x}}<br>"
            f"μ: %{{y:.6f}}<br>"
            f"SHAP: %{{customdata[0]:.6f}}<br>"
            f"|SHAP| rank: %{{customdata[1]}}<extra></extra>"
        ),
        customdata=np.stack([shap[topk_idx], np.arange(1, len(topk_idx) + 1)], axis=1),
    )
)

layout = go.Layout(
    title=cfg.title or "SHAP × μ Overlay",
    xaxis=dict(title=x_label),
    yaxis=dict(title="μ" if not cfg.mu_norm or cfg.mu_norm == "none" else f"μ ({cfg.mu_norm})"),
    yaxis2=dict(title="|SHAP|" if not cfg.shap_norm or cfg.shap_norm == "none" else f"|SHAP| ({cfg.shap_norm})",
                overlaying="y", side="right", showgrid=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=60, t=60, b=60),
    hovermode="x unified",
    template="plotly_white",
)

fig = go.Figure(data=traces, layout=layout)
plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")

return {
    "figure": "plotly",
    "path": str(out_html),
    "width_px": 1200,  # indicative
    "height_px": 650,
}
```

def \_write\_csv(path: Path, header: List\[str], rows: List\[List\[Any]]) -> None:
"""Write rows to CSV with UTF-8 encoding."""
df = pd.DataFrame(rows, columns=header)
df.to\_csv(path, index=False)

def \_append\_debug\_log(log\_path: Path, text: str) -> None:
"""Append a line to v50\_debug\_log.md (or provided log path)."""
with open(log\_path, "a", encoding="utf-8") as f:
f.write(text.rstrip() + "\n")

def \_compute\_run\_hash(cfg: "OverlayConfig", mu: np.ndarray, shap: np.ndarray, wl: Optional\[np.ndarray]) -> str:
"""Compute a SHA256 run hash across inputs and config (stable JSON)."""
cfg\_json = json.dumps(dataclasses.asdict(cfg), sort\_keys=True).encode("utf-8")
parts = \[
cfg\_json,
mu.astype(np.float64).tobytes(),
shap.astype(np.float64).tobytes(),
]
if wl is not None:
parts.append(wl.astype(np.float64).tobytes())
return \_sha256\_bytes(\*parts)

def \_open\_in\_default\_app(path: Path) -> None:
"""Best-effort open a file in the OS default viewer (non-blocking)."""
try:
if sys.platform.startswith("darwin"):
os.system(f"open '{path}' >/dev/null 2>&1 &")
elif os.name == "nt":
os.startfile(str(path))  # type: ignore\[attr-defined]
else:
os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")
except Exception:
\# Silently ignore if environment disallows opening
pass

# ----------------------------- Config Dataclass ------------------------------

@dataclass
class OverlayConfig:
"""
Overlay configuration and CLI-mappable settings.

```
Fields:
    mu_path: Path to μ spectrum file (.npy/.csv/.txt).
    shap_path: Path to SHAP values file (.npy/.csv/.txt).
    wavelengths_path: Optional path to wavelength centers (.npy/.csv/.txt).
    metadata_path: Optional path to JSON/CSV metadata.

    top_k: Number of top |SHAP| bins to annotate/export.
    mu_norm: One of {"none", "zscore", "minmax"}; normalization for μ.
    shap_norm: One of {"none", "zscore", "minmax"}; normalization for SHAP.
    abs_shap: If True, rank by |SHAP| (recommended). Always used for ranking/area.
    savgol_window: Optional odd window length for Savitzky–Golay smoothing (visual).
    savgol_poly: Polynomial order for Savitzky–Golay (must be < window).

    out_dir: Output directory for artifacts.
    title: Optional plot title override.
    axis_label: Axis label for wavelengths (e.g., "Wavelength (μm)").
    basename: Basename for outputs (default derived from input names).
    png: Emit PNG figure.
    html: Emit Plotly HTML (requires plotly).
    csv: Emit CSV of top-K bins.
    json: Emit JSON summary with stats + manifest.
    open_html: Open HTML in default browser after write.
    log_path: Debug log path (default: v50_debug_log.md at repo root).
    seed: Optional RNG seed for deterministic operations (not strictly needed here).
"""
mu_path: str
shap_path: str
wavelengths_path: Optional[str] = None
metadata_path: Optional[str] = None

top_k: int = 25
mu_norm: str = "none"
shap_norm: str = "none"
abs_shap: bool = True
savgol_window: Optional[int] = 0
savgol_poly: Optional[int] = 2

out_dir: str = "artifacts/diagnostics/shap"
title: Optional[str] = None
axis_label: str = "Wavelength"
basename: Optional[str] = None
png: bool = True
html: bool = True
csv: bool = True
json: bool = True
open_html: bool = False
log_path: str = "v50_debug_log.md"
seed: Optional[int] = None
```

# ------------------------------- Core Routine --------------------------------

def generate\_overlay(cfg: OverlayConfig) -> Dict\[str, Any]:
"""
Execute the full overlay pipeline:
1\) Load μ, SHAP, and optional wavelengths/metadata.
2\) Normalize as requested (μ and/or SHAP).
3\) Compute Top-K bins by |SHAP|.
4\) Render Matplotlib PNG and/or Plotly HTML.
5\) Export CSV/JSON with summary and manifest.
6\) Log CLI call into v50\_debug\_log.md.

```
Returns a manifest dict containing paths and run metadata.
"""
# Seed (defensive; not strictly used here, but keeps framework consistent).
if cfg.seed is not None:
    np.random.seed(cfg.seed)

# I/O load
mu = _read_vector(cfg.mu_path)
shap = _read_vector(cfg.shap_path)
wl = _read_vector(cfg.wavelengths_path) if cfg.wavelengths_path else None
meta_df = _read_metadata(cfg.metadata_path)

if mu is None or shap is None:
    raise ValueError("Both --mu and --shap must be provided.")
if mu.shape != shap.shape:
    raise ValueError(f"Shape mismatch: μ {mu.shape} vs SHAP {shap.shape}. They must be equal length.")
if wl is not None and wl.shape != mu.shape:
    raise ValueError(f"Wavelength shape mismatch: wl {wl.shape} vs μ {mu.shape}.")

# Normalization (visualization aid only)
mu_vis = _normalize(mu.astype(float), cfg.mu_norm)
shap_vis = _normalize(shap.astype(float), cfg.shap_norm)
shap_for_rank = np.abs(shap_vis) if cfg.abs_shap else shap_vis

# Top-K indices by |SHAP|
topk_idx = _topk_indices_by_abs(shap_vis, cfg.top_k)

# Output directory & basename
out_dir = _ensure_out_dir(cfg.out_dir)
if cfg.basename:
    base = _safe_filename(cfg.basename)
else:
    left = Path(cfg.mu_path).stem
    right = Path(cfg.shap_path).stem
    base = _safe_filename(f"{left}__{right}")

# Compute run hash (config + arrays)
run_hash = _compute_run_hash(cfg, mu_vis, shap_vis, wl)

# Exports
manifest: Dict[str, Any] = {
    "tool": "shap_overlay",
    "version": "v50",
    "timestamp": _now_iso(),
    "run_hash": run_hash,
    "config": dataclasses.asdict(cfg),
    "inputs": {
        "mu": str(Path(cfg.mu_path).resolve()),
        "shap": str(Path(cfg.shap_path).resolve()),
        "wavelengths": str(Path(cfg.wavelengths_path).resolve()) if cfg.wavelengths_path else None,
        "metadata": str(Path(cfg.metadata_path).resolve()) if cfg.metadata_path else None,
    },
    "artifacts": {},
    "stats": {},
    "topk": {},
    "meta_preview": {},
}

# Basic stats
def _stats(vec: np.ndarray, name: str) -> Dict[str, Any]:
    return {
        f"{name}_min": float(np.min(vec)),
        f"{name}_max": float(np.max(vec)),
        f"{name}_mean": float(np.mean(vec)),
        f"{name}_std": float(np.std(vec)),
        f"{name}_l2": float(np.linalg.norm(vec)),
    }

manifest["stats"].update(_stats(mu_vis, "mu"))
manifest["stats"].update(_stats(shap_vis, "shap"))

# Top-K table
shap_abs = np.abs(shap_vis)
ranks = np.argsort(-shap_abs, kind="stable")  # descending by |SHAP|
topk_sorted_global = ranks[: len(topk_idx)]
top_rows: List[List[Any]] = []
for rpos, idx in enumerate(topk_sorted_global, start=1):
    row = {
        "rank": rpos,
        "index": int(idx),
        "wavelength": float(wl[idx]) if wl is not None else float(idx),
        "mu": float(mu_vis[idx]),
        "shap": float(shap_vis[idx]),
        "abs_shap": float(shap_abs[idx]),
    }
    top_rows.append([row[k] for k in ["rank", "index", "wavelength", "mu", "shap", "abs_shap"]])

# CSV export
csv_path = out_dir / f"{base}__top{len(topk_idx)}.csv"
if cfg.csv:
    _write_csv(
        csv_path,
        header=["rank", "index", "wavelength_or_bin", "mu", "shap", "abs_shap"],
        rows=top_rows,
    )
    manifest["artifacts"]["csv"] = str(csv_path)

# PNG plot
if cfg.png:
    png_path = out_dir / f"{base}.png"
    fig_meta = _render_matplotlib_overlay(wl, mu_vis, shap_vis, topk_idx, cfg, png_path)
    manifest["artifacts"]["png"] = fig_meta

# HTML plot
if cfg.html:
    html_path = out_dir / f"{base}.html"
    fig_meta_html = _render_plotly_overlay(wl, mu_vis, shap_vis, topk_idx, cfg, html_path)
    manifest["artifacts"]["html"] = fig_meta_html
    if cfg.open_html:
        _open_in_default_app(html_path)

# JSON manifest
json_path = out_dir / f"{base}.json"
manifest["topk"] = {
    "k": len(topk_idx),
    "indices_sorted_by_abs_shap": [int(i) for i in topk_sorted_global.tolist()],
    "wavelengths": [float(wl[i]) if wl is not None else float(i) for i in topk_sorted_global.tolist()],
    "mu": [float(mu_vis[i]) for i in topk_sorted_global.tolist()],
    "shap": [float(shap_vis[i]) for i in topk_sorted_global.tolist()],
    "abs_shap": [float(shap_abs[i]) for i in topk_sorted_global.tolist()],
}
if meta_df is not None:
    preview = meta_df.head(1).to_dict(orient="records")[0]
    manifest["meta_preview"] = preview

if cfg.json:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["artifacts"]["json"] = str(json_path)

# Human-readable debug log append
log_path = Path(cfg.log_path)
log_path.parent.mkdir(parents=True, exist_ok=True)
cli_line = " ".join(
    [Path(sys.argv[0]).name]
    + [a if " " not in a else f'"{a}"' for a in sys.argv[1:]]
)
log_entry = f"""\
```

### SHAP OVERLAY | {\_now\_iso()}

* run\_hash: {run\_hash}
* out\_dir: {out\_dir}
* inputs: mu={cfg.mu\_path} shap={cfg.shap\_path} wl={cfg.wavelengths\_path or 'None'}
* config: top\_k={cfg.top\_k} mu\_norm={cfg.mu\_norm} shap\_norm={cfg.shap\_norm} savgol=({cfg.savgol\_window},{cfg.savgol\_poly})
* artifacts: png={manifest\['artifacts'].get('png','-')} html={manifest\['artifacts'].get('html','-')} csv={manifest\['artifacts'].get('csv','-')} json={manifest\['artifacts'].get('json','-')}
* cli: {cli\_line}
  """
  \_append\_debug\_log(log\_path, log\_entry)

  return manifest

# --------------------------------- CLI Layer ---------------------------------

def \_build\_argparser() -> argparse.ArgumentParser:
"""
Build an argparse CLI compatible with SpectraMind's Typer-driven ecosystem.
(We use argparse here to keep this module standalone-invocable via python -m.)
"""
p = argparse.ArgumentParser(
prog="shap-overlay",
description="Generate SHAP × μ overlay plots and exports (PNG/HTML/CSV/JSON).",
formatter\_class=argparse.ArgumentDefaultsHelpFormatter,
)
\# Inputs
p.add\_argument("--mu", dest="mu\_path", required=True, help="Path to μ vector (.npy/.csv/.txt).")
p.add\_argument("--shap", dest="shap\_path", required=True, help="Path to SHAP vector (.npy/.csv/.txt).")
p.add\_argument("--wavelengths", dest="wavelengths\_path", default=None, help="Optional wavelengths vector (.npy/.csv/.txt).")
p.add\_argument("--meta", dest="metadata\_path", default=None, help="Optional metadata (.json/.csv).")

```
# Visualization controls
p.add_argument("--top-k", type=int, default=25, help="Number of top |SHAP| bins to annotate/export.")
p.add_argument("--mu-norm", default="none", choices=["none", "zscore", "minmax"], help="Normalization for μ (visualization).")
p.add_argument("--shap-norm", default="none", choices=["none", "zscore", "minmax"], help="Normalization for SHAP (visualization).")
p.add_argument("--no-abs", dest="abs_shap", action="store_false", help="Use raw SHAP values for ranking instead of |SHAP|.")
p.add_argument("--savgol-window", type=int, default=0, help="Odd window size for Savitzky–Golay smoothing (0 disables).")
p.add_argument("--savgol-poly", type=int, default=2, help="Polynomial order for Savitzky–Golay (must be < window).")
p.add_argument("--axis-label", default="Wavelength (μm)", help="X-axis label when wavelengths are provided.")
p.add_argument("--title", default=None, help="Optional plot title override.")

# Outputs
p.add_argument("--out-dir", default="artifacts/diagnostics/shap", help="Output directory for artifacts.")
p.add_argument("--basename", default=None, help="Basename for artifact filenames (defaults to derived from input names).")
p.add_argument("--png", dest="png", action="store_true", help="Write PNG static figure.")
p.add_argument("--no-png", dest="png", action="store_false", help="Disable PNG export.")
p.add_argument("--html", dest="html", action="store_true", help="Write interactive HTML (requires plotly).")
p.add_argument("--no-html", dest="html", action="store_false", help="Disable HTML export.")
p.add_argument("--csv", dest="csv", action="store_true", help="Write CSV of top-K rows.")
p.add_argument("--no-csv", dest="csv", action="store_false", help="Disable CSV export.")
p.add_argument("--json", dest="json", action="store_true", help="Write JSON manifest and summary.")
p.add_argument("--no-json", dest="json", action="store_false", help="Disable JSON export.")
p.add_argument("--open-html", dest="open_html", action="store_true", help="Open HTML in default browser after export.")

# Logging & determinism
p.add_argument("--log-path", default="v50_debug_log.md", help="Debug log path to append run metadata.")
p.add_argument("--seed", type=int, default=None, help="Optional RNG seed for determinism.")
# Defaults for bool flags
p.set_defaults(png=True, html=True, csv=True, json=True, abs_shap=True, open_html=False)
return p
```

def main(argv: Optional\[List\[str]] = None) -> int:
"""
Entry point for CLI execution. Returns exit code (0 on success).
"""
argv = argv if argv is not None else sys.argv\[1:]
parser = \_build\_argparser()
args = parser.parse\_args(argv)

```
cfg = OverlayConfig(
    mu_path=args.mu_path,
    shap_path=args.shap_path,
    wavelengths_path=args.wavelengths_path,
    metadata_path=args.metadata_path,
    top_k=args.top_k,
    mu_norm=args.mu_norm,
    shap_norm=args.shap_norm,
    abs_shap=args.abs_shap,
    savgol_window=args.savgol_window,
    savgol_poly=args.savgol_poly,
    out_dir=args.out_dir,
    title=args.title,
    axis_label=args.axis_label,
    basename=args.basename,
    png=args.png,
    html=args.html,
    csv=args.csv,
    json=args.json,
    open_html=args.open_html,
    log_path=args.log_path,
    seed=args.seed,
)

try:
    manifest = generate_overlay(cfg)
    # Simple stdout summary for pipelines
    print(json.dumps({"ok": True, "run_hash": manifest["run_hash"], "artifacts": manifest.get("artifacts", {})}, indent=2))
    return 0
except Exception as e:
    # Write error to stderr and append to debug log
    err_text = f"[{_now_iso()}] SHAP OVERLAY ERROR: {e}\n{traceback.format_exc()}"
    sys.stderr.write(err_text + "\n")
    try:
        _append_debug_log(Path(getattr(cfg, "log_path", "v50_debug_log.md")), err_text)
    except Exception:
        pass
    # Also print a machine-readable failure for CI
    print(json.dumps({"ok": False, "error": str(e)}))
    return 1
```

# ------------------------------- Module Runner -------------------------------

if **name** == "**main**":
sys.exit(main())
