# src/diagnostics/spectral/smoothness/map.py

# =============================================================================

# 🧵 SpectraMind V50 — μ Spectral Smoothness Map (per-bin gradients/curvature)

# -----------------------------------------------------------------------------

# Purpose

# Compute per-bin smoothness diagnostics for μ spectra across planets and

# export a heatmap + CSV/JSON summaries. Metrics include:

# • |∇μ| (first-difference magnitude, "grad\_mag")

# • |∇²μ| (second-difference magnitude, "curv\_mag")

# • Total Variation TV = Σ |∇μ|

# • Optional Savitzky–Golay (or moving-average) smoothed reference and

# residual |μ − μ̂| ("resid\_mag")

#

# Outputs

# • artifacts/smoothness/heatmap.html (interactive Plotly)

# • artifacts/smoothness/heatmap.png   (if kaleido installed)

# • artifacts/smoothness/summary.csv   (per-planet aggregates)

# • artifacts/smoothness/summary.json  (per-planet + global stats)

# • Optional per-planet panel HTMLs (μ, μ̂, |∇μ|, |∇²μ|, |μ−μ̂|)

#

# Reproducibility

# • Read-only visualization/diagnostics on μ produced elsewhere

# • Deterministic given same inputs; appends a row to v50\_debug\_log.md

#

# CLI (examples)

# spectramind diagnose smoothness run \\

# --mu artifacts/mu\_preds.npy \\

# --labels artifacts/meta.csv \\

# --out-dir artifacts/smoothness \\

# --heatmap artifacts/smoothness/heatmap.html \\

# --png artifacts/smoothness/heatmap.png \\

# --per-planet-panels 4 \\

# --smooth.ref "savgol" --smooth.window 11 --smooth.poly 3 \\

# --flags.grad\_mag\_p95 1.75 --flags.curv\_mag\_p95 1.5

#

# Notes

# • This module computes diagnostics only; it does not modify μ or model state.

# • Savitzky–Golay is attempted via scipy.signal if available; otherwise a

# centered moving-average fallback is used.

# =============================================================================

from **future** import annotations

import json
import logging
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph\_objects as go
import plotly.io as pio

# Optional PNG export

try:
import kaleido  # noqa: F401
\_KALEIDO\_AVAILABLE = True
except Exception:
\_KALEIDO\_AVAILABLE = False

# Optional Savitzky–Golay smoothing

try:
from scipy.signal import savgol\_filter  # type: ignore
\_SCIPY\_AVAILABLE = True
except Exception:
\_SCIPY\_AVAILABLE = False

# Optional Typer CLI

try:
import typer
except Exception:
typer = None

# =============================================================================

# Constants

# =============================================================================

DEFAULT\_LOG\_PATH = Path("v50\_debug\_log.md")
DEFAULT\_OUT\_DIR = Path("artifacts") / "smoothness"
DEFAULT\_HEATMAP\_HTML = DEFAULT\_OUT\_DIR / "heatmap.html"
DEFAULT\_HEATMAP\_PNG = None
DEFAULT\_SUMMARY\_CSV = DEFAULT\_OUT\_DIR / "summary.csv"
DEFAULT\_SUMMARY\_JSON = DEFAULT\_OUT\_DIR / "summary.json"

PLANET\_ID\_COL = "planet\_id"
LABEL\_COL = "label"
CONFIDENCE\_COL = "confidence"
ENTROPY\_COL = "entropy"
SHAP\_MAG\_COL = "shap"
GLL\_COL = "gll"

DEFAULT\_SEED = 1337

# =============================================================================

# Dataclasses

# =============================================================================

@dataclass
class SmoothRefConfig:
"""Configuration for reference smoothing μ̂."""
\# ref: "none" | "savgol" | "moving\_avg"
ref: str = "savgol"
\# Savitzky–Golay parameters (used if ref == "savgol" and scipy available)
window: int = 11
poly: int = 3
\# Moving average window (used if ref == "moving\_avg" or scipy unavailable)
ma\_window: int = 9
\# Clamp window to valid odd >= poly+2 when applying
enforce\_valid: bool = True

@dataclass
class FlagConfig:
"""Thresholds for flagging potential roughness anomalies (per-planet)."""
\# Cumulative flags based on distribution across bins
grad\_mag\_p95: float = 1.75
curv\_mag\_p95: float = 1.25
resid\_mag\_p95: float = 1.50
\# Optional absolute clip thresholds (set <=0 to disable)
grad\_mag\_abs: float = 0.0
curv\_mag\_abs: float = 0.0
resid\_mag\_abs: float = 0.0

@dataclass
class OverlayConfig:
"""Optional overlays (per-planet metadata)."""
labels\_csv: Optional\[Path] = None
symbolic\_overlays\_path: Optional\[Path] = None
symbolic\_score\_key: str = "violation\_score"
symbolic\_label\_key: str = "top\_rule"
map\_score\_to: Optional\[str] = "symbolic\_score"
map\_label\_to: Optional\[str] = "symbolic\_label"

@dataclass
class OutputConfig:
"""Artifact outputs."""
out\_dir: Path = DEFAULT\_OUT\_DIR
heatmap\_html: Path = DEFAULT\_HEATMAP\_HTML
heatmap\_png: Optional\[Path] = DEFAULT\_HEATMAP\_PNG
summary\_csv: Path = DEFAULT\_SUMMARY\_CSV
summary\_json: Path = DEFAULT\_SUMMARY\_JSON
per\_planet\_panels: int = 0
open\_browser: bool = False
title: str = "SpectraMind V50 — μ Spectral Smoothness Map"

@dataclass
class PipelineLogContext:
"""Append audit metadata to Markdown log."""
log\_path: Path = DEFAULT\_LOG\_PATH
cli\_name: str = "spectramind diagnose smoothness"
config\_hash\_path: Optional\[Path] = Path("run\_hash\_summary\_v50.json")

# =============================================================================

# Utilities

# =============================================================================

def \_now\_iso() -> str:
return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def \_ensure\_parent(path: Path) -> None:
path = Path(path)
if path.parent and not path.parent.exists():
path.parent.mkdir(parents=True, exist\_ok=True)

def setup\_logging(level: int = logging.INFO) -> None:
logging.basicConfig(
level=level,
format="%(asctime)s | %(levelname)-7s | %(message)s",
datefmt="%H:%M:%S",
)

def \_coerce\_str\_id(s: pd.Series) -> pd.Series:
return s.astype(str).str.strip()

def \_read\_csv(path: Path) -> pd.DataFrame:
return pd.read\_csv(path)

def \_read\_json(path: Path) -> Any:
return json.loads(Path(path).read\_text())

def append\_v50\_log(ctx: PipelineLogContext, metadata: Dict\[str, Any]) -> None:
"""Append a Markdown table row to the debug log."""
try:
\_ensure\_parent(ctx.log\_path)
config\_hash = ""
if ctx.config\_hash\_path and Path(ctx.config\_hash\_path).exists():
try:
obj = json.loads(Path(ctx.config\_hash\_path).read\_text())
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
        f"| {row['timestamp']} | {row['cli']} | {row['config_hash']} "
        f"| {row.get('mu','')} | {row.get('labels','')} | {row.get('symbolic','')} "
        f"| {row.get('heatmap','')} | {row.get('png','')} |\n"
    )
    if not ctx.log_path.exists() or ctx.log_path.stat().st_size == 0:
        header = (
            "# SpectraMind V50 — Debug Log (Smoothness Map)\n\n"
            "| timestamp | cli | config_hash | mu | labels | symbolic | heatmap | png |\n"
            "|---|---|---|---|---|---|---|---|\n"
        )
        ctx.log_path.write_text(header + line)
    else:
        with open(ctx.log_path, "a", encoding="utf-8") as f:
            f.write(line)
except Exception as e:
    logging.warning(f"Failed to append to log {ctx.log_path}: {e}")
```

# =============================================================================

# Loading μ

# =============================================================================

def load\_mu\_matrix(mu\_path: Path, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""
Load μ into tall DataFrame with columns: planet\_id, bin, mu.

```
Accepts:
  • .npy  -> (N_planets × N_bins)
  • .npz  -> arrays 'mu' (N×B), optional 'planet_id' (N,)
  • .csv/.txt
        - tall: columns {planet_id, bin, mu}
        - wide: planet_id + numeric columns (assumed per-bin μ)
"""
mu_path = Path(mu_path)
if not mu_path.exists():
    raise FileNotFoundError(f"μ file not found: {mu_path}")

ext = mu_path.suffix.lower()
if ext == ".npy":
    mat = np.load(mu_path)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D μ in {mu_path}, got {mat.shape}")
    n, b = mat.shape
    dfw = pd.DataFrame(mat, columns=[f"f{i}" for i in range(b)])
    dfw[planet_id_col] = [f"P{i:04d}" for i in range(n)]
    tall = dfw.melt(id_vars=[planet_id_col], var_name="bin_col", value_name="mu")
    tall["bin"] = tall["bin_col"].str.replace("f", "", regex=False).astype(int)
    tall = tall.drop(columns=["bin_col"])

elif ext == ".npz":
    npz = np.load(mu_path)
    if "mu" not in npz:
        raise KeyError("NPZ is missing 'mu' array.")
    mat = npz["mu"]
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D μ in {mu_path}, got {mat.shape}")
    n, b = mat.shape
    ids = [str(x) for x in npz["planet_id"][:n]] if "planet_id" in npz else [f"P{i:04d}" for i in range(n)]
    dfw = pd.DataFrame(mat, columns=[f"f{i}" for i in range(b)])
    dfw[planet_id_col] = ids
    tall = dfw.melt(id_vars=[planet_id_col], var_name="bin_col", value_name="mu")
    tall["bin"] = tall["bin_col"].str.replace("f", "", regex=False).astype(int)
    tall = tall.drop(columns=["bin_col"])

elif ext in (".csv", ".txt"):
    df = _read_csv(mu_path)
    cols_lower = {c.lower() for c in df.columns}
    # Tall format
    if {"planet_id", "bin", "mu"}.issubset(cols_lower):
        ren = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "planet_id": ren[c] = planet_id_col
            elif cl == "bin": ren[c] = "bin"
            elif cl == "mu": ren[c] = "mu"
        tall = df.rename(columns=ren).copy()
        tall[planet_id_col] = _coerce_str_id(tall[planet_id_col])
        tall["bin"] = tall["bin"].astype(int)
    else:
        # Wide format: planet_id + numeric columns
        if planet_id_col not in df.columns:
            alt = [c for c in df.columns if c.lower() in ("planet_id", "planet", "id")]
            if alt:
                df = df.rename(columns={alt[0]: planet_id_col})
            else:
                df[planet_id_col] = [f"P{i:04d}" for i in range(len(df))]
        df[planet_id_col] = _coerce_str_id(df[planet_id_col])
        tall = df.melt(id_vars=[planet_id_col], var_name="bin_col", value_name="mu")
        # try to parse bin index
        def _to_idx(s: str) -> int:
            s = str(s)
            if s.startswith("f") or s.startswith("b"):
                return int(s[1:]) if s[1:].isdigit() else 0
            try:
                return int(s)
            except Exception:
                digits = "".join(ch for ch in s if ch.isdigit())
                return int(digits) if digits else 0
        tall["bin"] = tall["bin_col"].map(_to_idx).astype(int)
        tall = tall.drop(columns=["bin_col"])

else:
    raise ValueError(f"Unsupported μ file type: {ext}")

tall[planet_id_col] = _coerce_str_id(tall[planet_id_col])
tall = tall[[planet_id_col, "bin", "mu"]].sort_values([planet_id_col, "bin"]).reset_index(drop=True)
return tall
```

# =============================================================================

# Core Smoothness Metrics

# =============================================================================

def \_first\_diff(x: np.ndarray) -> np.ndarray:
"""Forward difference with same length (pad left with 0)."""
d = np.diff(x, prepend=x\[:1])
return d

def \_second\_diff(x: np.ndarray) -> np.ndarray:
"""Second difference with same length (pad at ends)."""
d1 = \_first\_diff(x)
d2 = np.diff(d1, prepend=d1\[:1])
return d2

def \_moving\_average(x: np.ndarray, window: int) -> np.ndarray:
"""Centered moving average with reflection padding; handle odd/even gracefully."""
if window <= 1:
return x.copy()
k = int(window)
if k % 2 == 0:
k += 1
pad = k // 2
xp = np.pad(x, (pad, pad), mode="reflect")
w = np.ones(k, dtype=float) / k
y = np.convolve(xp, w, mode="valid")
return y.astype(float)

def smooth\_reference(mu: np.ndarray, cfg: SmoothRefConfig) -> np.ndarray:
"""Compute μ̂ (smoothed reference) given config and availability of SciPy."""
if cfg.ref.lower() in ("none", "off", "false", "no"):
return mu.copy()

```
if cfg.ref.lower() == "savgol" and _SCIPY_AVAILABLE:
    # Ensure odd window >= poly+2 (S-G requirement: window > poly and window odd)
    w = cfg.window
    p = cfg.poly
    if cfg.enforce_valid:
        if w <= p:
            w = p + 3 if (p % 2 == 0) else p + 2
        if w % 2 == 0:
            w += 1
    w = max(3, w)
    try:
        return savgol_filter(mu, window_length=w, polyorder=p, mode="mirror")
    except Exception as e:
        logging.warning(f"savgol_filter failed ({e}); falling back to moving average.")
        return _moving_average(mu, cfg.ma_window)

# Fallback or explicit moving average
return _moving_average(mu, cfg.ma_window)
```

def compute\_smoothness\_fields(mu\_vec: np.ndarray, ref\_cfg: SmoothRefConfig) -> Dict\[str, np.ndarray]:
"""
Return a dict of per-bin fields:
• grad: ∇μ
• grad\_mag: |∇μ|
• curv: ∇²μ
• curv\_mag: |∇²μ|
• resid: μ − μ̂
• resid\_mag: |μ − μ̂|
• mu\_hat: μ̂ reference
"""
mu = mu\_vec.astype(float)
grad = \_first\_diff(mu)
curv = \_second\_diff(mu)
mu\_hat = smooth\_reference(mu, ref\_cfg)
resid = mu - mu\_hat

```
fields = {
    "grad": grad,
    "grad_mag": np.abs(grad),
    "curv": curv,
    "curv_mag": np.abs(curv),
    "resid": resid,
    "resid_mag": np.abs(resid),
    "mu_hat": mu_hat,
}
return fields
```

def summarize\_planet(fields: Dict\[str, np.ndarray]) -> Dict\[str, float]:
"""Aggregate per-planet scalar summaries."""
grad\_mag = fields\["grad\_mag"]
curv\_mag = fields\["curv\_mag"]
resid\_mag = fields\["resid\_mag"]

```
def _p(v: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(v, q)) if v.size else 0.0

tv = float(np.nansum(grad_mag))
s = {
    "tv": tv,
    "grad_mag_mean": float(np.nanmean(grad_mag)),
    "grad_mag_p95": _p(grad_mag, 95),
    "curv_mag_mean": float(np.nanmean(curv_mag)),
    "curv_mag_p95": _p(curv_mag, 95),
    "resid_mag_mean": float(np.nanmean(resid_mag)),
    "resid_mag_p95": _p(resid_mag, 95),
}
return s
```

def compute\_flags(summary: Dict\[str, float], flags: FlagConfig) -> Dict\[str, int]:
"""Return binary flags based on thresholds."""
out = {
"flag\_grad\_p95": int(summary\["grad\_mag\_p95"] > flags.grad\_mag\_p95),
"flag\_curv\_p95": int(summary\["curv\_mag\_p95"] > flags.curv\_mag\_p95),
"flag\_resid\_p95": int(summary\["resid\_mag\_p95"] > flags.resid\_mag\_p95),
}
if flags.grad\_mag\_abs > 0:
out\["flag\_grad\_abs"] = int(summary\["grad\_mag\_mean"] > flags.grad\_mag\_abs)
if flags.curv\_mag\_abs > 0:
out\["flag\_curv\_abs"] = int(summary\["curv\_mag\_mean"] > flags.curv\_mag\_abs)
if flags.resid\_mag\_abs > 0:
out\["flag\_resid\_abs"] = int(summary\["resid\_mag\_mean"] > flags.resid\_mag\_abs)
return out

# =============================================================================

# Overlays

# =============================================================================

def merge\_overlays(
tall\_mu: pd.DataFrame,
overlay\_cfg: OverlayConfig,
planet\_id\_col: str = PLANET\_ID\_COL,
) -> pd.DataFrame:
"""Merge labels CSV and symbolic overlays JSON by planet\_id (many:1)."""
df = tall\_mu.copy()

```
# Labels CSV (per-planet metrics)
if overlay_cfg.labels_csv and Path(overlay_cfg.labels_csv).exists():
    meta = _read_csv(Path(overlay_cfg.labels_csv))
    if planet_id_col not in meta.columns:
        alt = [c for c in meta.columns if c.lower() in ("planet_id", "planet", "id")]
        if alt:
            meta = meta.rename(columns={alt[0]: planet_id_col})
        else:
            raise KeyError("labels CSV missing planet_id.")
    meta[planet_id_col] = _coerce_str_id(meta[planet_id_col])
    df = df.merge(meta, on=planet_id_col, how="left", validate="many_to_one")

# Symbolic overlays JSON (per planet)
if overlay_cfg.symbolic_overlays_path and Path(overlay_cfg.symbolic_overlays_path).exists():
    obj = _read_json(Path(overlay_cfg.symbolic_overlays_path))
    if isinstance(obj, dict):
        recs = []
        for pid, payload in obj.items():
            if isinstance(payload, dict):
                recs.append({
                    planet_id_col: str(pid),
                    (overlay_cfg.map_score_to or "symbolic_score"): payload.get(overlay_cfg.symbolic_score_key),
                    (overlay_cfg.map_label_to or "symbolic_label"): payload.get(overlay_cfg.symbolic_label_key),
                })
        sym = pd.DataFrame.from_records(recs)
    elif isinstance(obj, list):
        sym = pd.DataFrame(obj)
        if planet_id_col not in sym.columns:
            alt = [c for c in sym.columns if c.lower() in ("planet_id", "planet", "id")]
            if alt:
                sym = sym.rename(columns={alt[0]: planet_id_col})
            else:
                raise KeyError("symbolic overlay list missing planet_id.")
        if overlay_cfg.symbolic_score_key in sym.columns and overlay_cfg.map_score_to:
            sym = sym.rename(columns={overlay_cfg.symbolic_score_key: overlay_cfg.map_score_to})
        if overlay_cfg.symbolic_label_key in sym.columns and overlay_cfg.map_label_to:
            sym = sym.rename(columns={overlay_cfg.symbolic_label_key: overlay_cfg.map_label_to})
    else:
        raise ValueError("Unsupported symbolic overlay JSON format.")

    sym[planet_id_col] = _coerce_str_id(sym[planet_id_col])
    df = df.merge(sym, on=planet_id_col, how="left", validate="many_to_one")

return df
```

# =============================================================================

# Figures

# =============================================================================

def make\_heatmap\_figure(planet\_ids: List\[str], bins: np.ndarray, Z: np.ndarray, title: str, zlabel: str) -> go.Figure:
"""
Build an image heatmap (planets × bins) with interactive hover.
Z is shape (N\_planets, N\_bins).
"""
fig = px.imshow(
Z,
labels=dict(x="bin", y="planet", color=zlabel),
x=bins,
y=planet\_ids,
aspect="auto",
origin="upper",
title=title,
)
fig.update\_layout(
template="plotly\_white",
margin=dict(l=40, r=40, t=80, b=40),
coloraxis\_colorbar=dict(title=zlabel),
)
return fig

def make\_per\_planet\_panel(planet\_id: str, bins: np.ndarray, mu: np.ndarray, mu\_hat: np.ndarray,
grad\_mag: np.ndarray, curv\_mag: np.ndarray, resid\_mag: np.ndarray) -> go.Figure:
"""Per-planet panel showing μ, μ̂, and magnitudes."""
fig = go.Figure()
fig.add\_trace(go.Scatter(x=bins, y=mu, mode="lines", name="μ"))
fig.add\_trace(go.Scatter(x=bins, y=mu\_hat, mode="lines", name="μ̂ (ref)"))
fig.add\_trace(go.Scatter(x=bins, y=grad\_mag, mode="lines", name="|∇μ|"))
fig.add\_trace(go.Scatter(x=bins, y=curv\_mag, mode="lines", name="|∇²μ|"))
fig.add\_trace(go.Scatter(x=bins, y=resid\_mag, mode="lines", name="|μ−μ̂|"))
fig.update\_layout(
title=f"{planet\_id} — μ vs ref and smoothness magnitudes",
xaxis\_title="bin",
yaxis\_title="value",
legend=dict(orientation="h"),
template="plotly\_white",
margin=dict(l=60, r=60, t=60, b=40),
)
return fig

# =============================================================================

# Orchestration

# =============================================================================

def run\_smoothness\_pipeline(
mu\_path: Path,
out\_cfg: OutputConfig,
ref\_cfg: SmoothRefConfig,
flag\_cfg: FlagConfig,
overlay\_cfg: Optional\[OverlayConfig] = None,
planet\_id\_col: str = PLANET\_ID\_COL,
log\_ctx: Optional\[PipelineLogContext] = None,
) -> Dict\[str, Any]:
"""Load μ → overlays → per-planet fields → heatmaps + summaries."""
setup\_logging()
t0 = time.time()

```
# Load μ (tall)
tall = load_mu_matrix(mu_path, planet_id_col=planet_id_col)
if overlay_cfg:
    tall = merge_overlays(tall, overlay_cfg, planet_id_col=planet_id_col)

# Gather shapes
bins_arr = np.sort(tall["bin"].unique())
planets = [str(pid) for pid in sorted(tall[planet_id_col].unique())]
n_planets, n_bins = len(planets), len(bins_arr)

# Storage for heatmaps
H_grad_mag = np.zeros((n_planets, n_bins), dtype=float)
H_curv_mag = np.zeros((n_planets, n_bins), dtype=float)
H_resid_mag = np.zeros((n_planets, n_bins), dtype=float)

# Per-planet summaries
rows: List[Dict[str, Any]] = []
per_planet_panels = []

for i, pid in enumerate(planets):
    grp = tall[tall[planet_id_col] == pid].sort_values("bin")
    mu_vec = grp["mu"].to_numpy(dtype=float)
    fields = compute_smoothness_fields(mu_vec, ref_cfg)
    summ = summarize_planet(fields)
    flags = compute_flags(summ, flag_cfg)

    # Fill heatmaps
    H_grad_mag[i, :] = fields["grad_mag"]
    H_curv_mag[i, :] = fields["curv_mag"]
    H_resid_mag[i, :] = fields["resid_mag"]

    # Row with overlays
    row: Dict[str, Any] = {
        PLANET_ID_COL: pid,
        "tv": summ["tv"],
        "grad_mag_mean": summ["grad_mag_mean"],
        "grad_mag_p95": summ["grad_mag_p95"],
        "curv_mag_mean": summ["curv_mag_mean"],
        "curv_mag_p95": summ["curv_mag_p95"],
        "resid_mag_mean": summ["resid_mag_mean"],
        "resid_mag_p95": summ["resid_mag_p95"],
        **flags,
    }
    # include common overlay cols if present
    for c in (LABEL_COL, CONFIDENCE_COL, ENTROPY_COL, SHAP_MAG_COL, GLL_COL, "symbolic_score", "symbolic_label"):
        if c in grp.columns:
            row[c] = grp[c].dropna().iloc[0] if grp[c].notna().any() else None
    rows.append(row)

    # Optional per-planet panel export
    if out_cfg.per_planet_panels and len(per_planet_panels) < out_cfg.per_planet_panels:
        fig = make_per_planet_panel(pid, grp["bin"].to_numpy(), mu_vec, fields["mu_hat"], fields["grad_mag"], fields["curv_mag"], fields["resid_mag"])
        panel_path = out_cfg.out_dir / f"panel_{pid}.html"
        _ensure_parent(panel_path)
        pio.write_html(fig, file=str(panel_path), auto_open=False, include_plotlyjs="cdn")
        per_planet_panels.append(str(panel_path))

# Build summary dataframe
summary_df = pd.DataFrame.from_records(rows).sort_values(PLANET_ID_COL).reset_index(drop=True)

# Heatmap figure (choose one primary Z for the main heatmap; grad_mag by default)
fig_heat = make_heatmap_figure(planets, bins_arr, H_grad_mag, out_cfg.title + " — |∇μ|", "|∇μ|")
_ensure_parent(out_cfg.heatmap_html)
pio.write_html(fig_heat, file=str(out_cfg.heatmap_html), auto_open=False, include_plotlyjs="cdn")

png_path = None
if out_cfg.heatmap_png:
    _ensure_parent(out_cfg.heatmap_png)
    if _KALEIDO_AVAILABLE:
        fig_heat.write_image(str(out_cfg.heatmap_png), format="png", scale=2, width=1280, height=800)
        png_path = str(out_cfg.heatmap_png)
    else:
        logging.warning("kaleido not installed; PNG export skipped.")

# Save CSV/JSON
_ensure_parent(out_cfg.summary_csv)
summary_df.to_csv(out_cfg.summary_csv, index=False)

global_stats = {
    "n_planets": int(n_planets),
    "n_bins": int(n_bins),
    "grad_mag_mean_global": float(np.nanmean(H_grad_mag)),
    "curv_mag_mean_global": float(np.nanmean(H_curv_mag)),
    "resid_mag_mean_global": float(np.nanmean(H_resid_mag)),
}
_ensure_parent(out_cfg.summary_json)
Path(out_cfg.summary_json).write_text(json.dumps({
    "globals": global_stats,
    "ref_config": asdict(ref_cfg),
    "flag_config": asdict(flag_cfg),
}, indent=2))

# Append audit log
if log_ctx:
    append_v50_log(
        log_ctx,
        metadata={
            "mu": str(mu_path),
            "labels": str(overlay_cfg.labels_csv) if overlay_cfg and overlay_cfg.labels_csv else "",
            "symbolic": str(overlay_cfg.symbolic_overlays_path) if overlay_cfg and overlay_cfg.symbolic_overlays_path else "",
            "heatmap": str(out_cfg.heatmap_html),
            "png": str(out_cfg.heatmap_png) if out_cfg.heatmap_png else "",
            "duration_sec": f"{time.time() - t0:.2f}",
            "n_planets": str(n_planets),
        },
    )

result = {
    "heatmap_html": str(out_cfg.heatmap_html),
    "heatmap_png": png_path,
    "summary_csv": str(out_cfg.summary_csv),
    "summary_json": str(out_cfg.summary_json),
    "per_planet_panels": per_planet_panels,
    "kaleido_available": _KALEIDO_AVAILABLE,
    "scipy_available": _SCIPY_AVAILABLE,
    "duration_sec": time.time() - t0,
}

# Optional: open browser
if out_cfg.open_browser:
    try:
        import webbrowser
        webbrowser.open_new_tab(out_cfg.heatmap_html.as_uri())
    except Exception as e:
        logging.warning(f"Failed to open browser: {e}")

return result
```

# =============================================================================

# CLI

# =============================================================================

def \_build\_typer\_app() -> "typer.Typer":
if typer is None:
raise RuntimeError(
"Typer is not installed. Install with `pip install typer[all]` "
"or import run\_smoothness\_pipeline() programmatically."
)

```
app = typer.Typer(
    add_completion=True,
    help="SpectraMind V50 — μ Spectral Smoothness Map",
    no_args_is_help=True,
)

@app.command("run")
def cli_run(
    mu: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to μ spectra (.npy/.npz/.csv/.txt)."),
    labels: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional labels/metrics CSV."),
    symbolic_overlays: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional symbolic overlays JSON (per planet)."),
    symbolic_score_key: str = typer.Option("violation_score", help="Key in symbolic JSON to map as score."),
    symbolic_label_key: str = typer.Option("top_rule", help="Key in symbolic JSON to map as label."),
    map_score_to: str = typer.Option("symbolic_score", help="Column name to store symbolic score."),
    map_label_to: str = typer.Option("symbolic_label", help="Column name to store symbolic label."),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, help="Directory for outputs."),
    heatmap: Path = typer.Option(DEFAULT_HEATMAP_HTML, help="Heatmap HTML path."),
    png: Optional[Path] = typer.Option(DEFAULT_HEATMAP_PNG, help="Optional heatmap PNG path (requires kaleido)."),
    summary_csv: Path = typer.Option(DEFAULT_SUMMARY_CSV, help="Per-planet summary CSV path."),
    summary_json: Path = typer.Option(DEFAULT_SUMMARY_JSON, help="Global summary JSON path."),
    per_planet_panels: int = typer.Option(0, help="Export up to N per-planet panel HTMLs."),
    open_browser: bool = typer.Option(False, help="Open the heatmap in a browser after generation."),
    title: str = typer.Option("SpectraMind V50 — μ Spectral Smoothness Map", help="Heatmap title."),
    # Smoothing ref config
    smooth_ref: str = typer.Option("savgol", help="Reference type: 'none' | 'savgol' | 'moving_avg'."),
    smooth_window: int = typer.Option(11, help="Savgol window or moving-average window (odd preferred)."),
    smooth_poly: int = typer.Option(3, help="Savgol polynomial order."),
    # Flag thresholds
    grad_mag_p95: float = typer.Option(1.75, help="95th percentile threshold for |∇μ|."),
    curv_mag_p95: float = typer.Option(1.25, help="95th percentile threshold for |∇²μ|."),
    resid_mag_p95: float = typer.Option(1.50, help="95th percentile threshold for |μ−μ̂|."),
    grad_mag_abs: float = typer.Option(0.0, help="Absolute mean threshold for |∇μ| (0 to disable)."),
    curv_mag_abs: float = typer.Option(0.0, help="Absolute mean threshold for |∇²μ| (0 to disable)."),
    resid_mag_abs: float = typer.Option(0.0, help="Absolute mean threshold for |μ−μ̂| (0 to disable)."),
    # Misc
    planet_id_col: str = typer.Option(PLANET_ID_COL, help="Planet ID column name."),
    log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="v50_debug_log.md path."),
    config_hash_path: Optional[Path] = typer.Option(Path("run_hash_summary_v50.json"), help="Optional run hash JSON path."),
):
    """Run the smoothness diagnostics pipeline."""
    ref_cfg = SmoothRefConfig(
        ref=smooth_ref,
        window=smooth_window,
        poly=smooth_poly,
        ma_window=smooth_window,
    )
    flag_cfg = FlagConfig(
        grad_mag_p95=grad_mag_p95,
        curv_mag_p95=curv_mag_p95,
        resid_mag_p95=resid_mag_p95,
        grad_mag_abs=grad_mag_abs,
        curv_mag_abs=curv_mag_abs,
        resid_mag_abs=resid_mag_abs,
    )
    overlay = OverlayConfig(
        labels_csv=labels,
        symbolic_overlays_path=symbolic_overlays,
        symbolic_score_key=symbolic_score_key,
        symbolic_label_key=symbolic_label_key,
        map_score_to=map_score_to,
        map_label_to=map_label_to,
    )
    out_cfg = OutputConfig(
        out_dir=out_dir,
        heatmap_html=heatmap,
        heatmap_png=png,
        summary_csv=summary_csv,
        summary_json=summary_json,
        per_planet_panels=per_planet_panels,
        open_browser=open_browser,
        title=title,
    )
    log_ctx = PipelineLogContext(
        log_path=log_path,
        cli_name="spectramind diagnose smoothness",
        config_hash_path=config_hash_path,
    )

    result = run_smoothness_pipeline(
        mu_path=mu,
        out_cfg=out_cfg,
        ref_cfg=ref_cfg,
        flag_cfg=flag_cfg,
        overlay_cfg=overlay,
        planet_id_col=planet_id_col,
        log_ctx=log_ctx,
    )
    typer.echo(json.dumps(result, indent=2))

@app.command("selftest")
def cli_selftest(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR / "_selftest", help="Directory for selftest artifacts."),
    n_planets: int = typer.Option(24, help="Number of synthetic planets."),
    n_bins: int = typer.Option(283, help="Bins per μ."),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed."),
    per_planet_panels: int = typer.Option(3, help="Export first N per-planet panels."),
):
    """Generate synthetic μ and run the smoothness analyzer end-to-end."""
    rng = np.random.default_rng(seed)
    mu_mat = []
    for i in range(n_planets):
        x = np.linspace(0, 8 * np.pi, n_bins)
        base = 0.6 * np.sin(x * (1.0 + 0.05 * rng.standard_normal())) + 0.25 * np.cos(1.5 * x + 0.3)
        trend = 0.05 * (x / x.max())
        noise = 0.04 * rng.standard_normal(n_bins)
        # add occasional sharp blips to test flags
        if i % 5 == 0:
            blip_idx = rng.integers(low=n_bins//4, high=3*n_bins//4)
            noise[blip_idx:blip_idx+2] += 0.5
        mu_vec = base + trend + noise + 1.0
        mu_mat.append(mu_vec)
    mu_mat = np.stack(mu_mat, axis=0)
    dfw = pd.DataFrame(mu_mat, columns=[f"f{i}" for i in range(n_bins)])
    dfw[PLANET_ID_COL] = [f"P{i:04d}" for i in range(n_planets)]
    mu_csv = out_dir / "mu.csv"
    _ensure_parent(mu_csv)
    dfw.to_csv(mu_csv, index=False)

    result = run_smoothness_pipeline(
        mu_path=mu_csv,
        out_cfg=OutputConfig(
            out_dir=out_dir,
            heatmap_html=out_dir / "heatmap.html",
            heatmap_png=out_dir / "heatmap.png",
            summary_csv=out_dir / "summary.csv",
            summary_json=out_dir / "summary.json",
            per_planet_panels=per_planet_panels,
            open_browser=False,
            title="Selftest — Smoothness Map",
        ),
        ref_cfg=SmoothRefConfig(ref="savgol", window=11, poly=3),
        flag_cfg=FlagConfig(grad_mag_p95=1.5, curv_mag_p95=1.0, resid_mag_p95=1.2),
        overlay_cfg=OverlayConfig(labels_csv=None),
    )
    typer.echo(json.dumps(result, indent=2))

return app
```

# =============================================================================

# Entrypoint

# =============================================================================

if **name** == "**main**":
if typer is None:
print(
"Typer is not installed. Install with `pip install typer[all]` or "
"import run\_smoothness\_pipeline() from this module.",
file=sys.stderr,
)
sys.exit(2)
\_app = \_build\_typer\_app()
\_app()
