# src/diagnostics/analyze/fft/autocorr/mu.py

# =============================================================================

# 🔬 SpectraMind V50 — μ Spectrum FFT & Autocorrelation Analyzer

# -----------------------------------------------------------------------------

# Purpose

# Perform frequency-domain (FFT) and time-domain (autocorrelation) diagnostics

# on per-planet μ spectra (283 bins for Ariel Challenge), with optional

# overlays from symbolic-rule outputs and SHAP/entropy/GLL metrics.

#

# Capabilities

# • Inputs: .npy/.npz/.csv matrices or tall CSVs (planet\_id, λ\_bin\_*, μ\_* …)

# • FFT: magnitude & power, dominant frequency peaks, spectral centroid/bandwidth

# • Autocorrelation: normalized ACF, dominant lag(s), decay time, periodicity hints

# • Molecular templates (optional): H₂O / CO₂ / CH₄ wavelength masks or fingerprints

# • Overlays: symbolic score/label, entropy, SHAP magnitude, GLL

# • Outputs:

# - JSON summary per planet (fft peaks, acf peaks, centroids)

# - CSV table of key diagnostics

# - Plotly HTML dashboard (interactive) + optional PNG

# - (Optional) per-planet mini-figures

# • Reproducibility:

# - Appends audit row to v50\_debug\_log.md

# - Deterministic computations given same inputs

#

# CLI Examples

# spectramind diagnose fft-autocorr-mu run \\

# --mu artifacts/mu\_preds.npy \\

# --planet-ids artifacts/planet\_ids.csv \\

# --labels artifacts/meta.csv \\

# --symbolic-overlays artifacts/symbolic\_violation\_summary.json \\

# --out-dir artifacts/fft\_autocorr \\

# --html artifacts/fft\_autocorr/overview\.html \\

# --png artifacts/fft\_autocorr/overview\.png \\

# --open-browser

#

# Notes

# • This module computes *diagnostics only* on μ already produced by the

# modeling pipeline. It does not modify μ or recompute scientific analytics.

# • Designed to integrate with generate\_html\_report.py and CLI dashboard.

# =============================================================================

from **future** import annotations

import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Plotly for interactive HTML dashboards

import plotly.express as px
import plotly.graph\_objects as go
import plotly.io as pio

# Optional PNG export backend

try:
import kaleido  # noqa: F401
\_KALEIDO\_AVAILABLE = True
except Exception:
\_KALEIDO\_AVAILABLE = False

# Typer CLI (optional at import; keep library-usable if missing)

try:
import typer
except Exception:
typer = None

# =============================================================================

# Constants & Defaults

# =============================================================================

DEFAULT\_LOG\_PATH = Path("v50\_debug\_log.md")
DEFAULT\_OUT\_DIR = Path("artifacts") / "fft\_autocorr"
DEFAULT\_HTML\_OUT = DEFAULT\_OUT\_DIR / "overview\.html"
DEFAULT\_PNG\_OUT = None  # can be set via CLI
DEFAULT\_JSON\_SUMMARY = DEFAULT\_OUT\_DIR / "summary.json"
DEFAULT\_CSV\_TABLE = DEFAULT\_OUT\_DIR / "summary.csv"

PLANET\_ID\_COL = "planet\_id"
LABEL\_COL = "label"
CONFIDENCE\_COL = "confidence"
ENTROPY\_COL = "entropy"
SHAP\_MAG\_COL = "shap"
GLL\_COL = "gll"

# Expected Ariel bins \~283; keep flexible

DEFAULT\_BINS = 283
DEFAULT\_SEED = 1337

# =============================================================================

# Dataclasses (Config & Results)

# =============================================================================

@dataclass
class OverlayConfig:
"""Paths and keys for optional overlay metadata."""
labels\_csv: Optional\[Path] = None
symbolic\_overlays\_path: Optional\[Path] = None
symbolic\_score\_key: str = "violation\_score"
symbolic\_label\_key: str = "top\_rule"
map\_score\_to: Optional\[str] = "symbolic\_score"
map\_label\_to: Optional\[str] = "symbolic\_label"

@dataclass
class TemplateConfig:
"""Optional molecular template config."""
\# Either provide per-bin boolean masks or sparse line lists; this module
\# consumes a CSV with columns: bin (int) and each molecule column in {0,1}.
\# Example columns: bin, H2O, CO2, CH4
bin\_template\_csv: Optional\[Path] = None
columns: Tuple\[str, ...] = ("H2O", "CO2", "CH4")
\# If provided, we'll compute per-molecule average power within mask ranges
\# and include them in output diagnostics.

@dataclass
class OutputConfig:
"""Artifact outputs."""
out\_dir: Path = DEFAULT\_OUT\_DIR
html\_out: Path = DEFAULT\_HTML\_OUT
png\_out: Optional\[Path] = DEFAULT\_PNG\_OUT
json\_summary: Path = DEFAULT\_JSON\_SUMMARY
csv\_table: Path = DEFAULT\_CSV\_TABLE
open\_browser: bool = False
title: str = "SpectraMind V50 — μ FFT & Autocorr Diagnostics"

@dataclass
class PipelineLogContext:
"""Append audit metadata to a Markdown log."""
log\_path: Path = DEFAULT\_LOG\_PATH
cli\_name: str = "spectramind diagnose fft-autocorr-mu"
config\_hash\_path: Optional\[Path] = Path("run\_hash\_summary\_v50.json")

@dataclass
class FFTParams:
"""Parameters for FFT/ACF computation."""
detrend: bool = True          # subtract mean before FFT/ACF
window: Optional\[str] = "hann"  # None|"hann"|"hamming"|"blackman"
zero\_pad: int = 0             # extra zeros appended for finer freq grid
acf\_max\_lag: Optional\[int] = None  # default: N-1
peak\_k: int = 5               # number of peaks to record
seed: int = DEFAULT\_SEED      # for any randomized steps (none by default)

@dataclass
class SeriesDiagnostics:
"""FFT/ACF results for a single planet."""
planet\_id: str
n\_bins: int
fft\_freq: List\[float]
fft\_power: List\[float]
fft\_peaks\_idx: List\[int]
fft\_peaks\_val: List\[float]
spectral\_centroid: float
spectral\_bandwidth: float
acf: List\[float]
acf\_peaks\_idx: List\[int]
acf\_peaks\_val: List\[float]
acf\_decay\_index: Optional\[int]  # first lag where acf < 1/e
molecule\_band\_power: Dict\[str, float] = field(default\_factory=dict)  # e.g., {"H2O": 0.12, ...}

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
"""Append a Markdown row to v50\_debug\_log.md for auditability."""
try:
\_ensure\_parent(ctx.log\_path)
config\_hash = ""
if ctx.config\_hash\_path and Path(ctx.config\_hash\_path).exists():
try:
obj = json.loads(Path(ctx.config\_hash\_path).read\_text())
config\_hash = obj.get("config\_hash") or obj.get("hash") or obj.get("run\_hash") or ""
except Exception:
config\_hash = ""
row = {
"timestamp": \_now\_iso(),
"cli": ctx.cli\_name,
"config\_hash": config\_hash,
\*\*metadata,
}
line = (
f"| {row\['timestamp']} | {row\['cli']} | {row\['config\_hash']} "
f"| {row\.get('mu','')} | {row\.get('planet\_ids','')} | {row\.get('labels','')} "
f"| {row\.get('symbolic','')} | {row\.get('html','')} | {row\.get('png','')} |\n"
)
if not ctx.log\_path.exists() or ctx.log\_path.stat().st\_size == 0:
header = (
"# SpectraMind V50 — Debug Log (μ FFT & Autocorr)\n\n"
"| timestamp | cli | config\_hash | mu | planet\_ids | labels | symbolic | html | png |\n"
"|---|---|---|---|---|---|---|---|---|\n"
)
ctx.log\_path.write\_text(header + line)
else:
with open(ctx.log\_path, "a", encoding="utf-8") as f:
f.write(line)
except Exception as e:
logging.warning(f"Failed to append to log {ctx.log\_path}: {e}")

# =============================================================================

# Loading μ and Metadata

# =============================================================================

def load\_mu\_matrix(mu\_path: Path, planet\_ids: Optional\[Path] = None, planet\_id\_col: str = PLANET\_ID\_COL) -> pd.DataFrame:
"""
Load μ spectra into a tall DataFrame with columns:
planet\_id, bin, mu

```
Accepts:
  - .npy: shape (N_planets, N_bins)
  - .npz: arrays 'mu' (N×B), optional 'planet_id' (N,), else numeric indices
  - .csv: either wide (planet_id + columns f0..fB-1) or tall (planet_id, bin, mu)
  - .txt: treated like CSV
"""
mu_path = Path(mu_path)
if not mu_path.exists():
    raise FileNotFoundError(f"μ file not found: {mu_path}")
ext = mu_path.suffix.lower()

def _attach_ids(n: int, df: pd.DataFrame) -> pd.DataFrame:
    if planet_ids and Path(planet_ids).exists():
        ids_df = _read_csv(Path(planet_ids))
        if planet_id_col not in ids_df.columns:
            alt = [c for c in ids_df.columns if c.lower() in ("planet_id", "planet", "id")]
            if alt:
                ids_df = ids_df.rename(columns={alt[0]: planet_id_col})
            else:
                raise KeyError("planet_ids CSV missing planet_id column.")
        ids_df[planet_id_col] = _coerce_str_id(ids_df[planet_id_col])
        if len(ids_df) != n:
            logging.warning(f"planet_ids count {len(ids_df)} != μ rows {n}; will truncate/pad by index.")
        ids = list(ids_df[planet_id_col].values[:n]) + [f"PID_{i}" for i in range(n - len(ids_df))]
    else:
        ids = [f"P{i:04d}" for i in range(n)]
    df[planet_id_col] = ids
    return df

if ext == ".npy":
    mat = np.load(mu_path)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D μ in {mu_path}, got shape={mat.shape}")
    n, b = mat.shape
    dfw = pd.DataFrame(mat, columns=[f"f{i}" for i in range(b)])
    dfw = _attach_ids(n, dfw)
    tall = dfw.melt(id_vars=[planet_id_col], var_name="bin_col", value_name="mu")
    tall["bin"] = tall["bin_col"].str.replace("f", "", regex=False).astype(int)
    tall = tall.drop(columns=["bin_col"])
    tall[planet_id_col] = _coerce_str_id(tall[planet_id_col])
    return tall[[planet_id_col, "bin", "mu"]].sort_values([planet_id_col, "bin"]).reset_index(drop=True)

if ext == ".npz":
    npz = np.load(mu_path)
    if "mu" not in npz:
        raise KeyError("NPZ is missing 'mu' array.")
    mat = npz["mu"]
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D μ in {mu_path}, got shape={mat.shape}")
    n, b = mat.shape
    if "planet_id" in npz:
        ids = [str(x) for x in npz["planet_id"]][:n]
    else:
        ids = [f"P{i:04d}" for i in range(n)]
    dfw = pd.DataFrame(mat, columns=[f"f{i}" for i in range(b)])
    dfw[planet_id_col] = ids
    dfw[planet_id_col] = _coerce_str_id(dfw[planet_id_col])
    tall = dfw.melt(id_vars=[planet_id_col], var_name="bin_col", value_name="mu")
    tall["bin"] = tall["bin_col"].str.replace("f", "", regex=False).astype(int)
    tall = tall.drop(columns=["bin_col"])
    return tall[[planet_id_col, "bin", "mu"]].sort_values([planet_id_col, "bin"]).reset_index(drop=True)

if ext in (".csv", ".txt"):
    df = _read_csv(mu_path)
    cols = [c.lower() for c in df.columns]
    # Tall format?
    if {"planet_id", "bin", "mu"}.issubset(set(cols)):
        # Normalize column names
        ren = {}
        for c in df.columns:
            cl = c.lower()
            if cl == "planet_id": ren[c] = planet_id_col
            elif cl == "bin": ren[c] = "bin"
            elif cl == "mu": ren[c] = "mu"
        df = df.rename(columns=ren)
        df[planet_id_col] = _coerce_str_id(df[planet_id_col])
        df["bin"] = df["bin"].astype(int)
        return df[[planet_id_col, "bin", "mu"]].sort_values([planet_id_col, "bin"]).reset_index(drop=True)
    # Wide format: must contain planet_id and numeric feature columns
    if planet_id_col not in df.columns:
        alt = [c for c in df.columns if c.lower() in ("planet_id", "planet", "id")]
        if alt:
            df = df.rename(columns={alt[0]: planet_id_col})
        else:
            # synthesize id
            df[planet_id_col] = [f"P{i:04d}" for i in range(len(df))]
    df[planet_id_col] = _coerce_str_id(df[planet_id_col])
    # Use numeric columns as μ bins
    feat_cols = [c for c in df.columns if c != planet_id_col and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        raise ValueError("μ CSV has no numeric bin columns.")
    # Rename fNN if necessary → extract bin index
    tall = df.melt(id_vars=[planet_id_col], var_name="bin_col", value_name="mu")
    # If bin_col looks like fN or bN or raw integers
    def _to_idx(s: str) -> int:
        s = str(s)
        if s.startswith("f") or s.startswith("b"):
            return int(s[1:])
        try:
            return int(s)
        except Exception:
            # unknown: try to parse last digits
            digits = "".join([ch for ch in s if ch.isdigit()])
            return int(digits) if digits else 0
    tall["bin"] = tall["bin_col"].map(_to_idx).astype(int)
    tall = tall.drop(columns=["bin_col"])
    return tall[[planet_id_col, "bin", "mu"]].sort_values([planet_id_col, "bin"]).reset_index(drop=True)

raise ValueError(f"Unsupported μ file type: {ext}")
```

def merge\_overlays(
tall\_mu: pd.DataFrame,
overlay\_cfg: OverlayConfig,
planet\_id\_col: str = PLANET\_ID\_COL,
) -> pd.DataFrame:
"""Merge labels CSV and symbolic overlays JSON onto (planet\_id,bin,mu) table (join on planet\_id)."""
df = tall\_mu.copy()

```
# Labels CSV (metrics per planet)
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

def load\_templates(template\_cfg: TemplateConfig, n\_bins: Optional\[int] = None) -> Optional\[pd.DataFrame]:
"""
Load per-bin molecular masks from CSV with columns:
bin, H2O, CO2, CH4 (subset allowed)
Returns a DataFrame indexed by bin with boolean/int mask columns.
"""
if not template\_cfg.bin\_template\_csv:
return None
path = Path(template\_cfg.bin\_template\_csv)
if not path.exists():
logging.warning(f"Template CSV not found: {path}; skipping molecular overlays.")
return None
df = \_read\_csv(path)
if "bin" not in df.columns:
raise KeyError("Template CSV must include a 'bin' column.")
\# Keep only known columns if present
keep = \["bin"] + \[c for c in template\_cfg.columns if c in df.columns]
df = df\[keep].copy()
df\["bin"] = df\["bin"].astype(int)
if n\_bins is not None:
df = df\[(df\["bin"] >= 0) & (df\["bin"] < n\_bins)]
df = df.set\_index("bin").sort\_index()
return df

# =============================================================================

# Core Diagnostics (FFT & ACF)

# =============================================================================

def \_apply\_window(x: np.ndarray, name: Optional\[str]) -> np.ndarray:
"""Apply a window to reduce spectral leakage (if requested)."""
if name is None:
return x
n = x.shape\[0]
if name == "hann":
w = np.hanning(n)
elif name == "hamming":
w = np.hamming(n)
elif name == "blackman":
w = np.blackman(n)
else:
logging.warning(f"Unknown window '{name}', ignoring.")
return x
return x \* w

def fft\_power\_spectrum(mu\_1d: np.ndarray, params: FFTParams) -> Tuple\[np.ndarray, np.ndarray]:
"""
Compute one-sided FFT frequency grid and power spectrum for a single μ vector.
Returns (freq, power). Frequency is normalized to \[0, 0.5] (Nyquist) if bins are unit-spaced.
"""
x = mu\_1d.astype(np.float64)
if params.detrend:
x = x - np.nanmean(x)
x = \_apply\_window(x, params.window)
if params.zero\_pad and params.zero\_pad > 0:
x = np.pad(x, (0, int(params.zero\_pad)), mode="constant", constant\_values=0.0)

```
n = x.shape[0]
# rfft: frequencies from 0..Nyquist
spec = np.fft.rfft(x, n=n)
power = (spec.real**2 + spec.imag**2) / n
freq = np.fft.rfftfreq(n, d=1.0)  # assume unit bin spacing
return freq, power
```

def spectral\_moments(freq: np.ndarray, power: np.ndarray) -> Tuple\[float, float]:
"""Return (spectral centroid, spectral bandwidth) on normalized freq grid."""
p = np.clip(power, 0.0, np.inf)
p\_sum = p.sum()
if p\_sum <= 0:
return 0.0, 0.0
centroid = float((freq \* p).sum() / p\_sum)
\# bandwidth (RMS around centroid)
var = float((p \* (freq - centroid) \*\* 2).sum() / p\_sum)
bandwidth = math.sqrt(max(0.0, var))
return centroid, bandwidth

def autocorr(x: np.ndarray, params: FFTParams) -> np.ndarray:
"""Normalized (biased) autocorrelation using FFT convolution trick."""
y = x.astype(np.float64)
if params.detrend:
y = y - np.nanmean(y)
n = y.shape\[0]
\# Next power of 2 for speed
nfft = 1 << (n - 1).bit\_length()
fy = np.fft.rfft(y, n=2 \* nfft)
ac = np.fft.irfft(fy \* np.conj(fy))\[:n]
\# Normalize by ac\[0]
if ac\[0] != 0:
ac = ac / ac\[0]
\# Optionally truncate to max lag
if params.acf\_max\_lag is not None:
ac = ac\[: max(1, params.acf\_max\_lag + 1)]
return ac

def \_top\_k\_peaks(y: np.ndarray, k: int, exclude\_zero\_idx: bool = True) -> Tuple\[List\[int], List\[float]]:
"""
Return indices and values for top-k peaks (by y value). Very simple heuristic:
sort by y descending; optionally ignore index 0 (DC).
"""
idxs = np.arange(len(y))
if exclude\_zero\_idx and len(y) > 0:
idxs = idxs\[1:]
vals = y\[1:]
else:
vals = y
order = np.argsort(vals)\[::-1]
top\_i = idxs\[order]\[:k].tolist()
top\_v = vals\[order]\[:k].tolist()
return top\_i, \[float(v) for v in top\_v]

def \_first\_below\_threshold(y: np.ndarray, thr: float) -> Optional\[int]:
"""Return first index where y < thr; None if never below."""
for i, v in enumerate(y):
if v < thr:
return i
return None

def analyze\_series(mu\_1d: np.ndarray, params: FFTParams, templates: Optional\[pd.DataFrame] = None) -> Tuple\[SeriesDiagnostics, np.ndarray, np.ndarray]:
"""
Analyze a single μ spectrum vector.
Returns:
SeriesDiagnostics, freq array, ACF array
"""
n = len(mu\_1d)
freq, power = fft\_power\_spectrum(mu\_1d, params)
centroid, bandwidth = spectral\_moments(freq, power)
k\_idx, k\_val = \_top\_k\_peaks(power, params.peak\_k, exclude\_zero\_idx=True)

```
acf = autocorr(mu_1d, params)
ac_idx, ac_val = _top_k_peaks(acf, params.peak_k, exclude_zero_idx=True)
decay = _first_below_threshold(acf, 1.0 / math.e)

molecule_power: Dict[str, float] = {}
if templates is not None and len(templates) > 0:
    # Compute mean power within bins flagged by each molecule mask (using original bin grid)
    # Map FFT power to bins by inverse FFT? Instead, approximate by mean μ within masks and correlate:
    # For a simple diagnostic, compute mean |Δμ| (first diff magnitude) inside mask.
    mu = np.asarray(mu_1d, dtype=float)
    dmu = np.abs(np.diff(mu, prepend=mu[:1]))
    for col in templates.columns:
        if col == "bin":
            continue
        mask = templates[col].astype(int).reindex(range(n), fill_value=0).to_numpy(dtype=int)
        if mask.sum() > 0:
            molecule_power[col] = float(np.mean(dmu[mask == 1]))
        else:
            molecule_power[col] = 0.0

diag = SeriesDiagnostics(
    planet_id="",
    n_bins=n,
    fft_freq=freq.tolist(),
    fft_power=power.tolist(),
    fft_peaks_idx=k_idx,
    fft_peaks_val=k_val,
    spectral_centroid=float(centroid),
    spectral_bandwidth=float(bandwidth),
    acf=acf.tolist(),
    acf_peaks_idx=ac_idx,
    acf_peaks_val=ac_val,
    acf_decay_index=decay,
    molecule_band_power=molecule_power,
)
return diag, freq, acf
```

# =============================================================================

# Plotly Figures

# =============================================================================

def make\_overview\_figure(summary\_df: pd.DataFrame, out\_title: str) -> go.Figure:
"""
Build a multi-panel style figure:
\- Scatter: spectral centroid vs bandwidth (size=top FFT peak, color=symbolic\_score or label)
\- Bar: average ACF decay (aggregated)
"""
\# Choose color dimension
color\_dim = None
for candidate in ("symbolic\_score", "symbolic\_label", LABEL\_COL):
if candidate in summary\_df.columns:
color\_dim = candidate
break

```
size_dim = "fft_peak0_val" if "fft_peak0_val" in summary_df.columns else None

fig = px.scatter(
    summary_df,
    x="spectral_centroid",
    y="spectral_bandwidth",
    size=size_dim,
    color=color_dim,
    hover_data=[PLANET_ID_COL, "acf_decay_index", "fft_peak0_idx", "fft_peak0_val"],
    title=out_title,
)
fig.update_layout(
    template="plotly_white",
    margin=dict(l=40, r=40, t=80, b=40),
)
return fig
```

def make\_series\_panel(planet\_id: str, bins: np.ndarray, mu: np.ndarray, freq: np.ndarray, power: np.ndarray, acf: np.ndarray) -> go.Figure:
"""Per-planet panel with μ(bins), FFT power, and ACF curves."""
\# Subplots without importing make\_subplots to keep deps minimal: stack traces
fig = go.Figure()
\# μ
fig.add\_trace(go.Scatter(x=bins, y=mu, mode="lines", name="μ"))
\# Power (frequency axis)
fig.add\_trace(go.Scatter(x=freq, y=power, mode="lines", name="FFT power", yaxis="y2"))
\# ACF
lags = np.arange(len(acf))
fig.add\_trace(go.Scatter(x=lags, y=acf, mode="lines", name="ACF", yaxis="y3"))

```
fig.update_layout(
    title=f"{planet_id} — μ, FFT power, ACF",
    xaxis=dict(title="bin"),
    yaxis=dict(title="μ"),
    yaxis2=dict(title="FFT power", overlaying="y", side="right"),
    yaxis3=dict(title="ACF", anchor="free", overlaying="y", side="left", position=0.0),
    template="plotly_white",
    legend=dict(orientation="h"),
    margin=dict(l=60, r=60, t=60, b=40),
)
return fig
```

# =============================================================================

# Orchestration

# =============================================================================

def run\_fft\_autocorr\_pipeline(
mu\_path: Path,
out\_cfg: OutputConfig,
params: FFTParams,
overlay\_cfg: Optional\[OverlayConfig] = None,
template\_cfg: Optional\[TemplateConfig] = None,
planet\_id\_col: str = PLANET\_ID\_COL,
per\_planet\_panels: int = 0,          # save first N per-planet panels
log\_ctx: Optional\[PipelineLogContext] = None,
) -> Dict\[str, Any]:
"""End-to-end pipeline: load μ → overlays → templates → per-planet analysis → exports."""
setup\_logging()

```
t0 = time.time()
_ensure_parent(out_cfg.out_dir / "dummy.txt")
tall = load_mu_matrix(mu_path, planet_ids=None, planet_id_col=planet_id_col)

# Merge overlays (per planet metadata)
if overlay_cfg:
    tall = merge_overlays(tall, overlay_cfg, planet_id_col=planet_id_col)

# Determine number of bins & templates
n_bins = int(tall["bin"].max() + 1) if len(tall) else DEFAULT_BINS
templates = load_templates(template_cfg or TemplateConfig(), n_bins=n_bins)

# Analyze per planet
summary_rows: List[Dict[str, Any]] = []
per_planet_json: Dict[str, Any] = {}
figures_saved = []

for pid, grp in tall.groupby(planet_id_col, sort=True):
    grp = grp.sort_values("bin")
    mu_vec = grp["mu"].to_numpy(dtype=float)
    diag, freq, acf = analyze_series(mu_vec, params, templates=templates)

    # Attach planet_id
    diag.planet_id = str(pid)
    per_planet_json[str(pid)] = asdict(diag)

    # Summary row
    row: Dict[str, Any] = {
        PLANET_ID_COL: str(pid),
        "n_bins": diag.n_bins,
        "spectral_centroid": diag.spectral_centroid,
        "spectral_bandwidth": diag.spectral_bandwidth,
        "acf_decay_index": diag.acf_decay_index if diag.acf_decay_index is not None else -1,
    }
    # Flatten top FFT peak (0th)
    if diag.fft_peaks_idx:
        row["fft_peak0_idx"] = diag.fft_peaks_idx[0]
        row["fft_peak0_val"] = diag.fft_peaks_val[0]
    else:
        row["fft_peak0_idx"] = -1
        row["fft_peak0_val"] = 0.0

    # Add molecule band power metrics
    for k, v in diag.molecule_band_power.items():
        row[f"mol_{k}_band_power"] = v

    # Carry overlays if present (planet-level)
    take_cols = [LABEL_COL, CONFIDENCE_COL, ENTROPY_COL, SHAP_MAG_COL, GLL_COL, "symbolic_score", "symbolic_label"]
    for c in take_cols:
        if c in grp.columns:
            # Use first non-null value for the planet
            val = grp[c].dropna().iloc[0] if grp[c].notna().any() else None
            row[c] = val

    summary_rows.append(row)

    # Optional per-planet panel export
    if per_planet_panels > 0 and len(figures_saved) < per_planet_panels:
        fig = make_series_panel(str(pid), grp["bin"].to_numpy(), mu_vec, freq, np.array(diag.fft_power), np.array(diag.acf))
        panel_path = out_cfg.out_dir / f"panel_{pid}.html"
        _ensure_parent(panel_path)
        pio.write_html(fig, file=str(panel_path), auto_open=False, include_plotlyjs="cdn")
        figures_saved.append(str(panel_path))

# Build summary DataFrame
summary_df = pd.DataFrame.from_records(summary_rows).sort_values(PLANET_ID_COL).reset_index(drop=True)

# Overview plot
fig_overview = make_overview_figure(summary_df, out_cfg.title)
_ensure_parent(out_cfg.html_out)
pio.write_html(fig_overview, file=str(out_cfg.html_out), auto_open=False, include_plotlyjs="cdn")

png_path = None
if out_cfg.png_out:
    _ensure_parent(out_cfg.png_out)
    if _KALEIDO_AVAILABLE:
        fig_overview.write_image(str(out_cfg.png_out), format="png", scale=2, width=1280, height=800)
        png_path = str(out_cfg.png_out)
    else:
        logging.warning("kaleido not installed; PNG export skipped.")

# Save JSON & CSV summaries
_ensure_parent(out_cfg.json_summary)
Path(out_cfg.json_summary).write_text(json.dumps({"planets": per_planet_json}, indent=2))
_ensure_parent(out_cfg.csv_table)
summary_df.to_csv(out_cfg.csv_table, index=False)

# Audit log
if log_ctx:
    append_v50_log(
        log_ctx,
        metadata={
            "mu": str(mu_path),
            "planet_ids": "",
            "labels": str(overlay_cfg.labels_csv) if overlay_cfg and overlay_cfg.labels_csv else "",
            "symbolic": str(overlay_cfg.symbolic_overlays_path) if overlay_cfg and overlay_cfg.symbolic_overlays_path else "",
            "html": str(out_cfg.html_out),
            "png": str(out_cfg.png_out) if out_cfg.png_out else "",
            "duration_sec": f"{time.time() - t0:.2f}",
            "n_planets": str(summary_df.shape[0]),
        },
    )

result = {
    "html": str(out_cfg.html_out),
    "png": png_path,
    "json_summary": str(out_cfg.json_summary),
    "csv_table": str(out_cfg.csv_table),
    "n_planets": int(summary_df.shape[0]),
    "per_planet_panels": figures_saved,
    "kaleido_available": _KALEIDO_AVAILABLE,
    "duration_sec": time.time() - t0,
}

# Optionally open browser
if out_cfg.open_browser:
    try:
        import webbrowser
        webbrowser.open_new_tab(out_cfg.html_out.as_uri())
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
"or import run\_fft\_autocorr\_pipeline() programmatically."
)

```
app = typer.Typer(
    add_completion=True,
    help="SpectraMind V50 — μ FFT & Autocorrelation Diagnostics",
    no_args_is_help=True,
)

@app.command("run")
def cli_run(
    mu: Path = typer.Option(..., exists=True, dir_okay=False, help="Path to μ spectra (.npy/.npz/.csv/.txt)."),
    labels: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional labels/metrics CSV (planet_id keyed)."),
    symbolic_overlays: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional symbolic overlays JSON (per planet)."),
    symbolic_score_key: str = typer.Option("violation_score", help="Key in symbolic JSON to map as score."),
    symbolic_label_key: str = typer.Option("top_rule", help="Key in symbolic JSON to map as label."),
    map_score_to: str = typer.Option("symbolic_score", help="Column name to store symbolic score."),
    map_label_to: str = typer.Option("symbolic_label", help="Column name to store symbolic label."),
    template_csv: Optional[Path] = typer.Option(None, exists=True, dir_okay=False, help="Optional molecular bin template CSV: bin,H2O,CO2,CH4."),
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR, help="Directory for outputs (HTML/PNG/CSV/JSON)."),
    html: Path = typer.Option(DEFAULT_HTML_OUT, help="Overview HTML path."),
    png: Optional[Path] = typer.Option(DEFAULT_PNG_OUT, help="Optional overview PNG path (requires kaleido)."),
    json_summary: Path = typer.Option(DEFAULT_JSON_SUMMARY, help="Per-planet JSON summary path."),
    csv_table: Path = typer.Option(DEFAULT_CSV_TABLE, help="CSV table summary path."),
    open_browser: bool = typer.Option(False, help="Open overview HTML after generation."),
    title: str = typer.Option("SpectraMind V50 — μ FFT & Autocorr Diagnostics", help="Overview plot title."),
    detrend: bool = typer.Option(True, help="Subtract mean before FFT/ACF."),
    window: Optional[str] = typer.Option("hann", help="Window function: None|'hann'|'hamming'|'blackman'."),
    zero_pad: int = typer.Option(0, help="Zero padding length appended to μ before FFT."),
    acf_max_lag: Optional[int] = typer.Option(None, help="Max lag to keep in ACF (default N-1)."),
    peak_k: int = typer.Option(5, help="Number of top peaks to extract from FFT/ACF."),
    per_planet_panels: int = typer.Option(0, help="Export up to N per-planet panel HTMLs."),
    planet_id_col: str = typer.Option(PLANET_ID_COL, help="Name of planet ID column."),
    log_path: Path = typer.Option(DEFAULT_LOG_PATH, help="v50_debug_log.md path."),
    config_hash_path: Optional[Path] = typer.Option(Path("run_hash_summary_v50.json"), help="Optional run hash JSON path."),
):
    """Run μ FFT & autocorrelation diagnostics."""
    params = FFTParams(
        detrend=detrend,
        window=None if (window is None or str(window).lower() in ("", "none", "null")) else window,
        zero_pad=zero_pad,
        acf_max_lag=acf_max_lag,
        peak_k=peak_k,
    )
    overlay_cfg = OverlayConfig(
        labels_csv=labels,
        symbolic_overlays_path=symbolic_overlays,
        symbolic_score_key=symbolic_score_key,
        symbolic_label_key=symbolic_label_key,
        map_score_to=map_score_to,
        map_label_to=map_label_to,
    )
    template_cfg = TemplateConfig(
        bin_template_csv=template_csv,
    )
    out_cfg = OutputConfig(
        out_dir=out_dir,
        html_out=html,
        png_out=png,
        json_summary=json_summary,
        csv_table=csv_table,
        open_browser=open_browser,
        title=title,
    )
    log_ctx = PipelineLogContext(
        log_path=log_path,
        cli_name="spectramind diagnose fft-autocorr-mu",
        config_hash_path=config_hash_path,
    )

    result = run_fft_autocorr_pipeline(
        mu_path=mu,
        out_cfg=out_cfg,
        params=params,
        overlay_cfg=overlay_cfg,
        template_cfg=template_cfg,
        planet_id_col=planet_id_col,
        per_planet_panels=per_planet_panels,
        log_ctx=log_ctx,
    )
    typer.echo(json.dumps(result, indent=2))

@app.command("selftest")
def cli_selftest(
    out_dir: Path = typer.Option(DEFAULT_OUT_DIR / "_selftest", help="Directory to write selftest artifacts."),
    n_planets: int = typer.Option(32, help="Number of synthetic planets."),
    n_bins: int = typer.Option(283, help="Bins per μ."),
    seed: int = typer.Option(DEFAULT_SEED, help="Random seed."),
    per_planet_panels: int = typer.Option(3, help="Export first N per-planet panels."),
):
    """Generate a synthetic μ dataset and run the analyzer end-to-end."""
    rng = np.random.default_rng(seed)
    # Synthetic μ: mixture of smooth + periodic components
    mu_mat = []
    for i in range(n_planets):
        x = np.linspace(0, 6 * np.pi, n_bins)
        base = 0.5 * np.sin(x * (1.0 + 0.1 * rng.standard_normal())) + 0.3 * np.cos(2.0 * x + 0.5)
        trend = 0.1 * (x / x.max())
        noise = 0.05 * rng.standard_normal(n_bins)
        mu_vec = base + trend + noise + 1.0
        mu_mat.append(mu_vec)
    mu_mat = np.stack(mu_mat, axis=0)
    mu_csv = out_dir / "mu.csv"
    _ensure_parent(mu_csv)
    dfw = pd.DataFrame(mu_mat, columns=[f"f{i}" for i in range(n_bins)])
    dfw[PLANET_ID_COL] = [f"P{i:04d}" for i in range(n_planets)]
    dfw.to_csv(mu_csv, index=False)

    # Minimal labels
    labels_csv = out_dir / "labels.csv"
    lab = pd.DataFrame({
        PLANET_ID_COL: dfw[PLANET_ID_COL],
        LABEL_COL: np.where(np.arange(n_planets) % 2 == 0, "A", "B"),
        ENTROPY_COL: rng.random(n_planets),
        SHAP_MAG_COL: np.abs(rng.normal(0.2, 0.1, size=n_planets)),
        GLL_COL: np.abs(rng.normal(0.0, 1.0, size=n_planets)),
    })
    lab.to_csv(labels_csv, index=False)

    # Minimal templates (fake masks)
    tmpl_csv = out_dir / "templates.csv"
    tmpl = pd.DataFrame({
        "bin": np.arange(n_bins, dtype=int),
        "H2O": ((np.arange(n_bins) % 7) == 0).astype(int),
        "CO2": ((np.arange(n_bins) % 11) == 0).astype(int),
        "CH4": ((np.arange(n_bins) % 13) == 0).astype(int),
    })
    tmpl.to_csv(tmpl_csv, index=False)

    result = run_fft_autocorr_pipeline(
        mu_path=mu_csv,
        out_cfg=OutputConfig(
            out_dir=out_dir,
            html_out=out_dir / "overview.html",
            png_out=out_dir / "overview.png",
            json_summary=out_dir / "summary.json",
            csv_table=out_dir / "summary.csv",
            open_browser=False,
            title="Selftest — μ FFT & Autocorr",
        ),
        params=FFTParams(detrend=True, window="hann", zero_pad=256, peak_k=5),
        overlay_cfg=OverlayConfig(labels_csv=labels_csv),
        template_cfg=TemplateConfig(bin_template_csv=tmpl_csv),
        per_planet_panels=per_planet_panels,
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
"import run\_fft\_autocorr\_pipeline() from this module.",
file=sys.stderr,
)
sys.exit(2)
\_app = \_build\_typer\_app()
\_app()
