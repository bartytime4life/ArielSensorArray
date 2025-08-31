# src/diagnostics/generate/diagnostic/summary.py

# =============================================================================

# 🧪 SpectraMind V50 — Diagnostic Summary Generator

# -----------------------------------------------------------------------------

# Responsibilities

# • Load μ (mean spectra), σ (uncertainties), optional ground-truth y, and metadata.

# • Compute per-planet + per-bin diagnostics: GLL (if y provided), entropy, FFT, TV/L2 smoothness,

# and simple symbolic-aware overlays (placeholders/hooks).

# • Aggregate to CSV/JSON; write a single diagnostic\_summary.json and tabular exports.

# • Capture reproducibility data (CLI args, file hashes, env info) and a run manifest.

#

# Notes

# • This module is CLI-first (Typer), with Rich logging for UX. No GUI code inside.

# • Safe to run even if some inputs are missing; it will skip dependent metrics rather than fail.

# • No mutation of inputs — diagnostics only read, analyze, and log artifacts.

# =============================================================================

from **future** import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import textwrap
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Third-party (soft) dependencies

try:
import numpy as np
except Exception as e:  # pragma: no cover
raise RuntimeError("numpy is required for diagnostics") from e

try:
import pandas as pd
except Exception as e:  # pragma: no cover
raise RuntimeError("pandas is required for diagnostics") from e

try:
from scipy.fft import rfft, rfftfreq
from scipy.signal import correlate
except Exception:
rfft = None
rfftfreq = None
correlate = None

try:
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TimeElapsedColumn
except Exception:  # pragma: no cover
\# Fallback minimal logger/CLI if Rich/Typer unavailable
typer = None
Console = None

# ----------------------------- Utilities & I/O --------------------------------

console = Console(stderr=True, highlight=True) if Console else None

def log\_info(msg: str) -> None:
if console:
console.log(f"\[bold cyan]ℹ\[/] {msg}")
else:
print(f"\[INFO] {msg}", file=sys.stderr)

def log\_warn(msg: str) -> None:
if console:
console.log(f"\[bold yellow]⚠\[/] {msg}")
else:
print(f"\[WARN] {msg}", file=sys.stderr)

def log\_err(msg: str) -> None:
if console:
console.log(f"\[bold red]✖\[/] {msg}")
else:
print(f"\[ERROR] {msg}", file=sys.stderr)

def ensure\_dir(path: Path) -> None:
path.mkdir(parents=True, exist\_ok=True)

def file\_sha256(path: Optional\[Path]) -> Optional\[str]:
if path is None or not path.exists():
return None
h = hashlib.sha256()
with path.open("rb") as f:
for chunk in iter(lambda: f.read(1024 \* 1024), b""):
h.update(chunk)
return h.hexdigest()

def save\_json(obj: Any, path: Path) -> None:
ensure\_dir(path.parent)
with path.open("w", encoding="utf-8") as f:
json.dump(obj, f, indent=2, ensure\_ascii=False)

def save\_csv(df: pd.DataFrame, path: Path) -> None:
ensure\_dir(path.parent)
df.to\_csv(path, index=False)

def now\_ts() -> str:
return time.strftime("%Y-%m-%dT%H-%M-%S")

# ------------------------------- Data Models ----------------------------------

@dataclass
class InputsConfig:
mu\_path: Path
sigma\_path: Optional\[Path] = None
y\_path: Optional\[Path] = None
meta\_path: Optional\[Path] = None

@dataclass
class FFTConfig:
enable: bool = True
cutoff\_freq: Optional\[float] = None  # report-only, not filtering
compute\_autocorr: bool = True

@dataclass
class SmoothnessConfig:
enable: bool = True
compute\_grad1: bool = True
compute\_grad2: bool = True
tv: bool = True  # total variation
normalize\_by\_range: bool = True
eps: float = 1e-12

@dataclass
class EntropyConfig:
enable: bool = True
method: str = "shannon"
bins: int = 32
per\_bin: bool = True  # entropy over planets for each wavelength

@dataclass
class FlagsConfig:
grad\_p95: float = 1.5
curv\_p95: float = 1.25
tv\_max: float = 0.30

@dataclass
class OutputConfig:
out\_dir: Path
per\_planet\_csv: Path
per\_bin\_csv: Path
summary\_json: Path
manifest\_json: Path

@dataclass
class RunConfig:
inputs: InputsConfig
fft: FFTConfig = FFTConfig()
smooth: SmoothnessConfig = SmoothnessConfig()
entropy: EntropyConfig = EntropyConfig()
flags: FlagsConfig = FlagsConfig()
outputs: Optional\[OutputConfig] = None
\# free-form for future extensions
extras: Dict\[str, Any] = dataclasses.field(default\_factory=dict)

# ----------------------------- Loading Functions ------------------------------

def load\_numpy(path: Optional\[Path], name: str) -> Optional\[np.ndarray]:
if path is None:
return None
if not path.exists():
log\_warn(f"{name} not found at {path}")
return None
arr = np.load(path)
if not isinstance(arr, np.ndarray):
raise ValueError(f"{name} at {path} is not a numpy array")
return arr

def load\_meta(path: Optional\[Path]) -> Optional\[pd.DataFrame]:
if path is None:
return None
if not path.exists():
log\_warn(f"meta not found at {path}")
return None
return pd.read\_csv(path)

# ---------------------------- Diagnostic Metrics ------------------------------

def gll\_per\_planet(mu: np.ndarray, sigma: np.ndarray, y: np.ndarray, eps: float = 1e-9) -> np.ndarray:
"""
Gaussian log-likelihood per-planet, averaged over bins:
L = -0.5 \* \[ log(2πσ^2) + (y-μ)^2 / σ^2 ]
Returns per-planet mean GLL.
"""
if sigma is None:
raise ValueError("σ required for GLL")
if y is None:
raise ValueError("y required for GLL")
sigma2 = np.clip(sigma, eps, None) \*\* 2
term = -0.5 \* (np.log(2 \* np.pi \* sigma2) + (y - mu) \*\* 2 / sigma2)
return term.mean(axis=1)

def grad\_curv\_tv(mu: np.ndarray, normalize\_by\_range: bool = True, eps: float = 1e-12) -> Tuple\[np.ndarray, np.ndarray, np.ndarray]:
"""
Compute first derivative magnitude (central diff), second derivative magnitude, and total variation per bin.
Returns (grad1\_mag \[N,K], grad2\_mag \[N,K], tv\_per\_bin \[N,K]).
"""
\# Assume mu shape \[N, K]
d1 = np.zeros\_like(mu)
d2 = np.zeros\_like(mu)

```
# First derivative (central)
d1[:, 1:-1] = (mu[:, 2:] - mu[:, :-2]) / 2.0
d1[:, 0] = mu[:, 1] - mu[:, 0]
d1[:, -1] = mu[:, -1] - mu[:, -2]

# Second derivative (central)
d2[:, 1:-1] = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
d2[:, 0] = d2[:, 1]
d2[:, -1] = d2[:, -2]

# TV per bin
tv = np.zeros_like(mu)
tv[:, 1:] = np.abs(mu[:, 1:] - mu[:, :-1])

if normalize_by_range:
    # scale per-planet by (max-min)+eps for stability
    ptp = np.clip(mu.max(axis=1) - mu.min(axis=1), eps, None)  # [N]
    d1 = d1 / ptp[:, None]
    d2 = d2 / ptp[:, None]
    tv = tv / ptp[:, None]

return np.abs(d1), np.abs(d2), tv
```

def spectral\_entropy(mu: np.ndarray, bins: int = 32, per\_bin: bool = True, eps: float = 1e-12) -> Tuple\[np.ndarray, np.ndarray]:
"""
Estimate Shannon entropy (base e) either per-planet (across bins) or per-bin (across planets).
Returns (H\_per\_planet \[N], H\_per\_bin \[K]) where one may be np.nan if disabled.
"""
N, K = mu.shape
Hp = np.full(N, np.nan, dtype=float)
Hb = np.full(K, np.nan, dtype=float)

```
# Per-planet entropy across wavelength bins
for i in range(N):
    x = mu[i]
    hist, _ = np.histogram(x, bins=bins, density=True)
    p = hist / (hist.sum() + eps)
    Hp[i] = -(p * np.log(p + eps)).sum()

# Per-bin entropy across planets
if per_bin:
    for j in range(K):
        x = mu[:, j]
        hist, _ = np.histogram(x, bins=bins, density=True)
        p = hist / (hist.sum() + eps)
        Hb[j] = -(p * np.log(p + eps)).sum()

return Hp, Hb
```

def fft\_autocorr\_features(mu: np.ndarray) -> Tuple\[Optional\[np.ndarray], Optional\[np.ndarray], Optional\[np.ndarray]]:
"""
Compute simple FFT power spectrum summaries and autocorrelation along wavelength axis per planet.
Returns (peak\_freq\[N], power\_mean\[N], autocorr\_max\[N]) or (None, None, None) if scipy missing.
"""
if rfft is None or rfftfreq is None or correlate is None:
log\_warn("scipy not available — skipping FFT/autocorr diagnostics.")
return None, None, None

```
N, K = mu.shape
peak_freq = np.zeros(N)
power_mean = np.zeros(N)
ac_max = np.zeros(N)

# Use dummy unit spacing; absolute freq not critical for relative diagnostics
freq = rfftfreq(K, d=1.0)
for i in range(N):
    x = mu[i] - mu[i].mean()
    X = rfft(x)
    P = (X.conj() * X).real
    # Exclude zero frequency when searching peak to avoid DC
    if P.shape[0] > 1:
        idx = np.argmax(P[1:]) + 1
    else:
        idx = 0
    peak_freq[i] = freq[idx]
    power_mean[i] = P.mean()

    # Autocorrelation (normalized)
    ac = correlate(x, x, mode="full")
    ac = ac[ac.size // 2 :]
    if ac[0] != 0:
        ac = ac / ac[0]
    ac_max[i] = np.nanmax(ac[1:]) if ac.size > 1 else 0.0

return peak_freq, power_mean, ac_max
```

# ---------------------------- Aggregation Helpers -----------------------------

def agg\_stats(arr: np.ndarray, axis: int) -> Dict\[str, np.ndarray]:
return {
"mean": np.nanmean(arr, axis=axis),
"p95": np.nanpercentile(arr, 95, axis=axis),
"max": np.nanmax(arr, axis=axis),
}

def safe\_col(vec: Optional\[np.ndarray], name: str, length: int) -> pd.Series:
if vec is None:
return pd.Series(\[np.nan] \* length, name=name)
return pd.Series(vec, name=name)

# ------------------------------- Main Routine ---------------------------------

def run\_diagnostics(cfg: RunConfig) -> Dict\[str, Any]:
\# Prepare outputs
if cfg.outputs is None:
out\_dir = Path("artifacts/diagnostics") / now\_ts()
outputs = OutputConfig(
out\_dir=out\_dir,
per\_planet\_csv=out\_dir / "per\_planet\_metrics.csv",
per\_bin\_csv=out\_dir / "per\_bin\_metrics.csv",
summary\_json=out\_dir / "diagnostic\_summary.json",
manifest\_json=out\_dir / "manifest.json",
)
cfg.outputs = outputs
ensure\_dir(cfg.outputs.out\_dir)

```
# Load inputs
mu = load_numpy(cfg.inputs.mu_path, "μ")
if mu is None:
    raise FileNotFoundError(f"μ not found at {cfg.inputs.mu_path}")
sigma = load_numpy(cfg.inputs.sigma_path, "σ") if cfg.inputs.sigma_path else None
y = load_numpy(cfg.inputs.y_path, "y") if cfg.inputs.y_path else None
meta = load_meta(cfg.inputs.meta_path)

N, K = mu.shape
log_info(f"Loaded μ with shape [N={N}, K={K}]")
if sigma is not None:
    log_info(f"Loaded σ with shape {sigma.shape}")
if y is not None:
    log_info(f"Loaded y with shape {y.shape}")
if meta is not None:
    log_info(f"Loaded meta with {len(meta)} rows")

# Smoothness metrics
grad1_mag = curv_mag = tv_per_bin = None
if cfg.smooth.enable:
    grad1_mag, curv_mag, tv_per_bin = grad_curv_tv(
        mu,
        normalize_by_range=cfg.smooth.normalize_by_range,
        eps=cfg.smooth.eps,
    )

# Entropy
H_planet = H_bin = None
if cfg.entropy.enable:
    H_planet, H_bin = spectral_entropy(
        mu,
        bins=cfg.entropy.bins,
        per_bin=cfg.entropy.per_bin,
    )

# FFT / Autocorr
peak_freq, power_mean, ac_max = (None, None, None)
if cfg.fft.enable:
    peak_freq, power_mean, ac_max = fft_autocorr_features(mu)

# GLL
gll_vec = None
if y is not None and sigma is not None:
    try:
        gll_vec = gll_per_planet(mu, sigma, y)
    except Exception as e:
        log_warn(f"GLL computation failed: {e}")

# Per-planet aggregation
per_planet = pd.DataFrame({
    "planet_idx": np.arange(N, dtype=int),
    "gll_mean": safe_col(gll_vec, "gll_mean", N),
    "fft_peak_freq": safe_col(peak_freq, "fft_peak_freq", N),
    "fft_power_mean": safe_col(power_mean, "fft_power_mean", N),
    "autocorr_max": safe_col(ac_max, "autocorr_max", N),
    "entropy_planet": safe_col(H_planet, "entropy_planet", N),
})

if grad1_mag is not None:
    per_planet["grad_mean"] = np.nanmean(grad1_mag, axis=1)
    per_planet["grad_p95"] = np.nanpercentile(grad1_mag, 95, axis=1)
    per_planet["grad_max"] = np.nanmax(grad1_mag, axis=1)

if curv_mag is not None:
    per_planet["curv_mean"] = np.nanmean(curv_mag, axis=1)
    per_planet["curv_p95"] = np.nanpercentile(curv_mag, 95, axis=1)
    per_planet["curv_max"] = np.nanmax(curv_mag, axis=1)

if tv_per_bin is not None:
    per_planet["tv_mean"] = np.nanmean(tv_per_bin, axis=1)
    per_planet["tv_p95"] = np.nanpercentile(tv_per_bin, 95, axis=1)
    per_planet["tv_max"] = np.nanmax(tv_per_bin, axis=1)

# Merge meta if available
if meta is not None:
    # Try to align by index if a column planet_id exists
    idx_col = None
    for c in ["planet_idx", "planet_id", "id", "pid"]:
        if c in meta.columns:
            idx_col = c
            break
    if idx_col is not None:
        try:
            per_planet = per_planet.merge(meta, left_on="planet_idx", right_on=idx_col, how="left")
        except Exception as e:
            log_warn(f"Failed to merge meta: {e}")

# Per-bin aggregation (across planets)
per_bin_records: Dict[str, Any] = {"bin_idx": np.arange(K, dtype=int)}
if H_bin is not None:
    per_bin_records["entropy_bin"] = H_bin
if grad1_mag is not None:
    per_bin_records["grad_bin_mean"] = np.nanmean(grad1_mag, axis=0)
    per_bin_records["grad_bin_p95"] = np.nanpercentile(grad1_mag, 95, axis=0)
    per_bin_records["grad_bin_max"] = np.nanmax(grad1_mag, axis=0)
if curv_mag is not None:
    per_bin_records["curv_bin_mean"] = np.nanmean(curv_mag, axis=0)
    per_bin_records["curv_bin_p95"] = np.nanpercentile(curv_mag, 95, axis=0)
    per_bin_records["curv_bin_max"] = np.nanmax(curv_mag, axis=0)
if tv_per_bin is not None:
    per_bin_records["tv_bin_mean"] = np.nanmean(tv_per_bin, axis=0)
    per_bin_records["tv_bin_p95"] = np.nanpercentile(tv_per_bin, 95, axis=0)
    per_bin_records["tv_bin_max"] = np.nanmax(tv_per_bin, axis=0)

per_bin = pd.DataFrame(per_bin_records)

# Flags (simple heuristics)
flags = {
    "grad_exceed": (per_planet.get("grad_p95", pd.Series([np.nan] * N)) > cfg.flags.grad_p95).fillna(False),
    "curv_exceed": (per_planet.get("curv_p95", pd.Series([np.nan] * N)) > cfg.flags.curv_p95).fillna(False),
    "tv_exceed": (per_planet.get("tv_max", pd.Series([np.nan] * N)) > cfg.flags.tv_max).fillna(False),
}
per_planet["flag_grad"] = flags["grad_exceed"].astype(bool)
per_planet["flag_curv"] = flags["curv_exceed"].astype(bool)
per_planet["flag_tv"] = flags["tv_exceed"].astype(bool)

# Summary JSON
summary: Dict[str, Any] = {
    "shape": {"N_planets": int(N), "K_bins": int(K)},
    "metrics": {
        "gll": {
            "available": gll_vec is not None,
            "mean": float(np.nanmean(gll_vec)) if gll_vec is not None else None,
            "p95": float(np.nanpercentile(gll_vec, 95)) if gll_vec is not None else None,
        },
        "fft": {
            "available": peak_freq is not None,
            "power_mean_avg": float(np.nanmean(power_mean)) if power_mean is not None else None,
            "autocorr_max_avg": float(np.nanmean(ac_max)) if ac_max is not None else None,
        },
        "entropy": {
            "planet_mean": float(np.nanmean(H_planet)) if H_planet is not None else None,
            "bin_mean": float(np.nanmean(H_bin)) if H_bin is not None else None,
        },
        "smoothness": {
            "grad_p95_mean": float(np.nanmean(per_planet["grad_p95"])) if "grad_p95" in per_planet else None,
            "curv_p95_mean": float(np.nanmean(per_planet["curv_p95"])) if "curv_p95" in per_planet else None,
            "tv_max_mean": float(np.nanmean(per_planet["tv_max"])) if "tv_max" in per_planet else None,
        },
        "flags": {
            "n_grad": int(per_planet["flag_grad"].sum()),
            "n_curv": int(per_planet["flag_curv"].sum()),
            "n_tv": int(per_planet["flag_tv"].sum()),
        },
    },
    "paths": {
        "per_planet_csv": str(cfg.outputs.per_planet_csv),
        "per_bin_csv": str(cfg.outputs.per_bin_csv),
        "summary_json": str(cfg.outputs.summary_json),
        "manifest_json": str(cfg.outputs.manifest_json),
    },
    "config": asdict(cfg),
}

# Save artifacts
save_csv(per_planet, cfg.outputs.per_planet_csv)
save_csv(per_bin, cfg.outputs.per_bin_csv)
save_json(summary, cfg.outputs.summary_json)

# Manifest (reproducibility)
manifest = {
    "timestamp": now_ts(),
    "platform": {
        "python": sys.version,
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
    },
    "inputs": {
        "mu_path": str(cfg.inputs.mu_path),
        "sigma_path": str(cfg.inputs.sigma_path) if cfg.inputs.sigma_path else None,
        "y_path": str(cfg.inputs.y_path) if cfg.inputs.y_path else None,
        "meta_path": str(cfg.inputs.meta_path) if cfg.inputs.meta_path else None,
        "mu_sha256": file_sha256(cfg.inputs.mu_path),
        "sigma_sha256": file_sha256(cfg.inputs.sigma_path) if cfg.inputs.sigma_path else None,
        "y_sha256": file_sha256(cfg.inputs.y_path) if cfg.inputs.y_path else None,
        "meta_sha256": file_sha256(cfg.inputs.meta_path) if cfg.inputs.meta_path else None,
    },
    "outputs": {
        "dir": str(cfg.outputs.out_dir),
        "per_planet_csv": str(cfg.outputs.per_planet_csv),
        "per_bin_csv": str(cfg.outputs.per_bin_csv),
        "summary_json": str(cfg.outputs.summary_json),
    },
    "cli": " ".join(map(str, sys.argv)),
    "config": asdict(cfg),
}
save_json(manifest, cfg.outputs.manifest_json)

# Console summary
if console:
    tbl = Table(title="Diagnostics Summary", show_header=True, header_style="bold magenta")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("N planets", str(N))
    tbl.add_row("K bins", str(K))
    gll_mean = summary["metrics"]["gll"]["mean"]
    tbl.add_row("GLL mean", f"{gll_mean:.6f}" if gll_mean is not None else "—")
    fft_pm = summary["metrics"]["fft"]["power_mean_avg"]
    tbl.add_row("FFT power mean (avg)", f"{fft_pm:.6f}" if fft_pm is not None else "—")
    ent_p = summary["metrics"]["entropy"]["planet_mean"]
    tbl.add_row("Entropy (planet mean)", f"{ent_p:.6f}" if ent_p is not None else "—")
    console.print(Panel(tbl, title="SpectraMind V50 — Diagnostics"))

return summary
```

# ---------------------------------- CLI ---------------------------------------

def build\_argparser() -> argparse.ArgumentParser:
p = argparse.ArgumentParser(
prog="spectramind-diagnostic-summary",
description="Generate diagnostic summary (GLL, entropy, FFT, smoothness) from μ/σ(/y) arrays.",
formatter\_class=argparse.ArgumentDefaultsHelpFormatter,
)
p.add\_argument("--mu", type=Path, required=True, help="Path to μ.npy (N,K)")
p.add\_argument("--sigma", type=Path, default=None, help="Path to σ.npy (N,K)")
p.add\_argument("--y", type=Path, default=None, help="Path to ground truth y.npy (N,K) (optional)")
p.add\_argument("--meta", type=Path, default=None, help="Path to meta.csv (optional)")
p.add\_argument("--outdir", type=Path, default=Path("artifacts/diagnostics"), help="Base output directory")
p.add\_argument("--fft", action="store\_true", help="Enable FFT/autocorr diagnostics")
p.add\_argument("--no-fft", dest="fft", action="store\_false", help="Disable FFT/autocorr diagnostics")
p.add\_argument("--entropy", action="store\_true", help="Enable entropy diagnostics")
p.add\_argument("--no-entropy", dest="entropy", action="store\_false", help="Disable entropy diagnostics")
p.add\_argument("--smooth", action="store\_true", help="Enable smoothness diagnostics")
p.add\_argument("--no-smooth", dest="smooth", action="store\_false", help="Disable smoothness diagnostics")
p.add\_argument("--bins", type=int, default=32, help="Entropy histogram bins")
p.add\_argument("--grad-norm", action="store\_true", help="Normalize gradients by (max-min) per planet")
p.add\_argument("--no-grad-norm", dest="grad\_norm", action="store\_false", help="Disable gradient normalization")
p.set\_defaults(fft=True, entropy=True, smooth=True, grad\_norm=True)
return p

def main\_cli(argv: Optional\[List\[str]] = None) -> int:
ap = build\_argparser()
args = ap.parse\_args(argv)

```
ts_dir = args.outdir / now_ts()

cfg = RunConfig(
    inputs=InputsConfig(
        mu_path=args.mu,
        sigma_path=args.sigma,
        y_path=args.y,
        meta_path=args.meta,
    ),
    fft=FFTConfig(enable=bool(args.fft)),
    smooth=SmoothnessConfig(enable=bool(args.smooth), normalize_by_range=bool(args.grad_norm)),
    entropy=EntropyConfig(enable=bool(args.entropy), bins=int(args.bins)),
)
cfg.outputs = OutputConfig(
    out_dir=ts_dir,
    per_planet_csv=ts_dir / "per_planet_metrics.csv",
    per_bin_csv=ts_dir / "per_bin_metrics.csv",
    summary_json=ts_dir / "diagnostic_summary.json",
    manifest_json=ts_dir / "manifest.json",
)

try:
    run_diagnostics(cfg)
    log_info(f"Artifacts written to {cfg.outputs.out_dir}")
    return 0
except Exception as e:
    log_err(f"Diagnostics failed: {e}")
    return 1
```

if **name** == "**main**":
sys.exit(main\_cli())
