# src/diagnostics/shap/symbolic/overlay.py

# =============================================================================

# 🛰️ SpectraMind V50 — SHAP × Symbolic × μ Spectrum Overlay Generator

# -----------------------------------------------------------------------------

# Purpose

# • Generate science-ready overlays combining SHAP importance and symbolic

# rule–derived importance over predicted μ spectra (Ariel, 283 bins).

# • Export static (PNG) and interactive (HTML) visualizations, plus CSV/JSON.

# • Provide CLI-first, Hydra-friendly, reproducible execution with run hashing.

#

# Design Tenets

# • NASA-grade reproducibility: config snapshot, run hash, artifact manifest.

# • CLI-first: invocable via Typer wrapper or `python -m` with explicit flags.

# • Safe & robust: strict I/O validation, clear error messages, rich logging.

#

# Inputs (CSV/TSV/TXT/NPY for vectors; JSON/CSV for rules; wavelengths optional):

# • --mu              Path to μ spectrum (shape \[B] for a single planet).

# • --shap            Path to SHAP values aligned to μ bins (shape \[B]).

# • --symbolic        Optional per-bin symbolic importance/violation vector \[B].

# • --rules-json      Optional JSON describing per-rule masks/weights.

# • --rules-csv       Optional CSV describing per-rule contributions.

# • --wavelengths     Optional wavelength centers (shape \[B]) in μm or nm.

# • --meta            Optional JSON/CSV with planet metadata (id, labels).

#

# Symbolic Inputs (choose one path — precedence symbolic > rules-json > rules-csv):

# 1) --symbolic VECTOR

# Direct per-bin importance/violation strengths. Any nonnegative scale.

#

# 2) --rules-json JSON\_FILE

# Expected schema (flexible, best-effort parser):

# {

# "rules": \[

# {"name": "smoothness\_violation", "weight": 1.0,

# "mask": \[0, 0.2, 1.0, ...]  // length B, 0..1 or floats

# },

# {"name": "nonneg", "weight": 0.5, "mask": \[ ... ]},

# ...

# ]

# }

# We compute symbolic\_vec = Σ\_r weight\_r \* normalize(mask\_r)  (clipped ≥0)

#

# 3) --rules-csv CSV\_FILE

# Two supported shapes:

# a) Wide vector: header is bin indices or wavelength; single row or col.

# b) Long triplets: columns \[rule, bin, value] (+ optional weight).

# Long format is aggregated as:

# symbolic\_vec\[bin] += weight \* value   (weight defaults to 1 if absent)

#

# Fusion Modes (combine |SHAP| with SYMBOLIC):

# • product:         fused = |SHAP| \* sym

# • normalized\_prod: fused = zscore(|SHAP|) \* zscore(sym), shifted to ≥ 0

# • weighted\_sum:    fused = α·|SHAP| + (1-α)·sym   (see --alpha)

# • harmonic\_mean:   fused = 2 \* |SHAP| \* sym / (|SHAP| + sym + ε)

# • max:             fused = max(|SHAP|, sym)

# • min:             fused = min(|SHAP|, sym)

#

# Key Features

# • Top-K bin finder by fused importance with CSV and JSON exports.

# • Static Matplotlib overlay (μ line + |SHAP| area + Symbolic area + Fused markers).

# • Interactive Plotly overlay (hover shows μ, |SHAP|, Symbolic, Fused).

# • Optional Savitzky–Golay reference smoothing (visual aid only).

# • Normalization options for μ, SHAP, Symbolic (zscore/minmax/none).

# • Rule attribution at Top-K bins (if rules-\* provided).

# • Deterministic styling (headless-safe) and comprehensive debug logging.

#

# CLI Example

# spectramind diagnose shap-symbolic-overlay \\

# --mu runs/predict/planet\_0042\_mu.npy \\

# --shap runs/explain/planet\_0042\_shap.npy \\

# --rules-json runs/symbolic/planet\_0042\_rules.json \\

# --wavelengths data/wavelengths\_283.csv \\

# --out-dir artifacts/diagnostics/shap\_symbolic/0042 \\

# --fusion product --top-k 25 --png --html --csv --json --open-html

#

# Notes

# • This tool does not compute SHAP or symbolic rules; it visualizes arrays

# produced by upstream pipeline stages (explainability & symbolic logic).

# • If wavelengths are absent, bin indices are used on the x-axis.

# • Conforms to SpectraMind V50 logging & artifact conventions.

# =============================================================================

from **future** import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Headless for CI/Kaggle/servers
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Plotly (optional)

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
return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def \_sha256\_bytes(\*chunks: bytes) -> str:
h = hashlib.sha256()
for c in chunks:
h.update(c)
return h.hexdigest()

def \_read\_vector(path: Optional\[Union\[str, Path]]) -> Optional\[np.ndarray]:
if path is None:
return None
p = Path(path)
if not p.exists():
raise FileNotFoundError(f"Input file not found: {p}")
suf = p.suffix.lower()
if suf == ".npy":
arr = np.load(p)
return np.asarray(arr).reshape(-1).astype(float)
if suf in (".csv", ".tsv", ".txt"):
df = pd.read\_csv(p, sep=None, engine="python", header=None, comment="#")
return df.values.reshape(-1).astype(float)
raise ValueError(f"Unsupported vector extension {p.suffix}; use .npy/.csv/.tsv/.txt")

def \_read\_metadata(path: Optional\[Union\[str, Path]]) -> Optional\[pd.DataFrame]:
if path is None:
return None
p = Path(path)
if not p.exists():
raise FileNotFoundError(f"Metadata file not found: {p}")
if p.suffix.lower() == ".json":
with open(p, "r", encoding="utf-8") as f:
data = json.load(f)
if isinstance(data, dict):
return pd.DataFrame(\[data])
if isinstance(data, list):
return pd.DataFrame(data)
raise ValueError("Unsupported JSON structure for metadata (expect dict or list).")
if p.suffix.lower() in (".csv", ".tsv"):
return pd.read\_csv(p)
raise ValueError(f"Unsupported metadata extension {p.suffix}; use .json/.csv/.tsv")

def \_ensure\_out\_dir(path: Union\[str, Path]) -> Path:
out = Path(path)
out.mkdir(parents=True, exist\_ok=True)
return out

def *safe\_filename(s: str) -> str:
return "".join(c if c.isalnum() or c in "-*." else "\_" for c in s)\[:200]

def \_normalize(arr: np.ndarray, mode: Optional\[str]) -> np.ndarray:
if mode is None or mode.lower() == "none":
return arr
eps = 1e-8
m = mode.lower()
if m == "zscore":
mu, sd = float(np.mean(arr)), float(np.std(arr))
sd = sd if sd > eps else 1.0
return (arr - mu) / sd
if m == "minmax":
lo, hi = float(np.min(arr)), float(np.max(arr))
rng = hi - lo
if rng <= eps:
return np.zeros\_like(arr)
return (arr - lo) / rng
raise ValueError(f"Unknown normalization mode: {mode}")

def \_apply\_savgol(arr: np.ndarray, window: int, poly: int) -> np.ndarray:
if not \_HAS\_SCIPY or window <= 2 or window % 2 == 0 or poly >= window:
return arr
try:
return savgol\_filter(arr, window\_length=window, polyorder=poly, mode="interp")
except Exception:
return arr

def \_stats(vec: np.ndarray, prefix: str) -> Dict\[str, float]:
return {
f"{prefix}\_min": float(np.min(vec)),
f"{prefix}\_max": float(np.max(vec)),
f"{prefix}\_mean": float(np.mean(vec)),
f"{prefix}\_std": float(np.std(vec)),
f"{prefix}\_l2": float(np.linalg.norm(vec)),
}

def \_append\_debug\_log(log\_path: Path, text: str) -> None:
log\_path.parent.mkdir(parents=True, exist\_ok=True)
with open(log\_path, "a", encoding="utf-8") as f:
f.write(text.rstrip() + "\n")

def \_open\_default(path: Path) -> None:
try:
if sys.platform.startswith("darwin"):
os.system(f"open '{path}' >/dev/null 2>&1 &")
elif os.name == "nt":
os.startfile(str(path))  # type: ignore\[attr-defined]
else:
os.system(f"xdg-open '{path}' >/dev/null 2>&1 &")
except Exception:
pass

# ---------------------- Symbolic Vector Construction -------------------------

def \_load\_symbolic\_from\_rules\_json(p: Path, bins: int) -> Tuple\[np.ndarray, Dict\[int, List\[str]]]:
"""
Build a per-bin symbolic importance vector from a rules JSON.
Returns:
symbolic\_vec: np.ndarray shape \[B], nonnegative.
rule\_hits: dict bin\_index -> list of rule names contributing (>0 after weight).
"""
with open(p, "r", encoding="utf-8") as f:
data = json.load(f)

```
rules = data.get("rules", data.get("Rules", data.get("RULES", [])))
if not isinstance(rules, list) or len(rules) == 0:
    raise ValueError("rules-json has no 'rules' list.")

symbolic = np.zeros(bins, dtype=float)
rule_hits: Dict[int, List[str]] = {}
for r in rules:
    name = str(r.get("name", r.get("rule", "rule")))
    weight = float(r.get("weight", 1.0))
    mask = np.asarray(r.get("mask", r.get("vector", r.get("values", []))), dtype=float).reshape(-1)
    if mask.shape[0] != bins:
        raise ValueError(f"Rule '{name}' mask length {mask.shape[0]} != bins {bins}")
    # Normalize mask to [0,1] if positive range; otherwise zscore-guard
    mnorm = _normalize(mask, "minmax")
    contrib = np.clip(weight, -1e9, 1e9) * mnorm
    symbolic += contrib
    nz = np.nonzero(contrib > 0)[0]
    for idx in nz:
        rule_hits.setdefault(int(idx), []).append(name)
symbolic = np.maximum(symbolic, 0.0)
return symbolic, rule_hits
```

def \_load\_symbolic\_from\_rules\_csv(p: Path, bins: int) -> Tuple\[np.ndarray, Dict\[int, List\[str]]]:
"""
Accepts two common CSV shapes:
1\) Long format: columns include \['rule','bin','value'] with optional 'weight'
2\) Wide vector: one row/col of length B (then treated as direct symbolic vec)
"""
df = pd.read\_csv(p)
cols = {c.lower(): c for c in df.columns}

```
# Long format
if {"rule", "bin", "value"} <= set(cols.keys()):
    rule_col = cols["rule"]; bin_col = cols["bin"]; val_col = cols["value"]
    weight_col = cols.get("weight", None)
    vec = np.zeros(bins, dtype=float)
    rule_hits: Dict[int, List[str]] = {}
    for _, row in df.iterrows():
        idx = int(row[bin_col])
        if idx < 0 or idx >= bins:
            continue
        val = float(row[val_col])
        w = float(row[weight_col]) if weight_col and not pd.isna(row[weight_col]) else 1.0
        contrib = w * val
        if contrib > 0:
            vec[idx] += contrib
            rule_hits.setdefault(idx, []).append(str(row[rule_col]))
    vec = np.maximum(vec, 0.0)
    return vec, rule_hits

# Wide vector fallback: use first row or column as vector
values = None
if df.shape[0] == 1 and df.shape[1] >= bins:
    values = df.iloc[0, :bins].to_numpy(dtype=float)
elif df.shape[0] >= bins and df.shape[1] == 1:
    values = df.iloc[:bins, 0].to_numpy(dtype=float)
elif df.shape[0] * df.shape[1] == bins:
    values = df.values.reshape(-1)[:bins].astype(float)

if values is None:
    raise ValueError("rules-csv is neither long format with (rule,bin,value) nor a vector of length bins.")
return np.maximum(values, 0.0), {}
```

def \_build\_symbolic\_vector(
bins: int,
symbolic\_path: Optional\[str],
rules\_json: Optional\[str],
rules\_csv: Optional\[str],
) -> Tuple\[np.ndarray, Dict\[int, List\[str]]]:
"""
Determine symbolic per-bin vector from available inputs.
Precedence: symbolic\_path > rules\_json > rules\_csv.
Returns (symbolic\_vec\_nonneg, rule\_hits\_dict).
"""
if symbolic\_path:
vec = \_read\_vector(symbolic\_path)
if vec is None:
raise ValueError("--symbolic path provided but could not be read.")
if vec.shape\[0] != bins:
raise ValueError(f"--symbolic length {vec.shape\[0]} != bins {bins}")
return np.maximum(vec, 0.0), {}

```
if rules_json:
    return _load_symbolic_from_rules_json(Path(rules_json), bins)

if rules_csv:
    return _load_symbolic_from_rules_csv(Path(rules_csv), bins)

raise ValueError("No symbolic inputs provided. Supply --symbolic or --rules-json or --rules-csv.")
```

# ----------------------------- Fusion Functions ------------------------------

def \_nonneg(x: np.ndarray) -> np.ndarray:
return np.maximum(x, 0.0)

def \_fuse\_importance(
shap\_vec: np.ndarray,
sym\_vec: np.ndarray,
mode: str,
alpha: float,
) -> np.ndarray:
eps = 1e-8
s = np.abs(shap\_vec)  # SHAP magnitude
z = sym\_vec

```
if mode == "product":
    fused = s * z
    return _nonneg(fused)

if mode == "normalized_prod":
    s_z = _normalize(s, "zscore")
    z_z = _normalize(z, "zscore")
    fused = s_z * z_z
    fused = fused - float(np.min(fused))
    return _nonneg(fused)

if mode == "weighted_sum":
    alpha = float(np.clip(alpha, 0.0, 1.0))
    fused = alpha * s + (1.0 - alpha) * z
    return _nonneg(fused)

if mode == "harmonic_mean":
    fused = (2.0 * s * z) / (s + z + eps)
    return _nonneg(fused)

if mode == "max":
    return np.maximum(s, z)

if mode == "min":
    return _nonneg(np.minimum(s, z))

raise ValueError(f"Unknown fusion mode: {mode}")
```

def \_topk\_indices(arr: np.ndarray, k: int) -> np.ndarray:
k = max(1, min(int(k), arr.shape\[0]))
idx = np.argpartition(arr, -k)\[-k:]
idx = idx\[np.argsort(-arr\[idx], kind="stable")]
return np.sort(idx)

# ----------------------------- Plotting (Matplotlib) -------------------------

def \_plot\_matplotlib(
x: np.ndarray,
mu: np.ndarray,
shap\_abs: np.ndarray,
sym: np.ndarray,
fused: np.ndarray,
topk\_idx: np.ndarray,
x\_label: str,
cfg: "OverlayConfig",
out\_png: Path,
) -> Dict\[str, Any]:
fig, ax1 = plt.subplots(figsize=(12, 6), dpi=150)

```
ax1.plot(x, mu, linewidth=1.8, label="μ spectrum")
if cfg.savgol_window and cfg.savgol_window > 2 and cfg.savgol_poly is not None:
    mu_sg = _apply_savgol(mu, cfg.savgol_window, cfg.savgol_poly or 2)
    if not np.allclose(mu_sg, mu):
        ax1.plot(x, mu_sg, linestyle="--", linewidth=1.2, label="μ (Savitzky–Golay)")

ax1.set_xlabel(x_label)
ax1.set_ylabel("μ" if cfg.mu_norm in (None, "none") else f"μ ({cfg.mu_norm})")
ax1.grid(True, alpha=0.2)
ax1.xaxis.set_major_locator(MaxNLocator(nbins=8))

ax2 = ax1.twinx()
ax2.fill_between(x, 0, shap_abs, alpha=0.25, step="pre", label="|SHAP|")
ax2.plot(x, sym, linewidth=1.0, alpha=0.85, label="Symbolic")
ax2.set_ylabel("Importance" if cfg.shap_norm in (None, "none") and cfg.symbolic_norm in (None, "none") else "Importance (norm)")

# Top-K fused markers
ax1.scatter(
    x[topk_idx],
    mu[topk_idx],
    s=36,
    marker="o",
    edgecolors="black",
    linewidths=0.5,
    label=f"Top-{cfg.top_k} fused",
)
for xi in x[topk_idx]:
    ax1.axvline(x=xi, ymin=0, ymax=1, alpha=0.08, linestyle=":", linewidth=0.8)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="best", frameon=True)

ax1.set_title(cfg.title or f"SHAP × Symbolic × μ Overlay — fusion={cfg.fusion}")

fig.tight_layout()
fig.savefig(out_png)
plt.close(fig)

return {"figure": "matplotlib", "path": str(out_png), "width_px": 1800, "height_px": 900}
```

# ----------------------------- Plotting (Plotly) -----------------------------

def \_plot\_plotly(
x: np.ndarray,
mu: np.ndarray,
shap\_abs: np.ndarray,
sym: np.ndarray,
fused: np.ndarray,
topk\_idx: np.ndarray,
x\_label: str,
cfg: "OverlayConfig",
out\_html: Path,
topk\_rules: Optional\[Dict\[int, List\[str]]] = None,
) -> Dict\[str, Any]:
if not \_HAS\_PLOTLY:
raise RuntimeError("Plotly is not installed. Re-run with --no-html or install plotly.")

```
traces = [
    go.Scatter(
        x=x, y=mu,
        mode="lines",
        name="μ spectrum",
        hovertemplate=f"{x_label}: %{{x}}<br>μ: %{{y:.6f}}<extra></extra>",
    )
]

if cfg.savgol_window and cfg.savgol_window > 2 and cfg.savgol_poly is not None:
    mu_sg = _apply_savgol(mu, cfg.savgol_window, cfg.savgol_poly or 2)
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

traces.append(
    go.Scatter(
        x=x, y=sym,
        name="Symbolic",
        mode="lines",
        yaxis="y2",
        hovertemplate=f"{x_label}: %{{x}}<br>Symbolic: %{{y:.6f}}<extra></extra>",
    )
)

# Build customdata for top-k hover: fused, |SHAP|, Symbolic, rules
rules_text = []
for i in topk_idx:
    if topk_rules and int(i) in topk_rules:
        rules_text.append(", ".join(sorted(set(topk_rules[int(i)])))[:200])
    else:
        rules_text.append("")

traces.append(
    go.Scatter(
        x=x[topk_idx],
        y=mu[topk_idx],
        mode="markers",
        name=f"Top-{cfg.top_k} fused",
        marker=dict(size=8, line=dict(width=0.5, color="black")),
        customdata=np.stack([
            fused[topk_idx],
            shap_abs[topk_idx],
            sym[topk_idx],
            np.array(rules_text, dtype=object),
        ], axis=1),
        hovertemplate=(
            f"{x_label}: %{{x}}<br>"
            f"μ: %{{y:.6f}}<br>"
            f"Fused: %{{customdata[0]:.6f}}<br>"
            f"|SHAP|: %{{customdata[1]:.6f}}<br>"
            f"Symbolic: %{{customdata[2]:.6f}}<br>"
            f"Rules: %{{customdata[3]}}<extra></extra>"
        ),
    )
)

layout = go.Layout(
    title=cfg.title or f"SHAP × Symbolic × μ Overlay — fusion={cfg.fusion}",
    xaxis=dict(title=x_label),
    yaxis=dict(title="μ" if cfg.mu_norm in (None, "none") else f"μ ({cfg.mu_norm})"),
    yaxis2=dict(
        title="Importance" if (cfg.shap_norm in (None, "none") and cfg.symbolic_norm in (None, "none")) else "Importance (norm)",
        overlaying="y",
        side="right",
        showgrid=False
    ),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=60, t=60, b=60),
    hovermode="x unified",
    template="plotly_white",
)

fig = go.Figure(data=traces, layout=layout)
plotly_plot(fig, filename=str(out_html), auto_open=False, include_plotlyjs="cdn")

return {"figure": "plotly", "path": str(out_html), "width_px": 1200, "height_px": 650}
```

# ----------------------------- Config Dataclass ------------------------------

@dataclass
class OverlayConfig:
mu\_path: str
shap\_path: str
wavelengths\_path: Optional\[str] = None
metadata\_path: Optional\[str] = None

```
# Symbolic sources (one of these required)
symbolic_path: Optional[str] = None
rules_json: Optional[str] = None
rules_csv: Optional[str] = None

top_k: int = 25

mu_norm: str = "none"           # {"none","zscore","minmax"}
shap_norm: str = "none"         # {"none","zscore","minmax"}
symbolic_norm: str = "none"     # {"none","zscore","minmax"}

fusion: str = "product"         # {"product","normalized_prod","weighted_sum","harmonic_mean","max","min"}
alpha: float = 0.5              # used by weighted_sum

savgol_window: Optional[int] = 0
savgol_poly: Optional[int] = 2

out_dir: str = "artifacts/diagnostics/shap_symbolic"
title: Optional[str] = None
axis_label: str = "Wavelength (μm)"
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
if cfg.seed is not None:
np.random.seed(cfg.seed)

```
mu = _read_vector(cfg.mu_path)
shap = _read_vector(cfg.shap_path)
wl = _read_vector(cfg.wavelengths_path) if cfg.wavelengths_path else None
meta_df = _read_metadata(cfg.metadata_path)

if mu is None or shap is None:
    raise ValueError("Require --mu and --shap vectors.")
if mu.shape != shap.shape:
    raise ValueError(f"Shape mismatch: μ{mu.shape} vs SHAP{shap.shape}; lengths must match.")
if wl is not None and wl.shape != mu.shape:
    raise ValueError(f"Wavelength shape mismatch: wl {wl.shape} vs μ {mu.shape}.")

bins = int(mu.shape[0])

# Build symbolic per-bin vector (and rule hits if available)
sym_vec_raw, rule_hits = _build_symbolic_vector(
    bins=bins,
    symbolic_path=cfg.symbolic_path,
    rules_json=cfg.rules_json,
    rules_csv=cfg.rules_csv,
)

# Visualization normalization
mu_vis = _normalize(mu.astype(float), cfg.mu_norm)
shap_vis = _normalize(shap.astype(float), cfg.shap_norm)
sym_vis = _normalize(sym_vec_raw.astype(float), cfg.symbolic_norm)

fused = _fuse_importance(shap_vis, sym_vis, mode=cfg.fusion, alpha=cfg.alpha)
fused_nonneg = _nonneg(fused)
topk_idx = _topk_indices(fused_nonneg, cfg.top_k)

out_dir = _ensure_out_dir(cfg.out_dir)
if cfg.basename:
    base = _safe_filename(cfg.basename)
else:
    bits = [Path(cfg.mu_path).stem, Path(cfg.shap_path).stem]
    if cfg.symbolic_path:
        bits.append(Path(cfg.symbolic_path).stem)
    elif cfg.rules_json:
        bits.append(Path(cfg.rules_json).stem)
    elif cfg.rules_csv:
        bits.append(Path(cfg.rules_csv).stem)
    base = _safe_filename("__".join(bits))

# Run hash across config + arrays (+ wl if present)
cfg_json = json.dumps(dataclasses.asdict(cfg), sort_keys=True).encode("utf-8")
parts = [
    cfg_json,
    mu_vis.astype(np.float64).tobytes(),
    shap_vis.astype(np.float64).tobytes(),
    sym_vis.astype(np.float64).tobytes(),
]
if wl is not None:
    parts.append(wl.astype(np.float64).tobytes())
run_hash = _sha256_bytes(*parts)

manifest: Dict[str, Any] = {
    "tool": "shap_symbolic_overlay",
    "version": "v50",
    "timestamp": _now_iso(),
    "run_hash": run_hash,
    "config": dataclasses.asdict(cfg),
    "inputs": {
        "mu": str(Path(cfg.mu_path).resolve()),
        "shap": str(Path(cfg.shap_path).resolve()),
        "symbolic": str(Path(cfg.symbolic_path).resolve()) if cfg.symbolic_path else None,
        "rules_json": str(Path(cfg.rules_json).resolve()) if cfg.rules_json else None,
        "rules_csv": str(Path(cfg.rules_csv).resolve()) if cfg.rules_csv else None,
        "wavelengths": str(Path(cfg.wavelengths_path).resolve()) if cfg.wavelengths_path else None,
        "metadata": str(Path(cfg.metadata_path).resolve()) if cfg.metadata_path else None,
    },
    "stats": {},
    "artifacts": {},
    "topk": {},
    "meta_preview": {},
}

# Stats
manifest["stats"].update(_stats(mu_vis, "mu"))
manifest["stats"].update(_stats(np.abs(shap_vis), "shap_abs"))
manifest["stats"].update(_stats(sym_vis, "symbolic"))
manifest["stats"].update(_stats(fused_nonneg, "fused"))

# Prepare x-axis
x = wl if wl is not None else np.arange(mu_vis.shape[0], dtype=float)
x_label = cfg.axis_label if wl is not None else "Bin Index"

# CSV (top-K by fused)
order = np.argsort(-fused_nonneg, kind="stable")[: len(topk_idx)]
if cfg.csv:
    csv_path = out_dir / f"{base}__top{len(topk_idx)}.csv"
    rows: List[List[Any]] = []
    for r, i in enumerate(order, start=1):
        rules_here = ", ".join(sorted(set(rule_hits.get(int(i), [])))) if rule_hits else ""
        rows.append([
            r,
            int(i),
            float(x[i]),
            float(mu_vis[i]),
            float(abs(shap_vis[i])),
            float(sym_vis[i]),
            float(fused_nonneg[i]),
            rules_here,
        ])
    df = pd.DataFrame(rows, columns=["rank", "index", "wavelength_or_bin", "mu", "abs_shap", "symbolic", "fused", "rules"])
    df.to_csv(csv_path, index=False)
    manifest["artifacts"]["csv"] = str(csv_path)

# PNG
if cfg.png:
    png_path = out_dir / f"{base}.png"
    fig_meta = _plot_matplotlib(
        x=x,
        mu=mu_vis,
        shap_abs=np.abs(shap_vis),
        sym=sym_vis,
        fused=fused_nonneg,
        topk_idx=topk_idx,
        x_label=x_label,
        cfg=cfg,
        out_png=png_path,
    )
    manifest["artifacts"]["png"] = fig_meta

# HTML
if cfg.html:
    html_path = out_dir / f"{base}.html"
    html_meta = _plot_plotly(
        x=x,
        mu=mu_vis,
        shap_abs=np.abs(shap_vis),
        sym=sym_vis,
        fused=fused_nonneg,
        topk_idx=topk_idx,
        x_label=x_label,
        cfg=cfg,
        out_html=html_path,
        topk_rules=rule_hits if rule_hits else None,
    )
    manifest["artifacts"]["html"] = html_meta
    if cfg.open_html:
        _open_default(html_path)

# JSON manifest (+ top-k block)
if cfg.json:
    json_path = out_dir / f"{base}.json"
    manifest["topk"] = {
        "k": int(len(topk_idx)),
        "indices_by_fused": [int(i) for i in order.tolist()],
        "x_values": [float(x[i]) for i in order.tolist()],
        "mu": [float(mu_vis[i]) for i in order.tolist()],
        "abs_shap": [float(abs(shap_vis[i])) for i in order.tolist()],
        "symbolic": [float(sym_vis[i]) for i in order.tolist()],
        "fused": [float(fused_nonneg[i]) for i in order.tolist()],
        "rules": [sorted(set(rule_hits.get(int(i), []))) if rule_hits else [] for i in order.tolist()],
    }
    if meta_df is not None:
        try:
            manifest["meta_preview"] = meta_df.head(1).to_dict(orient="records")[0]
        except Exception:
            manifest["meta_preview"] = {}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    manifest["artifacts"]["json"] = str(json_path)

# Debug log
cli_line = " ".join([Path(sys.argv[0]).name] + [a if " " not in a else f'"{a}"' for a in sys.argv[1:]])
log_entry = f"""\
```

### SHAP×SYMBOLIC OVERLAY | {\_now\_iso()}

* run\_hash: {run\_hash}
* out\_dir: {out\_dir}
* inputs: mu={cfg.mu\_path} shap={cfg.shap\_path} symbolic={cfg.symbolic\_path or 'None'} rules\_json={cfg.rules\_json or 'None'} rules\_csv={cfg.rules\_csv or 'None'} wl={cfg.wavelengths\_path or 'None'}
* config: top\_k={cfg.top\_k} mu\_norm={cfg.mu\_norm} shap\_norm={cfg.shap\_norm} symbolic\_norm={cfg.symbolic\_norm} fusion={cfg.fusion} alpha={cfg.alpha} savgol=({cfg.savgol\_window},{cfg.savgol\_poly})
* artifacts: png={manifest\['artifacts'].get('png','-')} html={manifest\['artifacts'].get('html','-')} csv={manifest\['artifacts'].get('csv','-')} json={manifest\['artifacts'].get('json','-')}
* cli: {cli\_line}
  """
  \_append\_debug\_log(Path(cfg.log\_path), log\_entry)

  return manifest

# --------------------------------- CLI Layer ---------------------------------

def \_build\_argparser() -> argparse.ArgumentParser:
p = argparse.ArgumentParser(
prog="shap-symbolic-overlay",
description="Generate SHAP × Symbolic × μ overlay plots and exports (PNG/HTML/CSV/JSON).",
formatter\_class=argparse.ArgumentDefaultsHelpFormatter,
)
\# Required inputs
p.add\_argument("--mu", dest="mu\_path", required=True, help="Path to μ vector (.npy/.csv/.tsv/.txt).")
p.add\_argument("--shap", dest="shap\_path", required=True, help="Path to SHAP vector (.npy/.csv/.tsv/.txt).")

```
# Symbolic sources (one required)
p.add_argument("--symbolic", dest="symbolic_path", default=None, help="Per-bin symbolic vector (.npy/.csv/.tsv/.txt).")
p.add_argument("--rules-json", dest="rules_json", default=None, help="Rules JSON (masks+weights).")
p.add_argument("--rules-csv", dest="rules_csv", default=None, help="Rules CSV (long: rule,bin,value[,weight] or wide vector).")

# Optional
p.add_argument("--wavelengths", dest="wavelengths_path", default=None, help="Optional wavelengths vector (.npy/.csv/.tsv/.txt).")
p.add_argument("--meta", dest="metadata_path", default=None, help="Optional metadata (.json/.csv).")

# Visualization & normalization
p.add_argument("--top-k", type=int, default=25, help="Number of top fused bins to annotate/export.")
p.add_argument("--mu-norm", default="none", choices=["none", "zscore", "minmax"], help="Normalization for μ (visualization).")
p.add_argument("--shap-norm", default="none", choices=["none", "zscore", "minmax"], help="Normalization for SHAP (visualization).")
p.add_argument("--symbolic-norm", dest="symbolic_norm", default="none", choices=["none", "zscore", "minmax"], help="Normalization for Symbolic (visualization).")

# Fusion
p.add_argument("--fusion", default="product", choices=["product", "normalized_prod", "weighted_sum", "harmonic_mean", "max", "min"], help="Fusion mode for |SHAP| and Symbolic.")
p.add_argument("--alpha", type=float, default=0.5, help="Weight for weighted_sum fusion (α for |SHAP|, 1-α for Symbolic).")

# Smoothing & labels
p.add_argument("--savgol-window", type=int, default=0, help="Odd window size for Savitzky–Golay smoothing (0 disables).")
p.add_argument("--savgol-poly", type=int, default=2, help="Polynomial order for Savitzky–Golay (must be < window).")
p.add_argument("--axis-label", default="Wavelength (μm)", help="X-axis label when wavelengths are provided.")
p.add_argument("--title", default=None, help="Optional plot title override.")

# Outputs
p.add_argument("--out-dir", default="artifacts/diagnostics/shap_symbolic", help="Output directory for artifacts.")
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
p.set_defaults(png=True, html=True, csv=True, json=True, open_html=False)
return p
```

def main(argv: Optional\[List\[str]] = None) -> int:
argv = argv if argv is not None else sys.argv\[1:]
parser = \_build\_argparser()
args = parser.parse\_args(argv)

```
# Validate symbolic source presence before constructing config
if not (args.symbolic_path or args.rules_json or args.rules_csv):
    parser.error("One of --symbolic, --rules-json, or --rules-csv is required.")

cfg = OverlayConfig(
    mu_path=args.mu_path,
    shap_path=args.shap_path,
    wavelengths_path=args.wavelengths_path,
    metadata_path=args.metadata_path,
    symbolic_path=args.symbolic_path,
    rules_json=args.rules_json,
    rules_csv=args.rules_csv,
    top_k=args.top_k,
    mu_norm=args.mu_norm,
    shap_norm=args.shap_norm,
    symbolic_norm=args.symbolic_norm,
    fusion=args.fusion,
    alpha=args.alpha,
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
    print(json.dumps({"ok": True, "run_hash": manifest["run_hash"], "artifacts": manifest.get("artifacts", {})}, indent=2))
    return 0
except Exception as e:
    err_text = f"[{_now_iso()}] SHAP×SYMBOLIC OVERLAY ERROR: {e}\n{traceback.format_exc()}"
    sys.stderr.write(err_text + "\n")
    try:
        _append_debug_log(Path(getattr(cfg, "log_path", "v50_debug_log.md")), err_text)
    except Exception:
        pass
    print(json.dumps({"ok": False, "error": str(e)}))
    return 1
```

# ------------------------------- Module Runner -------------------------------

if **name** == "**main**":
sys.exit(main())
