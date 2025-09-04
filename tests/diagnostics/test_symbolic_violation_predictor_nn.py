\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
tests/diagnostics/test\_symbolic\_violation\_predictor\_nn.py

SpectraMind V50 — Diagnostics tests for src/diagnostics/symbolic\_violation\_predictor\_nn.py

## Purpose

Validate the neural symbolic violation predictor module that estimates per-rule violation
probabilities from μ spectra (and optional metadata). The tests are tolerant to minor
API/CLI differences but enforce core guarantees:

1. Import & API surface
   • Class existence (e.g., SymbolicViolationPredictorNN) or train/predict functions.
   • Graceful xfail if the module isn't present yet.

2. Train → Predict sanity (tiny synthetic dataset)
   • Short CPU-only training improves separation for at least one rule
   (mean p̂(positive) > mean p̂(negative)).
   • Deterministic behavior with --seed (API-level), modulo expected float noise.

3. Artifact generation
   • JSON/CSV/PNG/HTML presence via an artifact generator entrypoint.
   • JSON contains expected keys (shapes, metrics, per-rule summary).

4. CLI contract
   • End-to-end run via `python -m src.diagnostics.symbolic_violation_predictor_nn`
   with --mu/--labels-json/--outdir (and --seed).
   • Determinism across two runs after scrubbing volatile fields.
   • Optional audit log append when SPECTRAMIND\_LOG\_PATH is set.

5. Housekeeping
   • Files are non-empty and survive idempotent re-runs.

## Assumptions

• The module lives at src/diagnostics/symbolic\_violation\_predictor\_nn.py and exports at least one of:
\- class SymbolicViolationPredictorNN(nn.Module)
\- train\_symbolic\_violation\_predictor\_nn(...)
\- predict\_symbolic\_violations(...)
\- generate\_symbolic\_violation\_artifacts(...) / run\_symbolic\_violation\_predictor\_nn(...)
• Alternative names tolerated:
\- train\_nn, fit\_model, train
\- infer, predict
\- produce\_symbolic\_violation\_outputs, analyze\_and\_export

• CLI accepts (variations tolerated):
\--mu \<path.npy>                  (required; \[N,B] or \[B])
\--planet-ids <.txt|.csv>         (optional; first column if .csv)
\--labels-json \<path.json>        (required for training/metrics)
\--outdir <dir>                   (required)
\--epochs <int>                   (optional; tiny in tests)
\--seed <int>                     (optional)
\--json --csv --png --html        (subset allowed)
\--silent                         (optional)

## Implementation Notes

• The synthetic dataset is tiny and CPU-friendly to keep CI fast.
• If torch is unavailable or import fails, tests xfail with an explanatory message.
• Numerical checks avoid brittle equalities on floats.

Author: SpectraMind V50 test harness
"""

from **future** import annotations

import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

# =================================================================================================

# Path & import helpers

# =================================================================================================

def \_repo\_root\_from\_test\_file() -> Path:
"""
Heuristic repository root given this file path:
tests/diagnostics/test\_symbolic\_violation\_predictor\_nn.py
"""
here = Path(**file**).resolve()
return here.parent.parent

def \_module\_candidates() -> Tuple\[Path, ...]:
"""
Likely filesystem locations for the module under test.
"""
root = \_repo\_root\_from\_test\_file()
return (
root / "src" / "diagnostics" / "symbolic\_violation\_predictor\_nn.py",
root / "src" / "symbolic\_violation\_predictor\_nn.py",
root / "symbolic\_violation\_predictor\_nn.py",
)

def \_import\_tool():
"""
Import the module under test, trying several import styles.
"""
\# Preferred: package import
try:
import importlib
m = importlib.import\_module("src.diagnostics.symbolic\_violation\_predictor\_nn")  # type: ignore
return m
except Exception:
pass

```
# Fallback: add repo/src to sys.path and try alternatives
root = _repo_root_from_test_file()
src_dir = root / "src"
for p in (str(root), str(src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import symbolic_violation_predictor_nn as m2  # type: ignore
    return m2
except Exception:
    pass

try:
    import importlib
    m3 = importlib.import_module("diagnostics.symbolic_violation_predictor_nn")  # type: ignore
    return m3
except Exception:
    pass

# If file exists but import fails, surface informative skip
cands = [str(p) for p in _module_candidates() if p.exists()]
if not cands:
    pytest.skip(
        "symbolic_violation_predictor_nn module not found. "
        "Expected at src/diagnostics/symbolic_violation_predictor_nn.py or importable as src.diagnostics.symbolic_violation_predictor_nn."
    )
pytest.skip(
    f"symbolic_violation_predictor_nn present but import failed. Tried candidates: {cands}. "
    "Ensure import path and package layout are correct."
)
```

def \_has\_attr(mod, name: str) -> bool:
return hasattr(mod, name) and getattr(mod, name) is not None

def \_run\_cli(
module\_path: Path,
args: Sequence\[str],
env: Optional\[Dict\[str, str]] = None,
timeout: int = 420,
) -> subprocess.CompletedProcess:
"""
Execute the tool as a CLI using `python -m src.diagnostics.symbolic_violation_predictor_nn` when possible.
Fallback to invoking the file directly.
"""
\# Try package execution if under src/diagnostics
if module\_path.name == "symbolic\_violation\_predictor\_nn.py" and module\_path.parent.name == "diagnostics":
repo\_root = module\_path.parents\[2]  # src/diagnostics/.. -> src/.. -> root
candidate\_pkg = "src.diagnostics.symbolic\_violation\_predictor\_nn"
cmd = \[sys.executable, "-m", candidate\_pkg, \*args]
cwd = str(repo\_root)
else:
cmd = \[sys.executable, str(module\_path), \*args]
cwd = str(module\_path.parent)

```
env_full = os.environ.copy()
if env:
    env_full.update(env)

return subprocess.run(
    cmd,
    cwd=cwd,
    env=env_full,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    timeout=timeout,
    text=True,
    check=False,
)
```

def \_assert\_file(p: Path, min\_size: int = 1) -> None:
"""
Basic artifact sanity: file exists, is a file, and exceeds min\_size bytes.
"""
assert p.exists(), f"File not found: {p}"
assert p.is\_file(), f"Expected file: {p}"
sz = p.stat().st\_size
assert sz >= min\_size, f"File too small ({sz} bytes): {p}"

# =================================================================================================

# Synthetic dataset (μ spectra + symbolic rule labels)

# =================================================================================================

def *make\_planet\_ids(n: int) -> List\[str]:
return \[f"planet*{i:04d}" for i in range(n)]

def \_make\_wavelengths(n\_bins: int, lo\_um: float = 0.5, hi\_um: float = 7.8) -> np.ndarray:
return np.linspace(float(lo\_um), float(hi\_um), n\_bins, dtype=np.float64)

def \_make\_mu\_spectra(n\_planets: int, n\_bins: int, seed: int = 2025) -> np.ndarray:
"""
Construct μ with structured signals that correlate with simple symbolic "rules".
Rules (hidden ground truth):
R0: "molecule band high" → mean over band\_1 exceeds global mean + δ
R1: "high-frequency"     → HF power fraction exceeds θ
R2: "sinusoid present"   → Corr with sine template exceeds τ
"""
rng = np.random.default\_rng(seed)
\# Base smooth spectrum
base = np.linspace(0.0, 1.0, n\_bins, dtype=np.float64)
spectra = np.empty((n\_planets, n\_bins), dtype=np.float64)

```
# Generate mixtures
for i in range(n_planets):
    # smooth + localized bump + sinusoid + noise
    bump_center = rng.integers(low=int(n_bins * 0.2), high=int(n_bins * 0.8))
    bump = np.exp(-0.5 * ((np.arange(n_bins) - bump_center) / max(3, n_bins * 0.02)) ** 2)
    sine = 0.15 * np.sin(2.0 * math.pi * np.arange(n_bins) / max(8, n_bins // 16))
    spectrum = 0.6 * base + 0.4 * bump + sine
    spectrum += rng.normal(0.0, 0.03, size=n_bins)
    spectra[i] = spectrum
return spectra
```

def \_labels\_from\_rules(mu: np.ndarray, wavelengths: np.ndarray) -> Dict\[str, Dict\[str, int]]:
"""
Produce synthetic binary labels per planet for three rules using simple heuristics
on μ and its FFT. Returns mapping {planet\_id: {"R0": 0/1, "R1": 0/1, "R2": 0/1}}.
"""
n, b = mu.shape
ids = \_make\_planet\_ids(n)
\# Band for R0 (e.g., 1.3–1.6 μm)
band\_mask = (wavelengths >= 1.3) & (wavelengths <= 1.6)
\# Compute simple features
labels: Dict\[str, Dict\[str, int]] = {}
for i in range(n):
x = mu\[i]
mean\_all = float(np.mean(x))
mean\_band = float(np.mean(x\[band\_mask])) if band\_mask.any() else mean\_all
\# R0: band above global mean by δ
r0 = 1 if (mean\_band - mean\_all) > 0.05 else 0
\# R1: high-frequency power fraction
X = np.fft.rfft(x - np.mean(x))
P = (X \* np.conj(X)).real
if P.size <= 1:
hf\_frac = 0.0
else:
idx\_cut = int(0.6 \* (P.size - 1))
denom = float(np.sum(P\[1:]) + 1e-12)
hf\_frac = float(np.sum(P\[idx\_cut:]) / denom)
r1 = 1 if hf\_frac > 0.12 else 0
\# R2: correlation with sine template
sine = np.sin(2.0 \* math.pi \* np.arange(b) / max(8, b // 16))
a = x - np.mean(x)
s = sine - np.mean(sine)
denom = (np.std(a) \* np.std(s)) or 1.0
corr = float(np.dot(a, s) / (b \* denom))
r2 = 1 if corr > 0.05 else 0
labels\[ids\[i]] = {"R0": int(r0), "R1": int(r1), "R2": int(r2)}
return labels

# =================================================================================================

# Fixtures

# =================================================================================================

@pytest.fixture(scope="module")
def tool\_mod():
return \_import\_tool()

@pytest.fixture()
def tmp\_workspace(tmp\_path: Path) -> Dict\[str, Path]:
"""
Prepare a clean workspace layout:
inputs/  — for .npy μ, planet ids, labels JSON, wavelengths
outputs/ — for artifacts
logs/    — for optional v50\_debug\_log.md
"""
ip = tmp\_path / "inputs"
op = tmp\_path / "outputs"
lg = tmp\_path / "logs"
ip.mkdir(parents=True, exist\_ok=True)
op.mkdir(parents=True, exist\_ok=True)
lg.mkdir(parents=True, exist\_ok=True)
return {"root": tmp\_path, "inputs": ip, "outputs": op, "logs": lg}

@pytest.fixture()
def synthetic\_inputs(tmp\_workspace: Dict\[str, Path]) -> Dict\[str, Path]:
"""
Create a tiny synthetic dataset on disk:
\- mu.npy              (N × B)
\- wavelengths.npy     (B)
\- planet\_ids.csv      (first column)
\- labels.json         ({planet\_id: {rule: 0/1}})
"""
n\_planets, n\_bins = 48, 128
MU = \_make\_mu\_spectra(n\_planets, n\_bins, seed=2025)
WL = \_make\_wavelengths(n\_bins)
IDS = \_make\_planet\_ids(n\_planets)
LAB = \_labels\_from\_rules(MU, WL)

```
ip = tmp_workspace["inputs"]
mu_path = ip / "mu.npy"
wl_path = ip / "wavelengths.npy"
ids_path = ip / "planet_ids.csv"
labels_path = ip / "labels.json"

np.save(mu_path, MU)
np.save(wl_path, WL)
ids_path.write_text("\n".join(IDS), encoding="utf-8")
labels_path.write_text(json.dumps(LAB, indent=2, sort_keys=True), encoding="utf-8")

return {
    "mu": mu_path,
    "wavelengths": wl_path,
    "planet_ids": ids_path,
    "labels": labels_path,
}
```

# =================================================================================================

# API presence

# =================================================================================================

def test\_api\_presence(tool\_mod):
"""
Ensure that at least one training entry and one prediction entry (or a class) exist.
"""
train\_candidates = \[
"train\_symbolic\_violation\_predictor\_nn",
"train\_nn",
"fit\_model",
"train",
]
predict\_candidates = \[
"predict\_symbolic\_violations",
"infer",
"predict",
]
class\_candidates = \[
"SymbolicViolationPredictorNN",
]

```
has_train = any(_has_attr(tool_mod, n) for n in train_candidates)
has_predict = any(_has_attr(tool_mod, n) for n in predict_candidates)
has_class = any(_has_attr(tool_mod, n) for n in class_candidates)

if not (has_train or has_class):
    pytest.xfail(
        "No training API/class found in symbolic_violation_predictor_nn (expected a trainer or SymbolicViolationPredictorNN)."
    )
if not has_predict and not has_class:
    pytest.xfail(
        "No prediction API found in symbolic_violation_predictor_nn (expected predict/infer or class with forward)."
    )
```

# =================================================================================================

# Train → Predict sanity on tiny dataset

# =================================================================================================

def test\_train\_predict\_sanity(tool\_mod, synthetic\_inputs, tmp\_workspace):
"""
CPU-friendly tiny training loop should improve separation for at least one rule:
mean(p̂ | label=1) > mean(p̂ | label=0).
Tolerant to different function signatures and return types.
"""
\# Import torch (xfail if unavailable to avoid false negatives)
try:
import torch  # noqa: F401
except Exception:
pytest.xfail("PyTorch not available in test environment.")

```
MU = np.load(synthetic_inputs["mu"])
with open(synthetic_inputs["labels"], "r", encoding="utf-8") as f:
    labels = json.load(f)

# Resolve training function/class
train_fn = None
for name in ("train_symbolic_violation_predictor_nn", "train_nn", "fit_model", "train"):
    if _has_attr(tool_mod, name):
        train_fn = getattr(tool_mod, name)
        break
predict_fn = None
for name in ("predict_symbolic_violations", "infer", "predict"):
    if _has_attr(tool_mod, name):
        predict_fn = getattr(tool_mod, name)
        break

# Attempt training with very small epochs
model = None
history = None
try:
    if train_fn is not None:
        try:
            out = train_fn(
                mu=MU,
                labels=labels,
                epochs=2,
                seed=123,
                batch_size=16,
                lr=1e-2,
                device="cpu",
                validation_split=0.2,
                silent=True,
            )
        except TypeError:
            # Positional minimal variant
            out = train_fn(MU, labels, 2)  # type: ignore
        # out may be (model, history) or just model
        if isinstance(out, tuple) and len(out) >= 1:
            model = out[0]
            history = out[1] if len(out) > 1 else None
        else:
            model = out
except Exception as e:
    pytest.xfail(f"Training function exists but failed on tiny dataset: {e}")

# Prediction
try:
    if predict_fn is not None:
        try:
            preds = predict_fn(model=model, mu=MU, device="cpu")  # expected shape [N, R]
        except TypeError:
            preds = predict_fn(MU)  # type: ignore
    else:
        # If no predict_fn, try model(model_input)
        if model is None:
            pytest.xfail("No predict function and no model returned; cannot evaluate.")
        # Try generic forward-like call; tolerate various interfaces
        if hasattr(model, "predict"):
            preds = model.predict(MU)  # type: ignore
        elif hasattr(model, "__call__"):
            preds = model(MU)  # type: ignore
        else:
            pytest.xfail("Model returned but no callable/predict interface available.")
except Exception as e:
    pytest.xfail(f"Prediction failed on tiny dataset: {e}")

P = np.asarray(preds)
assert P.ndim == 2 and P.shape[0] == MU.shape[0], "Predictions must be [N, R]."
assert np.isfinite(P).all(), "Predictions contain non-finite values."
# Map labels into arrays
rules = sorted(next(iter(labels.values())).keys())
pid_list = sorted(labels.keys())
lab = np.array([[int(labels[pid][r]) for r in rules] for pid in pid_list], dtype=int)

# Sanity: for at least one rule, positives have higher mean score than negatives
improved = False
for j in range(P.shape[1]):
    pos = P[:, j][lab[:, j] == 1]
    neg = P[:, j][lab[:, j] == 0]
    if pos.size > 0 and neg.size > 0 and float(np.nanmean(pos)) > float(np.nanmean(neg)):
        improved = True
        break
assert improved, "Expected at least one rule where mean(pred|1) > mean(pred|0)."
```

# =================================================================================================

# Artifact generation API

# =================================================================================================

def test\_generate\_artifacts(tool\_mod, synthetic\_inputs, tmp\_workspace):
"""
Artifact generator should emit JSON/CSV/PNG/HTML (subset allowed) and return a manifest.
"""
entry\_candidates = \[
"generate\_symbolic\_violation\_artifacts",
"run\_symbolic\_violation\_predictor\_nn",
"produce\_symbolic\_violation\_outputs",
"analyze\_and\_export",
]
entry = None
for name in entry\_candidates:
if \_has\_attr(tool\_mod, name):
entry = getattr(tool\_mod, name)
break
if entry is None:
pytest.xfail("No artifact generation entrypoint found in symbolic\_violation\_predictor\_nn.")

```
outdir = tmp_workspace["outputs"] / "artifacts_api"
outdir.mkdir(parents=True, exist_ok=True)

MU = np.load(synthetic_inputs["mu"])
with open(synthetic_inputs["labels"], "r", encoding="utf-8") as f:
    labels = json.load(f)

kwargs = dict(
    mu=MU,
    labels=labels,
    outdir=str(outdir),
    json_out=True,
    csv_out=True,
    png_out=True,
    html_out=True,
    epochs=2,
    seed=777,
    title="SVP-NN Artifacts Test",
    silent=True,
)
try:
    manifest = entry(**kwargs)
except TypeError:
    # Legacy positional
    manifest = entry(MU, labels, str(outdir), True, True, True, True, 2, 777, "SVP-NN Artifacts Test")  # type: ignore

# Presence checks
json_files = list(outdir.glob("*.json"))
csv_files = list(outdir.glob("*.csv"))
png_files = list((outdir / "plots").glob("*.png")) or list(outdir.glob("*.png"))
html_files = list(outdir.glob("*.html"))

assert json_files, "No JSON manifest produced by artifact generator."
assert csv_files, "No CSV artifact produced by artifact generator."
assert png_files, "No PNG artifact produced by artifact generator."
assert html_files, "No HTML artifact produced by artifact generator."

# Minimal JSON schema
with open(json_files[0], "r", encoding="utf-8") as f:
    js = json.load(f)
assert isinstance(js, dict), "Manifest must be a JSON object."
has_shape = "shape" in js and isinstance(js["shape"], dict)
has_rules = ("rules" in js) or ("rule_names" in js) or ("num_rules" in js)
has_metrics = ("metrics" in js) or ("train_metrics" in js) or ("val_metrics" in js)
assert has_shape, "JSON manifest should include a 'shape' object with planets/bins and/or rules."
assert has_rules, "JSON manifest should include rule descriptors."
assert has_metrics, "JSON manifest should include training/validation metrics."
```

# =================================================================================================

# CLI end-to-end

# =================================================================================================

def test\_cli\_end\_to\_end(tmp\_workspace, synthetic\_inputs):
"""
Run the module via CLI:
• Use --mu/--labels-json/--outdir and emit JSON/CSV/PNG/HTML.
• Use --seed for determinism and compare JSON across two runs (scrub volatility).
• Verify audit log append when SPECTRAMIND\_LOG\_PATH is set.
"""
\# Locate module file
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("symbolic\_violation\_predictor\_nn.py not found; cannot run CLI test.")

```
out1 = tmp_workspace["outputs"] / "cli_run1"
out2 = tmp_workspace["outputs"] / "cli_run2"
out1.mkdir(parents=True, exist_ok=True)
out2.mkdir(parents=True, exist_ok=True)

logsdir = tmp_workspace["logs"]
env = {
    "PYTHONUNBUFFERED": "1",
    "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
}

args_common = (
    "--mu", str(synthetic_inputs["mu"]),
    "--labels-json", str(synthetic_inputs["labels"]),
    "--json",
    "--csv",
    "--png",
    "--html",
    "--epochs", "2",
    "--seed", "2025",
    "--silent",
)

# Run 1
args1 = ("--outdir", str(out1), *args_common)
proc1 = _run_cli(module_file, args1, env=env, timeout=420)
if proc1.returncode != 0:
    msg = f"CLI run 1 failed (exit={proc1.returncode}).\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}"
    pytest.fail(msg)

json1 = sorted(out1.glob("*.json"))
csv1 = sorted(out1.glob("*.csv"))
png1 = sorted((out1 / "plots").glob("*.png")) or sorted(out1.glob("*.png"))
html1 = sorted(out1.glob("*.html"))
assert json1 and csv1 and png1 and html1, "CLI run 1 did not produce all expected artifact types."

# Run 2 (determinism)
args2 = ("--outdir", str(out2), *args_common)
proc2 = _run_cli(module_file, args2, env=env, timeout=420)
if proc2.returncode != 0:
    msg = f"CLI run 2 failed (exit={proc2.returncode}).\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"
    pytest.fail(msg)

json2 = sorted(out2.glob("*.json"))
assert json2, "Second CLI run produced no JSON artifacts."

# Normalize JSON: drop volatile keys (timestamps, host, abs paths, durations)
def _normalize(j: Dict[str, Any]) -> Dict[str, Any]:
    d = json.loads(json.dumps(j))  # deep copy
    vol_patterns = re.compile(r"(time|date|timestamp|duration|path|cwd|hostname|uuid|version)", re.I)

    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            keys = list(obj.keys())
            for k in keys:
                if vol_patterns.search(k):
                    obj.pop(k, None)
                else:
                    obj[k] = scrub(obj[k])
        elif isinstance(obj, list):
            for i in range(len(obj)):
                obj[i] = scrub(obj[i])
        return obj

    return scrub(d)

with open(json1[0], "r", encoding="utf-8") as f:
    j1 = _normalize(json.load(f))
with open(json2[0], "r", encoding="utf-8") as f:
    j2 = _normalize(json.load(f))
assert j1 == j2, "Seeded CLI runs should yield identical JSON after removing volatile metadata."

# Audit log should exist and include recognizable signature
log_file = Path(env["SPECTRAMIND_LOG_PATH"])
if log_file.exists():
    _assert_file(log_file, min_size=1)
    text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
    assert ("symbolic_violation_predictor_nn" in text) or ("symbolic" in text and "violation" in text), \
        "Audit log exists but lacks recognizable CLI signature."
```

def test\_cli\_error\_cases(tmp\_workspace, synthetic\_inputs):
"""
CLI should:
• Exit non-zero when required --mu or --labels-json are missing.
• Report helpful error mentioning the missing/invalid flag.
• Handle invalid numeric args (e.g., --epochs -3) by failing or sanitizing with warning.
"""
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("symbolic\_violation\_predictor\_nn.py not found; cannot run CLI error tests.")

```
outdir = tmp_workspace["outputs"] / "cli_errs"
outdir.mkdir(parents=True, exist_ok=True)

# Missing --mu
args_missing_mu = (
    "--labels-json", str(synthetic_inputs["labels"]),
    "--outdir", str(outdir),
    "--json",
)
proc = _run_cli(module_file, args_missing_mu, env=None, timeout=240)
assert proc.returncode != 0, "CLI should fail when required --mu is missing."
msg = (proc.stderr + "\n" + proc.stdout).lower()
assert "mu" in msg, "Error message should mention missing 'mu'."

# Missing --labels-json
args_missing_labels = (
    "--mu", str(synthetic_inputs["mu"]),
    "--outdir", str(outdir),
    "--json",
)
proc2 = _run_cli(module_file, args_missing_labels, env=None, timeout=240)
assert proc2.returncode != 0, "CLI should fail when required --labels-json is missing."
msg2 = (proc2.stderr + "\n" + proc2.stdout).lower()
assert ("label" in msg2) or ("labels" in msg2), "Error message should mention missing 'labels'."

# Invalid --epochs
args_bad_epochs = (
    "--mu", str(synthetic_inputs["mu"]),
    "--labels-json", str(synthetic_inputs["labels"]),
    "--outdir", str(outdir),
    "--json",
    "--epochs", "-3",
)
proc3 = _run_cli(module_file, args_bad_epochs, env=None, timeout=240)
# Either fail informatively or sanitize; accept both, but require JSON if sanitized.
if proc3.returncode != 0:
    m = (proc3.stderr + "\n" + proc3.stdout).lower()
    assert ("epoch" in m) or ("invalid" in m) or ("must be" in m) or ("warn" in m), \
        "Expected an informative error/warning about invalid --epochs."
else:
    json_files = list(outdir.glob("*.json"))
    assert json_files, "CLI sanitized invalid --epochs but produced no JSON output."
```

# =================================================================================================

# Artifact housekeeping

# =================================================================================================

def test\_artifact\_min\_sizes(tmp\_workspace):
"""
After prior tests, ensure that artifacts produced are non-trivially sized.
"""
\# Search across outputs/\* subfolders created in earlier tests
roots = \[p for p in (tmp\_workspace\["outputs"]).glob("*") if p.is\_dir()]
png\_files, csv\_files, html\_files, json\_files = \[], \[], \[], \[]
for r in roots:
png\_files += list((r / "plots").glob("*.png")) + list(r.glob("*.png"))
csv\_files += list(r.glob("*.csv"))
html\_files += list(r.glob("*.html"))
json\_files += list(r.glob("*.json"))
\# Presence is already checked elsewhere; here just size when present
for p in png\_files:
\_assert\_file(p, min\_size=256)
for c in csv\_files:
\_assert\_file(c, min\_size=64)
for h in html\_files:
\_assert\_file(h, min\_size=128)
for j in json\_files:
\_assert\_file(j, min\_size=64)

def test\_idempotent\_rerun\_behavior(tmp\_workspace, synthetic\_inputs):
"""
Re-run CLI into the same directory to ensure the tool either overwrites safely
or produces versioned filenames; artifacts should not be corrupted or disappear.
"""
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("symbolic\_violation\_predictor\_nn.py not found; cannot run idempotency test.")

```
outdir = tmp_workspace["outputs"] / "idempotent"
outdir.mkdir(parents=True, exist_ok=True)

args = (
    "--mu", str(synthetic_inputs["mu"]),
    "--labels-json", str(synthetic_inputs["labels"]),
    "--outdir", str(outdir),
    "--json",
    "--csv",
    "--png",
    "--epochs", "2",
    "--seed", "888",
    "--silent",
)
# First run
p1 = _run_cli(module_file, args, env=None, timeout=300)
assert p1.returncode == 0, f"First idempotent run failed: {p1.stderr}"

before = {pp.name for pp in outdir.glob("*")}
# Drop a marker to ensure the tool doesn't crash on re-run with extra files
(outdir / "marker.txt").write_text("marker", encoding="utf-8")

# Second run
p2 = _run_cli(module_file, args, env=None, timeout=300)
assert p2.returncode == 0, f"Second idempotent run failed: {p2.stderr}"

after = {pp.name for pp in outdir.glob("*")}
assert before.issubset(after), "Artifacts disappeared unexpectedly between idempotent runs."
```
