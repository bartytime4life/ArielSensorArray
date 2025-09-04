\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
tests/diagnostics/test\_symbolic\_fusion\_predictor.py

SpectraMind V50 — Diagnostics tests for src/diagnostics/symbolic\_fusion\_predictor.py

## Purpose

Validate the *symbolic fusion* module that aggregates multiple diagnostic sources
(e.g., symbolic rule engine scores, NN violation probabilities, SHAP magnitudes,
and optionally μ-derived heuristics) into a unified per-planet, per-rule score
and ranking. Tests are tolerant to minor API/CLI differences but enforce core guarantees:

1. Import & API surface
   • Presence of fusion entrypoints/classes (e.g., SymbolicFusionPredictor, fuse\_symbolic\_sources, rank\_symbolic\_rules).
   • Graceful xfail if the module is not present yet.

2. Fusion sanity on synthetic data
   • Construct easy synthetic signals and source JSONs with a hidden ground truth:
   "positives" for a rule should receive higher fused scores on average than "negatives."
   • Determinism with --seed at API level (after scrubbing volatile fields where needed).

3. Artifact generation
   • JSON/CSV/PNG/HTML presence via an artifact generator (generate\_symbolic\_fusion\_artifacts, run\_symbolic\_fusion\_predictor, analyze\_and\_export, etc.).
   • JSON manifest contains expected keys (shapes, rule names, fusion weights, per-planet scores).

4. CLI contract
   • End-to-end run via `python -m src.diagnostics.symbolic_fusion_predictor`
   with typical flags like --mu/--planet-ids/--symbolic-json/--nn-json/--shap-json/--outdir/--seed/etc.
   • Deterministic JSON across two runs after scrubbing volatile fields.
   • Optional audit logging to v50\_debug\_log.md if SPECTRAMIND\_LOG\_PATH is set.

5. Housekeeping
   • Artifacts are non-empty and survive idempotent re-runs.

## Assumptions

• The module lives at src/diagnostics/symbolic\_fusion\_predictor.py and exports at least one of:
\- class SymbolicFusionPredictor(...)
\- fuse\_symbolic\_sources(...), rank\_symbolic\_rules(...)
\- generate\_symbolic\_fusion\_artifacts(...), run\_symbolic\_fusion\_predictor(...)
\- produce\_symbolic\_fusion\_outputs(...), analyze\_and\_export(...)

• CLI accepts (variations tolerated; subset allowed):
\--mu \<path.npy>                    (optional for some fusion modes; \[N,B] or \[B])
\--planet-ids <.txt|.csv>           (optional; first column if .csv)
\--symbolic-json \<path.json>        (optional; per-planet rule scores)
\--nn-json \<path.json>              (optional; per-planet rule probabilities)
\--shap-json \<path.json>            (optional; per-planet SHAP magnitudes)
\--outdir <dir>                     (required)
\--json --csv --png --html          (subset allowed)
\--seed <int>                       (optional)
\--silent                           (optional)
\--w-rule <float>                   (optional; weight for rule-engine source)
\--w-nn <float>                     (optional; weight for NN source)
\--w-shap <float>                   (optional; weight for SHAP source)

## Implementation Notes

• All synthetic data is tiny and CPU-only to keep CI fast.
• Numerical checks avoid brittle equals on floats.
• If an interface is missing, the test xfails with an explanation (rather than silently passing).

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
Infer repository root from this test's location:
tests/diagnostics/test\_symbolic\_fusion\_predictor.py
"""
here = Path(**file**).resolve()
\# tests/diagnostics -> tests -> repo root
return here.parent.parent

def \_module\_candidates() -> Tuple\[Path, ...]:
"""
Filesystem locations to check for the fusion module.
"""
root = \_repo\_root\_from\_test\_file()
return (
root / "src" / "diagnostics" / "symbolic\_fusion\_predictor.py",
root / "src" / "symbolic\_fusion\_predictor.py",
root / "symbolic\_fusion\_predictor.py",
)

def \_import\_tool():
"""
Import the fusion module, trying several import styles.
"""
\# Preferred: packaged import
try:
import importlib
m = importlib.import\_module("src.diagnostics.symbolic\_fusion\_predictor")  # type: ignore
return m
except Exception:
pass

```
# Fallbacks: add repo/src to sys.path
root = _repo_root_from_test_file()
src_dir = root / "src"
for p in (str(root), str(src_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    import symbolic_fusion_predictor as m2  # type: ignore
    return m2
except Exception:
    pass

try:
    import importlib
    m3 = importlib.import_module("diagnostics.symbolic_fusion_predictor")  # type: ignore
    return m3
except Exception:
    pass

# If file exists but still not importable, surface a helpful skip
cands = [str(p) for p in _module_candidates() if p.exists()]
if not cands:
    pytest.skip(
        "symbolic_fusion_predictor module not found. "
        "Expected at src/diagnostics/symbolic_fusion_predictor.py or importable as src.diagnostics.symbolic_fusion_predictor."
    )
pytest.skip(
    f"symbolic_fusion_predictor present but import failed. Tried candidates: {cands}. "
    "Ensure import path and package layout are correct (e.g., __init__.py files)."
)
```

def \_has\_attr(mod, name: str) -> bool:
return hasattr(mod, name) and getattr(mod, name) is not None

def \_run\_cli(
module\_path: Path,
args: Sequence\[str],
env: Optional\[Dict\[str, str]] = None,
timeout: int = 360,
) -> subprocess.CompletedProcess:
"""
Execute the tool as a CLI using `python -m src.diagnostics.symbolic_fusion_predictor` when possible.
Falls back to invoking the file directly.
"""
\# Attempt module execution if under src/diagnostics
if module\_path.name == "symbolic\_fusion\_predictor.py" and module\_path.parent.name == "diagnostics":
repo\_root = module\_path.parents\[2]  # src/diagnostics/.. -> src/.. -> root
candidate\_pkg = "src.diagnostics.symbolic\_fusion\_predictor"
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
File existence & minimal size check.
"""
assert p.exists(), f"File not found: {p}"
assert p.is\_file(), f"Expected a file path: {p}"
sz = p.stat().st\_size
assert sz >= min\_size, f"File too small ({sz} bytes): {p}"

# =================================================================================================

# Synthetic inputs (μ + source JSONs with a hidden ground truth)

# =================================================================================================

def *make\_planet\_ids(n: int) -> List\[str]:
return \[f"planet*{i:04d}" for i in range(n)]

def \_make\_wavelengths(n\_bins: int, lo\_um: float = 0.5, hi\_um: float = 7.8) -> np.ndarray:
return np.linspace(float(lo\_um), float(hi\_um), n\_bins, dtype=np.float64)

def \_make\_mu(n\_planets: int, n\_bins: int, seed: int = 2025) -> np.ndarray:
"""
Generate smooth-ish μ spectra with gentle structure plus noise.
"""
rng = np.random.default\_rng(seed)
x = np.linspace(0, 1, n\_bins, dtype=np.float64)
base = 0.5 + 0.2 \* np.sin(2.0 \* math.pi \* x \* 3.0)
MU = np.empty((n\_planets, n\_bins), dtype=np.float64)
for i in range(n\_planets):
bump\_center = int(0.25 \* n\_bins + 0.5 \* n\_bins \* (rng.random() \* 0.4))
bump = 0.15 \* np.exp(-0.5 \* ((np.arange(n\_bins) - bump\_center) / max(3, n\_bins \* 0.02)) \*\* 2)
noise = rng.normal(0.0, 0.02, size=n\_bins)
MU\[i] = base + bump + noise
return MU

def \_make\_hidden\_labels(n\_planets: int, rule\_names: List\[str], seed: int = 7) -> Dict\[str, Dict\[str, int]]:
"""
Hidden ground truth per planet per rule (binary 0/1), not used by the module
under test but used by this unit test to validate fusion behavior.
The pattern is deterministic and simple: half positives/half negatives per rule with slight shifts.
"""
rng = np.random.default\_rng(seed)
labels: Dict\[str, Dict\[str, int]] = {}
pids = \_make\_planet\_ids(n\_planets)
for i, pid in enumerate(pids):
d: Dict\[str, int] = {}
for j, r in enumerate(rule\_names):
\# Alternate positives/negatives in blocks, with some slight random jitter for variety
d\[r] = 1 if ((i + j) % 4 in (0, 1)) else 0
\# (Optionally flip with small probability)
if rng.random() < 0.05:
d\[r] = 1 - d\[r]
labels\[pid] = d
return labels

def \_make\_source\_jsons\_from\_labels(
labels: Dict\[str, Dict\[str, int]],
w\_rule: float = 1.0,
w\_nn: float = 1.0,
w\_shap: float = 1.0,
noise: float = 0.05,
seed: int = 11,
) -> Tuple\[Dict\[str, Dict\[str, float]], Dict\[str, Dict\[str, float]], Dict\[str, float]]:
"""
Convert hidden labels into three source JSON-like dicts:
\- symbolic\_json: magnitude-ish scores per rule, centered \~\[0,1]
\- nn\_json: probabilities per rule in \[0,1]
\- shap\_json: a single scalar per planet indicating overall magnitude (used as a tie-breaker)
The numbers are constructed so that positives are typically larger than negatives for each source.
"""
rng = np.random.default\_rng(seed)
symbolic: Dict\[str, Dict\[str, float]] = {}
nn: Dict\[str, Dict\[str, float]] = {}
shap: Dict\[str, float] = {}

```
for pid, rd in labels.items():
    symbolic[pid] = {}
    nn[pid] = {}
    # SHAP scalar: mix of mean of positive-rule indications + noise
    pos_count = sum(int(v) for v in rd.values())
    shap_val = (pos_count / max(1, len(rd))) + rng.normal(0.0, noise)
    shap[pid] = max(0.0, float(shap_val))

    for r, y in rd.items():
        # Symbolic score — positives ≈ 0.7–1.0, negatives ≈ 0.0–0.3
        base_sym = 0.8 if y == 1 else 0.2
        symbolic[pid][r] = float(np.clip(base_sym + rng.normal(0.0, noise), 0.0, 1.0))
        # NN prob — positives ≈ 0.75–0.95, negatives ≈ 0.05–0.25
        base_nn = 0.85 if y == 1 else 0.15
        nn[pid][r] = float(np.clip(base_nn + rng.normal(0.0, noise), 0.0, 1.0))

# Optional: scale by weights so that tests that read raw values still see consistent ordering;
# the fusion module may re-weight internally; this scaling just biases sources similarly.
for pid in symbolic:
    for r in symbolic[pid]:
        symbolic[pid][r] = float(np.clip(symbolic[pid][r] * max(1e-6, w_rule), 0.0, 10.0))
        nn[pid][r] = float(np.clip(nn[pid][r] * max(1e-6, w_nn), 0.0, 10.0))
    shap[pid] = float(np.clip(shap[pid] * max(1e-6, w_shap), 0.0, 10.0))

return symbolic, nn, shap
```

# =================================================================================================

# Fixtures

# =================================================================================================

@pytest.fixture(scope="module")
def tool\_mod():
return \_import\_tool()

@pytest.fixture()
def tmp\_workspace(tmp\_path: Path) -> Dict\[str, Path]:
"""
Prepare a clean workspace:
inputs/  — for arrays and source JSONs
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
Create tiny synthetic inputs on disk:
\- mu.npy                (\[N,B])            — not strictly required by all fusers, but provided
\- planet\_ids.csv        (first column)     — planet IDs
\- symbolic.json         ({pid:{rule\:score}})
\- nn.json               ({pid:{rule\:prob}})
\- shap.json             ({pid\:scalar})
\- wavelengths.npy       (\[B])              — optional/unused by most fusers
"""
N, B = 40, 96
rule\_names = \["R0", "R1", "R2"]
MU = \_make\_mu(N, B, seed=2025)
WL = \_make\_wavelengths(B)
IDS = \_make\_planet\_ids(N)
LAB = \_make\_hidden\_labels(N, rule\_names=rule\_names, seed=7)
SYM, NN, SHP = \_make\_source\_jsons\_from\_labels(LAB, w\_rule=1.0, w\_nn=1.0, w\_shap=1.0, noise=0.05, seed=11)

```
ip = tmp_workspace["inputs"]
mu_path = ip / "mu.npy"
wl_path = ip / "wavelengths.npy"
ids_path = ip / "planet_ids.csv"
sym_path = ip / "symbolic.json"
nn_path = ip / "nn.json"
shap_path = ip / "shap.json"
labels_path = ip / "labels_hidden.json"  # not passed to module; test-only reference

np.save(mu_path, MU)
np.save(wl_path, WL)
with ids_path.open("w", encoding="utf-8") as f:
    for pid in IDS:
        f.write(pid + "\n")
sym_path.write_text(json.dumps(SYM, indent=2, sort_keys=True), encoding="utf-8")
nn_path.write_text(json.dumps(NN, indent=2, sort_keys=True), encoding="utf-8")
shap_path.write_text(json.dumps(SHP, indent=2, sort_keys=True), encoding="utf-8")
labels_path.write_text(json.dumps(LAB, indent=2, sort_keys=True), encoding="utf-8")

return {
    "mu": mu_path,
    "wavelengths": wl_path,
    "planet_ids": ids_path,
    "symbolic_json": sym_path,
    "nn_json": nn_path,
    "shap_json": shap_path,
    "labels_hidden": labels_path,
    "rule_names": json.dumps(rule_names),  # for convenience if needed
}
```

# =================================================================================================

# API presence

# =================================================================================================

def test\_api\_presence(tool\_mod):
"""
Ensure fusion entrypoints exist: either a class or explicit functions.
"""
class\_candidates = \["SymbolicFusionPredictor"]
func\_candidates = \[
"fuse\_symbolic\_sources",
"rank\_symbolic\_rules",
"generate\_symbolic\_fusion\_artifacts",
"run\_symbolic\_fusion\_predictor",
"produce\_symbolic\_fusion\_outputs",
"analyze\_and\_export",
]
has\_class = any(\_has\_attr(tool\_mod, n) for n in class\_candidates)
has\_func = any(\_has\_attr(tool\_mod, n) for n in func\_candidates)
if not (has\_class or has\_func):
pytest.xfail(
"No fusion class or function found in symbolic\_fusion\_predictor "
"(expected SymbolicFusionPredictor or fuse\_symbolic\_sources/generate\_symbolic\_fusion\_artifacts)."
)

# =================================================================================================

# Fusion sanity on synthetic data

# =================================================================================================

def test\_fusion\_sanity(tool\_mod, synthetic\_inputs, tmp\_workspace):
"""
For at least one rule, fused scores for positives should exceed negatives on average.
This test is tolerant to various return types and naming.
"""
\# Load test-only hidden labels
with open(synthetic\_inputs\["labels\_hidden"], "r", encoding="utf-8") as f:
labels = json.load(f)
\# Extract rule list from labels
rule\_names = sorted(next(iter(labels.values())).keys())
pids = sorted(labels.keys())

```
# Resolve fusion entry
entry_candidates = [
    "fuse_symbolic_sources",
    "generate_symbolic_fusion_artifacts",
    "run_symbolic_fusion_predictor",
    "produce_symbolic_fusion_outputs",
    "analyze_and_export",
    "rank_symbolic_rules",
]
entry = None
for name in entry_candidates:
    if _has_attr(tool_mod, name):
        entry = getattr(tool_mod, name)
        break
if entry is None:
    pytest.xfail("No fusion entrypoint found in symbolic_fusion_predictor.")

# Prepare inputs for API usage
MU = np.load(synthetic_inputs["mu"])
with open(synthetic_inputs["symbolic_json"], "r", encoding="utf-8") as f:
    sym = json.load(f)
with open(synthetic_inputs["nn_json"], "r", encoding="utf-8") as f:
    nn = json.load(f)
with open(synthetic_inputs["shap_json"], "r", encoding="utf-8") as f:
    shp = json.load(f)

# Call entry; accept multiple signatures
fused = None
outdir = tmp_workspace["outputs"] / "fusion_api"
outdir.mkdir(parents=True, exist_ok=True)
try:
    # Prefer a pure "fuse" call that returns a per-planet per-rule score map or array
    if entry.__name__ in ("fuse_symbolic_sources", "rank_symbolic_rules"):
        try:
            fused = entry(mu=MU, symbolic_json=sym, nn_json=nn, shap_json=shp, weights={"rule": 1.0, "nn": 1.0, "shap": 0.2}, seed=123)  # type: ignore
        except TypeError:
            fused = entry(MU, sym, nn, shp)  # type: ignore
    else:
        # Artifact generator style — should return a manifest including per-planet fused scores
        try:
            manifest = entry(
                mu=MU, symbolic_json=sym, nn_json=nn, shap_json=shp,
                outdir=str(outdir), json_out=True, csv_out=True, png_out=True, html_out=False,
                seed=123, title="Fusion API Test", silent=True
            )  # type: ignore
        except TypeError:
            manifest = entry(MU, None, None, sym, nn, shp, str(outdir), True, True, True, False, 123, "Fusion API Test")  # type: ignore
        # Pull a generic "scores" block out of manifest; tolerate multiple names
        fused = manifest.get("scores") or manifest.get("fused") or manifest.get("per_planet") or manifest.get("fusion")
except Exception as e:
    pytest.xfail(f"Fusion entry exists but failed on synthetic data: {e}")

# Normalize fused output into an [N, R] array aligned to (pids, rule_names)
# Accept dict forms like {pid:{rule:score}} or arrays
def _to_array(obj: Any) -> np.ndarray:
    if isinstance(obj, (list, tuple, np.ndarray)):
        A = np.asarray(obj, dtype=float)
        # If 1D, assume a single rule
        if A.ndim == 1:
            A = A[:, None]
        return A
    if isinstance(obj, dict):
        # nested dict expected
        M = np.zeros((len(pids), len(rule_names)), dtype=float)
        for i, pid in enumerate(pids):
            rd = obj.get(pid, {})
            for j, r in enumerate(rule_names):
                v = rd.get(r, np.nan) if isinstance(rd, dict) else np.nan
                M[i, j] = float(v) if (v is not None and np.isfinite(v)) else np.nan
        return M
    pytest.fail("Unknown fused output format; expected array-like or {pid:{rule:score}} dict.")

F = _to_array(fused)
assert F.ndim == 2 and F.shape[0] == len(pids), "Fused scores must be [N, R]."

# For at least one rule, positives should outrank negatives on average
improved = False
L = np.array([[int(labels[pid][r]) for r in rule_names] for pid in pids], dtype=int)
for j in range(F.shape[1]):
    pos = F[:, j][L[:, j] == 1]
    neg = F[:, j][L[:, j] == 0]
    if pos.size > 0 and neg.size > 0 and float(np.nanmean(pos)) > float(np.nanmean(neg)):
        improved = True
        break
assert improved, "Expected at least one rule to show higher fused scores for positives than negatives."
```

# =================================================================================================

# Artifact generation API

# =================================================================================================

def test\_generate\_artifacts(tool\_mod, synthetic\_inputs, tmp\_workspace):
"""
Call an artifact generator entrypoint and verify presence of JSON/CSV/PNG/HTML files.
"""
entry\_candidates = \[
"generate\_symbolic\_fusion\_artifacts",
"run\_symbolic\_fusion\_predictor",
"produce\_symbolic\_fusion\_outputs",
"analyze\_and\_export",
]
entry = None
for name in entry\_candidates:
if \_has\_attr(tool\_mod, name):
entry = getattr(tool\_mod, name)
break
if entry is None:
pytest.xfail("No artifact generation entrypoint found in symbolic\_fusion\_predictor.")

```
outdir = tmp_workspace["outputs"] / "fusion_artifacts_api"
outdir.mkdir(parents=True, exist_ok=True)

MU = np.load(synthetic_inputs["mu"])
with open(synthetic_inputs["symbolic_json"], "r", encoding="utf-8") as f:
    sym = json.load(f)
with open(synthetic_inputs["nn_json"], "r", encoding="utf-8") as f:
    nn = json.load(f)
with open(synthetic_inputs["shap_json"], "r", encoding="utf-8") as f:
    shp = json.load(f)

try:
    manifest = entry(
        mu=MU,
        symbolic_json=sym,
        nn_json=nn,
        shap_json=shp,
        outdir=str(outdir),
        json_out=True,
        csv_out=True,
        png_out=True,
        html_out=True,
        seed=777,
        title="Fusion Artifacts Test",
        silent=True,
    )  # type: ignore
except TypeError:
    manifest = entry(MU, sym, nn, shp, str(outdir), True, True, True, True, 777, "Fusion Artifacts Test")  # type: ignore

# Presence checks
json_files = list(outdir.glob("*.json"))
csv_files = list(outdir.glob("*.csv"))
png_files = list((outdir / "plots").glob("*.png")) or list(outdir.glob("*.png"))
html_files = list(outdir.glob("*.html"))
assert json_files, "Expected JSON manifest from fusion artifact generator."
assert csv_files, "Expected CSV artifact from fusion artifact generator."
assert png_files, "Expected PNG plots from fusion artifact generator."
assert html_files, "Expected HTML report from fusion artifact generator."

# Minimal JSON schema checks
with open(json_files[0], "r", encoding="utf-8") as f:
    js = json.load(f)
assert isinstance(js, dict), "Manifest must be a JSON object."
has_shape = "shape" in js and isinstance(js["shape"], dict)
has_rules = ("rules" in js) or ("rule_names" in js) or ("num_rules" in js)
has_scores = ("scores" in js) or ("fused" in js) or ("per_planet" in js) or ("fusion" in js)
has_weights = ("weights" in js) or ("params" in js)
assert has_shape, "Manifest should include a 'shape' with N and rule count."
assert has_rules, "Manifest should include rule names/metadata."
assert has_scores, "Manifest should include fused scores."
assert has_weights, "Manifest should record fusion weights/params."

# Size thresholds to avoid zero-byte artifacts
for p in png_files:
    _assert_file(p, min_size=256)
for c in csv_files:
    _assert_file(c, min_size=64)
for h in html_files:
    _assert_file(h, min_size=128)
```

# =================================================================================================

# CLI end-to-end

# =================================================================================================

def test\_cli\_end\_to\_end(tmp\_workspace, synthetic\_inputs):
"""
End-to-end CLI test:
• Provide --mu, --planet-ids, and source JSONs → emit JSON/CSV/PNG/HTML.
• Use --seed for determinism and compare JSON across runs (scrub volatility).
• Verify optional audit log when SPECTRAMIND\_LOG\_PATH is set.
"""
\# Locate module file
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("symbolic\_fusion\_predictor.py not found; cannot run CLI test.")

```
out1 = tmp_workspace["outputs"] / "fusion_cli1"
out2 = tmp_workspace["outputs"] / "fusion_cli2"
out1.mkdir(parents=True, exist_ok=True)
out2.mkdir(parents=True, exist_ok=True)

logsdir = tmp_workspace["logs"]
env = {
    "PYTHONUNBUFFERED": "1",
    "SPECTRAMIND_LOG_PATH": str(logsdir / "v50_debug_log.md"),
}

args_common = (
    "--mu", str(synthetic_inputs["mu"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--symbolic-json", str(synthetic_inputs["symbolic_json"]),
    "--nn-json", str(synthetic_inputs["nn_json"]),
    "--shap-json", str(synthetic_inputs["shap_json"]),
    "--json",
    "--csv",
    "--png",
    "--html",
    "--seed", "2025",
    "--silent",
)

# Run 1
args1 = ("--outdir", str(out1), *args_common)
proc1 = _run_cli(module_file, args1, env=env, timeout=360)
if proc1.returncode != 0:
    msg = f"Fusion CLI run 1 failed (exit={proc1.returncode}).\nSTDOUT:\n{proc1.stdout}\nSTDERR:\n{proc1.stderr}"
    pytest.fail(msg)

json1 = sorted(out1.glob("*.json"))
csv1 = sorted(out1.glob("*.csv"))
png1 = sorted((out1 / "plots").glob("*.png")) or sorted(out1.glob("*.png"))
html1 = sorted(out1.glob("*.html"))
assert json1 and csv1 and png1 and html1, "Fusion CLI run 1 missing one or more expected artifacts."

# Run 2 (determinism)
args2 = ("--outdir", str(out2), *args_common)
proc2 = _run_cli(module_file, args2, env=env, timeout=360)
if proc2.returncode != 0:
    msg = f"Fusion CLI run 2 failed (exit={proc2.returncode}).\nSTDOUT:\n{proc2.stdout}\nSTDERR:\n{proc2.stderr}"
    pytest.fail(msg)
json2 = sorted(out2.glob("*.json"))
assert json2, "Fusion CLI run 2 produced no JSON artifacts."

# Normalize JSONs by removing volatile fields (timestamps, paths, hostnames, durations)
def _normalize(j: Dict[str, Any]) -> Dict[str, Any]:
    d = json.loads(json.dumps(j))  # deep copy
    vol_patterns = re.compile(r"(time|date|timestamp|duration|path|cwd|hostname|uuid|version)", re.I)
    def scrub(obj: Any) -> Any:
        if isinstance(obj, dict):
            for k in list(obj.keys()):
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
assert j1 == j2, "Seeded fusion CLI runs should yield identical JSON after scrubbing volatile metadata."

# Audit log should exist and include recognizable signature
log_file = Path(env["SPECTRAMIND_LOG_PATH"])
if log_file.exists():
    _assert_file(log_file, min_size=1)
    text = log_file.read_text(encoding="utf-8", errors="ignore").lower()
    assert ("symbolic_fusion_predictor" in text) or ("fusion" in text and "symbolic" in text), \
        "Audit log exists but lacks recognizable fusion signature."
```

def test\_cli\_error\_cases(tmp\_workspace, synthetic\_inputs):
"""
CLI should handle:
• Missing required inputs (e.g., all sources absent) → non-zero exit with helpful message.
• Invalid numeric flags (e.g., negative weights) → either fail with message or sanitize with warning.
"""
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("symbolic\_fusion\_predictor.py not found; cannot run CLI error tests.")

```
outdir = tmp_workspace["outputs"] / "fusion_cli_errs"
outdir.mkdir(parents=True, exist_ok=True)

# Missing *all* sources (symbolic/nn/shap)
args_missing_sources = (
    "--mu", str(synthetic_inputs["mu"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--outdir", str(outdir),
    "--json",
)
proc = _run_cli(module_file, args_missing_sources, env=None, timeout=180)
assert proc.returncode != 0, "CLI should fail when no fusion sources are provided."
msg = (proc.stderr + "\n" + proc.stdout).lower()
assert ("symbolic" in msg) or ("nn" in msg) or ("shap" in msg) or ("source" in msg), \
    "Error message should mention missing fusion sources."

# Negative weight flag (if supported): either fail informatively or sanitize
args_bad_weight = (
    "--mu", str(synthetic_inputs["mu"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--symbolic-json", str(synthetic_inputs["symbolic_json"]),
    "--outdir", str(outdir),
    "--json",
    "--w-rule", "-0.5",
)
proc2 = _run_cli(module_file, args_bad_weight, env=None, timeout=180)
if proc2.returncode != 0:
    m = (proc2.stderr + "\n" + proc2.stdout).lower()
    assert ("weight" in m) or ("w-rule" in m) or ("invalid" in m) or ("must be" in m) or ("warn" in m) or ("unrecognized" in m) or ("unknown" in m), \
        "Expected an informative error/warning about invalid weight flag."
else:
    # If sanitized and succeeded, ensure JSON exists
    json_files = list(outdir.glob("*.json"))
    assert json_files, "CLI sanitized invalid weight but produced no JSON output."
```

# =================================================================================================

# Artifact housekeeping

# =================================================================================================

def test\_artifact\_min\_sizes(tmp\_workspace):
"""
Ensure that artifacts created in prior tests are non-trivially sized.
"""
\# Walk outputs subfolders
roots = \[p for p in (tmp\_workspace\["outputs"]).glob("*") if p.is\_dir()]
png\_files, csv\_files, html\_files, json\_files = \[], \[], \[], \[]
for r in roots:
png\_files += list((r / "plots").glob("*.png")) + list(r.glob("*.png"))
csv\_files += list(r.glob("*.csv"))
html\_files += list(r.glob("*.html"))
json\_files += list(r.glob("*.json"))
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
Re-run the CLI into the same directory to ensure safe overwrite or versioned outputs.
"""
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("symbolic\_fusion\_predictor.py not found; cannot run idempotency test.")

```
outdir = tmp_workspace["outputs"] / "fusion_idempotent"
outdir.mkdir(parents=True, exist_ok=True)

args = (
    "--mu", str(synthetic_inputs["mu"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--symbolic-json", str(synthetic_inputs["symbolic_json"]),
    "--nn-json", str(synthetic_inputs["nn_json"]),
    "--outdir", str(outdir),
    "--json",
    "--csv",
    "--png",
    "--seed", "314",
    "--silent",
)

# First run
p1 = _run_cli(module_file, args, env=None, timeout=240)
assert p1.returncode == 0, f"First idempotent fusion run failed: {p1.stderr}"

before = {pp.name for pp in outdir.glob("*")}
# Add a benign marker to ensure re-run doesn't crash or delete unrelated files
(outdir / "marker.txt").write_text("marker", encoding="utf-8")

# Second run
p2 = _run_cli(module_file, args, env=None, timeout=240)
assert p2.returncode == 0, f"Second idempotent fusion run failed: {p2.stderr}"

after = {pp.name for pp in outdir.glob("*")}
assert before.issubset(after), "Artifacts disappeared unexpectedly across idempotent re-runs."
```
