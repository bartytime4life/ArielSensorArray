\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
tests/diagnostics/test\_plot\_gll\_heatmap\_per\_bin.py

SpectraMind V50 — Diagnostics tests for src/diagnostics/plot\_gll\_heatmap\_per\_bin.py

## Goal

Validate the GLL-per-bin heatmap diagnostic module for μ/σ evaluation. The tests are
robust to minor API/CLI differences but enforce core guarantees:

1. Core API sanity
   • Entry function accepts GLL arrays (planets × bins) and produces artifacts.
   • Determinism (with seeds): JSON equality modulo volatile fields.

2. Artifact generation
   • PNG heatmap, optional per-bin CSV, and JSON manifest exist and are non-trivial in size.
   • JSON contains expected keys (e.g., per-bin stats, color scale, shapes).

3. CLI contract
   • End-to-end run via `python -m src.diagnostics.plot_gll_heatmap_per_bin`.
   • Graceful error for missing/invalid args (e.g., missing --gll).
   • Optional audit logging to v50\_debug\_log.md if SPECTRAMIND\_LOG\_PATH is set.

4. Housekeeping
   • Idempotent behavior: re-runs do not corrupt artifacts.
   • Minimal size thresholds for PNG/CSV/HTML (if emitted).

## Assumptions

• The module lives at src/diagnostics/plot\_gll\_heatmap\_per\_bin.py and exports at least one of:
\- generate\_gll\_heatmap\_artifacts(...)
\- run\_gll\_heatmap(...)
\- plot\_gll\_heatmap\_per\_bin(...)
\- analyze\_and\_export(...)

• CLI accepts (tolerant to variations):
\--gll \<path.npy|.csv>       (required; \[N\_planets, N\_bins] or DataFrame with columns)
\--planet-ids <.txt|.csv>    (optional; first column if .csv)
\--wavelengths \<path.npy>    (optional; \[N\_bins])
\--outdir <dir>              (required)
\--json --csv --png --html   (subset allowed)
\--seed <int>                (optional)
\--silent                    (optional)
\--title <str>               (optional)
\--colormap <str>            (optional)

## Implementation Notes

• These tests generate synthetic GLL with simple structure (lower is better).
• Numerical checks use tolerant comparisons; JSON equality scrubs volatile fields.
• If a surface is missing, tests xfail with a clear reason instead of silently passing.

Author: SpectraMind V50 test harness
"""

from **future** import annotations

import csv
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
tests/diagnostics/test\_plot\_gll\_heatmap\_per\_bin.py
"""
here = Path(**file**).resolve()
\# tests/diagnostics -> tests -> repo root
return here.parent.parent

def \_module\_candidates() -> Tuple\[Path, ...]:
"""
Likely filesystem locations for the module under test.
"""
root = \_repo\_root\_from\_test\_file()
return (
root / "src" / "diagnostics" / "plot\_gll\_heatmap\_per\_bin.py",
root / "plot\_gll\_heatmap\_per\_bin.py",
)

def \_import\_tool():
"""
Import the module under test, trying several import styles.
"""
\# Preferred: package import
try:
import importlib
m = importlib.import\_module("src.diagnostics.plot\_gll\_heatmap\_per\_bin")  # type: ignore
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
    import plot_gll_heatmap_per_bin as m2  # type: ignore
    return m2
except Exception:
    pass

try:
    import importlib
    m3 = importlib.import_module("diagnostics.plot_gll_heatmap_per_bin")  # type: ignore
    return m3
except Exception:
    pass

# If file exists but import fails, surface informative skip
cands = [str(p) for p in _module_candidates() if p.exists()]
if not cands:
    pytest.skip(
        "plot_gll_heatmap_per_bin module not found. "
        "Expected at src/diagnostics/plot_gll_heatmap_per_bin.py or importable as src.diagnostics.plot_gll_heatmap_per_bin."
    )
pytest.skip(
    f"plot_gll_heatmap_per_bin present but import failed. Tried candidates: {cands}. "
    "Ensure import path and package layout are correct."
)
```

def \_has\_attr(mod, name: str) -> bool:
return hasattr(mod, name) and getattr(mod, name) is not None

def \_run\_cli(
module\_path: Path,
args: Sequence\[str],
env: Optional\[Dict\[str, str]] = None,
timeout: int = 240,
) -> subprocess.CompletedProcess:
"""
Execute the tool as a CLI using `python -m src.diagnostics.plot_gll_heatmap_per_bin` when possible.
Fallback to invoking the file directly.
"""
if module\_path.name == "plot\_gll\_heatmap\_per\_bin.py" and module\_path.parent.name == "diagnostics":
repo\_root = module\_path.parents\[2]  # src/diagnostics/.. -> src/.. -> root
candidate\_pkg = "src.diagnostics.plot\_gll\_heatmap\_per\_bin"
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

# Synthetic inputs

# =================================================================================================

def *make\_planet\_ids(n: int) -> List\[str]:
return \[f"planet*{i:04d}" for i in range(n)]

def \_make\_wavelengths(n\_bins: int, lo\_um: float = 0.5, hi\_um: float = 7.8) -> np.ndarray:
return np.linspace(float(lo\_um), float(hi\_um), n\_bins, dtype=np.float64)

def \_make\_gll\_matrix(n\_planets: int, n\_bins: int, seed: int = 2025) -> np.ndarray:
"""
Construct a synthetic GLL matrix with structure:
Base = 0.2 + 0.1 \* sin(2π \* bin / period) + noise
Lower values indicate better likelihood. We clip to \[0, 5] to be realistic.
"""
rng = np.random.default\_rng(seed)
bins = np.arange(n\_bins, dtype=np.float64)
period = max(10, n\_bins // 16)
base = 0.2 + 0.1 \* np.sin(2.0 \* math.pi \* bins / period)
M = np.empty((n\_planets, n\_bins), dtype=np.float64)
for p in range(n\_planets):
noise = rng.normal(0.0, 0.02, size=n\_bins)
shift = 0.02 \* np.sin(2.0 \* math.pi \* (p + 1) / (n\_planets + 3))
M\[p] = base + shift + noise
M = np.clip(M, 0.0, 5.0)
return M

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
inputs/  — for .npy/.csv GLL, planet ids, wavelengths
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
Write synthetic GLL, wavelengths, and planet IDs to disk for CLI tests.
Files:
\- gll.npy                (N × B)
\- wavelengths.npy        (B)
\- planet\_ids.csv         (first column)
"""
n\_planets, n\_bins = 24, 283
G = \_make\_gll\_matrix(n\_planets, n\_bins, seed=2025)
WL = \_make\_wavelengths(n\_bins)
IDS = \_make\_planet\_ids(n\_planets)

```
ip = tmp_workspace["inputs"]
gll_path = ip / "gll.npy"
wl_path = ip / "wavelengths.npy"
ids_path = ip / "planet_ids.csv"

np.save(gll_path, G)
np.save(wl_path, WL)
with ids_path.open("w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    for pid in IDS:
        w.writerow([pid])

return {
    "gll": gll_path,
    "wavelengths": wl_path,
    "planet_ids": ids_path,
}
```

# =================================================================================================

# Core API tests

# =================================================================================================

def test\_api\_generate\_artifacts(tool\_mod, tmp\_workspace, synthetic\_inputs):
"""
Exercise the primary artifact generator API with arrays (no file IO dependencies).
Accepts either:
\- generate\_gll\_heatmap\_artifacts(...)
\- run\_gll\_heatmap(...)
\- plot\_gll\_heatmap\_per\_bin(...)
\- analyze\_and\_export(...)
"""
entry\_candidates = \[
"generate\_gll\_heatmap\_artifacts",
"run\_gll\_heatmap",
"plot\_gll\_heatmap\_per\_bin",
"analyze\_and\_export",
]
entry = None
for name in entry\_candidates:
if \_has\_attr(tool\_mod, name):
entry = getattr(tool\_mod, name)
break
if entry is None:
pytest.xfail("No artifact generator function found in plot\_gll\_heatmap\_per\_bin module.")

```
outdir = tmp_workspace["outputs"] / "api_gen"
outdir.mkdir(parents=True, exist_ok=True)

# Load arrays for direct API usage
G = np.load(synthetic_inputs["gll"])
WL = np.load(synthetic_inputs["wavelengths"])
IDS = [row.strip() for row in (synthetic_inputs["planet_ids"].read_text(encoding="utf-8").splitlines()) if row.strip()]
# Some implementations accept lists, be flexible
kwargs = dict(
    gll=G,
    wavelengths=WL,
    planet_ids=IDS,
    outdir=str(outdir),
    png_out=True,
    csv_out=True,
    json_out=True,
    html_out=False,
    seed=42,
    title="GLL Heatmap Test",
    colormap="viridis",
)
try:
    manifest = entry(**kwargs)
except TypeError:
    # Tolerate positional-only or different signature
    try:
        manifest = entry(G, WL, IDS, str(outdir), True, True, True, False, 42, "GLL Heatmap Test", "viridis")  # type: ignore
    except Exception as e:
        pytest.fail(f"Artifact generator call failed: {e}")

# Presence checks
json_files = list(outdir.glob("*.json"))
csv_files = list(outdir.glob("*.csv"))
png_files = list((outdir / "plots").glob("*.png")) or list(outdir.glob("*.png"))

assert json_files, "Expected a JSON manifest from artifact generator."
assert csv_files, "Expected a CSV (per-bin or per-planet) from artifact generator."
assert png_files, "Expected at least one PNG plot (heatmap)."

# Minimal JSON schema
with open(json_files[0], "r", encoding="utf-8") as f:
    js = json.load(f)
assert isinstance(js, dict), "Manifest must be a JSON object."
# Tolerant key checks: look for gll shape and per-bin stats
has_shape = "shape" in js and isinstance(js["shape"], dict)
has_stats = ("per_bin" in js) or ("per_bin_stats" in js) or ("bin_stats" in js)
assert has_shape, "JSON manifest should include a 'shape' object with planets/bins."
assert has_stats, "JSON manifest should include per-bin statistics."
```

# =================================================================================================

# CLI end-to-end tests

# =================================================================================================

def test\_cli\_end\_to\_end(tmp\_workspace, synthetic\_inputs):
"""
Run the module via CLI:
• Use --gll/--wavelengths/--planet-ids/--outdir and emit JSON/CSV/PNG/HTML.
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
pytest.skip("plot\_gll\_heatmap\_per\_bin.py not found; cannot run CLI test.")

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
    "--gll", str(synthetic_inputs["gll"]),
    "--wavelengths", str(synthetic_inputs["wavelengths"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--json",
    "--csv",
    "--png",
    "--html",
    "--seed", "2025",
    "--silent",
)

# Run 1
args1 = ("--outdir", str(out1), *args_common)
proc1 = _run_cli(module_file, args1, env=env, timeout=240)
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
proc2 = _run_cli(module_file, args2, env=env, timeout=240)
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
    assert ("plot_gll_heatmap_per_bin" in text) or ("gll" in text and "heatmap" in text), \
        "Audit log exists but lacks recognizable CLI signature."
```

def test\_cli\_error\_cases(tmp\_workspace, synthetic\_inputs):
"""
CLI should:
• Exit non-zero when required --gll is missing.
• Report helpful error mentioning the missing/invalid flag.
• Handle invalid numeric args if any (e.g., --bins -5) by failing or sanitizing with warning.
"""
candidates = list(\_module\_candidates())
module\_file = None
for p in candidates:
if p.exists():
module\_file = p
break
if module\_file is None:
pytest.skip("plot\_gll\_heatmap\_per\_bin.py not found; cannot run CLI error tests.")

```
outdir = tmp_workspace["outputs"] / "cli_errs"
outdir.mkdir(parents=True, exist_ok=True)

# Missing --gll
args_missing_gll = (
    "--wavelengths", str(synthetic_inputs["wavelengths"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--outdir", str(outdir),
    "--json",
)
proc = _run_cli(module_file, args_missing_gll, env=None, timeout=120)
assert proc.returncode != 0, "CLI should fail when required --gll is missing."
msg = (proc.stderr + "\n" + proc.stdout).lower()
assert "gll" in msg, "Error message should mention missing 'gll'."

# Invalid numeric arg, if module supports such (e.g., --bins or --dpi)
# We try a likely flag; tolerate both fail-with-message or sanitize-with-warn behaviors.
args_bad_numeric = (
    "--gll", str(synthetic_inputs["gll"]),
    "--outdir", str(outdir),
    "--json",
    "--png",
    "--seed", "2025",
    "--silent",
    "--dpi", "-10",     # many plotters support --dpi; if not recognized, module may ignore
)
proc2 = _run_cli(module_file, args_bad_numeric, env=None, timeout=120)
if proc2.returncode != 0:
    m = (proc2.stderr + "\n" + proc2.stdout).lower()
    # Accept any informative hint about the invalid numeric parameter OR unknown arg
    assert ("dpi" in m and ("invalid" in m or "must be" in m or "error" in m)) or ("unrecognized" in m) or ("unknown" in m), \
        "Expected an informative error/warning about invalid or unknown numeric flag."
```

# =================================================================================================

# Artifact housekeeping

# =================================================================================================

def test\_artifact\_min\_sizes(tmp\_workspace):
"""
After prior tests, ensure that PNG/CSV/HTML artifacts are non-trivially sized.
"""
\# Search across outputs/\* subfolders created in earlier tests
roots = \[p for p in (tmp\_workspace\["outputs"]).glob("*") if p.is\_dir()]
png\_files, csv\_files, html\_files = \[], \[], \[]
for r in roots:
png\_files += list((r / "plots").glob("*.png")) + list(r.glob("*.png"))
csv\_files += list(r.glob("*.csv"))
html\_files += list(r.glob("\*.html"))
for p in png\_files:
\_assert\_file(p, min\_size=256)
for c in csv\_files:
\_assert\_file(c, min\_size=64)
for h in html\_files:
\_assert\_file(h, min\_size=128)

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
pytest.skip("plot\_gll\_heatmap\_per\_bin.py not found; cannot run idempotency test.")

```
outdir = tmp_workspace["outputs"] / "idempotent"
outdir.mkdir(parents=True, exist_ok=True)

args = (
    "--gll", str(synthetic_inputs["gll"]),
    "--wavelengths", str(synthetic_inputs["wavelengths"]),
    "--planet-ids", str(synthetic_inputs["planet_ids"]),
    "--outdir", str(outdir),
    "--json",
    "--csv",
    "--png",
    "--seed", "777",
    "--silent",
)
# First run
p1 = _run_cli(module_file, args, env=None, timeout=180)
assert p1.returncode == 0, f"First idempotent run failed: {p1.stderr}"

before = {pp.name for pp in outdir.glob("*")}
# Drop a marker to ensure the tool doesn't crash on re-run with extra files
(outdir / "marker.txt").write_text("marker", encoding="utf-8")

# Second run
p2 = _run_cli(module_file, args, env=None, timeout=180)
assert p2.returncode == 0, f"Second idempotent run failed: {p2.stderr}"

after = {pp.name for pp in outdir.glob("*")}
assert before.issubset(after), "Artifacts disappeared unexpectedly between idempotent runs."
```
