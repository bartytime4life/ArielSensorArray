\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
SpectraMind V50 — Symbolic-Aware Ablation Engine (Upgraded, Challenge-Grade)
============================================================================

## Purpose

Run a *deterministic, resumable, parallel* set of ablations over your Hydra configs
with a focus on **symbolic loss knobs** and **diagnostics quality**. It executes
a base command (e.g., `spectramind train` or `spectramind submit`) with mutated
overrides, collects metrics from diagnostics JSONs, ranks results, and exports:
• CSV / Markdown / HTML / JSON leaderboards
• Optional Top-N ZIP
• JSONL event stream for CI and post-hoc analysis
• Full run manifest with config hashes for reproducibility

## Design Highlights

• CLI-first (Typer) + Rich progress UI
• Two ablation modes: *one-at-a-time* or *cartesian*
• Deterministic short IDs, per-run seed injection, `hydra.run.dir` isolation
• Resume support (`.done` flag) + retries + parallelism + graceful SIGINT
• Timeouts per run + hard kill of process trees (no zombie jobs)
• Filters: include/exclude by glob to constrain the run set
• Metrics ingestion preference order:
1\) run\_dir/diagnostic\_summary.json
2\) run\_dir/metrics.json
3\) recursive fallback (first match)
• Scientific metrics (if provided by pipeline):
\- gll\_mean, rmse\_mean, mae\_mean
\- coverage\_p95, z\_abs\_mean, z\_abs\_std
\- fft\_hf\_fraction\_mean, symbolic\_agg\_mean
• Leaderboard exports: CSV, MD, HTML, JSON; optional Top-N ZIP bundle
• Events: events.jsonl (state machine for each run; ready for CI ingestion)
• Manifest + per-run `overrides.txt` + `run_spec.json` for full reproducibility
• Optional DVC tracking of the ablation output directory (one command toggle)

## Quick Start

# 1) Run an ablation with built-in *symbolic loss* sweep (one-at-a-time)

python auto\_ablate\_v50.py&#x20;
\--command "spectramind train"&#x20;
\--base "data=nominal" --base "trainer=default"&#x20;
\--preset loss-sweep&#x20;
\--parallel 2 --retries 1

# 2) Provide a YAML spec for custom keys and values (see format below)

python auto\_ablate\_v50.py&#x20;
\--command "spectramind train"&#x20;
\--grid-yaml grids/ablate\_v50.yaml&#x20;
\--mode one-at-a-time&#x20;
\--parallel 4

# 3) Dry-run to preview runs

python auto\_ablate\_v50.py --dry-run --preset loss-sweep

# 4) Resume a partially completed ablation

python auto\_ablate\_v50.py --preset loss-sweep --resume

# 5) Rank by RMSE ascending and export Top-5 bundle

python auto\_ablate\_v50.py --preset loss-sweep --sort-key rmse\_mean --asc --top-n 5

# 6) Track the ablation directory in DVC for reproducibility

python auto\_ablate\_v50.py --preset loss-sweep --dvc-track

## YAML Grid Format

# grids/ablate\_v50.yaml

command: "spectramind train"
base\_overrides:

* "data=nominal"
* "trainer=default"
  ablate:

# Choose generation mode: "one-at-a-time" (default) or "cartesian"

mode: "one-at-a-time"

# Explicit sets to mutate:

sets:
\- key: "losses.symbolic.smooth\_w"
values: \[0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
\- key: "losses.symbolic.nonneg\_w"
values: \[0.0, 0.05, 0.1, 0.2]
\- key: "losses.symbolic.asym\_w"
values: \[0.0, 0.05, 0.1]
\- key: "losses.symbolic.fft\_w"
values: \[0.0, 0.02, 0.05, 0.1]
options:
name: "v50\_symbolic\_ablate"
retries: 1
parallel: 2

## License

MIT
"""

from **future** import annotations

import csv
import fnmatch
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as\_completed
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn
from rich.table import Table
from rich import box

try:
import yaml
except Exception:
yaml = None  # YAML support optional unless you pass --grid-yaml

# -----------------------------------------------------------------------------

# Typer CLI app and console (Rich)

# -----------------------------------------------------------------------------

app = typer.Typer(add\_completion=False, no\_args\_is\_help=True)
console = Console()

# -----------------------------------------------------------------------------

# Utilities (deterministic hashing, paths, shell, time)

# -----------------------------------------------------------------------------

def now\_ts() -> str:
"""UTC timestamp string for filenames/logs."""
return time.strftime("%Y%m%d-%H%M%S", time.gmtime())

def ensure\_dir(p: Path) -> Path:
"""Create directory if missing; return the Path."""
p.mkdir(parents=True, exist\_ok=True)
return p

def shell\_join(parts: Sequence\[str]) -> str:
"""
Safely join argv parts into a shell-visible line for logging/exporting.
Avoid using it to *execute*; keep it for display or sh scripts only.
"""
return " ".join(shlex.quote(p) for p in parts)

def short\_hash(items: Sequence\[str], salt: str) -> str:
"""
Stable, deterministic 8-byte hash (hex) from ordered items + salt.
Used to build short run IDs and grid IDs.
"""
h = blake2b(digest\_size=8)
for s in items:
h.update(str(s).encode("utf-8"))
h.update(b"\x00")
h.update(salt.encode("utf-8"))
return h.hexdigest()

def to\_seed(h: str) -> int:
"""
Convert short hex hash to a 31-bit deterministic seed.
"""
return int(h, 16) % (2\*\*31 - 1)

def load\_yaml(p: Optional\[Path]) -> dict:
"""Load a YAML file (or return empty dict)."""
if p is None:
return {}
if yaml is None:
raise typer.BadParameter("PyYAML not installed; cannot use --grid-yaml.")
with p.open("r", encoding="utf-8") as f:
return yaml.safe\_load(f) or {}

def glob\_one(p: Path, patterns: Sequence\[str]) -> Optional\[Path]:
"""Return the first found path that matches patterns within p."""
for pat in patterns:
for q in p.glob(pat):
return q
return None

def which(cmd: str) -> Optional\[str]:
"""Simple 'which' to check external tools availability."""
for path\_dir in os.environ.get("PATH", "").split(os.pathsep):
candidate = Path(path\_dir) / cmd
if candidate.is\_file() and os.access(candidate, os.X\_OK):
return str(candidate)
return None

def kill\_process\_tree(pid: int) -> None:
"""
Best-effort terminate and kill a process tree (POSIX).
On Windows, falls back to taskkill if available.
"""
try:
if os.name == "nt":
\# /T = kill child processes, /F = force
subprocess.run(\["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
else:
\# Send SIGTERM to the whole process group, then SIGKILL
os.killpg(pid, signal.SIGTERM)
time.sleep(1.0)
with contextlib.suppress(Exception):
os.killpg(pid, signal.SIGKILL)
except Exception:
pass

# -----------------------------------------------------------------------------

# Built-in presets (symbolic-focused defaults)

# -----------------------------------------------------------------------------

PRESET\_SETS: Dict\[str, List\[Tuple\[str, List\[str]]]] = {
\# Symbolic loss weight sweep; safe ranges to explore O(10) trials
"loss-sweep": \[
("losses.symbolic.smooth\_w", \[0.0, 0.02, 0.05, 0.1, 0.2]),
("losses.symbolic.nonneg\_w", \[0.0, 0.02, 0.05, 0.1]),
("losses.symbolic.asym\_w",   \[0.0, 0.02, 0.05, 0.1]),
("losses.symbolic.fft\_w",    \[0.0, 0.01, 0.02, 0.05]),
],
\# Module toggles (boolean switches)
"toggle-symbolic": \[
("modules.symbolic.enable", \["false", "true"]),
("modules.smooth.enable",   \["false", "true"]),
("modules.fft.enable",      \["false", "true"]),
],
\# Curriculum stages
"curriculum": \[
("training.curriculum.stage", \["none", "warmup", "full"]),
],
}

# -----------------------------------------------------------------------------

# Run specification and grid construction

# -----------------------------------------------------------------------------

@dataclass
class RunSpec:
index: int
id\_short: str
base\_cmd: str
base\_overrides: List\[str]
combo\_overrides: List\[str]
run\_dir: Path
seed: int
retries: int
timeout\_sec: Optional\[int] = None

```
def hydra_overrides(self) -> List[str]:
    """
    Build Hydra overrides list, ensuring deterministic seed and per-run hydra.run.dir isolation.
    """
    ov = list(self.base_overrides) + list(self.combo_overrides)
    has_seed = any("=" in x and x.split("=", 1)[0] == "seed" for x in ov)
    if not has_seed:
        ov.append(f"seed={self.seed}")
    ov.append(f"hydra.run.dir={str(self.run_dir)}")
    return ov
```

def one\_at\_a\_time(sets: List\[Tuple\[str, List\[str]]]) -> List\[List\[str]]:
"""Generate runs mutating one key at a time, keeping others default."""
combos: List\[List\[str]] = \[]
for k, vals in sets:
for v in vals:
combos.append(\[f"{k}={v}"])
return combos

def cartesian(sets: List\[Tuple\[str, List\[str]]]) -> List\[List\[str]]:
"""Cartesian product of sets; each element is a list of k=v overrides."""
if not sets:
return \[\[]]
keys = \[k for k, \_ in sets]
value\_lists = \[vals for \_, vals in sets]
out: List\[List\[str]] = \[]
def rec(i: int, acc: List\[str]):
if i == len(keys):
out.append(list(acc))
return
for v in value\_lists\[i]:
acc.append(f"{keys\[i]}={v}")
rec(i+1, acc)
acc.pop()
rec(0, \[])
return out

def filter\_combos(
combos: List\[List\[str]],
include\_globs: List\[str],
exclude\_globs: List\[str],
) -> List\[List\[str]]:
"""
Filter combos by include/exclude glob patterns applied to the string
'k1=v1,k2=v2,...'. If include is given, only those matching any pattern are kept.
Exclude always removes matches.
"""
if not include\_globs and not exclude\_globs:
return combos

```
kept: List[List[str]] = []
for c in combos:
    s = ",".join(c)
    inc_ok = True
    if include_globs:
        inc_ok = any(fnmatch.fnmatch(s, pat) for pat in include_globs)
    exc_ok = not any(fnmatch.fnmatch(s, pat) for pat in exclude_globs)
    if inc_ok and exc_ok:
        kept.append(c)
return kept
```

# -----------------------------------------------------------------------------

# Subprocess execution with logging, retries, and timeouts

# -----------------------------------------------------------------------------

def write\_text(path: Path, text: str) -> None:
path.parent.mkdir(parents=True, exist\_ok=True)
path.write\_text(text, encoding="utf-8")

def append\_jsonl(path: Path, record: dict) -> None:
path.parent.mkdir(parents=True, exist\_ok=True)
with path.open("a", encoding="utf-8") as f:
f.write(json.dumps(record, ensure\_ascii=False) + "\n")

def run\_once(
spec: RunSpec,
extra\_env: Dict\[str, str],
dry: bool,
events\_path: Optional\[Path] = None,
) -> Tuple\[int, str]:
"""
Execute one run spec once:
• writes per-run files (overrides.txt, run\_spec.json, command.sh)
• runs subprocess with optional timeout
• logs stdout/stderr to file
• emits JSONL event records (start/end/error)
"""
ensure\_dir(spec.run\_dir)

```
# Persist run recipe and spec
overrides_txt = "\n".join(spec.hydra_overrides())
write_text(spec.run_dir / "overrides.txt", overrides_txt)
write_text(
    spec.run_dir / "run_spec.json",
    json.dumps(
        {
            "index": spec.index,
            "id_short": spec.id_short,
            "seed": spec.seed,
            "base_cmd": spec.base_cmd,
            "base_overrides": spec.base_overrides,
            "combo_overrides": spec.combo_overrides,
            "timeout_sec": spec.timeout_sec,
        },
        indent=2,
    ),
)

done_flag = spec.run_dir / ".done"
if done_flag.exists():
    return (0, "skipped (done)")

parts = [sys.executable, "-m"] + spec.base_cmd.split() + spec.hydra_overrides()
cmd_str = shell_join(parts)
cmd_sh = "#!/usr/bin/env bash\nset -euo pipefail\n" + cmd_str + "\n"
script_path = spec.run_dir / "command.sh"
write_text(script_path, cmd_sh)
os.chmod(script_path, 0o755)

if dry:
    if events_path:
        append_jsonl(
            events_path,
            {
                "ts": now_ts(),
                "run_dir": str(spec.run_dir),
                "id": spec.id_short,
                "event": "dry_run",
                "cmd": parts,
            },
        )
    return (0, "dry-run")

# Start process in its own group for robust termination
env = os.environ.copy()
env.update(extra_env)
log_path = spec.run_dir / "stdout_stderr.log"

if events_path:
    append_jsonl(
        events_path,
        {
            "ts": now_ts(),
            "run_dir": str(spec.run_dir),
            "id": spec.id_short,
            "event": "start",
            "cmd": parts,
        },
    )

# Launch
preexec = None
creationflags = 0
if os.name != "nt":
    preexec = os.setsid  # create new process group on POSIX

with log_path.open("wb") as logf:
    try:
        proc = subprocess.Popen(
            parts,
            stdout=logf,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=preexec,
            creationflags=creationflags,
        )
        # Wait with optional timeout
        if spec.timeout_sec and spec.timeout_sec > 0:
            deadline = time.time() + spec.timeout_sec
            while True:
                rc = proc.poll()
                if rc is not None:
                    break
                if time.time() > deadline:
                    # timeout -> kill process tree
                    if events_path:
                        append_jsonl(
                            events_path,
                            {
                                "ts": now_ts(),
                                "run_dir": str(spec.run_dir),
                                "id": spec.id_short,
                                "event": "timeout",
                                "timeout_sec": spec.timeout_sec,
                            },
                        )
                    try:
                        if os.name == "nt":
                            subprocess.run(
                                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                                stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL,
                            )
                        else:
                            os.killpg(proc.pid, signal.SIGTERM)
                            time.sleep(1.0)
                            with contextlib.suppress(Exception):
                                os.killpg(proc.pid, signal.SIGKILL)
                    except Exception:
                        pass
                    return (124, f"timeout (>{spec.timeout_sec}s)")
                time.sleep(0.25)
            rc = proc.returncode
        else:
            rc = proc.wait()
    except Exception as e:
        if events_path:
            append_jsonl(
                events_path,
                {
                    "ts": now_ts(),
                    "run_dir": str(spec.run_dir),
                    "id": spec.id_short,
                    "event": "exception",
                    "error": repr(e),
                },
            )
        return (98, f"exception: {e}")

# Success path
if rc == 0:
    write_text(done_flag, now_ts())
    if events_path:
        append_jsonl(
            events_path,
            {
                "ts": now_ts(),
                "run_dir": str(spec.run_dir),
                "id": spec.id_short,
                "event": "done",
                "rc": rc,
            },
        )
    return (0, "ok")

# Failure path
if events_path:
    append_jsonl(
        events_path,
        {
            "ts": now_ts(),
            "run_dir": str(spec.run_dir),
            "id": spec.id_short,
            "event": "error",
            "rc": rc,
        },
    )
return (rc, f"fail (rc={rc})")
```

def run\_with\_retries(
spec: RunSpec,
extra\_env: Dict\[str, str],
dry: bool,
max\_retries: int,
backoff\_sec: float,
events\_path: Optional\[Path] = None,
) -> Tuple\[int, str]:
"""
Retry wrapper over run\_once with linear backoff and cap.
"""
attempts = max(0, max\_retries) + 1
for i in range(attempts):
rc, status = run\_once(spec, extra\_env, dry=dry, events\_path=events\_path)
if rc == 0:
return rc, status
if i < attempts - 1:
time.sleep(min(backoff\_sec \* (i + 1), 30.0))
return rc, status

# -----------------------------------------------------------------------------

# Metrics parsing and leaderboard assembly

# -----------------------------------------------------------------------------

METRIC\_KEYS = \[
"gll\_mean", "rmse\_mean", "mae\_mean",
"coverage\_p95", "z\_abs\_mean", "z\_abs\_std",
"fft\_hf\_fraction\_mean",
"symbolic\_agg\_mean",
]

def read\_json(path: Path) -> Optional\[dict]:
try:
return json.loads(path.read\_text(encoding="utf-8"))
except Exception:
return None

def discover\_metrics(run\_dir: Path) -> Dict\[str, Optional\[float]]:
"""
Heuristic: prefer diagnostic\_summary.json, fallback to metrics.json.
Also search recursively to catch subtool outputs.
"""
out: Dict\[str, Optional\[float]] = {k: None for k in METRIC\_KEYS}

```
# Preferred locations
candidates: List[Path] = []
for p in (run_dir / "diagnostic_summary.json", run_dir / "metrics.json"):
    if p.exists():
        candidates.append(p)

# Recursive fallback
if not candidates:
    for root, _, files in os.walk(run_dir):
        for name in files:
            if name in ("diagnostic_summary.json", "metrics.json"):
                candidates.append(Path(root) / name)
        if candidates:
            break

def _set_from_obj(obj: dict):
    # Compact metrics.json
    if "gll_mean" in obj:
        for k in METRIC_KEYS:
            if k in obj and isinstance(obj[k], (int, float)):
                out[k] = float(obj[k])
    # Rich summary structure
    if "metrics" in obj and isinstance(obj["metrics"], dict):
        m = obj["metrics"]
        for k in ("gll_mean", "rmse_mean", "mae_mean", "coverage_p95", "z_abs_mean", "z_abs_std"):
            if k in m and isinstance(m[k], (int, float)):
                out[k] = float(m[k])
    # Optional top-level probes
    if "fft_hf_fraction_mean" in obj and isinstance(obj["fft_hf_fraction_mean"], (int, float)):
        out["fft_hf_fraction_mean"] = float(obj["fft_hf_fraction_mean"])
    if "symbolic_agg_mean" in obj and isinstance(obj["symbolic_agg_mean"], (int, float)):
        out["symbolic_agg_mean"] = float(obj["symbolic_agg_mean"])

for c in candidates:
    obj = read_json(c)
    if obj:
        _set_from_obj(obj)

return out
```

def fmt(x: Optional\[float]) -> str:
"""Human-compact metric string (empty on None/NaN)."""
if x is None:
return ""
try:
if math.isnan(x):
return ""
return f"{x:.6g}"
except Exception:
return ""

def rank\_key\_factory(sort\_key: str):
"""
Return a function extracting a numeric key from leaderboard rows for sorting.
Missing/invalid values are treated as +inf so they sink for ascending order
and rise for descending (we'll invert accordingly).
"""
def \_key(row: Dict\[str, str]) -> float:
try:
return float(row\.get(sort\_key, "") or "inf")
except Exception:
return float("inf")
return \_key

# -----------------------------------------------------------------------------

# Exports (CSV, MD, HTML, JSON), Top-N ZIP, optional DVC tracking

# -----------------------------------------------------------------------------

def write\_csv(rows: List\[Dict\[str, str]], dest: Path) -> None:
if not rows:
return
ensure\_dir(dest.parent)
cols = list(rows\[0].keys())
with dest.open("w", encoding="utf-8", newline="") as f:
w = csv.DictWriter(f, fieldnames=cols)
w\.writeheader()
for r in rows:
w\.writerow(r)

def write\_md(rows: List\[Dict\[str, str]], dest: Path, title: str) -> None:
ensure\_dir(dest.parent)
if not rows:
dest.write\_text(f"# {title}\n\n\_No rows.\_\n", encoding="utf-8")
return
cols = list(rows\[0].keys())
lines = \[f"# {title}\n", "", "|" + "|".join(cols) + "|", "|" + "|".join(\["---"] \* len(cols)) + "|"]
for r in rows:
lines.append("|" + "|".join(str(r\[c]) for c in cols) + "|")
dest.write\_text("\n".join(lines) + "\n", encoding="utf-8")

def write\_html(rows: List\[Dict\[str, str]], dest: Path, title: str) -> None:
ensure\_dir(dest.parent)
if not rows:
dest.write\_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title><h1>{title}</h1><p>No rows.</p>", encoding="utf-8")
return
cols = list(rows\[0].keys())
head = f"""<!doctype html><html><head><meta charset="utf-8">

<title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
:root{{--bg:#0b1220;--fg:#e6edf3;--muted:#8aa;--card:#121a2a;--border:#22304a}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:ui-sans-serif,system-ui}}
.container{{max-width:1200px;margin:0 auto;padding:20px}}
table{{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--border)}}
th,td{{padding:8px;border-bottom:1px solid var(--border);font-size:14px}}
th{{text-align:left;color:#a7bde8}}
h1{{font-size:20px;margin:0 0 12px}}
.small{{color:var(--muted);font-size:12px}}
tr:hover td{{background:#0f1a32}}
</style></head><body>
<div class="container">
<h1>{title}</h1>
<table><thead><tr>"""
    for c in cols:
        head += f"<th>{c}</th>"
    head += "</tr></thead><tbody>"
    body = ""
    for r in rows:
        body += "<tr>" + "".join(f"<td>{r[c]}</td>" for c in cols) + "</tr>"
    tail = f"</tbody></table><div class='small'>Generated by auto_ablate_v50.py • {time.ctime()}</div></div></body></html>"
    dest.write_text(head + body + tail, encoding="utf-8")

def write\_json(rows: List\[Dict\[str, str]], dest: Path) -> None:
ensure\_dir(dest.parent)
dest.write\_text(json.dumps(rows, indent=2), encoding="utf-8")

def zip\_top\_n(run\_dirs: List\[Path], dest\_zip: Path) -> None:
import shutil, tempfile
ensure\_dir(dest\_zip.parent)
with tempfile.TemporaryDirectory() as tmpd:
tmp\_root = Path(tmpd)
for rd in run\_dirs:
dst = tmp\_root / rd.name
shutil.copytree(rd, dst, dirs\_exist\_ok=True)
shutil.make\_archive(dest\_zip.with\_suffix(""), "zip", tmp\_root)

def dvc\_track(path: Path) -> None:
"""
Optionally track an ablation directory with DVC (best-effort).
Creates/updates .gitignore as needed and runs 'dvc add'.
"""
if which("dvc") is None:
console.print("\[yellow]DVC not found in PATH; skipping dvc add.\[/yellow]")
return
try:
subprocess.run(\["dvc", "add", str(path)], check=False)
except Exception:
console.print("\[yellow]dvc add failed (continuing).\[/yellow]")

# -----------------------------------------------------------------------------

# Main CLI command

# -----------------------------------------------------------------------------

@app.command()
def main(
\# What to run
command: Optional\[str] = typer.Option(None, "--command", "-c", help='Module command to run (e.g., "spectramind train").'),
base: List\[str] = typer.Option(None, "--base", "-b", help="Base overrides (repeatable)."),
\# Ablation inputs
grid\_yaml: Optional\[Path] = typer.Option(None, "--grid-yaml", "-y", help="YAML ablation spec (see header)."),
preset: Optional\[str] = typer.Option(None, "--preset", help=f"Built-in sets: {', '.join(PRESET\_SETS.keys())}"),
mode: str = typer.Option("one-at-a-time", "--mode", help="Ablation mode: 'one-at-a-time' or 'cartesian'"),
\# Filters
include: List\[str] = typer.Option(None, "--include", help="Only keep combos whose 'k=v,...' matches any of these globs."),
exclude: List\[str] = typer.Option(None, "--exclude", help="Drop combos whose 'k=v,...' matches any of these globs."),
\# Runtime behavior
name: Optional\[str] = typer.Option(None, "--name", "-n", help="Ablation name (used in output path)."),
out\_root: Path = typer.Option(Path("outputs/ablate"), "--out-root", help="Root output directory."),
parallel: int = typer.Option(2, "--parallel", "-p", help="Parallel workers."),
retries: int = typer.Option(0, "--retries", help="Retries per failed run."),
retry\_backoff: float = typer.Option(3.0, "--retry-backoff", help="Seconds backoff base between retries."),
limit: Optional\[int] = typer.Option(None, "--limit", help="Max runs to execute (after filters)."),
resume: bool = typer.Option(False, "--resume", help="Skip runs with .done flag."),
dry\_run: bool = typer.Option(False, "--dry-run", help="Preview only (no subprocesses)."),
env\_file: Optional\[Path] = typer.Option(None, "--env-file", help="Optional KEY=VALUE lines to inject into env."),
timeout\_sec: Optional\[int] = typer.Option(None, "--timeout-sec", help="Kill a run if it exceeds this many seconds."),
\# Leaderboard & ranking
sort\_key: str = typer.Option("gll\_mean", "--sort-key", help=f"Metric to rank by (default: gll\_mean). Known: {', '.join(METRIC\_KEYS)}"),
asc: bool = typer.Option(False, "--asc", help="Sort ascending (default descending)."),
\# Exports
md: bool = typer.Option(True, "--md/--no-md", help="Export Markdown leaderboard."),
html: bool = typer.Option(True, "--html/--no-html", help="Export HTML leaderboard."),
json\_out: bool = typer.Option(True, "--json/--no-json", help="Export JSON leaderboard."),
open\_html: bool = typer.Option(False, "--open", help="Try to open the HTML leaderboard with system opener."),
top\_n: Optional\[int] = typer.Option(None, "--top-n", help="Bundle top-N runs into a ZIP."),
zip\_path: Optional\[Path] = typer.Option(None, "--zip", help="Path for Top-N ZIP (default topN\_bundle.zip)."),
\# DVC
dvc\_track\_out: bool = typer.Option(False, "--dvc-track", help="Run 'dvc add' on the ablation output directory."),
):
"""
SpectraMind V50 — Symbolic-Aware Ablation Engine (Upgraded).
Provide either --grid-yaml or a --preset. You may also add --base overrides and --command.
"""

```
console.rule("[bold cyan]auto_ablate_v50")

# ----------------------------------------
# Validate base command presence minimally
# ----------------------------------------
# Common path: "spectramind train" etc. Allow any "python -m pkg ..." too.
if not command:
    # Infer from grid file or fallback
    y = load_yaml(grid_yaml)
    command_local = y.get("command") if y else None
    command = command_local or "spectramind train"

# Quick preflight for known entrypoint
if command.split()[0] == "spectramind" and which("python") is None:
    console.print("[yellow]python not found in PATH; ensure your venv is active.[/yellow]")

# ----------------------------------------
# Load ENV injection
# ----------------------------------------
extra_env: Dict[str, str] = {}
if env_file and env_file.exists():
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        extra_env[k.strip()] = v.strip()
    if extra_env:
        table = Table("ENV", "VALUE", title="Injected ENV", box=box.SIMPLE_HEAVY)
        for k, v in extra_env.items():
            table.add_row(k, v)
        console.print(table)

# ----------------------------------------
# Load YAML or preset and construct sets
# ----------------------------------------
y = load_yaml(grid_yaml)
base_overrides = list(y.get("base_overrides", [])) + (base or [])

sets_spec: List[Tuple[str, List[str]]] = []
active_mode = (y.get("ablate", {}) or {}).get("mode") if "ablate" in y else None
if "ablate" in y:
    abl = y["ablate"] or {}
    mode = (active_mode or mode).strip().lower()
    for item in abl.get("sets", []):
        k = item.get("key")
        vals = item.get("values", [])
        if k and isinstance(vals, list) and len(vals) > 0:
            sets_spec.append((k, [str(v) for v in vals]))
elif preset:
    if preset not in PRESET_SETS:
        raise typer.BadParameter(f"Unknown --preset {preset}. Choices: {', '.join(PRESET_SETS.keys())}")
    sets_spec = PRESET_SETS[preset]
else:
    # Default to a gentle symbolic sweep if nothing is given
    sets_spec = PRESET_SETS["loss-sweep"]

if mode not in ("one-at-a-time", "cartesian"):
    raise typer.BadParameter("--mode must be one of {'one-at-a-time','cartesian'}")

all_runs: List[List[str]] = one_at_a_time(sets_spec) if mode == "one-at-a-time" else cartesian(sets_spec)

# Filters (optional)
all_runs = filter_combos(all_runs, include_globs=include or [], exclude_globs=exclude or [])

# Limit (optional)
if limit is not None and limit >= 0:
    all_runs = all_runs[:limit]

# ----------------------------------------
# Determine ablate name + output root
# ----------------------------------------
derivation = [command] + sorted(base_overrides) + [f"{k}={','.join(vs)}" for k, vs in sets_spec]
grid_id = short_hash(derivation, salt="v50_ablate")
ablate_name = name or y.get("options", {}).get("name") or f"ablate-{grid_id[:8]}-{mode}"
root = ensure_dir(out_root / f"{ablate_name}-{now_ts()}")

# Create run events JSONL stream
events_path = root / "events.jsonl"

# Write manifest
manifest = {
    "ablate_name": ablate_name,
    "generated_at": now_ts(),
    "base_cmd": command,
    "base_overrides": base_overrides,
    "mode": mode,
    "sets": [{"key": k, "values": vs} for k, vs in sets_spec],
    "filters": {"include": include or [], "exclude": exclude or []},
    "options": {
        "parallel": parallel,
        "retries": retries,
        "retry_backoff": retry_backoff,
        "resume": resume,
        "dry_run": dry_run,
        "limit": limit,
        "sort_key": sort_key,
        "ascending": asc,
        "timeout_sec": timeout_sec,
    },
    "env_file": str(env_file) if env_file else None,
    "grid_id": grid_id,
    "out_root": str(root),
}
write_text(root / "manifest.json", json.dumps(manifest, indent=2))

# ----------------------------------------
# Build RunSpecs
# ----------------------------------------
specs: List[RunSpec] = []
for idx, overrides in enumerate(all_runs):
    id_short = short_hash([command] + base_overrides + overrides, salt=grid_id)[:8]
    seed_val = to_seed(id_short)
    run_dir = root / f"run_{idx:04d}_{id_short}"
    specs.append(
        RunSpec(
            index=idx,
            id_short=id_short,
            base_cmd=command,
            base_overrides=base_overrides,
            combo_overrides=overrides,
            run_dir=run_dir,
            seed=seed_val,
            retries=retries,
            timeout_sec=timeout_sec,
        )
    )

# ----------------------------------------
# Preview
# ----------------------------------------
preview = Table("Idx", "ID", "Overrides", title="Ablation Preview", box=box.SIMPLE_HEAVY)
show_n = min(len(specs), 12)
for s in specs[:show_n]:
    preview.add_row(str(s.index), s.id_short, ", ".join(s.combo_overrides))
if len(specs) > show_n:
    preview.caption = f"Showing first {show_n} of {len(specs)} runs…"
console.print(preview)
if dry_run:
    console.print("[yellow]Dry-run: commands will not be executed.[/yellow]")

# ----------------------------------------
# Resume: filter out completed
# ----------------------------------------
results: List[Tuple[int, str]] = []
completed = 0
if resume and not dry_run:
    pending: List[RunSpec] = []
    for s in specs:
        if (s.run_dir / ".done").exists():
            results.append((0, "skipped (done)"))
            completed += 1
        else:
            pending.append(s)
    specs = pending
    if completed:
        console.print(f"[green]Resume: skipped {completed} already-completed runs[/green]")

total = len(specs) + completed

# ----------------------------------------
# SIGINT graceful handling
# ----------------------------------------
stop = False

def _sigint(signum, frame):
    nonlocal stop
    stop = True
    console.print("\n[red]Interrupt received — no new tasks will be started; waiting for running tasks.[/red]")

signal.signal(signal.SIGINT, _sigint)

# ----------------------------------------
# Execute runs
# ----------------------------------------
if total == 0:
    console.print("[green]Nothing to do.[/green]")
else:
    with Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as prog:
        task = prog.add_task("Running ablation…", total=total)
        if completed:
            prog.update(task, advance=completed)

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
            fut2spec = {}
            for s in specs:
                if stop:
                    break
                fut = ex.submit(
                    run_with_retries,
                    s,
                    extra_env,
                    dry_run,
                    s.retries,
                    retry_backoff,
                    events_path,
                )
                fut2spec[fut] = s

            for fut in as_completed(fut2spec):
                s = fut2spec[fut]
                try:
                    rc, status = fut.result()
                except Exception as e:
                    rc, status = (99, f"exception: {e}")
                    append_jsonl(
                        events_path,
                        {
                            "ts": now_ts(),
                            "run_dir": str(s.run_dir),
                            "id": s.id_short,
                            "event": "exception",
                            "traceback": traceback.format_exc(limit=1),
                        },
                    )
                results.append((rc, status))
                completed += 1
                prog.update(task, advance=1)
                color = "green" if rc == 0 else "red"
                console.print(f"[{color}]Run {s.index} ({s.id_short}): {status}[/{color}]")
                if stop:
                    break

# ----------------------------------------
# Collect metrics and build leaderboard
# ----------------------------------------
rows: List[Dict[str, str]] = []
for sdir in sorted(Path(root).glob("run_*_*")):
    spec_json = sdir / "run_spec.json"
    if not spec_json.exists():
        continue
    spec = json.loads(spec_json.read_text(encoding="utf-8"))
    m = discover_metrics(sdir)
    rows.append(
        {
            "run": sdir.name,
            "index": str(spec.get("index", "")),
            "id": spec.get("id_short", ""),
            "overrides": ", ".join(spec.get("combo_overrides", [])),
            "gll_mean": fmt(m["gll_mean"]),
            "rmse_mean": fmt(m["rmse_mean"]),
            "mae_mean": fmt(m["mae_mean"]),
            "coverage_p95": fmt(m["coverage_p95"]),
            "z_abs_mean": fmt(m["z_abs_mean"]),
            "z_abs_std": fmt(m["z_abs_std"]),
            "fft_hf_fraction_mean": fmt(m["fft_hf_fraction_mean"]),
            "symbolic_agg_mean": fmt(m["symbolic_agg_mean"]),
        }
    )

# Sort
key_fn = rank_key_factory(sort_key)
rows_sorted = sorted(rows, key=key_fn, reverse=(not asc))

# Exports
write_csv(rows_sorted, root / "leaderboard.csv")
if md:
    write_md(rows_sorted, root / "leaderboard.md", f"Ablation Leaderboard — {ablate_name}")
if html:
    write_html(rows_sorted, root / "leaderboard.html", f"Ablation Leaderboard — {ablate_name}")
    if open_html:
        # best effort: open with platform-specific opener
        try:
            html_path = str(root / "leaderboard.html")
            if os.name == "nt":
                os.startfile(html_path)  # nosec - intended UX feature
            elif sys.platform == "darwin":
                subprocess.run(["open", html_path], check=False)
            else:
                subprocess.run(["xdg-open", html_path], check=False)
        except Exception:
            pass
if json_out:
    write_json(rows_sorted, root / "leaderboard.json")

# Top-N ZIP
if top_n and top_n > 0:
    chosen = rows_sorted[:top_n]
    run_dirs = [root / r["run"] for r in chosen if (root / r["run"]).exists()]
    if not zip_path:
        zip_path = root / f"top{top_n}_bundle.zip"
    zip_top_n(run_dirs, zip_path)
    console.print(f"[cyan]Top-{top_n} bundle → {zip_path}[/cyan]")

# Optional DVC track
if dvc_track_out:
    dvc_track(root)

# Summary
ok = sum(1 for rc, _ in results if rc == 0)
fail = sum(1 for rc, _ in results if rc != 0)
console.rule("[bold]Summary")
console.print(f"[bold green]Success:[/bold green] {ok}")
console.print(f"[bold red]Failed:[/bold red]  {fail}")
console.print(f"[bold]Output root:[/bold] {root}")

# Exit code indicates if any runs failed
raise typer.Exit(code=0 if fail == 0 else 1)
```

# -----------------------------------------------------------------------------

# Entrypoint

# -----------------------------------------------------------------------------

if **name** == "**main**":
app()
