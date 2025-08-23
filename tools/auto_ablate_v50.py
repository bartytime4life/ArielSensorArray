#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_ablate_v50.py

SpectraMind V50 — Symbolic‑Aware Ablation Engine (Upgraded, Challenge‑Grade)
=============================================================================

Purpose
-------
Run a *deterministic, resumable, parallel* set of ablations over your Hydra configs
with a focus on **symbolic loss knobs** and **diagnostics quality**. It executes
a base command (e.g., `spectramind train` or `spectramind submit`) with mutated
overrides, collects metrics from diagnostics JSONs, ranks results, and exports
a **Markdown + HTML leaderboard** and an optional **Top‑N ZIP** bundle.

Design Highlights
-----------------
• CLI‑first (Typer) + Rich progress  
• Two ablation modes: *one‑at‑a‑time* or *cartesian*  
• Deterministic short IDs, per‑run seed injection, `hydra.run.dir` isolation  
• Resume support (`.done` flag) + retries + parallelism  
• Metrics ingestion from multiple sources (prefer `diagnostic_summary.json`, fall back to `metrics.json`)  
• Scientific metrics recorded when present:
  - GLL (mean), RMSE/MAE (mean), coverage@95%, |z| stats
  - Smoothness proxies: FFT high‑freq fraction (if available from your FFT tools)
  - Symbolic overlays (aggregated), entropy placeholders
• Leaderboard exports: CSV, Markdown, HTML; optional Top‑N ZIP  
• Manifest + per‑run `overrides.txt` + `run_spec.json` for full reproducibility

Quick Start
-----------
# 1) Run an ablation with built-in *symbolic loss* sweep (one-at-a-time):
python auto_ablate_v50.py \
  --command "spectramind train" \
  --base "data=nominal" --base "trainer=default" \
  --preset loss-sweep \
  --parallel 2 --retries 1

# 2) Provide a YAML spec for custom keys and values (see below):
python auto_ablate_v50.py \
  --command "spectramind train" \
  --grid-yaml grids/ablate_v50.yaml \
  --mode one-at-a-time \
  --parallel 4

# 3) Dry-run to preview runs:
python auto_ablate_v50.py --dry-run --preset loss-sweep

# 4) Resume a partially completed ablation:
python auto_ablate_v50.py --preset loss-sweep --resume

YAML Grid Format
----------------
# grids/ablate_v50.yaml
command: "spectramind train"
base_overrides:
  - "data=nominal"
  - "trainer=default"
ablate:
  # Choose generation mode: "one-at-a-time" (default) or "cartesian"
  mode: "one-at-a-time"
  # Explicit sets to mutate:
  sets:
    - key: "losses.symbolic.smooth_w"
      values: [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
    - key: "losses.symbolic.nonneg_w"
      values: [0.0, 0.05, 0.1, 0.2]
    - key: "losses.symbolic.asym_w"
      values: [0.0, 0.05, 0.1]
    - key: "losses.symbolic.fft_w"
      values: [0.0, 0.02, 0.05, 0.1]
options:
  name: "v50_symbolic_ablate"
  retries: 1
  parallel: 2

Metrics Discovery
-----------------
This tool looks for metrics in each run_dir using, in order:
1) {run_dir}/diagnostic_summary.json          # from tools/generate_diagnostic_summary.py
2) {run_dir}/metrics.json                     # a compact fallback (gll_mean, rmse_mean, etc.)
3) {run_dir}/**/diagnostic_summary.json       # recursive glob (safeguard)
4) {run_dir}/**/metrics.json

The leaderboard will include whatever is found. Unknown metrics are left blank.

HTML/Markdown Leaderboard
-------------------------
Exports to:
  {out_root}/{name}/leaderboard.csv
  {out_root}/{name}/leaderboard.md
  {out_root}/{name}/leaderboard.html

Top‑N Export
------------
Use --top-n and --zip to create a `topN_bundle.zip` containing selected run subdirs
(useful for CI artifacts or quick sharing).

License
-------
MIT
"""

from __future__ import annotations

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from hashlib import blake2b
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import typer
from rich.console import Console
from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich import box

try:
    import yaml
except Exception:
    yaml = None  # YAML support optional unless you pass --grid-yaml

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(p) for p in parts)

def short_hash(items: Sequence[str], salt: str) -> str:
    h = blake2b(digest_size=8)
    for s in items:
        h.update(s.encode("utf-8"))
        h.update(b"\x00")
    h.update(salt.encode("utf-8"))
    return h.hexdigest()

def to_seed(h: str) -> int:
    return int(h, 16) % (2**31 - 1)

def load_yaml(p: Optional[Path]) -> dict:
    if p is None:
        return {}
    if yaml is None:
        raise typer.BadParameter("PyYAML not installed; cannot use --grid-yaml.")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def glob_one(p: Path, patterns: Sequence[str]) -> Optional[Path]:
    for pat in patterns:
        for q in p.glob(pat):
            return q
    return None


# --------------------------------------------------------------------------
# Presets (built-in ablation sets)
# --------------------------------------------------------------------------

PRESET_SETS = {
    # Symbolic loss weight sweep; safe ranges to explore O(10) trials
    "loss-sweep": [
        ("losses.symbolic.smooth_w", [0.0, 0.02, 0.05, 0.1, 0.2]),
        ("losses.symbolic.nonneg_w", [0.0, 0.02, 0.05, 0.1]),
        ("losses.symbolic.asym_w",   [0.0, 0.02, 0.05, 0.1]),
        ("losses.symbolic.fft_w",    [0.0, 0.01, 0.02, 0.05]),
    ],
    # Switches for module toggles (e.g., on/off) to quickly test ablations
    "toggle-symbolic": [
        ("modules.symbolic.enable", ["false", "true"]),
        ("modules.smooth.enable",   ["false", "true"]),
        ("modules.fft.enable",      ["false", "true"]),
    ],
    # Curriculum stages
    "curriculum": [
        ("training.curriculum.stage", ["none", "warmup", "full"]),
    ],
}


# --------------------------------------------------------------------------
# Data classes
# --------------------------------------------------------------------------

@dataclass
class RunSpec:
    index: int
    id_short: str
    base_cmd: str
    base_overrides: List[str]
    combo_overrides: List[str]
    run_dir: Path
    seed: int
    retries: int

    def hydra_overrides(self) -> List[str]:
        ov = list(self.base_overrides) + list(self.combo_overrides)
        # inject seed if not present
        has_seed = any("=" in x and x.split("=", 1)[0] == "seed" for x in ov)
        if not has_seed:
            ov.append(f"seed={self.seed}")
        ov.append(f"hydra.run.dir={str(self.run_dir)}")
        return ov


# --------------------------------------------------------------------------
# Grid generation
# --------------------------------------------------------------------------

def one_at_a_time(sets: List[Tuple[str, List[str]]]) -> List[List[str]]:
    """Generate runs mutating one key at a time, keeping others default."""
    combos: List[List[str]] = []
    for k, vals in sets:
        for v in vals:
            combos.append([f"{k}={v}"])
    return combos

def cartesian(sets: List[Tuple[str, List[str]]]) -> List[List[str]]:
    """Cartesian product of sets; each element is a list of k=v overrides."""
    if not sets:
        return [[]]
    keys = [k for k, _ in sets]
    value_lists = [vals for _, vals in sets]
    out: List[List[str]] = []
    def rec(i: int, acc: List[str]):
        if i == len(keys):
            out.append(list(acc))
            return
        for v in value_lists[i]:
            acc.append(f"{keys[i]}={v}")
            rec(i+1, acc)
            acc.pop()
    rec(0, [])
    return out


# --------------------------------------------------------------------------
# Execution worker
# --------------------------------------------------------------------------

def run_once(spec: RunSpec, extra_env: Dict[str, str], dry: bool) -> Tuple[int, str]:
    ensure_dir(spec.run_dir)

    # Save overrides and spec
    (spec.run_dir / "overrides.txt").write_text("\n".join(spec.hydra_overrides()), encoding="utf-8")
    (spec.run_dir / "run_spec.json").write_text(json.dumps({
        "index": spec.index,
        "id_short": spec.id_short,
        "seed": spec.seed,
        "base_cmd": spec.base_cmd,
        "base_overrides": spec.base_overrides,
        "combo_overrides": spec.combo_overrides,
    }, indent=2), encoding="utf-8")

    done_flag = spec.run_dir / ".done"
    if done_flag.exists():
        return (0, "skipped (done)")

    parts = [sys.executable, "-m"] + spec.base_cmd.split() + spec.hydra_overrides()
    cmd_str = shell_join(parts)
    (spec.run_dir / "command.sh").write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + cmd_str + "\n", encoding="utf-8")
    os.chmod(spec.run_dir / "command.sh", 0o755)

    if dry:
        return (0, "dry-run")

    log_path = spec.run_dir / "stdout_stderr.log"
    env = os.environ.copy()
    env.update(extra_env)
    with log_path.open("wb") as logf:
        proc = subprocess.Popen(parts, stdout=logf, stderr=subprocess.STDOUT, env=env)
        rc = proc.wait()

    if rc == 0:
        done_flag.write_text(now_ts(), encoding="utf-8")
        return (0, "ok")
    return (rc, f"fail (rc={rc})")

def run_with_retries(spec: RunSpec, extra_env: Dict[str, str], dry: bool) -> Tuple[int, str]:
    attempts = spec.retries + 1
    for i in range(attempts):
        rc, status = run_once(spec, extra_env, dry=dry)
        if rc == 0:
            return rc, status
        time.sleep(min(5 * (i + 1), 30))
    return rc, status


# --------------------------------------------------------------------------
# Metrics parsing
# --------------------------------------------------------------------------

# Metrics keys we care about; we normalize into a flat dict for the leaderboard.
METRIC_KEYS = [
    "gll_mean", "rmse_mean", "mae_mean",
    "coverage_p95", "z_abs_mean", "z_abs_std",
    # smoothness proxies (if present in your pipeline)
    "fft_hf_fraction_mean",
    # symbolic aggregation (if present)
    "symbolic_agg_mean",
]

def read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def discover_metrics(run_dir: Path) -> Dict[str, Optional[float]]:
    """
    Heuristic: prefer diagnostic_summary.json, fallback to metrics.json.
    Also search recursively to catch subtool outputs.
    """
    out: Dict[str, Optional[float]] = {k: None for k in METRIC_KEYS}

    # 1) Local preferred paths
    p1 = run_dir / "diagnostic_summary.json"
    p2 = run_dir / "metrics.json"

    candidates: List[Path] = []
    if p1.exists(): candidates.append(p1)
    if p2.exists(): candidates.append(p2)

    # 2) Recursive glob fallback
    if not candidates:
        for root, _, files in os.walk(run_dir):
            for name in files:
                if name in ("diagnostic_summary.json", "metrics.json"):
                    candidates.append(Path(root) / name)
            if candidates:
                break

    def _set_from_metrics_obj(obj: dict):
        # Try compact metrics.json
        if "gll_mean" in obj:
            for k in METRIC_KEYS:
                if k in obj and isinstance(obj[k], (int, float)):
                    out[k] = float(obj[k])

        # Try diagnostic_summary.json
        if "metrics" in obj and isinstance(obj["metrics"], dict):
            m = obj["metrics"]
            for k in ("gll_mean", "rmse_mean", "mae_mean", "coverage_p95", "z_abs_mean", "z_abs_std"):
                if k in m and isinstance(m[k], (int, float)):
                    out[k] = float(m[k])

        # Template/smoothness proxies sometimes live on top-level or keyed subdicts
        if "template_means" in obj and isinstance(obj["template_means"], dict):
            # Not a single scalar, keep silent or choose a representative if wanted
            pass

        # FFT summary (if provided by your FFT tool)
        if "fft_hf_fraction_mean" in obj and isinstance(obj["fft_hf_fraction_mean"], (int, float)):
            out["fft_hf_fraction_mean"] = float(obj["fft_hf_fraction_mean"])

        # Symbolic aggregate (if encoded)
        if "symbolic_agg_mean" in obj and isinstance(obj["symbolic_agg_mean"], (int, float)):
            out["symbolic_agg_mean"] = float(obj["symbolic_agg_mean"])

    for c in candidates:
        obj = read_json(c)
        if obj:
            _set_from_metrics_obj(obj)

    return out


# --------------------------------------------------------------------------
# HTML/Markdown export
# --------------------------------------------------------------------------

def write_csv(rows: List[Dict[str, str]], dest: Path) -> None:
    if not rows:
        return
    ensure_dir(dest.parent)
    cols = list(rows[0].keys())
    with dest.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def write_md(rows: List[Dict[str, str]], dest: Path, title: str) -> None:
    ensure_dir(dest.parent)
    if not rows:
        dest.write_text(f"# {title}\n\n_No rows._\n", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    lines = [f"# {title}\n", "", "|" + "|".join(cols) + "|", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("|" + "|".join(str(r[c]) for c in cols) + "|")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")

def write_html(rows: List[Dict[str, str]], dest: Path, title: str) -> None:
    ensure_dir(dest.parent)
    if not rows:
        dest.write_text(f"<!doctype html><meta charset='utf-8'><title>{title}</title><h1>{title}</h1><p>No rows.</p>", encoding="utf-8")
        return
    cols = list(rows[0].keys())
    # Minimal, self-contained HTML
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


# --------------------------------------------------------------------------
# ZIP Top‑N bundle
# --------------------------------------------------------------------------

def zip_top_n(run_dirs: List[Path], dest_zip: Path) -> None:
    import shutil, tempfile
    ensure_dir(dest_zip.parent)
    # Create a temp folder with symlinked or copied dirs; to be safe for cross-platform, copy tree lightly
    with tempfile.TemporaryDirectory() as tmpd:
        tmp_root = Path(tmpd)
        for rd in run_dirs:
            dst = tmp_root / rd.name
            shutil.copytree(rd, dst, dirs_exist_ok=True)
        shutil.make_archive(dest_zip.with_suffix(""), "zip", tmp_root)


# --------------------------------------------------------------------------
# Main CLI
# --------------------------------------------------------------------------

@app.command()
def main(
    # What to run
    command: Optional[str] = typer.Option(None, "--command", "-c", help='Module command to run (e.g., "spectramind train").'),
    base: List[str] = typer.Option(None, "--base", "-b", help="Base overrides (repeatable)."),
    # Ablation inputs
    grid_yaml: Optional[Path] = typer.Option(None, "--grid-yaml", "-y", help="YAML ablation spec (see docstring)."),
    preset: Optional[str] = typer.Option(None, "--preset", help=f"Built-in sets: {', '.join(PRESET_SETS.keys())}"),
    mode: str = typer.Option("one-at-a-time", "--mode", help="Ablation mode: 'one-at-a-time' or 'cartesian'"),
    # Runtime behavior
    name: Optional[str] = typer.Option(None, "--name", "-n", help="Ablation name (used in output path)."),
    out_root: Path = typer.Option(Path("outputs/ablate"), "--out-root", help="Root folder."),
    parallel: int = typer.Option(2, "--parallel", "-p", help="Parallel workers."),
    retries: int = typer.Option(0, "--retries", help="Retries per failed run."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Max runs to execute."),
    resume: bool = typer.Option(False, "--resume", help="Skip runs with .done flag."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only."),
    env_file: Optional[Path] = typer.Option(None, "--env-file", help="Optional KEY=VALUE lines to inject."),
    # Leaderboard & ranking
    sort_key: str = typer.Option("gll_mean", "--sort-key", help=f"Metric to rank by (default: gll_mean). Known keys: {', '.join(METRIC_KEYS)}"),
    sort_desc: bool = typer.Option(True, "--desc/--asc", help="Sort order (desc by default)."),
    # Exports
    md: bool = typer.Option(True, "--md/--no-md", help="Export Markdown leaderboard."),
    html: bool = typer.Option(True, "--html/--no-html", help="Export HTML leaderboard."),
    top_n: Optional[int] = typer.Option(None, "--top-n", help="Bundle top‑N runs into a ZIP."),
    zip_path: Optional[Path] = typer.Option(None, "--zip", help="Path for Top‑N ZIP (default topN_bundle.zip)."),
):
    """
    SpectraMind V50 — Symbolic‑Aware Ablation Engine.

    Provide either --grid-yaml or a --preset. You may also add --base overrides and --command.
    """
    console.rule("[bold cyan]auto_ablate_v50")

    # Load env
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

    # Load YAML or preset
    y = load_yaml(grid_yaml)
    if not command:
        command = y.get("command") or "spectramind train"
    base_overrides = list(y.get("base_overrides", [])) + (base or [])

    # Compose sets from YAML or preset
    sets_spec: List[Tuple[str, List[str]]] = []
    if "ablate" in y:
        abl = y["ablate"] or {}
        mode = (abl.get("mode") or mode).strip().lower()
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

    # Generate combos
    if mode not in ("one-at-a-time", "cartesian"):
        raise typer.BadParameter("--mode must be one of {'one-at-a-time','cartesian'}")
    all_runs: List[List[str]] = one_at_a_time(sets_spec) if mode == "one-at-a-time" else cartesian(sets_spec)
    if limit is not None:
        all_runs = all_runs[:limit]

    # Determine ablate name + output root
    derivation = [command] + sorted(base_overrides) + [f"{k}={','.join(vs)}" for k, vs in sets_spec]
    grid_id = short_hash(derivation, salt="v50_ablate")
    ablate_name = name or y.get("options", {}).get("name") or f"ablate-{grid_id[:8]}-{mode}"
    root = ensure_dir(out_root / f"{ablate_name}-{now_ts()}")

    # Manifest
    manifest = {
        "ablate_name": ablate_name,
        "generated_at": now_ts(),
        "base_cmd": command,
        "base_overrides": base_overrides,
        "mode": mode,
        "sets": [{ "key": k, "values": vs } for k, vs in sets_spec],
        "options": {
            "parallel": parallel, "retries": retries, "resume": resume,
            "dry_run": dry_run, "limit": limit, "sort_key": sort_key, "sort_desc": sort_desc,
        },
        "env_file": str(env_file) if env_file else None,
        "grid_id": grid_id,
        "out_root": str(root),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # Build RunSpecs
    specs: List[RunSpec] = []
    for idx, overrides in enumerate(all_runs):
        id_short = short_hash([command] + base_overrides + overrides, salt=grid_id)[:8]
        seed_val = to_seed(id_short)
        run_dir = root / f"run_{idx:04d}_{id_short}"
        specs.append(RunSpec(
            index=idx,
            id_short=id_short,
            base_cmd=command,
            base_overrides=base_overrides,
            combo_overrides=overrides,
            run_dir=run_dir,
            seed=seed_val,
            retries=retries,
        ))

    # Preview table
    preview = Table("Idx", "ID", "Overrides", title="Ablation Preview", box=box.SIMPLE_HEAVY)
    show_n = min(len(specs), 12)
    for s in specs[:show_n]:
        preview.add_row(str(s.index), s.id_short, ", ".join(s.combo_overrides))
    if len(specs) > show_n:
        preview.caption = f"Showing first {show_n} of {len(specs)} runs…"
    console.print(preview)
    if dry_run:
        console.print("[yellow]Dry-run: commands will not be executed.[/yellow]")

    # Resume support: mark completed
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
    # SIGINT graceful handler
    stop = False
    def _sigint(signum, frame):
        nonlocal stop
        stop = True
        console.print("\n[red]Interrupt received — no new tasks will be started; waiting for running tasks.[/red]")
    signal.signal(signal.SIGINT, _sigint)

    # Run
    if total == 0:
        console.print("[green]Nothing to do.[/green]")
    else:
        with Progress(
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
                    fut = ex.submit(run_with_retries, s, extra_env, dry_run)
                    fut2spec[fut] = s

                for fut in as_completed(fut2spec):
                    s = fut2spec[fut]
                    try:
                        rc, status = fut.result()
                    except Exception as e:
                        rc, status = (99, f"exception: {e}")
                    results.append((rc, status))
                    completed += 1
                    prog.update(task, advance=1)
                    color = "green" if rc == 0 else "red"
                    console.print(f"[{color}]Run {s.index} ({s.id_short}): {status}[/{color}]")
                    if stop:
                        break

    # -----------------------------
    # Collect metrics and build leaderboard
    # -----------------------------
    rows: List[Dict[str, str]] = []
    for s in sorted(Path(root).glob("run_*_*")):
        spec_json = s / "run_spec.json"
        if not spec_json.exists():
            continue
        spec = json.loads(spec_json.read_text(encoding="utf-8"))

        # metrics discovery
        m = discover_metrics(s)
        row = {
            "run": s.name,
            "index": str(spec.get("index", "")),
            "id": spec.get("id_short", ""),
            "overrides": ", ".join(spec.get("combo_overrides", [])),
            # metrics
            "gll_mean": fmt(m["gll_mean"]),
            "rmse_mean": fmt(m["rmse_mean"]),
            "mae_mean": fmt(m["mae_mean"]),
            "coverage_p95": fmt(m["coverage_p95"]),
            "z_abs_mean": fmt(m["z_abs_mean"]),
            "z_abs_std": fmt(m["z_abs_std"]),
            "fft_hf_fraction_mean": fmt(m["fft_hf_fraction_mean"]),
            "symbolic_agg_mean": fmt(m["symbolic_agg_mean"]),
        }
        rows.append(row)

    # Rank by sort_key (default: gll_mean)
    def _key(row: Dict[str, str]) -> float:
        try:
            return float(row.get(sort_key, "") or "nan")
        except Exception:
            return float("nan")

    rows_sorted = sorted(rows, key=_key, reverse=sort_desc)

    # Exports
    csv_path = root / "leaderboard.csv"
    write_csv(rows_sorted, csv_path)

    if md:
        write_md(rows_sorted, root / "leaderboard.md", f"Ablation Leaderboard — {ablate_name}")
    if html:
        write_html(rows_sorted, root / "leaderboard.html", f"Ablation Leaderboard — {ablate_name}")

    # Top-N ZIP
    if top_n and top_n > 0:
        chosen = rows_sorted[:top_n]
        run_dirs = [root / r["run"] for r in chosen if (root / r["run"]).exists()]
        if not zip_path:
            zip_path = root / f"top{top_n}_bundle.zip"
        zip_top_n(run_dirs, zip_path)
        console.print(f"[cyan]Top‑{top_n} bundle → {zip_path}[/cyan]")

    # Summary to console
    ok = sum(1 for rc, _ in results if rc == 0)
    fail = sum(1 for rc, _ in results if rc != 0)
    console.rule("[bold]Summary")
    console.print(f"[bold green]Success:[/bold green] {ok}")
    console.print(f"[bold red]Failed:[/bold red]  {fail}")
    console.print(f"[bold]Output root:[/bold] {root}")

    # Exit code indicates if any runs failed
    raise typer.Exit(code=0 if fail == 0 else 1)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def fmt(x: Optional[float]) -> str:
    if x is None:
        return ""
    try:
        if math.isnan(x):
            return ""
        # Compact formatting with 6 sigfigs
        return f"{x:.6g}"
    except Exception:
        return ""


if __name__ == "__main__":
    app()
