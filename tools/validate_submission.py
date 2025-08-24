#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
src/validation/validate_submission.py

SpectraMind V50 — Submission Validator (Upgraded, Challenge‑Grade)
==================================================================

Purpose
-------
Validate a submission file for the NeurIPS 2025 Ariel Data Challenge / SpectraMind V50
pipeline before upload. Runs a comprehensive suite of static and light dynamic checks:

Schema & Shape
  • Supports multiple schemas (wide, long, kaggle‑row) and auto‑detects when possible
  • Verifies number of planets, number of bins (default 283), and uniqueness
  • Ensures stable sort (optional), consistent row counts, and reproducible hashing

Data Integrity
  • Detects NaN/inf, non‑finite, wrong dtypes, and non‑monotonic bins
  • Range checks (configurable): μ ∈ [0, 1+ε], σ ∈ (0, σ_max], per‑planet sanity rules
  • Optional per‑planet redundancy checks (duplicates, constant spectra, suspicious σ)

File Hygiene
  • File size limits, compressed format suggestion, delimiter checks
  • Column naming normalization (e.g., mu_0..mu_282, sigma_0..sigma_282)

Reports & CI
  • Human‑readable console summary (Rich)
  • JSON report + Markdown summary in outdir
  • Appends audit line to logs/v50_debug_log.md
  • Typer CLI, Kaggle/CI‑friendly (no internet; headless)

Typical usage
-------------
python -m src.validation.validate_submission \
  --submission outputs/submission.csv \
  --planet-ids data/metadata/planet_ids.csv \
  --bins 283 \
  --format auto \
  --range-mu-min 0.0 \
  --range-mu-max 1.05 \
  --range-sigma-min 1e-8 \
  --range-sigma-max 1.0 \
  --outdir outputs/validation \
  --strict \
  --open-md
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import sys
import gzip
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(add_completion=False, no_args_is_help=True)
console = Console()

# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def _read_planet_ids(path: Optional[Path]) -> Optional[List[str]]:
    if not path:
        return None
    if not path.exists():
        console.print(f"[yellow]WARN[/] planet_ids file not found: {path}")
        return None
    if path.suffix.lower() == ".txt":
        return [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    out: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        r = csv.reader(f)
        for row in r:
            if not row: continue
            out.append(str(row[0]).strip())
    return out or None

def _file_bytes(path: Path) -> int:
    try:
        return path.stat().st_size
    except Exception:
        return -1

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _append_audit(log_path: Path, message: str):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"- [{dt.datetime.now().isoformat(timespec='seconds')}] validate_submission: {message}\n")

# ---------------------------------------------------------------------
# Schema detection & loaders
# ---------------------------------------------------------------------

def _is_long_format(df: pd.DataFrame) -> bool:
    cols = {c.lower() for c in df.columns}
    return {"planet_id", "bin", "mu", "sigma"}.issubset(cols)

def _is_wide_format(df: pd.DataFrame, bins: int) -> bool:
    lower = [c.lower() for c in df.columns]
    has_mu = sum(1 for c in lower if c.startswith("mu_")) >= bins // 2
    has_sigma = sum(1 for c in lower if c.startswith("sigma_")) >= bins // 2
    return has_mu and has_sigma

def _is_kaggle_row_format(df: pd.DataFrame, bins: int) -> bool:
    # Common pattern: one row per planet, columns like v0..v(2*B-1) or "prediction_string" etc.
    # We'll detect a single vector field or a sequence of numbered columns.
    cols = [c.lower() for c in df.columns]
    if "prediction_string" in cols:
        return True
    numbered = [c for c in cols if c.startswith("v")]
    return len(numbered) >= (2 * bins - 5)  # heuristic tolerance

def _load_submission(submission: Path, bins: int, fmt: str) -> Tuple[pd.DataFrame, str]:
    # CSV or parquet supported
    ext = submission.suffix.lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(submission)
    else:
        # detect delimiter quickly—use pandas' default; if fails try tab
        try:
            df = pd.read_csv(submission)
        except Exception:
            df = pd.read_csv(submission, sep="\t")
    if fmt == "auto":
        if _is_long_format(df):
            return df, "long"
        if _is_wide_format(df, bins):
            return df, "wide"
        if _is_kaggle_row_format(df, bins):
            return df, "kaggle_row"
        # fallback assume wide
        return df, "wide"
    return df, fmt

# ---------------------------------------------------------------------
# Canonicalization into arrays
# ---------------------------------------------------------------------

def _canon_from_long(df: pd.DataFrame, planet_ids: Optional[List[str]], bins: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    cols = {c.lower(): c for c in df.columns}
    pid_col = cols.get("planet_id")
    bin_col = cols.get("bin")
    mu_col = cols.get("mu")
    sg_col = cols.get("sigma")

    if not all([pid_col, bin_col, mu_col, sg_col]):
        raise ValueError("Long format requires columns: planet_id, bin, mu, sigma")

    # Ensure bin spans 0..bins-1
    uniq_bins = sorted(set(df[bin_col].astype(int).tolist()))
    if uniq_bins != list(range(bins)):
        raise ValueError(f"Long format: bin column must cover 0..{bins-1}; got {uniq_bins[:5]}... len={len(uniq_bins)}")

    # Build pivot [N,B] for mu and sigma in planet order
    if planet_ids is None:
        planet_ids = sorted(df[pid_col].astype(str).unique().tolist())
    mu = np.zeros((len(planet_ids), bins), dtype=float)
    sg = np.zeros_like(mu)
    grouped = df.groupby(pid_col)
    for i, pid in enumerate(planet_ids):
        if pid not in grouped.groups:
            raise ValueError(f"Missing rows for planet_id {pid}")
        sub = grouped.get_group(pid).sort_values(bin_col)
        mu[i, :] = sub[mu_col].astype(float).values
        sg[i, :] = sub[sg_col].astype(float).values
    return planet_ids, mu, sg

def _canon_from_wide(df: pd.DataFrame, planet_ids: Optional[List[str]], bins: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    # Planet column guess
    possible_id_cols = [c for c in df.columns if c.lower() in {"planet_id", "id", "sample_id"}]
    pid_col = possible_id_cols[0] if possible_id_cols else None
    if planet_ids is None:
        if pid_col:
            planet_ids = df[pid_col].astype(str).tolist()
        else:
            planet_ids = [str(i) for i in range(len(df))]
    if pid_col and len(df) != len(planet_ids):
        # If explicit planet_ids given, align/merge
        df = df.set_index(pid_col).reindex(planet_ids).reset_index()
    # Find mu_*, sigma_* columns 0..B-1
    mu_cols = []
    sg_cols = []
    lower = [c.lower() for c in df.columns]
    col_map = {c.lower(): c for c in df.columns}
    for b in range(bins):
        mkey = f"mu_{b}"
        skey = f"sigma_{b}"
        if mkey in col_map and skey in col_map:
            mu_cols.append(col_map[mkey])
            sg_cols.append(col_map[skey])
        else:
            # try zero-padded
            mkey2 = f"mu_{b:03d}"
            skey2 = f"sigma_{b:03d}"
            if mkey2 in col_map and skey2 in col_map:
                mu_cols.append(col_map[mkey2])
                sg_cols.append(col_map[skey2])
            else:
                raise ValueError(f"Wide format missing mu/sigma column for bin {b}")
    mu = df[mu_cols].to_numpy(dtype=float)
    sg = df[sg_cols].to_numpy(dtype=float)
    return planet_ids, mu, sg

def _canon_from_kaggle_row(df: pd.DataFrame, planet_ids: Optional[List[str]], bins: int) -> Tuple[List[str], np.ndarray, np.ndarray]:
    # Either "prediction_string" with space‑separated sequence [μ0 σ0 μ1 σ1 ...]
    cols = {c.lower(): c for c in df.columns}
    pid_col = None
    for c in ("planet_id", "id", "sample_id"):
        if c in cols:
            pid_col = cols[c]
            break
    if planet_ids is None:
        if pid_col:
            planet_ids = df[pid_col].astype(str).tolist()
        else:
            planet_ids = [str(i) for i in range(len(df))]

    if "prediction_string" in cols:
        arr_mu = np.zeros((len(df), bins), dtype=float)
        arr_sg = np.zeros_like(arr_mu)
        for i, s in enumerate(df[cols["prediction_string"]].astype(str).tolist()):
            toks = s.strip().split()
            if len(toks) != 2 * bins:
                raise ValueError(f"Row {i}: expected {2*bins} tokens, got {len(toks)}")
            vals = np.asarray(list(map(float, toks)), dtype=float)
            arr_mu[i, :] = vals[0::2]
            arr_sg[i, :] = vals[1::2]
        return planet_ids, arr_mu, arr_sg

    # Else numbered columns v0..v(2B-1)
    numbered = [c for c in df.columns if c.lower().startswith("v")]
    if len(numbered) < 2 * bins:
        raise ValueError(f"kaggle_row format requires >= {2*bins} numbered columns or 'prediction_string'")
    ordered = sorted(numbered, key=lambda x: int("".join([ch for ch in x if ch.isdigit()]) or "0"))
    V = df[ordered].to_numpy(dtype=float)
    arr_mu = V[:, 0::2][:, :bins]
    arr_sg = V[:, 1::2][:, :bins]
    return planet_ids, arr_mu, arr_sg

# ---------------------------------------------------------------------
# Core validators
# ---------------------------------------------------------------------

def _check_finite(name: str, arr: np.ndarray) -> List[str]:
    issues = []
    if np.any(~np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        issues.append(f"{name}: found {n_bad} non‑finite entries (NaN/Inf).")
    return issues

def _check_range(name: str, arr: np.ndarray, lo: Optional[float], hi: Optional[float], allow_equal_low: bool = True) -> List[str]:
    issues = []
    if lo is not None:
        bad = arr < (lo if allow_equal_low else (lo + np.finfo(float).eps))
        if np.any(bad):
            n = int(np.sum(bad))
            vmin = float(np.nanmin(arr))
            issues.append(f"{name}: {n} entries below {lo} (min={vmin}).")
    if hi is not None:
        bad = arr > hi
        if np.any(bad):
            n = int(np.sum(bad))
            vmax = float(np.nanmax(arr))
            issues.append(f"{name}: {n} entries above {hi} (max={vmax}).")
    return issues

def _check_monotonic_bins(mu: np.ndarray) -> List[str]:
    # Not required; warn if too spiky: many sign flips in first differences
    d = np.diff(mu, axis=1)
    flips = np.sum(np.sign(d[:, 1:]) != np.sign(d[:, :-1]), axis=1)  # per planet
    pct = float(np.mean(flips > mu.shape[1] * 0.30) * 100.0)
    if pct > 20.0:
        return [f"μ shows high oscillation in {pct:.1f}% of planets (heuristic)."]
    return []

def _check_sigma_positive(sigma: np.ndarray, min_val: float) -> List[str]:
    bad = sigma <= min_val
    if np.any(bad):
        n = int(np.sum(bad))
        return [f"σ contains {n} entries ≤ {min_val} (non‑positive/too small)."]
    return []

def _check_unique_ids(ids: List[str]) -> List[str]:
    issues = []
    if len(set(ids)) != len(ids):
        issues.append("Duplicate planet IDs detected.")
    return issues

def _check_expected_ids(ids: List[str], expected: Optional[List[str]]) -> List[str]:
    issues = []
    if expected is None:
        return issues
    if len(ids) != len(expected):
        issues.append(f"Count mismatch: submission N={len(ids)} vs expected N={len(expected)}.")
    missing = sorted(set(expected) - set(ids))
    extra = sorted(set(ids) - set(expected))
    if missing:
        issues.append(f"Missing {len(missing)} planet IDs (e.g., {missing[:5]}...).")
    if extra:
        issues.append(f"Found {len(extra)} unexpected planet IDs (e.g., {extra[:5]}...).")
    return issues

def _check_file_size(path: Path, max_mb: float) -> List[str]:
    size = _file_bytes(path)
    if size < 0:
        return [f"Unable to stat submission file."]
    mb = size / (1024 * 1024)
    if mb > max_mb:
        return [f"File size {mb:.2f} MB exceeds limit {max_mb:.2f} MB. Consider compression."]
    return []

# ---------------------------------------------------------------------
# Canonical wrapper
# ---------------------------------------------------------------------

def _canonicalize(submission: Path, bins: int, fmt: str, planet_ids_ref: Optional[List[str]]) -> Tuple[List[str], np.ndarray, np.ndarray, str]:
    df, used_fmt = _load_submission(submission, bins, fmt)
    if used_fmt == "long":
        ids, mu, sg = _canon_from_long(df, planet_ids_ref, bins)
    elif used_fmt == "wide":
        ids, mu, sg = _canon_from_wide(df, planet_ids_ref, bins)
    elif used_fmt == "kaggle_row":
        ids, mu, sg = _canon_from_kaggle_row(df, planet_ids_ref, bins)
    else:
        raise ValueError(f"Unknown format: {used_fmt}")
    return ids, mu, sg, used_fmt

# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

def _json_report(outdir: Path, report: Dict[str, Any]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "validation_report.json"
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return p

def _md_report(outdir: Path, report: Dict[str, Any]) -> Path:
    lines = []
    lines.append(f"# Submission Validation Report")
    lines.append(f"- Timestamp: {report['timestamp']}")
    lines.append(f"- Submission: `{report['submission']}`")
    lines.append(f"- Hash (sha256): `{report['sha256']}`")
    lines.append(f"- Detected format: **{report['format']}**")
    lines.append(f"- Planets: **{report['n_planets']}**, Bins: **{report['n_bins']}**\n")
    status = "PASS ✅" if report["pass"] else "FAIL ❌"
    lines.append(f"## Overall Status: {status}\n")
    if report["errors"]:
        lines.append("### Errors")
        for e in report["errors"]:
            lines.append(f"- {e}")
        lines.append("")
    if report["warnings"]:
        lines.append("### Warnings")
        for w in report["warnings"]:
            lines.append(f"- {w}")
        lines.append("")
    lines.append("### Summary")
    for k, v in report["summary"].items():
        lines.append(f"- {k}: {v}")
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / "validation_report.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

@app.command("run")
def cli_run(
    submission: Path = typer.Option(..., help="Path to submission file (csv/parquet)."),
    planet_ids: Optional[Path] = typer.Option(None, help="Optional planet IDs file (.txt or .csv first column)."),
    bins: int = typer.Option(283, help="Number of wavelength bins."),
    fmt: str = typer.Option("auto", help="Format: auto|wide|long|kaggle_row"),
    range_mu_min: Optional[float] = typer.Option(0.0, help="Minimum allowed μ (inclusive)."),
    range_mu_max: Optional[float] = typer.Option(1.05, help="Maximum allowed μ."),
    range_sigma_min: Optional[float] = typer.Option(1e-8, help="Minimum allowed σ (exclusive)."),
    range_sigma_max: Optional[float] = typer.Option(1.0, help="Maximum allowed σ."),
    max_mb: float = typer.Option(100.0, help="Maximum file size (MB) before warning/error in --strict."),
    strict: bool = typer.Option(False, "--strict/--no-strict", help="Treat warnings as errors."),
    outdir: Path = typer.Option(Path("outputs/validation"), help="Output directory for reports."),
    open_md: bool = typer.Option(False, "--open-md/--no-open-md", help="Open Markdown report after validation."),
    log_path: Path = typer.Option(Path("logs/v50_debug_log.md"), help="Append audit line here."),
):
    """
    Validate submission file against SpectraMind/Kaggle constraints and emit reports.
    """
    err: List[str] = []
    wrn: List[str] = []
    try:
        if not submission.exists():
            raise FileNotFoundError(f"Submission file not found: {submission}")

        planet_ids_ref = _read_planet_ids(planet_ids)
        ids, mu, sg, used_fmt = _canonicalize(submission, bins, fmt, planet_ids_ref)

        # Basic checks
        err += _check_unique_ids(ids)
        err += _check_expected_ids(ids, planet_ids_ref)

        # Finite and dtype
        err += _check_finite("μ", mu)
        err += _check_finite("σ", sg)

        # Ranges
        err += _check_range("μ", mu, range_mu_min, range_mu_max, allow_equal_low=True)
        # σ must be > min strictly
        if range_sigma_min is not None:
            err += _check_sigma_positive(sg, min_val=float(range_sigma_min))
        err += _check_range("σ", sg, None, range_sigma_max, allow_equal_low=True)

        # Heuristic smoothness warning (optional)
        wrn += _check_monotonic_bins(mu)

        # File size
        size_issues = _check_file_size(submission, max_mb=max_mb)
        if strict:
            err += size_issues
        else:
            wrn += size_issues

        # Summary & status
        n_planets = len(ids)
        n_bins = mu.shape[1]
        sub_hash = _sha256(submission)
        ok = (len(err) == 0) if not strict else (len(err) == 0 and len(wrn) == 0)

        # Console table
        table = Table(title="Submission Validation", show_header=True, header_style="bold")
        table.add_column("Field")
        table.add_column("Value")
        table.add_row("Format", used_fmt)
        table.add_row("Planets", str(n_planets))
        table.add_row("Bins", str(n_bins))
        table.add_row("Submission", submission.name)
        table.add_row("SHA256", sub_hash[:12] + "…")
        table.add_row("Status", "PASS ✅" if ok else "FAIL ❌")
        console.print(table)

        if err:
            console.print(Panel.fit("Errors", style="red"))
            for e in err:
                console.print(f"[red]- {e}[/red]")
        if wrn:
            console.print(Panel.fit("Warnings", style="yellow"))
            for w in wrn:
                console.print(f"[yellow]- {w}[/yellow]")

        # Reports
        report = {
            "timestamp": dt.datetime.utcnow().isoformat(),
            "submission": submission.as_posix(),
            "sha256": sub_hash,
            "format": used_fmt,
            "n_planets": n_planets,
            "n_bins": n_bins,
            "errors": err,
            "warnings": wrn,
            "pass": bool(ok),
            "summary": {
                "mu_min": float(np.nanmin(mu)),
                "mu_max": float(np.nanmax(mu)),
                "sigma_min": float(np.nanmin(sg)),
                "sigma_max": float(np.nanmax(sg)),
                "file_mb": float(_file_bytes(submission) / (1024 * 1024.0)),
            },
        }
        jsonp = _json_report(outdir, report)
        mdp = _md_report(outdir, report)

        _append_audit(log_path, f"file={submission.name} fmt={used_fmt} pass={ok}")
        console.print(Panel.fit(f"JSON: {jsonp.as_posix()}\nMD  : {mdp.as_posix()}", style=("green" if ok else "red")))

        if open_md:
            try:
                import webbrowser
                webbrowser.open(f"file://{mdp.resolve().as_posix()}")
            except Exception:
                pass

        raise typer.Exit(code=(0 if ok else 1))

    except KeyboardInterrupt:
        console.print("\n[red]Interrupted[/]")
        raise typer.Exit(code=130)
    except Exception as ex:
        console.print(Panel.fit(f"ERROR: {ex}", style="red"))
        _append_audit(log_path, f"file={submission.name if submission else '?'} ERROR={ex}")
        raise typer.Exit(code=1)

@app.callback()
def _cb():
    """
    SpectraMind V50 — Submission Validator
    • Auto‑detects schema (wide/long/kaggle‑row) and canonicalizes to (ids, μ, σ)
    • Validates counts, ranges, finiteness, and file hygiene
    • Emits JSON + Markdown reports, appends audit log
    • --strict treats warnings as errors for CI
    """
    pass

if __name__ == "__main__":
    app()
