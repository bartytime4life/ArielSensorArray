#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_shap_metadata_v50.py  —  SpectraMind V50 /tools

Purpose
-------
Turn raw SHAP attributions into crisp, reproducible “explanation metadata”:
- Global and per-group importances
- Per-feature summary stats (|SHAP| mean/median/max, sign ratio, nulls)
- Quick plots (top-K bar, grouped stacked bar)
- Self-contained JSON/CSV metadata bundle suitable for dashboards & CI artifacts

Design notes
------------
• Zero-ML dependency: we consume SHAP arrays that your training/inference produced.
• File formats: .npy, .npz (key="shap"), .csv, .parquet supported.
• Optional feature names via file or inferred from headers.
• Optional grouping via JSON mapping (feature -> group) or regex prefix rules.
• CLI is Typer-based, with Rich logging and deterministic outputs.
• No seaborn; Matplotlib only (fast + minimal dep).
• Repro: writes a manifest (YAML) with config + hashes.

Example
-------
# Summarize and plot with explicit feature names and groups
python tools/explain_shap_metadata_v50.py report \
  --shap-path outputs/shap/train_shap.npy \
  --feature-names data/feature_names.txt \
  --group-map configs/feature_groups.json \
  --top-k 30 --out-dir outputs/explain/shap_metadata/run_001

# Validate shapes & NaNs (CI smoke test)
python tools/explain_shap_metadata_v50.py validate --shap-path outputs/shap/train_shap.parquet
"""

from __future__ import annotations

import os
import re
import io
import sys
import csv
import json
import math
import time
import uuid
import zlib
import hashlib
import logging
import textwrap
import datetime as dt
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from matplotlib import pyplot as plt
import yaml

app = typer.Typer(help="SpectraMind V50 — SHAP explanation metadata generator")
console = Console()
logger = logging.getLogger("shap_meta")


# -------------------------- Utilities & IO ---------------------------------- #

def _now_iso() -> str:
    return dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()


def _sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf)
    return h.hexdigest()


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _load_feature_names(path: Optional[Path], n_features: int) -> List[str]:
    if path is None:
        return [f"f{i}" for i in range(n_features)]
    ext = path.suffix.lower()
    if ext in {".txt", ".tsv", ".csv"}:
        names: List[str] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                name = line.strip().split(",")[0]
                if name:
                    names.append(name)
        if len(names) != n_features:
            raise ValueError(f"Feature names length {len(names)} != n_features {n_features}")
        return names
    if ext in {".json"}:
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            # assume index->name mapping
            names = [None] * n_features  # type: ignore
            for k, v in data.items():
                idx = int(k)
                if idx < 0 or idx >= n_features:
                    raise ValueError(f"Bad feature index {idx} in {path}")
                names[idx] = v
            if any(n is None for n in names):
                raise ValueError("Missing some indices in JSON feature map.")
            return list(names)  # type: ignore
        elif isinstance(data, list):
            if len(data) != n_features:
                raise ValueError(f"Feature names length {len(data)} != n_features {n_features}")
            return [str(x) for x in data]
    raise ValueError(f"Unsupported feature names format: {path}")


def _load_group_map(group_map: Optional[Path],
                    feature_names: List[str],
                    group_regex_prefix: Optional[str]) -> Dict[str, str]:
    """
    Returns mapping feature->group.
    Priority: explicit JSON mapping; else regex prefix capture group; else default group "all".
    """
    if group_map is not None:
        data = json.loads(group_map.read_text())
        if not isinstance(data, dict):
            raise ValueError("group-map JSON must be a dict of feature->group")
        # Accept partial maps; missing features go to "other"
        result: Dict[str, str] = {}
        for f in feature_names:
            result[f] = data.get(f, "other")
        return result

    if group_regex_prefix:
        pat = re.compile(group_regex_prefix)
        result = {}
        for f in feature_names:
            m = pat.match(f)
            result[f] = (m.group(1) if (m and m.groups()) else "other")
        return result

    return {f: "all" for f in feature_names}


def _load_shap(path: Path) -> np.ndarray:
    """
    Load SHAP attributions.
    Expected shape: (n_samples, n_features)
    """
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape} in {path}")
        return arr
    if ext == ".npz":
        data = np.load(path)
        key = "shap" if "shap" in data.files else list(data.files)[0]
        arr = data[key]
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D array, got shape {arr.shape} in {path}")
        return arr
    if ext in {".csv"}:
        df = pd.read_csv(path)
        return df.values.astype(float)
    if ext in {".parquet"}:
        df = pd.read_parquet(path)
        return df.values.astype(float)
    raise ValueError(f"Unsupported SHAP file type: {path}")


# --------------------------- Core computation ------------------------------- #

@dataclass
class SummaryRow:
    feature: str
    group: str
    abs_mean: float
    abs_median: float
    abs_max: float
    signed_mean: float
    signed_fraction_positive: float
    null_fraction: float


def summarize_shap(shap: np.ndarray,
                   feature_names: List[str],
                   feature_to_group: Dict[str, str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-feature and per-group summary tables.
    """
    if shap.ndim != 2:
        raise ValueError("shap must be 2-D (n_samples, n_features)")

    n_samples, n_features = shap.shape
    if len(feature_names) != n_features:
        raise ValueError("feature_names length mismatch")

    rows: List[SummaryRow] = []

    # We will allow NaNs (e.g., masked features), compute stats robustly
    with np.errstate(invalid="ignore"):
        abs_shap = np.abs(shap)
        abs_mean = np.nanmean(abs_shap, axis=0)
        abs_median = np.nanmedian(abs_shap, axis=0)
        abs_max = np.nanmax(abs_shap, axis=0)
        signed_mean = np.nanmean(shap, axis=0)
        pos_frac = np.nanmean((shap > 0).astype(float), axis=0)
        null_frac = np.mean(np.isnan(shap), axis=0)

    for j, fname in enumerate(feature_names):
        rows.append(SummaryRow(
            feature=fname,
            group=feature_to_group.get(fname, "other"),
            abs_mean=float(abs_mean[j]),
            abs_median=float(abs_median[j]),
            abs_max=float(abs_max[j]),
            signed_mean=float(signed_mean[j]),
            signed_fraction_positive=float(pos_frac[j]),
            null_fraction=float(null_frac[j])
        ))

    feat_df = pd.DataFrame([asdict(r) for r in rows])
    # Group-level importances (mean of |SHAP| across features in group)
    grp_df = (
        feat_df
        .groupby("group", as_index=False)
        .agg(abs_mean=("abs_mean", "sum"),
             features=("feature", "count"))
        .sort_values("abs_mean", ascending=False)
        .reset_index(drop=True)
    )

    # Normalize group abs_mean to proportion (sum to 1.0) if non-zero
    total = grp_df["abs_mean"].sum()
    if total > 0:
        grp_df["abs_mean_prop"] = grp_df["abs_mean"] / total
    else:
        grp_df["abs_mean_prop"] = 0.0

    return feat_df, grp_df


# ------------------------------ Plotting ------------------------------------ #

def plot_topk_bar(feat_df: pd.DataFrame, out_png: Path, top_k: int = 25, title: str = "Top Features by |SHAP| mean") -> None:
    top = feat_df.sort_values("abs_mean", ascending=False).head(top_k)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(top))))
    ax.barh(top["feature"][::-1], top["abs_mean"][::-1], color="#5B8FF9")
    ax.set_xlabel("|SHAP| mean")
    ax.set_title(title)
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_group_stacked(grp_df: pd.DataFrame, out_png: Path, title: str = "Group importance (sum |SHAP| mean)") -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    # Plot as a horizontal stacked bar representing proportions
    groups = grp_df["group"].tolist()
    props = grp_df["abs_mean_prop"].tolist()
    left = 0.0
    for g, p in zip(groups, props):
        ax.barh(["groups"], [p], left=left, label=g)
        left += p
    ax.set_xlim(0, 1)
    ax.set_xlabel("Proportion of total |SHAP|")
    ax.set_title(title)
    ax.legend(ncol=4, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# ------------------------------ Manifest ------------------------------------ #

def _run_hash(payload: Dict) -> str:
    data = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:16]


def write_manifest(out_dir: Path,
                   shap_path: Path,
                   n_samples: int,
                   n_features: int,
                   feature_names_path: Optional[Path],
                   group_map_path: Optional[Path],
                   config: Dict) -> Path:
    manifest = {
        "created_utc": _now_iso(),
        "tool": "explain_shap_metadata_v50",
        "version": "v1.0.0",
        "inputs": {
            "shap_path": str(shap_path),
            "shap_sha256": _sha256_file(shap_path) if shap_path.exists() else None,
            "feature_names_path": str(feature_names_path) if feature_names_path else None,
            "group_map_path": str(group_map_path) if group_map_path else None
        },
        "shape": {"n_samples": int(n_samples), "n_features": int(n_features)},
        "env": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "git_sha": os.environ.get("GIT_SHA"),
            "user": os.environ.get("USER") or os.environ.get("USERNAME")
        },
        "config": config
    }
    manifest["run_hash"] = _run_hash(manifest)
    path = out_dir / "manifest.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return path


# ------------------------------ CLI config ---------------------------------- #

@dataclass
class CommonArgs:
    shap_path: Path
    out_dir: Path
    feature_names: Optional[Path] = None
    group_map: Optional[Path] = None
    group_regex_prefix: Optional[str] = None
    top_k: int = 25
    debug: bool = False
    dry_run: bool = False


def _setup_logging(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def _print_head(feat_df: pd.DataFrame, grp_df: pd.DataFrame, top_k: int) -> None:
    table = Table(title="Top features by |SHAP| mean", box=box.SIMPLE)
    table.add_column("#", style="bold cyan", width=4)
    table.add_column("feature", style="bold")
    table.add_column("|SHAP| mean", justify="right")
    for i, row in enumerate(feat_df.sort_values("abs_mean", ascending=False).head(top_k).itertuples(False), start=1):
        table.add_row(str(i), row.feature, f"{row.abs_mean:.6g}")
    console.print(table)

    gt = Table(title="Groups by importance", box=box.SIMPLE)
    gt.add_column("#", style="bold cyan", width=4)
    gt.add_column("group", style="bold")
    gt.add_column("sum |SHAP|", justify="right")
    gt.add_column("prop", justify="right")
    for i, row in enumerate(grp_df.itertuples(False), start=1):
        gt.add_row(str(i), row.group, f"{row.abs_mean:.6g}", f"{row.abs_mean_prop:.2%}")
    console.print(gt)


# ------------------------------- Commands ----------------------------------- #

@app.command()
def summarize(
    shap_path: Path = typer.Option(..., help="Path to SHAP attributions (.npy/.npz/.csv/.parquet)"),
    out_dir: Path = typer.Option(..., help="Output directory for metadata bundle"),
    feature_names: Optional[Path] = typer.Option(None, help="Optional feature names file (.txt/.csv/.json)"),
    group_map: Optional[Path] = typer.Option(None, help="Optional JSON mapping: feature->group"),
    group_regex_prefix: Optional[str] = typer.Option(None, help=r"Optional regex with capture group, e.g. r'^(.*)_'"),
    top_k: int = typer.Option(25, min=1, help="How many top features to display/plot"),
    debug: bool = typer.Option(False, help="Verbose logs"),
    dry_run: bool = typer.Option(False, help="Do not write files (preview only)"),
):
    """
    Compute per-feature and per-group SHAP importance tables.
    """
    _setup_logging(debug)
    args = CommonArgs(shap_path, out_dir, feature_names, group_map, group_regex_prefix, top_k, debug, dry_run)

    console.rule("[bold]SHAP Metadata — summarize")
    logger.info(f"Loading SHAP from {args.shap_path}")
    shap = _load_shap(args.shap_path)
    n_samples, n_features = shap.shape
    logger.info(f"SHAP shape: {n_samples} x {n_features}")

    names = _load_feature_names(args.feature_names, n_features)
    fmap = _load_group_map(args.group_map, names, args.group_regex_prefix)

    feat_df, grp_df = summarize_shap(shap, names, fmap)
    _print_head(feat_df, grp_df, args.top_k)

    if args.dry_run:
        console.print(Panel("Dry-run mode, skipping writes.", style="yellow"))
        return

    _ensure_dir(args.out_dir)
    feat_df.to_csv(args.out_dir / "feature_importance.csv", index=False)
    grp_df.to_csv(args.out_dir / "group_importance.csv", index=False)
    (args.out_dir / "feature_importance.json").write_text(feat_df.to_json(orient="records"), encoding="utf-8")
    (args.out_dir / "group_importance.json").write_text(grp_df.to_json(orient="records"), encoding="utf-8")

    manifest = write_manifest(
        out_dir=args.out_dir,
        shap_path=args.shap_path,
        n_samples=n_samples,
        n_features=n_features,
        feature_names_path=args.feature_names,
        group_map_path=args.group_map,
        config={"group_regex_prefix": args.group_regex_prefix, "top_k": args.top_k}
    )
    console.print(f"[green]Wrote[/green] tables and manifest → {args.out_dir}")
    console.print(f"Manifest: {manifest}")


@app.command()
def plot(
    shap_path: Path = typer.Option(..., help="Path to SHAP attributions (.npy/.npz/.csv/.parquet)"),
    out_dir: Path = typer.Option(..., help="Output directory for plots"),
    feature_names: Optional[Path] = typer.Option(None, help="Optional feature names file (.txt/.csv/.json)"),
    group_map: Optional[Path] = typer.Option(None, help="Optional JSON mapping: feature->group"),
    group_regex_prefix: Optional[str] = typer.Option(None, help=r"Optional regex with capture group, e.g. r'^(.*)_'"),
    top_k: int = typer.Option(25, min=1),
    debug: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
):
    """
    Produce quick-look PNG plots (top-K features + group stacked bar).
    """
    _setup_logging(debug)
    args = CommonArgs(shap_path, out_dir, feature_names, group_map, group_regex_prefix, top_k, debug, dry_run)

    console.rule("[bold]SHAP Metadata — plot")
    shap = _load_shap(args.shap_path)
    n_samples, n_features = shap.shape
    names = _load_feature_names(args.feature_names, n_features)
    fmap = _load_group_map(args.group_map, names, args.group_regex_prefix)
    feat_df, grp_df = summarize_shap(shap, names, fmap)

    if args.dry_run:
        console.print(Panel("Dry-run mode, skipping plot writes.", style="yellow"))
        _print_head(feat_df, grp_df, args.top_k)
        return

    _ensure_dir(args.out_dir)
    plot_topk_bar(feat_df, args.out_dir / "top_features.png", top_k=args.top_k)
    plot_group_stacked(grp_df, args.out_dir / "groups_stacked.png")
    console.print(f"[green]Plots saved[/green] → {args.out_dir}")


@app.command()
def report(
    shap_path: Path = typer.Option(..., help="Path to SHAP attributions (.npy/.npz/.csv/.parquet)"),
    out_dir: Path = typer.Option(..., help="Output bundle dir (tables + plots + manifest)"),
    feature_names: Optional[Path] = typer.Option(None, help="Optional feature names file (.txt/.csv/.json)"),
    group_map: Optional[Path] = typer.Option(None, help="Optional JSON mapping: feature->group"),
    group_regex_prefix: Optional[str] = typer.Option(None, help=r"Optional regex with capture group, e.g. r'^(.*)_'"),
    top_k: int = typer.Option(25, min=1),
    debug: bool = typer.Option(False),
    dry_run: bool = typer.Option(False),
):
    """
    Full bundle: tables + plots + manifest (one-stop command for CI artifacts & dashboards).
    """
    _setup_logging(debug)
    summarize.callback(  # type: ignore
        shap_path=shap_path,
        out_dir=out_dir,
        feature_names=feature_names,
        group_map=group_map,
        group_regex_prefix=group_regex_prefix,
        top_k=top_k,
        debug=debug,
        dry_run=dry_run
    )
    plot.callback(  # type: ignore
        shap_path=shap_path,
        out_dir=out_dir,
        feature_names=feature_names,
        group_map=group_map,
        group_regex_prefix=group_regex_prefix,
        top_k=top_k,
        debug=debug,
        dry_run=dry_run
    )


@app.command()
def validate(
    shap_path: Path = typer.Option(..., help="Path to SHAP attributions"),
    max_nan_frac: float = typer.Option(0.2, min=0.0, max=1.0, help="Allowed NaN fraction per feature"),
    debug: bool = typer.Option(False)
):
    """
    Lightweight CI smoke: shape checks + NaN fraction thresholds.
    """
    _setup_logging(debug)
    console.rule("[bold]SHAP Metadata — validate")
    shap = _load_shap(shap_path)
    n_samples, n_features = shap.shape
    console.print(f"Loaded SHAP {shap_path} with shape [samples={n_samples}, features={n_features}]")

    nan_frac = np.mean(np.isnan(shap), axis=0)
    worst = float(np.max(nan_frac))
    idx = int(np.argmax(nan_frac))
    if worst > max_nan_frac:
        console.print(f"[red]FAIL[/red] Feature {idx} has NaN fraction {worst:.2%} > {max_nan_frac:.2%}")
        raise typer.Exit(code=2)
    console.print(f"[green]OK[/green] All features NaN fraction ≤ {max_nan_frac:.2%}")


# ------------------------------ Entry point --------------------------------- #

if __name__ == "__main__":
    app()
