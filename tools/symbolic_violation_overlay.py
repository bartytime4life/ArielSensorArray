#!/usr/bin/env python3
"""
symbolic_violation_overlay.py

NeurIPS Ariel Data Challenge 2025 — SpectraMind V50
-------------------------------------------------------------------
Generates symbolic-constraint violation overlays (heatmaps, CSV, JSON)
for predicted transmission spectra. Integrates Typer (CLI), Hydra
(config), Rich (console), and Matplotlib (reports).

Key ideas:
- Physics/logic constraints encoded as reusable "rules"
- Violation masks & scores per (sample × wavelength × rule)
- Saved artifacts: heatmaps, per-sample CSV summaries, JSON manifest
- Reproducible runs via Hydra config capture and hash display

References (design & rationale):
- CLI-first + Typer/Hydra integration and Rich logging
    
- NASA-grade reproducibility, config-as-code via Hydra + logs
   
- Symbolic constraints in loss and violation heatmaps/reporting
  (non-negative flux, ≤ stellar baseline, co-occurrence, smoothness)
   
- Spectroscopic/physical plausibility (no negative transit depth,
  consistent molecular bands, spectral smoothness)
   

Author: SpectraMind V50 team
"""

from __future__ import annotations

import json
import hashlib
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track

import typer
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig, OmegaConf


# =========================
# Rich console
# =========================
console = Console()


# =========================
# Config (Hydra)
# =========================

@dataclass
class RuleConfig:
    name: str
    type: str  # 'nonneg', 'upper_bound', 'smoothness', 'cooccur'
    # common
    weight: float = 1.0

    # nonneg
    min_value: float = 0.0

    # upper_bound
    # if per-sample/per-wavelength upper bound provided, specify a column path
    # otherwise use a scalar (e.g., 1.0)
    upper_value: Optional[float] = None
    upper_from_column: Optional[str] = None

    # smoothness
    # max allowed |delta| between adjacent wavelengths
    max_delta: Optional[float] = None

    # cooccur
    # groups: list of dicts {"name": "H2O_band", "channels": [indices]}
    # require if group A active -> group B min-activity; thresholds, etc.
    anchor_group: Optional[str] = None
    dependent_group: Optional[str] = None
    groups: List[Dict] = field(default_factory=list)
    anchor_threshold: float = 0.002  # activity threshold (e.g., mean depth)
    dependent_min_fraction: float = 0.5  # fraction of dependent channels passing activity


@dataclass
class IOConfig:
    preds_path: str  # CSV/Parquet with predictions (rows=samples, 283 wavelength columns)
    out_dir: str
    id_column: Optional[str] = "sample_id"  # if absent, use row index
    # Optional extra columns (e.g., stellar baseline, per-channel) if referenced by rules
    extra_columns: List[str] = field(default_factory=list)
    # Optionally, a separate file with upper bounds (N×283) when 'upper_from_column' is "upper_matrix"
    upper_matrix_path: Optional[str] = None
    # If predictions file contains non-spectral columns, only these columns are spectrum (ordered!):
    spectrum_columns: Optional[List[str]] = None


@dataclass
class PlotConfig:
    cmap: str = "magma"
    dpi: int = 140
    max_samples_in_heatmap: int = 400
    figsize: Tuple[float, float] = (10.0, 6.0)


@dataclass
class RunConfig:
    io: IOConfig
    plots: PlotConfig = PlotConfig()
    rules: List[RuleConfig] = field(default_factory=list)
    # If True, write per-rule CSV (violations) and combined CSV
    write_csv: bool = True
    # If True, write JSON manifest summarizing run and artifacts
    write_json: bool = True
    # If True, save combined heatmap and per-rule heatmaps
    save_heatmaps: bool = True
    # Threshold to list top-k violations per sample (for summary table)
    top_k_per_sample: int = 3


# Register config
cs = ConfigStore.instance()
cs.store(name="run_config", node=RunConfig)


# =========================
# Utility
# =========================

def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _hash_config(cfg: DictConfig) -> str:
    s = OmegaConf.to_yaml(cfg)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:12]


def _as_numpy(df: pd.DataFrame, columns: Optional[List[str]]) -> Tuple[np.ndarray, List[str]]:
    if columns is None:
        # infer spectral columns: all numeric except id/extra
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return df[numeric_cols].to_numpy(dtype=float), numeric_cols
    else:
        return df[columns].to_numpy(dtype=float), columns


# =========================
# Constraint base & rules
# =========================

class Constraint:
    """Base class for symbolic constraints."""

    def __init__(self, cfg: RuleConfig, n_wavelengths: int):
        self.cfg = cfg
        self.n_wl = n_wavelengths

    def name(self) -> str:
        return self.cfg.name

    def evaluate(
        self,
        spectra: np.ndarray,         # (N, W)
        extras: Dict[str, np.ndarray]  # optional extra arrays aligned with spectra or (N, W)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
          violation_mask: (N, W) boolean array, True where violation occurs
          violation_score: (N, W) float array, degree/magnitude of violation (>=0)
        """
        raise NotImplementedError


class NonNegConstraint(Constraint):
    def evaluate(self, spectra: np.ndarray, extras: Dict[str, np.ndarray]):
        min_val = float(self.cfg.min_value)
        # violation where spectra < min_val
        diff = np.minimum(0.0, spectra - min_val)  # negative or zero
        mask = diff < 0.0
        score = -diff  # magnitude below minimum
        # Physics: negative transit depth is non-physical for transmission spectra
        # 
        return mask, score


class UpperBoundConstraint(Constraint):
    def evaluate(self, spectra: np.ndarray, extras: Dict[str, np.ndarray]):
        if self.cfg.upper_from_column == "upper_matrix":
            upper = extras.get("upper_matrix", None)
            if upper is None:
                raise ValueError("upper_matrix not found in extras but referenced by rule.")
            if upper.shape != spectra.shape:
                raise ValueError("upper_matrix shape must match spectra.")
        elif self.cfg.upper_from_column:
            # fallback to per-wavelength scalar from another source if provided, but here treat as scalar per rule
            raise ValueError("upper_from_column other than 'upper_matrix' not implemented in this script.")
        else:
            upper = float(self.cfg.upper_value if self.cfg.upper_value is not None else 1.0)

        # violation where spectra > upper
        if np.isscalar(upper):
            diff = np.maximum(0.0, spectra - upper)
        else:
            diff = np.maximum(0.0, spectra - upper)

        mask = diff > 0.0
        score = diff
        # Physics: transit depth must not exceed stellar baseline/total light
        # 
        return mask, score


class SmoothnessConstraint(Constraint):
    def evaluate(self, spectra: np.ndarray, extras: Dict[str, np.ndarray]):
        max_delta = float(self.cfg.max_delta or 1e9)
        # Compute finite differences along wavelength
        # (N, W-1) for adjacent deltas
        deltas = np.diff(spectra, axis=1)
        # Violation if |delta| > max_delta; map back to (N, W) by placing at right index (wavelength i+1)
        viol = (np.abs(deltas) - max_delta)
        viol = np.maximum(0.0, viol)
        mask = np.zeros_like(spectra, dtype=bool)
        score = np.zeros_like(spectra, dtype=float)
        mask[:, 1:] = viol > 0.0
        score[:, 1:] = viol
        # Spectral smoothness prior embeds physical correlation across wavelengths
        #  
        return mask, score


class CoOccurrenceConstraint(Constraint):
    """
    If anchor group is active (mean depth >= anchor_threshold),
    dependent group must have at least dependent_min_fraction of channels active.
    Violations are assigned to dependent group channels that fail.
    """
    def __init__(self, cfg: RuleConfig, n_wavelengths: int):
        super().__init__(cfg, n_wavelengths)
        # Build group index maps
        self.group_map = {}
        for g in cfg.groups:
            name = g["name"]
            chans = list(map(int, g["channels"]))
            self.group_map[name] = np.array(chans, dtype=int)

        if cfg.anchor_group not in self.group_map or cfg.dependent_group not in self.group_map:
            raise ValueError("anchor_group/dependent_group must exist in groups.")

    def evaluate(self, spectra: np.ndarray, extras: Dict[str, np.ndarray]):
        anchor_idx = self.group_map[self.cfg.anchor_group]
        dep_idx = self.group_map[self.cfg.dependent_group]

        # "Activity" heuristic: positive absorption depth beyond tiny epsilon
        eps = float(self.cfg.anchor_threshold)
        # mean of anchor channels
        anchor_active = (np.mean(spectra[:, anchor_idx], axis=1) >= eps)

        # dependent channels "pass" if >= eps (activity)
        dep_active_mask = spectra[:, dep_idx] >= eps
        # count fraction active
        frac_active = np.mean(dep_active_mask, axis=1)  # (N,)

        need_frac = float(self.cfg.dependent_min_fraction)
        bad_samples = anchor_active & (frac_active < need_frac)

        # Build violation mask for dependent channels that are inactive
        mask = np.zeros_like(spectra, dtype=bool)
        score = np.zeros_like(spectra, dtype=float)
        # Where bad_samples, mark dependent channels that are inactive as violation
        inactive = (~dep_active_mask)  # (N, |dep|)
        # Only for bad samples
        bad_idx = np.where(bad_samples)[0]
        if bad_idx.size > 0:
            sub_inactive = inactive[bad_idx, :]
            # Simple score: shortfall amount (eps - value) clipped to >=0
            shortfall = np.maximum(0.0, eps - spectra[bad_idx[:, None], dep_idx])
            for i, sidx in enumerate(bad_idx):
                mask[sidx, dep_idx] = sub_inactive[i, :]
                score[sidx, dep_idx] = shortfall[i, :]

        # Domain: correlated molecular bands should co-occur (if one strong water band, others expected)
        # 
        return mask, score


# =========================
# Rule factory
# =========================

def build_constraint(rule_cfg: RuleConfig, n_wavelengths: int) -> Constraint:
    if rule_cfg.type == "nonneg":
        return NonNegConstraint(rule_cfg, n_wavelengths)
    if rule_cfg.type == "upper_bound":
        return UpperBoundConstraint(rule_cfg, n_wavelengths)
    if rule_cfg.type == "smoothness":
        return SmoothnessConstraint(rule_cfg, n_wavelengths)
    if rule_cfg.type == "cooccur":
        return CoOccurrenceConstraint(rule_cfg, n_wavelengths)
    raise ValueError(f"Unknown rule type '{rule_cfg.type}' for rule '{rule_cfg.name}'.")


# =========================
# Core runner
# =========================

def run_symbolic_overlay(cfg: RunConfig) -> Dict:
    _ensure_dir(cfg.io.out_dir)

    # Load predictions
    ext = os.path.splitext(cfg.io.preds_path)[1].lower()
    console.rule("[bold cyan]Load Predictions")
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(cfg.io.preds_path)
    else:
        df = pd.read_csv(cfg.io.preds_path)

    # Identify spectrum columns & ID
    spectrum, spectrum_cols = _as_numpy(df, cfg.io.spectrum_columns)
    n, w = spectrum.shape
    if w == 0:
        raise ValueError("No spectral columns found.")

    sample_ids = (
        df[cfg.io.id_column].astype(str).tolist()
        if cfg.io.id_column and cfg.io.id_column in df.columns
        else [str(i) for i in range(n)]
    )

    # Load extras if needed
    extras: Dict[str, np.ndarray] = {}
    if cfg.io.upper_matrix_path and any(r.upper_from_column == "upper_matrix" for r in cfg.rules):
        ext2 = os.path.splitext(cfg.io.upper_matrix_path)[1].lower()
        if ext2 in [".parquet", ".pq"]:
            df_u = pd.read_parquet(cfg.io.upper_matrix_path)
        else:
            df_u = pd.read_csv(cfg.io.upper_matrix_path)
        upper_mat, _ = _as_numpy(df_u, cfg.io.spectrum_columns)
        if upper_mat.shape != spectrum.shape:
            raise ValueError("upper_matrix and predictions must have same shape.")
        extras["upper_matrix"] = upper_mat

    console.print(
        Panel.fit(
            f"[bold]Samples[/]: {n}  |  [bold]Wavelengths[/]: {w}  |  [bold]Rules[/]: {len(cfg.rules)}",
            title="Dataset",
        )
    )

    # Build constraints
    constraints: List[Constraint] = [build_constraint(r, w) for r in cfg.rules]

    # Evaluate rules
    console.rule("[bold cyan]Evaluate Rules")
    all_masks = {}
    all_scores = {}
    per_rule_weight = {}
    for rule in constraints:
        console.print(f"• Evaluating rule: [bold]{rule.name()}[/] ({rule.cfg.type})")
        mask, score = rule.evaluate(spectrum, extras)
        all_masks[rule.name()] = mask
        all_scores[rule.name()] = score
        per_rule_weight[rule.name()] = float(rule.cfg.weight)

    # Combined weighted score and mask
    console.rule("[bold cyan]Aggregate Violations")
    combined_score = np.zeros_like(spectrum, dtype=float)
    combined_mask = np.zeros_like(spectrum, dtype=bool)
    for rname, score in all_scores.items():
        wgt = per_rule_weight.get(rname, 1.0)
        combined_score += wgt * score
        combined_mask |= all_masks[rname]

    # Per-sample severity (sum across wavelengths)
    sample_severity = combined_score.sum(axis=1)

    # =========================
    # Reports
    # =========================
    artifacts = []

    # CSVs
    if cfg.write_csv:
        console.rule("[bold cyan]Write CSVs")
        # Combined per-sample summary
        summary = pd.DataFrame({
            "sample_id": sample_ids,
            "combined_severity": sample_severity,
            "num_violated_wavelengths": combined_mask.sum(axis=1),
        })
        # top-k rules per sample
        top_k = cfg.top_k_per_sample
        for i, sid in enumerate(sample_ids):
            # aggregate per-rule severity for this sample
            per_rule = {r: all_scores[r][i, :].sum() for r in all_scores}
            # sort & add flattened string
            tops = sorted(per_rule.items(), key=lambda x: x[1], reverse=True)[:top_k]
            summary.loc[i, "top_rules"] = "; ".join([f"{r}:{v:.4g}" for r, v in tops])

        out_csv = os.path.join(cfg.io.out_dir, "violations_summary.csv")
        summary.to_csv(out_csv, index=False)
        artifacts.append(out_csv)
        console.print(f"[green]Saved[/] {out_csv}")

        # Optional per-rule CSV, wide format (can be large)
        for rname, score in all_scores.items():
            out_r = os.path.join(cfg.io.out_dir, f"violations_{rname}.csv")
            df_r = pd.DataFrame(score, columns=spectrum_cols)
            df_r.insert(0, "sample_id", sample_ids)
            df_r.to_csv(out_r, index=False)
            artifacts.append(out_r)
            console.print(f"[green]Saved[/] {out_r}")

    # Heatmaps
    if cfg.save_heatmaps:
        console.rule("[bold cyan]Heatmaps")
        # Downsample samples if too many for a readable heatmap
        idx = np.argsort(-sample_severity)
        take = min(cfg.plots.max_samples_in_heatmap, n)
        sel = idx[:take]
        sel_ids = [sample_ids[i] for i in sel]

        def _plot_heat(data: np.ndarray, title: str, fname: str):
            plt.figure(figsize=cfg.plots.figsize, dpi=cfg.plots.dpi)
            plt.imshow(
                data,
                aspect="auto",
                interpolation="nearest",
                cmap=cfg.plots.cmap,
            )
            plt.colorbar(label="Violation score")
            plt.xlabel("Wavelength index")
            plt.ylabel("Sample (sorted by severity)")
            plt.title(title)
            plt.tight_layout()
            plt.savefig(fname)
            plt.close()
            artifacts.append(fname)
            console.print(f"[green]Saved[/] {fname}")

        # Combined
        _plot_heat(
            combined_score[sel, :],
            "Combined Violation Heatmap (top severity samples)",
            os.path.join(cfg.io.out_dir, "heatmap_combined.png"),
        )
        # Per-rule
        for rname, score in all_scores.items():
            _plot_heat(
                score[sel, :],
                f"Violation Heatmap — {rname}",
                os.path.join(cfg.io.out_dir, f"heatmap_{rname}.png"),
            )

    # Manifest JSON
    if cfg.write_json:
        console.rule("[bold cyan]Manifest")
        manifest = {
            "config": OmegaConf.to_container(OmegaConf.structured(cfg), resolve=True),
            "config_hash": _hash_config(OmegaConf.structured(cfg)),
            "n_samples": int(n),
            "n_wavelengths": int(w),
            "rules": list(all_scores.keys()),
            "artifacts": artifacts,
        }
        out_json = os.path.join(cfg.io.out_dir, "manifest.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        console.print(f"[green]Saved[/] {out_json}")

    # Console summary table
    console.rule("[bold cyan]Summary (Top 10 by severity)")
    tab = Table(show_header=True, header_style="bold")
    tab.add_column("rank", justify="right")
    tab.add_column("sample_id", justify="left")
    tab.add_column("combined_severity", justify="right")
    tab.add_column("#viol_wl", justify="right")
    for rank, i in enumerate(idx[: min(10, n)], start=1):
        tab.add_row(
            str(rank),
            sample_ids[i],
            f"{sample_severity[i]:.5g}",
            str(int(combined_mask[i, :].sum())),
        )
    console.print(tab)

    console.rule("[bold green]Done")
    return {"artifacts": artifacts}


# =========================
# Typer CLI (Hydra compose)
# =========================

app = typer.Typer(add_completion=False, help="Symbolic Violation Overlay — SpectraMind V50")


@app.command("run")
def cli_run(
    config_path: str = typer.Option(
        ".", "--config-path", "-cp", help="Hydra config search path (folder)."
    ),
    config_name: str = typer.Option(
        "symbolic_overlay.yaml", "--config-name", "-cn", help="Hydra config name in the search path."
    ),
    overrides: List[str] = typer.Option(
        None,
        "--override",
        "-o",
        help="Hydra override(s), e.g. -o io.preds_path=preds.csv -o io.out_dir=out",
    ),
):
    """
    Execute overlay via Hydra config. Example:
      python symbolic_violation_overlay.py run -cp configs -cn symbolic_overlay.yaml \\
        -o io.preds_path=./preds.csv -o io.out_dir=./overlay_out
    """
    # Compose Hydra programmatically (Typer + Hydra pattern)
    # 
    with hydra.initialize_config_dir(config_dir=os.path.abspath(config_path), version_base=None):
        cfg = hydra.compose(config_name=config_name, overrides=overrides or [])
        # Show effective config
        console.rule("[bold]Hydra Config")
        console.print(OmegaConf.to_yaml(cfg))
        # Run
        run_symbolic_overlay(OmegaConf.to_object(cfg))


# =========================
# Hydra direct entry (optional)
# =========================

@hydra.main(config_name=None, version_base=None)
def _hydra_main(cfg: DictConfig) -> None:
    """
    Allows running directly via Hydra if invoked as:
      python symbolic_violation_overlay.py
    with a structured RunConfig passed via command line. Typically prefer the Typer entrypoint.
    """
    run_symbolic_overlay(OmegaConf.to_object(cfg))


# =========================
# Entry
# =========================

if __name__ == "__main__":
    # Prefer Typer CLI which composes Hydra configs cleanly.
    app()
