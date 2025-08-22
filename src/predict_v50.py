#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
predict_v50.py — SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge)

End-to-end inference for SpectraMind V50:
  • Loads Hydra config (config_v50.yaml) or CLI overrides
  • Restores best checkpoint (or provided weights)
  • Runs forward pass on the test/eval split to produce μ and σ
  • Optionally applies uncertainty calibration (temperature / COREL)
  • Writes submission CSV/Parquet and diagnostics artifacts
  • Logs a full, reproducible telemetry trail (JSONL + metrics.json)

Typical usage via CLI wrapper:
    spectramind predict \
        runtime.weights=outputs/checkpoints/best_xxxxx.pt \
        data.test_path=data/test.csv \
        outputs.submission_path=outputs/submission/submission.csv

Or direct:
    python -m src.predict_v50 runtime.weights=... data.test_path=...
"""

from __future__ import annotations

import os
import sys
import json
import math
import time
import hashlib
import logging
import zipfile
import traceback
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Headless plotting for optional quick diagnostics
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Hydra / OmegaConf
import hydra
from omegaconf import DictConfig, OmegaConf

# Optional MLflow
try:
    import mlflow
    _HAS_MLFLOW = True
except Exception:  # pragma: no cover
    _HAS_MLFLOW = False

# -----------------------------
# Local project imports
# -----------------------------
# Encoders / decoders should match train_v50.py shapes and config fields
from src.models.fgs1_mamba import FGS1MambaEncoder
from src.models.airs_gnn import AIRSGNNEncoder
from src.models.multi_scale_decoder import MultiScaleDecoder

# If you have COREL or temperature calibration modules, import here (optional)
# from src.calibration.temperature import TemperatureScaler
# from src.corel.corel_inference import CORELCalibrator

# -----------------------------
# Logging utilities (lightweight)
# -----------------------------

LOG = logging.getLogger("predict_v50")

def _setup_logging(log_dir: Path, level: str = "INFO") -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "predict.log"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(path, mode="a", encoding="utf-8")]
    )
    return path

def _hash_config(cfg: DictConfig) -> str:
    s = OmegaConf.to_yaml(cfg, resolve=True)
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]

def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)

# -----------------------------
# Minimal dataset for inference
# -----------------------------

class CsvInferenceDataset(Dataset):
    """
    Generic CSV inference dataset:
      • Reads inputs defined by cfg.data.input_cols
      • Returns features + ids (if available) for submission join
    """
    def __init__(self,
                 csv_path: Path,
                 input_cols: List[str],
                 id_col: Optional[str] = None,
                 delimiter: str = ",",
                 missing_ok: bool = False):
        self.csv_path = Path(csv_path)
        self.id_col = id_col
        if not self.csv_path.exists():
            if missing_ok:
                LOG.warning("Test CSV not found at %s — generating synthetic inputs for smoke test.", self.csv_path)
                n, d = 256, len(input_cols)
                self.X = np.random.randn(n, d).astype(np.float32)
                self.ids = np.arange(n).astype(np.int64) if id_col else None
                self.columns = list(input_cols) + ([id_col] if id_col else [])
                return
            raise FileNotFoundError(f"Test CSV not found: {self.csv_path}")

        import pandas as pd  # local import
        df = pd.read_csv(self.csv_path, delimiter=delimiter)

        for col in input_cols:
            if col not in df.columns:
                raise ValueError(f"Required input column '{col}' not in {self.csv_path}")
        self.X = df[input_cols].to_numpy(dtype=np.float32)
        self.ids = df[id_col].to_numpy() if (id_col is not None and id_col in df.columns) else None
        self.columns = df.columns.tolist()

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        out = {"x": torch.from_numpy(self.X[idx])}
        if self.ids is not None:
            out["id"] = torch.tensor(self.ids[idx])
        return out

# -----------------------------
# Model wrapper to keep parity with train_v50.py
# -----------------------------

class V50Model(nn.Module):
    """
    Thin wrapper: encoders + decoder consistent with train_v50.py.
    Expectation:
      - fgs1: FGS1MambaEncoder outputs fgs1_feat
      - airs: AIRSGNNEncoder outputs airs_feat
      - decoder: MultiScaleDecoder takes (fgs1_feat, airs_feat) -> (mu, sigma)
    """
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.fgs1 = FGS1MambaEncoder(cfg.model.fgs1)
        self.airs = AIRSGNNEncoder(cfg.model.airs)
        self.decoder = MultiScaleDecoder(cfg.model.decoder)

    @torch.no_grad()
    def forward(self,
                x_fgs1: torch.Tensor,
                x_airs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        fgs1_out = self.fgs1(x_fgs1)
        airs_out = self.airs(x_airs)
        mu, sigma = self.decoder(fgs1_out, airs_out)  # [B, 283] each
        return mu, sigma

# -----------------------------
# Utility: load checkpoint
# -----------------------------

def _restore_weights(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state, strict=False)
    LOG.info("Restored weights from %s", ckpt_path)

# -----------------------------
# Optional calibration stubs (keep modular)
# -----------------------------

def _apply_temperature_scaling(sigmas: np.ndarray, T: float) -> np.ndarray:
    """
    Simple global temperature scaling for σ (uncertainty).
    """
    if T is None or T == 1.0:
        return sigmas
    return np.clip(sigmas * float(T), 1e-8, 1e8)

def _apply_corel_calibration(mu: np.ndarray,
                             sigma: np.ndarray,
                             cfg: DictConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Placeholder for COREL or other correlation-aware calibration.
    Replace with real implementation if available in src.corel.
    """
    if not cfg.calibration.corel.enable:
        return mu, sigma
    LOG.warning("COREL calibration is enabled but no runtime implementation was provided; returning inputs.")
    return mu, sigma

# -----------------------------
# Inference core
# -----------------------------

@dataclass
class InferenceArtifacts:
    submission: Path
    predictions_npy: Path
    metrics_json: Path
    bundle_zip: Optional[Path]

def _build_inference_loaders(cfg: DictConfig) -> Tuple[DataLoader, Optional[List[Any]]]:
    """
    Build test/eval dataloader(s). This stub assumes two input views:
       - FGS1 columns set in cfg.data.fgs1_cols
       - AIRS columns set in cfg.data.airs_cols
    If your test file already concatenates both, adapt loader accordingly.
    """
    # Load the same CSV twice with different views, then zip in collation.
    test_ds_fgs1 = CsvInferenceDataset(
        csv_path=Path(cfg.data.test_path),
        input_cols=list(cfg.data.fgs1_cols),
        id_col=cfg.data.get("id_col", None),
        delimiter=cfg.data.get("delimiter", ","),
        missing_ok=cfg.data.get("missing_ok", False),
    )
    test_ds_airs = CsvInferenceDataset(
        csv_path=Path(cfg.data.test_path),
        input_cols=list(cfg.data.airs_cols),
        id_col=cfg.data.get("id_col", None),
        delimiter=cfg.data.get("delimiter", ","),
        missing_ok=cfg.data.get("missing_ok", False),
    )
    if len(test_ds_fgs1) != len(test_ds_airs):
        raise RuntimeError("FGS1 and AIRS view datasets have different lengths — check column splits.")

    class PairedDataset(Dataset):
        def __len__(self): return len(test_ds_fgs1)
        def __getitem__(self, idx: int):
            a = test_ds_fgs1[idx]
            b = test_ds_airs[idx]
            item = {
                "fgs1": a["x"],
                "airs": b["x"],
            }
            if "id" in a:
                item["id"] = a["id"]
            return item

    ds = PairedDataset()

    def _collate(batch: List[Dict[str, torch.Tensor]]):
        fgs1 = torch.stack([b["fgs1"] for b in batch], dim=0)
        airs = torch.stack([b["airs"] for b in batch], dim=0)
        ids = None
        if "id" in batch[0]:
            ids = torch.stack([b["id"] for b in batch], dim=0)
        return {"fgs1": fgs1, "airs": airs, "id": ids}

    loader = DataLoader(
        ds,
        batch_size=cfg.infer.batch_size,
        shuffle=False,
        num_workers=cfg.infer.num_workers,
        pin_memory=True,
        collate_fn=_collate,
        drop_last=False,
    )
    return loader, None

@torch.no_grad()
def _run_batches(model: V50Model,
                 loader: DataLoader,
                 device: torch.device,
                 amp: bool = True) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval().to(device)
    mu_list: List[np.ndarray] = []
    sg_list: List[np.ndarray] = []
    id_list: List[np.ndarray] = []

    for batch in loader:
        x_fgs1 = batch["fgs1"].to(device, non_blocking=True)
        x_airs = batch["airs"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(amp and device.type == "cuda")):
            mu, sigma = model(x_fgs1, x_airs)
        mu_list.append(mu.detach().cpu().numpy())
        sg_list.append(sigma.detach().cpu().numpy())
        if batch["id"] is not None:
            id_list.append(batch["id"].detach().cpu().numpy())

    mu_all = np.concatenate(mu_list, axis=0) if mu_list else np.zeros((0, 283), dtype=np.float32)
    sg_all = np.concatenate(sg_list, axis=0) if sg_list else np.zeros((0, 283), dtype=np.float32)
    ids_all = np.concatenate(id_list, axis=0).reshape(-1) if id_list else None
    return mu_all, sg_all, ids_all

def _write_submission(mu: np.ndarray,
                      sigma: np.ndarray,
                      ids: Optional[np.ndarray],
                      cfg: DictConfig) -> Path:
    """
    Writes the submission CSV in the shape expected by the competition.
    If a wide format is required (one row per example with 283 columns), we save that directly.
    If long format is required (id, wavelength_bin, mu, sigma), pivot accordingly.
    """
    import pandas as pd

    out_dir = Path(cfg.outputs.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / cfg.outputs.submission_filename

    # Build dataframe
    n, d = mu.shape
    if ids is None:
        ids = np.arange(n)

    if cfg.outputs.format == "wide":
        cols_mu = [f"mu_{i:03d}" for i in range(d)]
        cols_sg = [f"sigma_{i:03d}" for i in range(d)]
        df = pd.DataFrame(np.concatenate([mu, sigma], axis=1), columns=cols_mu + cols_sg)
        df.insert(0, cfg.data.get("id_col", "id"), ids)
    elif cfg.outputs.format == "long":
        recs = []
        for i in range(n):
            for j in range(d):
                recs.append({
                    cfg.data.get("id_col", "id"): int(ids[i]),
                    "bin": j,
                    "mu": float(mu[i, j]),
                    "sigma": float(sigma[i, j]),
                })
        df = pd.DataFrame.from_records(recs)
    else:
        raise ValueError(f"Unknown outputs.format: {cfg.outputs.format}")

    # Save
    if sub_path.suffix.lower() == ".csv":
        df.to_csv(sub_path, index=False)
    elif sub_path.suffix.lower() in (".parquet", ".pq"):
        df.to_parquet(sub_path, index=False)
    else:
        raise ValueError("Unsupported submission extension (use .csv or .parquet)")

    LOG.info("Submission written: %s  (shape=%s)", sub_path, tuple(df.shape))
    return sub_path

def _quick_plot(mu: np.ndarray, out_dir: Path) -> Optional[Path]:
    """Optional quick diagnostic plot of a few spectra."""
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "preview_spectra.png"
        plt.figure(figsize=(9, 5))
        k = min(5, len(mu))
        for i in range(k):
            plt.plot(mu[i], linewidth=2, alpha=0.9, label=f"sample {i}")
        plt.xlabel("Spectral bin")
        plt.ylabel("μ (mean)")
        plt.title("Predicted μ Preview")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        LOG.info("Saved spectra preview: %s", path)
        return path
    except Exception:  # non-fatal
        LOG.warning("Failed to write quick preview plot.", exc_info=True)
        return None

# -----------------------------
# Main inference pipeline
# -----------------------------

def run_inference(cfg: DictConfig) -> InferenceArtifacts:
    # Prepare paths
    out_dir = Path(cfg.runtime.out_dir).absolute()
    logs_dir = out_dir / "logs"
    plots_dir = out_dir / "plots"
    preds_dir = out_dir / "predictions"
    sub_dir = out_dir / "submission"

    # Logging
    log_path = _setup_logging(logs_dir, cfg.runtime.log_level)
    LOG.info("Predict logs at %s", log_path)

    # Device & seeds
    device = torch.device("cuda" if (torch.cuda.is_available() and cfg.runtime.cuda) else "cpu")
    torch.manual_seed(int(cfg.runtime.seed))
    np.random.seed(int(cfg.runtime.seed))
    LOG.info("Using device: %s", device)

    # Config hash
    run_hash = _hash_config(cfg)
    cfg_out = out_dir / "config_infer.yaml"
    cfg_out.parent.mkdir(parents=True, exist_ok=True)
    with cfg_out.open("w", encoding="utf-8") as f:
        f.write(OmegaConf.to_yaml(cfg, resolve=True))
    LOG.info("Config snapshot written: %s | run_hash=%s", cfg_out, run_hash)

    # Dataloader(s)
    loader, _ = _build_inference_loaders(cfg)

    # Build model
    model = V50Model(cfg)
    weights = Path(cfg.runtime.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint/weights not found: {weights}")
    _restore_weights(model, weights)
    model.to(device)

    # Inference
    mu, sigma, ids = _run_batches(model, loader, device, amp=bool(cfg.runtime.amp))

    # Calibration (temperature / COREL)
    sigma = _apply_temperature_scaling(sigma, cfg.calibration.temperature)
    mu, sigma = _apply_corel_calibration(mu, sigma, cfg)

    # Persist raw predictions (NumPy)
    preds_dir.mkdir(parents=True, exist_ok=True)
    mu_path = preds_dir / f"mu_{run_hash}.npy"
    sg_path = preds_dir / f"sigma_{run_hash}.npy"
    np.save(mu_path, mu)
    np.save(sg_path, sigma)
    LOG.info("Saved raw predictions: %s, %s", mu_path, sg_path)

    # Submission
    cfg.outputs.submission_dir = str(sub_dir)
    submission_path = _write_submission(mu, sigma, ids, cfg)

    # Quick plot (optional)
    _quick_plot(mu, plots_dir)

    # Metrics JSON
    metrics = {
        "num_samples": int(mu.shape[0]),
        "num_bins": int(mu.shape[1]) if mu.ndim == 2 else None,
        "weights": str(weights),
        "submission": str(submission_path),
        "run_hash": run_hash,
    }
    metrics_path = out_dir / "metrics_infer.json"
    _write_json(metrics_path, metrics)

    # Optional MLflow
    if cfg.mlflow.enable and _HAS_MLFLOW:
        mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
        mlflow.set_experiment(cfg.mlflow.experiment)
        mlflow.start_run(run_name=cfg.mlflow.run_name or f"predict_v50_{run_hash}")
        mlflow.log_params({
            "weights": str(weights),
            "batch_size": cfg.infer.batch_size,
            "num_workers": cfg.infer.num_workers,
            "temperature": cfg.calibration.temperature,
            "corel_enable": cfg.calibration.corel.enable,
            "outputs_format": cfg.outputs.format,
        })
        mlflow.log_artifact(str(cfg_out))
        if (plots_dir / "preview_spectra.png").exists():
            mlflow.log_artifact(str(plots_dir / "preview_spectra.png"))
        mlflow.log_artifact(str(submission_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.end_run()

    # Bundle (zip) convenience
    bundle_path = None
    if cfg.runtime.bundle_zip:
        bundle_dir = out_dir / "bundle"
        bundle_dir.mkdir(parents=True, exist_ok=True)
        bundle_path = bundle_dir / f"predict_bundle_{run_hash}.zip"
        with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(submission_path, arcname=submission_path.name)
            zf.write(mu_path, arcname=mu_path.name)
            zf.write(sg_path, arcname=sg_path.name)
            if (plots_dir / "preview_spectra.png").exists():
                zf.write(plots_dir / "preview_spectra.png", arcname="preview_spectra.png")
            zf.write(metrics_path, arcname=metrics_path.name)
            zf.write(cfg_out, arcname=cfg_out.name)
        LOG.info("Bundle created: %s", bundle_path)

    return InferenceArtifacts(
        submission=submission_path,
        predictions_npy=mu_path,
        metrics_json=metrics_path,
        bundle_zip=bundle_path
    )

# -----------------------------
# Hydra entrypoint
# -----------------------------

@hydra.main(config_path="../configs", config_name="config_v50.yaml", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    In-code defaults are assumed to be defined in configs/config_v50.yaml.
    You can override via CLI, e.g.:
      python -m src.predict_v50 \
          runtime.weights=outputs/checkpoints/best_abcdef.pt \
          data.test_path=data/test.csv \
          outputs.submission_filename=submission.csv
    """
    out_dir = Path(cfg.runtime.out_dir).absolute()
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        artifacts = run_inference(cfg)
        LOG.info("Submission: %s", artifacts.submission)
        LOG.info("Predictions: %s", artifacts.predictions_npy)
        LOG.info("Metrics: %s", artifacts.metrics_json)
        if artifacts.bundle_zip:
            LOG.info("Bundle: %s", artifacts.bundle_zip)
    except Exception as e:
        err_path = out_dir / "predict_error.txt"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with err_path.open("w", encoding="utf-8") as f:
            f.write("".join(traceback.format_exception(type(e), e, e.__traceback__)))
        LOG.exception("Prediction failed — details in %s", err_path)
        sys.exit(1)


if __name__ == "__main__":
    main()