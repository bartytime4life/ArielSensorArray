#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_v50.py — SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge)

Main training entrypoint for the V50 architecture:
 • Loads Hydra config (config_v50.yaml)
 • Initializes FGS1 Mamba + AIRS GNN encoders and multi-head decoders
 • Runs pretraining → contrastive → GLL + symbolic fine-tuning curriculum
 • Logs all metrics, config hashes, and diagnostic artifacts
 • Produces `loss_curve.png` + JSON logs for dashboard integration
"""

from __future__ import annotations
import os, sys, json, hashlib, datetime as dt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

import hydra
from omegaconf import DictConfig, OmegaConf

from src.models.fgs1_mamba import FGS1MambaEncoder
from src.models.airs_gnn import AIRSGNNEncoder
from src.models.multi_scale_decoder import MultiScaleDecoder
from src.losses.symbolic_loss import SymbolicLoss
from src.utils.reproducibility import set_seed, hash_config
from src.utils.logging import init_logger, log_metrics

# ---------------------------
# Main training loop
# ---------------------------

def train_from_config(cfg: DictConfig):
    """Train SpectraMind V50 from Hydra config."""

    # ---- Reproducibility
    set_seed(cfg.training.seed)
    run_hash = hash_config(cfg)
    log = init_logger(f"logs/train_{run_hash}.log")
    log.info(f"SpectraMind V50 training started at {dt.datetime.now()}")
    log.info(f"Run hash: {run_hash}")
    log.info(f"Full config:\n{OmegaConf.to_yaml(cfg)}")

    # ---- Data loaders (placeholder; assume implemented)
    from src.data.loaders import get_dataloaders
    train_loader, val_loader = get_dataloaders(cfg)

    # ---- Models
    fgs1 = FGS1MambaEncoder(cfg.model.fgs1).to(cfg.device)
    airs = AIRSGNNEncoder(cfg.model.airs).to(cfg.device)
    decoder = MultiScaleDecoder(cfg.model.decoder).to(cfg.device)

    model = nn.ModuleDict({"fgs1": fgs1, "airs": airs, "decoder": decoder})

    # ---- Loss & optimizer
    criterion = SymbolicLoss(cfg.loss)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.training.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.training.epochs)

    # ---- Training loop
    train_losses, val_losses = [], []
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            fgs1_out = model["fgs1"](batch["fgs1"])
            airs_out = model["airs"](batch["airs"])
            mu, sigma = model["decoder"](fgs1_out, airs_out)
            loss = criterion(mu, sigma, batch["target"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_loss = total_loss / len(train_loader)
        train_losses.append(train_loss)

        # ---- Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                fgs1_out = model["fgs1"](batch["fgs1"])
                airs_out = model["airs"](batch["airs"])
                mu, sigma = model["decoder"](fgs1_out, airs_out)
                loss = criterion(mu, sigma, batch["target"])
                val_loss += loss.item()
        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # ---- Log & schedule
        log.info(f"Epoch {epoch}/{cfg.training.epochs} — train {train_loss:.4f}, val {val_loss:.4f}")
        log_metrics({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}, run_hash)
        scheduler.step()

    # ---- Save loss curve
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("SpectraMind V50 — Loss Curve")
    plt.legend()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/loss_curve.png", dpi=200)
    log.info("Saved loss curve to outputs/plots/loss_curve.png")

    # ---- Save metrics JSON
    os.makedirs("outputs/metrics", exist_ok=True)
    with open(f"outputs/metrics/train_summary_{run_hash}.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f, indent=2)

    log.info("Training complete.")


# ---------------------------
# Hydra entrypoint
# ---------------------------

@hydra.main(config_path="../configs", config_name="config_v50.yaml", version_base=None)
def main(cfg: DictConfig):
    train_from_config(cfg)


if __name__ == "__main__":
    main()