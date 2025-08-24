# 🏗️ SpectraMind V50 — Architecture

> Neuro-symbolic, physics-informed AI pipeline for the **NeurIPS 2025 Ariel Data Challenge**.  
> This document describes the end-to-end architecture of SpectraMind V50, covering data flow,  
> modeling components, symbolic integration, configuration, and diagnostics.

---

## 🌌 High-Level Overview

The SpectraMind V50 system is built as a **modular, reproducible pipeline**:

1. **Data Ingestion & Calibration**  
   - Raw Ariel telescope simulations (FGS1 photometer + AIRS spectrometer).  
   - Calibration pipeline: ADC, dark, flat, CDS, photometry, trace extraction, normalization, phase alignment.  
   - Output: calibrated lightcurves and spectral bins ready for modeling.

2. **Modeling Engine**  
   - **FGS1 Encoder** — `fgs1_mamba.py` (Mamba SSM for long-sequence photometry).  
   - **AIRS Encoder** — `airs_gnn.py` (graph neural net over wavelength bins, with edge features for molecule type, detector region, proximity).  
   - **Decoder** — `multi_scale_decoder.py` (predicts μ and σ per bin; supports symbolic overlays, uncertainty heads).  

3. **Symbolic & Physics-Informed Modules**  
   - `symbolic_loss.py` — differentiable constraints (smoothness, non-negativity, FFT, asymmetry).  
   - `molecular_priors.py` — wavelength masks for CH₄, CO₂, H₂O bands.  
   - `photonic_alignment.py` — FGS1 transit shape anchors AIRS alignment.  
   - `corel.py` — conformal uncertainty calibration.  
   - `symbolic_logic_engine.py` — evaluates multi-rule symbolic programs, soft/hard enforcement.  

4. **Explainability & Diagnostics**  
   - SHAP overlays (`shap_overlay.py`, `shap_attention_overlay.py`).  
   - Symbolic violation prediction (`symbolic_violation_predictor.py`, `symbolic_violation_predictor_nn.py`, `symbolic_fusion_predictor.py`).  
   - Latent visualizations (UMAP, t-SNE, PCA).  
   - HTML dashboard (`generate_html_report.py`) with symbolic overlays, CLI logs, diagnostics summary.

5. **Orchestration & CLI**  
   - Unified Typer CLI: `spectramind.py` with subcommands (`train`, `predict`, `diagnose`, `submit`, `selftest`).  
   - Hydra config management (`configs/config_v50.yaml`).  
   - Data/version tracking with DVC + Git.  
   - CI/CD with GitHub Actions.

---

## 🔑 Core Components

### Encoders
- **FGS1 Mamba Encoder**  
  - Long photometric time-series (~135k × 32 × 32).  
  - Captures jitter, limb-darkening, transit morphology.  
  - Outputs compressed latent vector aligned to AIRS.

- **AIRS GNN Encoder**  
  - Spectral bins (~356 wavelengths).  
  - Graph nodes = bins; edges encode wavelength adjacency, molecule type, detector region.  
  - Supports GATConv/NNConv with edge features.

### Decoder
- **Multi-Scale Decoder**  
  - Outputs μ (mean transmission spectrum) and σ (uncertainty).  
  - Supports quantile heads, diffusion sampling, or flow-based σ modeling.  
  - Integrates symbolic overlays for physics constraints.

---

## ⚖️ Symbolic Integration

Symbolic modules act as “overlay losses” during training and diagnostics:

- **Smoothness loss** — penalize high-frequency oscillations in μ(λ).  
- **FFT symmetry loss** — enforce spectral symmetry where applicable.  
- **Molecular priors** — encourage expected band absorption depths.  
- **COREL calibration** — ensures calibrated σ matches residual error distribution.  
- **Symbolic logic engine** — multi-rule constraints (e.g., “if H₂O band dips > X, then adjacent CO₂ must not exceed Y”).

Outputs: symbolic violation scores, per-bin influence maps, per-rule loss decomposition.

---

## 📊 Diagnostics & Dashboard

- **Metrics** — Gaussian log-likelihood (GLL), RMSE, entropy, calibration coverage.  
- **Overlays** — SHAP × symbolic × entropy fusion plots.  
- **Visualizations** — FFT power spectra, latent UMAP/t-SNE, symbolic rule tables.  
- **HTML Report** — `generate_html_report.py` assembles diagnostics, CLI logs, symbolic overlays into a versioned dashboard (`diagnostic_report_v1.html`, etc.).

---

## ⚙️ Configuration & Reproducibility

- **Hydra configs** (`configs/`): hierarchical YAML for model, data, training, diagnostics.  
- **DVC**: tracks large data and checkpoints.  
- **Run Hashing**: config + git commit + dataset hash → `run_hash_summary_v50.json`.  
- **CI/CD**: GitHub Actions runs pipeline self-tests and diagnostics on PRs.  
- **Logs**: append-only `logs/v50_debug_log.md` + JSONL run history.

---

## 📐 Architecture Diagram

     ┌─────────────┐         ┌──────────────┐
     │   FGS1      │         │    AIRS      │
     │ Photometer  │         │ Spectrometer │
     └──────┬──────┘         └──────┬───────┘
            │                        │
  ┌─────────▼───────────┐   ┌───────▼──────────┐
  │ FGS1_Mamba Encoder  │   │   AIRS_GNN       │
  └─────────┬───────────┘   └───────┬──────────┘
            │                        │
            └───────────┬────────────┘
                        ▼
             ┌────────────────────────┐
             │ Multi-Scale Decoder     │
             │   (μ, σ predictions)   │
             └───────────┬────────────┘
                         ▼
         ┌────────────────────────────────┐
         │ Symbolic Modules                │
         │  – symbolic_loss.py             │
         │  – molecular_priors.py          │
         │  – corel.py (calibration)       │
         └────────────────────────────────┘
                         ▼
             ┌────────────────────────┐
             │ Diagnostics & Dashboard│
             │  – SHAP overlays       │
             │  – UMAP/t-SNE          │
             │  – HTML reports        │
             └────────────────────────┘

---

## 🚀 Key Principles

- **CLI-first** — everything runs through `spectramind` Typer CLI.  
- **Hydra-safe** — configs drive reproducibility, never hardcoded params.  
- **Glass-box** — all stages log intermediate artifacts, fully inspectable.  
- **NASA-grade reproducibility** — Git+DVC+CI guardrails, run hashes, config snapshots.  
- **Physics-informed AI** — symbolic rules + molecular priors + calibration kill chain.  
- **Challenge-ready** — optimized to process ~1,100 exoplanets in <9h on Kaggle GPU.

---