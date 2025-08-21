# SpectraMind V50 — ArielSensorArray

**Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw Ariel **FGS1/AIRS frames** → **calibrated light curves** → **μ/σ spectra across 283 bins** → **diagnostics & symbolic overlays** → **leaderboard-ready submission** — fully reproducible via CLI, Hydra configs, DVC, CI, and Kaggle integration.

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-76B900)
![Kaggle](https://img.shields.io/badge/platform-Kaggle-20BEFF)

⸻

## 0) Overview

**ArielSensorArray** is the engineering architecture of **SpectraMind V50**: a **NASA-grade, mission-critical pipeline** for the NeurIPS 2025 Ariel Data Challenge.  

It integrates **astrophysical calibration**, **symbolic physics-informed modeling**, and **deep learning architectures** into a reproducible, CLI-first workflow.

### 🛰️ Core Highlights
- **Calibration Kill Chain** — ADC, bias, dark, flat, nonlinearity, dead-pixel masking, CDS, wavelength alignment, jitter correction.  
- **Dual Encoders**:  
  • **FGS1 → Mamba SSM** (long-sequence transit modeling)  
  • **AIRS → Graph Neural Network** (edges = wavelength adjacency, molecule priors, detector regions)  
- **Decoders** — μ (mean spectrum), σ (uncertainty), quantile & diffusion heads.  
- **Uncertainty Calibration** — temperature scaling + **SpectralCOREL GNN**.  
- **Diagnostics** — GLL/entropy heatmaps, SHAP overlays, symbolic violation maps, FFT/UMAP/t-SNE, HTML dashboards.  
- **Symbolic Physics Layer** — smoothness, positivity, asymmetry, FFT suppression, radiative transfer, gravitational/micro-lensing corrections.  
- **Reproducibility** — Hydra configs, DVC/lakeFS, deterministic seeds, Git SHA + config hashes, CI pipelines.  
- **Unified CLI** — `spectramind` orchestrates everything (train, predict, calibrate, diagnose, ablate, submit, selftest, analyze-log, check-cli-map).  

⏱ Optimized for **≤9 hr runtime** on ~1,100 planets with Kaggle A100 GPUs.

---

## 1) System Architecture

```mermaid
flowchart TD
    A[Raw Data] --> B[Calibration Kill Chain]
    B --> C1[FGS1 → Mamba SSM]
    B --> C2[AIRS → Graph Neural Net]
    C1 --> D[Multi-Scale Fusion]
    C2 --> D[Multi-Scale Fusion]
    D --> E1[μ Decoder (mean spectrum)]
    D --> E2[σ Decoder (uncertainty)]
    E1 --> F1[Uncertainty Calibration (Temp Scaling)]
    E2 --> F1
    F1 --> G[Diagnostics Suite]
    G --> H1[GLL Heatmaps]
    G --> H2[SHAP Overlays]
    G --> H3[Symbolic Rule Scoring]
    G --> H4[FFT/UMAP/t-SNE]
    G --> I[HTML Dashboard]
    F1 --> J[Submission Bundle]
