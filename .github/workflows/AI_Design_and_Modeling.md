# AI Design and Modeling — SpectraMind V50
*ArielSensorArray · NeurIPS 2025 Ariel Data Challenge*

> From raw Ariel **FGS1/AIRS frames** → calibrated light curves → **μ/σ** spectra (283 bins) → diagnostics dashboards → leaderboard-ready submission.

---

## 🌌 North Star
- **Input:** Ariel telescope frames (FGS1 + AIRS).  
- **Output:** Mean (μ) and Uncertainty (σ) spectra across 283 bins.  
- **Pipeline:** Calibration → Encoders → Decoders → Symbolic/Uncertainty layers → Diagnostics → Submission.  
- **Guarantees:** Fully reproducible (CLI-first, Hydra configs, DVC/lakeFS, CI/CD, Docker/Poetry).:contentReference[oaicite:0]{index=0}

---

## 🏗️ Architecture Overview
- **FGS1 Encoder:** Mamba State-Space Model (long-sequence).:contentReference[oaicite:1]{index=1}  
- **AIRS Encoder:** Graph Neural Network with edges for wavelength adjacency, molecule groups, detector regions.:contentReference[oaicite:2]{index=2}  
- **Decoders:**  
  - Dual-head for μ (mean) and σ (uncertainty).  
  - Optional quantile/diffusion decoders for robust uncertainty.  
- **Curriculum:** Masked Autoencoder (MAE) pretrain → Contrastive fine-tuning → GLL + symbolic optimization.  
- **Losses:** Gaussian log-likelihood, FFT smoothness, asymmetry, non-negativity, symbolic constraints.:contentReference[oaicite:3]{index=3}

---

## 🎯 Uncertainty & Symbolic Integration
- **Temperature Scaling:** Global calibration of σ.  
- **Conformal Prediction (COREL):** Bin-wise coverage guarantees using spectral correlations.  
- **Symbolic Constraints:** Smoothness, non-negativity, asymmetry, FFT symmetry, photonic alignment:contentReference[oaicite:4]{index=4}.  

---

## 🔍 Explainability & Dashboard
- **SHAP Overlays:** For AIRS wavelengths and FGS1 frames.  
- **Symbolic Mapping:** Latent → rule overlays.  
- **Attention Fusion:** Decoder attention + symbolic influence scores.  
- **Interactive Dashboard:** HTML with UMAP/t-SNE, SHAP overlays, GLL heatmaps, symbolic violation matrices, CLI logs.  

---

## ⚙️ Engineering & Reproducibility
- **Configs:** Hydra YAMLs with defaults, overrides, multiruns.:contentReference[oaicite:5]{index=5}  
- **Data:** DVC + lakeFS tracked datasets, ensuring exact reproducibility.  
- **Experiments:** MLflow logging (metrics, configs, hashes, artifacts).  
- **Runtime:** Docker + CUDA + Poetry for hermetic builds.  
- **CI/CD:** GitHub Actions run smoke diagnostics, validate packaging, enforce reproducibility.:contentReference[oaicite:6]{index=6}

---

## 🏁 Kaggle Integration
- Execution constrained to ≤ 9 hours across ~1,100 planets on Kaggle GPU.:contentReference[oaicite:7]{index=7}  
- FGS1 transit shape aligns AIRS light curves; jitter injected for augmentation.  
- Submission workflow:  
  ```bash
  spectramind submit
