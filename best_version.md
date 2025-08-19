# SpectraMind V50 — Best Version Overview
**Neuro-Symbolic, Physics-Informed Pipeline for the NeurIPS 2025 Ariel Data Challenge**

---

## Current State (Best Version to Date)

### Repository & Infrastructure
- **Structure**: Modular repo (`src/`, `configs/`, `data/`, `outputs/`).
- **Configuration**: Hydra configs with full override support.
- **Environment**: Poetry for Python deps, Dockerfile for OS reproducibility, DVC for datasets and checkpoints.
- **CI/CD**: GitHub Actions runs env rebuild + unit tests + mini end-to-end run.
- **Logging**: Rich console UX + append-only audit log (`logs/v50_debug_log.md`).

---

### Pipeline & Modeling
- **Calibration**: NASA-grade corrections  
  - bias, dark, flat, wavelength calibration  
  - detrending, normalization  
  - physics-aware preprocessing for Ariel FGS1/AIRS.

- **Architecture**: Hybrid  
  - **FGS1**: Structured State-Space Model (Mamba SSM) for >100k timesteps.  
  - **AIRS**: Graph Neural Network (GNN) with edges encoding wavelength adjacency + molecular fingerprints.  
  - Decoders: predict μ and σ for 283 bins.

- **Training & Loss**:  
  - Gaussian log-likelihood loss (mean + variance).  
  - Symbolic constraints: non-negativity, smoothness, molecular co-occurrence.  
  - Modular symbolic loss hooks for overlays.

- **Uncertainty**:  
  - Predictive variance (σ).  
  - Temperature scaling calibration.  
  - Ensembling & Monte Carlo dropout.  
  - COREL & conformal prediction flagged for future integration.

---

### CLI & UX
- **Unified CLI**: `spectramind.py` (Typer-based)  
  Subcommands:  
  - `selftest` (integrity checks)  
  - `calibrate` (raw → calibrated)  
  - `train` (pipeline training)  
  - `predict` (submission-ready CSVs)  
  - `calibrate-temp` (temperature scaling)  
  - `corel-train` (graph conformal prediction)  
  - `diagnose dashboard` (HTML plots: UMAP/t-SNE/SHAP/symbolic overlays)  
  - `submit` (package bundle)  
  - `analyze-log` (summarize CLI calls)  
  - `check-cli-map` (command → module integrity)

- **Diagnostics**:  
  - Rich-powered tables + progress bars.  
  - HTML reports (dashboard with UMAP/t-SNE/SHAP).  
  - Append-only audit log with Git SHA, config hash, artifacts.

---

## Gaps & Future Expansions

### Symbolic & Physics
- Expand symbolic loss library (FFT asymmetry, transit-shape priors, photonic alignment).
- Add symbolic rule mining & violation predictors.

### Uncertainty
- Full COREL integration (graph-based conformal prediction).
- Diffusion-based uncertainty decoders.

### Explainability
- SHAP × symbolic overlays (already partial).
- Attention tracing in decoder heads.

### GUI / Dashboard (Optional Layer)
- Current: terminal-first (Rich) + HTML dashboard.
- Future optional overlays:
  - **Qt/PySide**: scientific/engineering native GUI.
  - **Electron/React**: web dashboard for diagnostics.
  - **Flutter**: GPU-accelerated cross-platform UI.

---

## Verdict
**This is the best version to date.**  
- ✅ CLI-first, reproducible, physics-aware, symbolic-ready.  
- ✅ FGS1 handled by SSM (Mamba), AIRS by GNN.  
- ✅ Uncertainty via μ/σ outputs, calibrated by temperature scaling.  
- ✅ CI, logging, diagnostics, submission bundle pipeline.  
- ⚠️ Symbolic and COREL extensions still ahead.  
- ⚠️ GUI layer optional — CLI + HTML currently preferred.

---

**North Star**: Deliver a challenge-winning, physics-informed, neuro-symbolic system that runs reproducibly in Kaggle’s 9-hour envelope and produces scientifically credible spectra with calibrated uncertainties.
