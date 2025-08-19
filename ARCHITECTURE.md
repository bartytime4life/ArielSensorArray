
# SpectraMind V50 — Engineering Vision & Master Architecture
**Neuro-Symbolic, Physics-Informed AI Pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** Deliver a reproducible, explainable, physics-informed system that ingests Ariel FGS1/AIRS cubes, outputs μ and σ for 283 bins, passes calibration and diagnostics, and packages a competition-valid submission — all within Kaggle’s 9-hour runtime envelope.

---

## 0) Purpose & Scope

This document is the **engineering blueprint** for SpectraMind V50:

- **For engineers:** defines modules, contracts, workflows, performance budgets, and acceptance criteria.  
- **For scientists:** encodes the physics and symbolic priors into the machine learning architecture.  
- **For ops/MLOps:** prescribes reproducibility, CI/CD, and data governance practices.  

---

## 1) System Architecture

SpectraMind V50 is structured as **four tightly coupled subsystems**:

1. **Calibration Kill Chain**  
   - ADC → bias → dark → flat → nonlinearity → trace extraction → wavelength alignment → normalization  
   - Outputs: clean, aligned lightcurves and AIRS/FGS1 spectra.  
   - Implemented in `src/asa/calib/`.

2. **Modeling Core**  
   - **Encoders:**  
     - FGS1 → Mamba SSM for long-sequence transit curves.  
     - AIRS → Graph Neural Network (edge types: wavelength adjacency, molecule priors, detector regions).  
   - **Decoders:**  
     - μ (mean) spectrum head  
     - σ (uncertainty) head (temperature scaling + COREL support)  
     - Optional diffusion/quantile decoders.  
   - Implemented in `src/asa/pipeline/`.

3. **Diagnostics & Explainability**  
   - GLL heatmaps, RMSE/entropy plots  
   - SHAP overlays, symbolic violation maps  
   - Latent UMAP/t-SNE visualizations  
   - Integrated HTML diagnostics dashboard  
   - Implemented in `src/asa/diagnostics/`.

4. **Unified CLI + Config + Logging**  
   - Typer CLI (`spectramind`) with subcommands: `train`, `predict`, `calibrate`, `diagnose`, `submit`, `ablate`, `selftest`, `analyze-log`, `corel-train`, `check-cli-map`.  
   - Hydra 1.3 configs in `configs/`  
   - DVC + lakeFS integration for data versioning  
   - Logging: console/file/JSONL, optional MLflow/W&B  
   - Reproducibility tracked via `v50_debug_log.md`  

---

## 2) Physics & Scientific Foundations

- **Ariel Mission Context:** ESA’s Ariel telescope observes ~1,000 exoplanets (0.5–7.8 μm).  
- **Observational Challenges:** spacecraft jitter, detector nonlinearity, cosmic noise.  
- **Spectral Physics Priors:**  
  - Radiative transfer: $I(ν) ≈ I₀(ν) e^{−τ(ν)}$  
  - Gas fingerprints: H₂O, CO₂, CH₄ across bins  
  - Smoothness & asymmetry constraints on spectra  
- **Symbolic AI Layer:**  
  - Smoothness, positivity, FFT frequency suppression  
  - Physics-informed rules embedded as loss terms  
  - Conformal prediction (COREL) for calibrated σ  

---

## 3) Data Flow

FGS1/AIRS raw cubes
│
▼
Calibration Kill Chain ──► calibrated lightcurves/spectra
│
▼
Encoders (Mamba + GNN)
│
▼
Decoders (μ, σ) ──► predictions + uncertainties
│
├──► Diagnostics (UMAP, SHAP, symbolic overlays, FFT)
│
└──► Submission packaging (CSV, NPZ, ZIP)

---

## 4) Reproducibility Standards

- **Hydra configs:** all params (data, model, training, diagnostics, calibration)  
- **DVC tracked data:** raw/calibrated cubes, checkpoints  
- **Deterministic seeds:** fixed RNG, ordered DataLoader (toy configs)  
- **Audit log:** every CLI call recorded in `v50_debug_log.md` (git SHA, config hash, host, CUDA info)  
- **CI pipeline:** GitHub Actions runs selftest + smoke pipeline on PRs  

---

## 5) Performance Budget

- **End-to-end:** ≤ 9 hours on Kaggle A100 (~1,100 planets)  
- **FGS1 encoder:** optimized SSM with downsampling  
- **AIRS encoder:** batched GNN with sparse adjacency  
- **Decoders:** lightweight heads (μ, σ), optional distillation  
- **Diagnostics:** parallelizable, HTML/PNG export, non-blocking  

---

## 6) CLI Coverage

All major functions are accessible via CLI:

- `spectramind train` — training entrypoint  
- `spectramind predict` — μ/σ inference + submission  
- `spectramind calibrate` — calibration kill chain  
- `spectramind calibrate-temp` — temperature scaling  
- `spectramind corel-train` — COREL conformal training  
- `spectramind diagnose` — diagnostics suite (HTML, plots, overlays)  
- `spectramind ablate` — automated ablations  
- `spectramind submit` — selftest → predict → validate → bundle  
- `spectramind analyze-log` — parse CLI logs  
- `spectramind check-cli-map` — verify CLI↔file integrity  

---

## 7) Acceptance Criteria

- **Scientific integrity:** smooth, physically plausible, uncertainty-calibrated predictions  
- **Reproducibility:** any leaderboard submission tied to git SHA + config hash + DVC data  
- **Explainability:** SHAP + symbolic overlays must accompany μ/σ outputs  
- **Efficiency:** full pipeline ≤ 9 hrs; diagnostics ≤ 1 hr  
- **CI/Testing:** `spectramind selftest` and GitHub Actions pipeline must pass  

---

## 8) Roadmap

- TorchScript/JIT optimization for inference  
- Expanded symbolic influence maps in HTML dashboard  
- Optional GUI (FastAPI + React)  
- Automated ablation sweeps with leaderboard export  
- Leaderboard automation with artifact promotion  

---

**Status:** V50 architecture frozen for NeurIPS 2025 competition  
