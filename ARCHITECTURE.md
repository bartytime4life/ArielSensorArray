```markdown
# SpectraMind V50 — Engineering Vision & Master Architecture
**Neuro-Symbolic, Physics-Informed AI Pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** Deliver a reproducible, explainable, physics-informed system that ingests Ariel FGS1/AIRS cubes, outputs μ and σ for 283 bins, passes calibration and diagnostics, and packages a competition-valid submission — all within Kaggle’s 9-hour runtime envelope:contentReference[oaicite:0]{index=0}.

---

## 0) Purpose & Scope

This document is the **engineering blueprint** for SpectraMind V50:

- **For engineers:** defines modules, contracts, workflows, performance budgets, and acceptance criteria.  
- **For scientists:** encodes the physics and symbolic priors into the machine learning architecture.  
- **For ops/MLOps:** prescribes reproducibility, CI/CD, and data governance practices:contentReference[oaicite:1]{index=1}.  

---

## 1) System Architecture

SpectraMind V50 is structured as **four tightly coupled subsystems**:

1. **Calibration Kill Chain**  
   - ADC → bias → dark → flat → nonlinearity → trace extraction → wavelength alignment → normalization  
   - Outputs: clean, aligned lightcurves and AIRS/FGS1 spectra.  
   - Implemented in `src/asa/calib/`.  
   - Mirrors **NASA/ESA spectroscopic calibration practice**:contentReference[oaicite:2]{index=2}.

2. **Modeling Core**  
   - **Encoders:**  
     - FGS1 → **Mamba SSM** for long-sequence transit curves (linear-time, long context).  
     - AIRS → **Graph Neural Network** (edge types: wavelength adjacency, molecule priors, detector regions).  
   - **Decoders:**  
     - μ (mean) spectrum head  
     - σ (uncertainty) head (temperature scaling + COREL conformal support)  
     - Optional diffusion/quantile decoders.  
   - Design informed by Kaggle baselines and deep MLP approaches, but upgraded with physics and uncertainty layers:contentReference[oaicite:3]{index=3}.  
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
   - DVC + lakeFS integration for data versioning:contentReference[oaicite:4]{index=4}  
   - Logging: console/file/JSONL, Rich console UI, optional MLflow/W&B:contentReference[oaicite:5]{index=5}  
   - Reproducibility tracked via `v50_debug_log.md`.  

---

## 2) Physics & Scientific Foundations

- **Ariel Mission Context:** ESA’s Ariel telescope observes ~1,000 exoplanets (0.5–7.8 μm) via transmission spectroscopy:contentReference[oaicite:6]{index=6}.  
- **Observational Challenges:** spacecraft jitter, detector nonlinearity, cosmic radiation, background noise:contentReference[oaicite:7]{index=7}.  
- **Spectral Physics Priors:**  
  - Radiative transfer: $I(ν) ≈ I₀(ν) e^{−τ(ν)}$  
  - Gas fingerprints: H₂O, CO₂, CH₄ across bins  
  - Smoothness & asymmetry constraints on spectra  
- **Symbolic AI Layer:**  
  - Smoothness, positivity, FFT suppression  
  - Radiative-transfer alignment rules embedded as loss terms:contentReference[oaicite:8]{index=8}  
  - Conformal prediction (COREL) for calibrated σ:contentReference[oaicite:9]{index=9}  

---

## 3) Data Flow

```

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

```

---

## 4) Reproducibility Standards

- **Hydra configs:** all params (data, model, training, diagnostics, calibration):contentReference[oaicite:10]{index=10}  
- **DVC tracked data:** raw/calibrated cubes, checkpoints, submission bundles:contentReference[oaicite:11]{index=11}  
- **Deterministic seeds:** fixed RNG, ordered DataLoader (toy configs)  
- **Audit log:** every CLI call recorded in `v50_debug_log.md` (git SHA, config hash, CUDA info):contentReference[oaicite:12]{index=12}  
- **CI pipeline:** GitHub Actions runs selftest + smoke pipeline on PRs:contentReference[oaicite:13]{index=13}  

---

## 5) Performance Budget

- **End-to-end:** ≤ 9 hours on Kaggle A100 (~1,100 planets):contentReference[oaicite:14]{index=14}  
- **FGS1 encoder:** optimized Mamba SSM with downsampling  
- **AIRS encoder:** batched GNN with sparse adjacency  
- **Decoders:** lightweight heads (μ, σ), optional distillation  
- **Diagnostics:** parallelizable, HTML/PNG export, non-blocking  

---

## 6) CLI Coverage

All major functions are accessible via CLI:contentReference[oaicite:15]{index=15}:

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

- **Scientific integrity:** smooth, physically plausible, uncertainty-calibrated predictions:contentReference[oaicite:16]{index=16}:contentReference[oaicite:17]{index=17}  
- **Reproducibility:** any leaderboard submission tied to git SHA + config hash + DVC data:contentReference[oaicite:18]{index=18}  
- **Explainability:** SHAP + symbolic overlays must accompany μ/σ outputs  
- **Efficiency:** full pipeline ≤ 9 hrs; diagnostics ≤ 1 hr:contentReference[oaicite:19]{index=19}  
- **CI/Testing:** `spectramind selftest` and GitHub Actions pipeline must pass:contentReference[oaicite:20]{index=20}  

---

## 8) Roadmap

- TorchScript/JIT optimization for inference:contentReference[oaicite:21]{index=21}  
- Expanded symbolic influence maps in HTML dashboard  
- Optional GUI (FastAPI + React, MVVM style):contentReference[oaicite:22]{index=22}  
- Automated ablation sweeps with leaderboard export  
- Leaderboard automation with artifact promotion (Kaggle submission bundling):contentReference[oaicite:23]{index=23}  

---

## 9) Relation to Kaggle Competitors

- **Baselines (0.329 LB MLP)** show that simple dense nets can perform decently but lack physics and uncertainty:contentReference[oaicite:24]{index=24}.  
- **Deep MLPs (80bl-128hd)** push capacity but are inefficient under 9-hour limits:contentReference[oaicite:25]{index=25}.  
- **Spectrum Regressor** demonstrates stable multi-output regression but no uncertainty:contentReference[oaicite:26]{index=26}.  
- **V50 integrates best ideas**: calibration-first, topology-aware encoders, calibrated uncertainties, symbolic physics, and diagnostics.  

---

**Status:** V50 architecture frozen for NeurIPS 2025 competition; extensions tracked in roadmap.  
```
