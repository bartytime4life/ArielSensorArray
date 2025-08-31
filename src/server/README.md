# 🌌 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

## 0) Mission

**SpectraMind V50** is a **neuro-symbolic, physics-informed AI system** built for the
[NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025).

Goal: Predict **exoplanet transmission spectra (μ)** and **uncertainty (σ)** across **283 bins** from simulated ESA Ariel telescope data (FGS1 photometer + AIRS spectrometer).

Design principles:

* 🚀 **CLI-first, GUI-optional** (Typer CLI + Hydra configs, GUI is thin wrapper)
* 🛰 **NASA-grade reproducibility** (DVC + Git + Hydra + MLflow logging)
* 🌌 **Physics & Symbolics** (smoothness, nonnegativity, FFT priors, molecular fingerprints)
* 🧩 **Modularity** (encoders, decoders, constraints, calibration layers swappable)
* 🏆 **Kaggle-safe** (≤ 9h runtime, ≤ 16 GB GPU, deterministic runs)

---

## 1) Architecture Overview

```mermaid
flowchart TD
    A0[Raw Telescope Data]:::data --> A1[Calibration Kill Chain]:::stage
    A1 --> A2[Preprocessing & Augmentation]:::stage
    A2 --> A3[FGS1 Mamba Encoder]:::model
    A2 --> A4[AIRS GNN Encoder]:::model
    A3 --> A5[Fusion Cross-Attention]:::fusion
    A4 --> A5
    A5 --> A6[Multi-Scale Decoders (μ, σ)]:::model
    A6 --> A7[Loss Engine]:::loss
    A7 --> A8[Diagnostics + Dashboard]:::diag
    A8 --> A9[Submission Bundle]:::artifact

classDef data fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20;
classDef stage fill:#ede7f6,stroke:#5e35b1,color:#4527a0;
classDef model fill:#e3f2fd,stroke:#1565c0,color:#0d47a1;
classDef fusion fill:#fff3e0,stroke:#ef6c00,color:#e65100;
classDef loss fill:#ffebee,stroke:#c62828,color:#b71c1c;
classDef diag fill:#ede7f6,stroke:#5e35b1,color:#311b92;
classDef artifact fill:#f3e5f5,stroke:#6a1b9a,color:#4a148c;
```

* **FGS1 (135k timesteps)** → Mamba SSM (long-context, linear-time)
* **AIRS (11k spectra × 356 bins)** → Graph Neural Net (edges by wavelength, molecule, detector region)
* **Fusion**: Cross-attention or gated sum
* **Decoders**: mean μ, uncertainty σ, with optional diffusion/quantile heads
* **Losses**: Gaussian Log-Likelihood (GLL) + FFT smoothness + symbolic constraints

---

## 2) Workflow & Reproducibility

```mermaid
graph LR
    C0[User CLI (spectramind)] -->|Hydra Compose| C1[Pipeline Stage]
    C1 -->|DVC + Git| C2[Data/Model Versioning]
    C1 -->|Logging| C3[v50_debug_log.md + MLflow]
    C1 -->|Artifacts| C4[HTML Dashboard, Plots, JSON]

classDef cli fill:#e3f2fd,stroke:#1565c0;
classDef repo fill:#ede7f6,stroke:#5e35b1;
classDef log fill:#fff3e0,stroke:#ef6c00;
classDef art fill:#f3e5f5,stroke:#6a1b9a;
```

* **CLI**: `spectramind calibrate|train|diagnose|submit`
* **Configs**: Hydra YAMLs in `/configs/` (`data/`, `model/`, `trainer/`, `loss/`)
* **Data & Models**: Versioned with **DVC** (raw, processed, checkpoints)
* **Logging**: JSON + Rich console dashboards + `v50_debug_log.md`
* **Artifacts**: UMAP/t-SNE, SHAP overlays, GLL heatmaps, HTML reports

---

## 3) Quickstart

### Environment

```bash
# Clone
git clone https://github.com/your-org/spectramind-v50
cd spectramind-v50

# Setup
poetry install         # or pip install -r requirements.txt
dvc pull               # fetch data & checkpoints
```

### CLI Usage

```bash
# Run calibration
spectramind calibrate data=nominal calib=nominal

# Train a model
spectramind train model=v50 optimizer=adamw trainer=kaggle_safe

# Diagnose (UMAP, SHAP, symbolic overlays, FFT)
spectramind diagnose dashboard

# Package for Kaggle
spectramind submit --selftest
```

---

## 4) Scientific Foundations

SpectraMind V50 is grounded in:

* **AI Processing & Decoding Methods**
* **Patterns, Algorithms, & Fractals** for spectral self-similarity
* **NASA-grade Modeling & Simulation Standards**
* **Astrophysical Physics**: spectroscopy, gravitational lensing, radiation physics
* **Advanced Python & Hydra/YAML configs**
* **GUI Programming & Optional Dashboards**
* **Kaggle Ecosystem & Model Comparisons**

---

## 5) Roadmap

Planned upgrades:

* 🔬 **Experiment tracking** (MLflow/W\&B integration)
* 🧠 **Pretrained models** (Hugging Face ViT/TimeSformer + LoRA)
* 📊 **Uncertainty calibration** (COREL, conformal prediction)
* 🧩 **Symbolic diagnostics** (rule violations, fingerprints, cycle checks)
* 🖥 **Optional GUI** (React/FastAPI dashboard, already scaffolded in `src/gui/`)

---

## 6) Citation

If you use this repository in research:

```
@misc{SpectraMindV50_NeurIPS2025,
  author = {Barta, A. et al.},
  title = {SpectraMind V50: Neuro-Symbolic Exoplanet Spectroscopy System},
  year = {2025},
  howpublished = {\url{https://github.com/your-org/spectramind-v50}},
  note = {NeurIPS 2025 Ariel Data Challenge}
}
```

---

✅ This README ties together **code, configs, CLI, DVC, Kaggle, physics, and symbolic methods** into a single reproducible framework.

---
