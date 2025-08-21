# SpectraMind V50 — ArielSensorArray Architecture

**Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge**

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](../.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-orange)

---

## 🚀 North Star

From **raw Ariel FGS1/AIRS frames** → **calibrated light curves** → **μ/σ spectra across 283 bins** → **diagnostics & symbolic overlays** → **leaderboard-ready submission**.  
Fully reproducible via CLI, Hydra configs, DVC, CI workflows, and Kaggle integration.

---

## 📊 Kaggle Model Insights (Context)

SpectraMind V50 integrates lessons from publicly shared Kaggle baselines:

- **Baseline (0.329 LB)** — Residual MLP, simple and reproducible, but lacked physics/uncertainty.  
- **80-layer Deep Net (0.322 LB)** — High capacity, captured subtle features, but variance/overfitting risks.  
- **Spectrum Regressor (0.318 LB)** — Multi-output regression, stable and interpretable.  

**Design responses in V50:**  
- Use **Mamba SSM (FGS1)** and **Graph NN (AIRS)** encoders instead of brute-force depth.  
- Add **physics-informed symbolic losses**: smoothness, FFT coherence, non-negativity, asymmetry, molecular alignment.  
- Explicit **uncertainty calibration** (temperature scaling + COREL GNN).  
- Maintain full **reproducibility stack**: Hydra configs, DVC, GitHub CI, selftest.  

---

## 🖼 Pipeline Overview

The full end-to-end flow of SpectraMind V50:  

![Pipeline Overview](diagrams/pipeline_overview.svg)

**Stages:**  
1. **Ingestion** → FGS1/AIRS frames + metadata.  
2. **Calibration** → Bias/dark/flat/CDS, trace extraction, jitter correction, normalization.  
3. **Modeling** → Encoders (Mamba SSM + GNN), latent fusion, μ/σ decoders.  
4. **Uncertainty Calibration** → Temperature scaling + COREL GNN.  
5. **Diagnostics** → Metrics, SHAP, symbolic overlays, latent projections.  
6. **Submission** → Validator → bundle → Kaggle upload.  
7. **Reproducibility & Ops** → Hydra, DVC, CI, logs.  

---

## 🖼 Architecture Stack

Layered architecture view of the system:  

![Architecture Stack](diagrams/architecture_stack.svg)

**Layers:**  
- L0 Entry Points — CLI, console, optional GUI.  
- L1 Orchestration — Hydra configs, Makefiles, Poetry/Docker.  
- L2 Data/Versioning — DVC pipelines, remotes, Git.  
- L3 Calibration — Light curve preparation.  
- L4 Modeling — Encoders, fusion, decoders.  
- L5 UQ — Temperature scaling + COREL.  
- L6 Diagnostics — Metrics, FFT, SHAP, symbolic, UMAP/t-SNE.  
- L7 Submission — Validator, bundler, Kaggle artifact.  
- L8 Observability/CI — Logs, telemetry, CI workflows.  
- L9 Runtime/Integrations — CUDA, Kaggle GPUs, Hugging Face, Ollama.  

---

## 🖼 Symbolic Logic Layers

Constraint engine and diagnostic overlays:  

![Symbolic Logic Layers](diagrams/symbolic_logic_layers.svg)

**Key Rule Families:**  
- **Non-negativity** — μ(λ) ≥ 0  
- **Smoothness** — spectral gradient penalties  
- **Asymmetry Guard** — rule out unphysical lobes  
- **FFT Coherence** — enforce frequency-domain plausibility  
- **Molecular Alignment** — peak positions within H₂O, CO₂, CH₄ bands  
- **Optional Monotonicity** — localized monotone segments  

**Evaluation & Diagnostics:**  
- Per-λ violation maps  
- Rule scoring & global symbolic loss  
- HTML overlays, rule tables, heatmaps  
- Training hooks: curriculum weighting, selective backprop  

---

## 🖼 Kaggle CI Pipeline

End-to-end continuous integration and leaderboard workflow:  

![Kaggle CI Pipeline](diagrams/kaggle_ci_pipeline.svg)

**Flow:**  
1. **GitHub Actions CI** — triggers on PRs/commits.  
2. **Selftest** — verifies configs, modules, CLI integrity.  
3. **Training** — reproducible with Hydra + DVC.  
4. **Diagnostics** — metrics, symbolic overlays, HTML dashboards.  
5. **Validation** — shape/bin checks, coverage.  
6. **Packaging** — CSV/ZIP + reports.  
7. **Submission** — automatic or manual Kaggle push.  
8. **Artifacts Registry** — models, plots, logs stored for reproducibility.  

---

## 📑 Reports & Dashboards

- **`report.html`** — Compact reproducibility log with pipeline + config snapshots.  
- **`diagnostics_dashboard.html`** — Rich interactive dashboard (symbolic overlays, SHAP, latent projections, calibration).  

---

## 🛠 Reproducibility & CI

- **Hydra configs** → parameterized runs.  
- **DVC** → dataset & model versioning.  
- **GitHub Actions** → diagnostics, selftest, mermaid-export.  
- **Logs** → append-only (`logs/v50_debug_log.md`, JSONL events).  
- **CI-tested diagrams** → embedded directly in docs.  

---

## 🔗 References

- [Pipeline Overview Diagram](diagrams/pipeline_overview.svg)  
- [Architecture Stack Diagram](diagrams/architecture_stack.svg)  
- [Symbolic Logic Layers Diagram](diagrams/symbolic_logic_layers.svg)  
- [Kaggle CI Pipeline Diagram](diagrams/kaggle_ci_pipeline.svg)  

---