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

## 🖼 Architecture Diagram

High-level stack (rendered from `assets/diagrams/architecture_stack.mmd`):

![Architecture Stack](diagrams/architecture_stack.svg)

---

## ⚙️ Pipeline Layers

1. **Entry Points (UX)** — Typer CLI, lightweight console, optional GUI dashboard hooks.  
2. **Configuration & Orchestration** — Hydra configs, Makefile targets, Poetry/Docker environments.  
3. **Data & Versioning** — DVC pipelines/remotes, Git commits, artifact tracking.  
4. **Calibration & Feature Build** — Bias/dark/flat/CDS correction, trace extraction, jitter correction, normalization.  
5. **Modeling** — FGS1 Mamba SSM, AIRS GNN, latent fusion, μ/σ decoders.  
6. **Uncertainty Calibration** — Temperature scaling + COREL conformal GNN.  
7. **Diagnostics & Explainability** — Metrics (GLL, RMSE, MAE), FFT/smoothness, SHAP/attention, symbolic logic, latent UMAP/t-SNE projections.  
8. **Packaging & Submission** — Validator → CSV/ZIP bundle → Kaggle submission.  
9. **Observability & CI** — Structured telemetry (JSONL), audit logs, GitHub Actions CI, artifact registry.  
10. **Runtime & Integrations** — CUDA/cuDNN, Kaggle GPUs, Hugging Face, Ollama LLM explainers.  

---

## 📑 Reports & Dashboards

- **`report.html`** — Compact reproducibility report with pipeline + config snapshots.  
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

- [Pipeline Overview](diagrams/pipeline_overview.svg)  
- [Symbolic Logic Layers](diagrams/symbolic_logic_layers.svg)  
- [Kaggle CI Pipeline](diagrams/kaggle_ci_pipeline.svg)  

---