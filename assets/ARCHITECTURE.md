---

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

From **raw Ariel FGS1/AIRS frames** → **calibrated light curves** → **μ/σ spectra (283 bins)** → **diagnostics & symbolic overlays** → **leaderboard-ready submission**.  
Every step is reproducible via CLI, Hydra configs, DVC pipelines, GitHub Actions, and Kaggle runtime integration.

---

## 📊 Kaggle Model Insights (Context)

SpectraMind V50 builds directly on lessons from Kaggle baselines:

- **Thang Do Duc “0.329 LB” Baseline**  
  • Residual MLP, simple preprocessing, no σ estimation.  
  • Robust and reproducible but limited physics grounding.  

- **V1ctorious3010 “80bl-128hd-impact” (0.322 LB)**  
  • 80 residual blocks, 128 hidden size.  
  • Captures subtle features, but variance/overfitting risks; lower interpretability.  

- **Fawad Awan “Spectrum Regressor” (0.318 LB)**  
  • Multi-output regression head (predicts all bins simultaneously).  
  • Stable, interpretable, consistent across spectrum.  

**Design responses in V50:**  
- Residual-style encoders with domain priors: **Mamba SSM (FGS1)** + **Graph NN (AIRS λ-graph)**.  
- **Physics-informed symbolic losses**: smoothness, FFT coherence, non-negativity, asymmetry, molecular alignment.  
- Explicit **uncertainty calibration**: temperature scaling + COREL GNN.  
- Full **reproducibility stack**: Hydra YAML configs, DVC-tracked data, GitHub Actions CI, selftest CLI.  
- **Dashboard-ready diagnostics**: SHAP, symbolic overlays, latent projections, FFT, z-score maps.

> For visual + narrative comparison of baselines vs V50, see **[COMPARISON_GUIDE.md](COMPARISON_GUIDE.md)** and `comparison_overview.png`.

---

## 🖼 Pipeline Overview

**End-to-end flow:**  

![Pipeline Overview](diagrams/pipeline_overview.svg)

---

## 🖼 Architecture Stack

![Architecture Stack](diagrams/architecture_stack.svg)

---

## 🖼 Symbolic Logic Layers

![Symbolic Logic Layers](diagrams/symbolic_logic_layers.svg)

---

## 🖼 Kaggle CI Pipeline

![Kaggle CI Pipeline](diagrams/kaggle_ci_pipeline.svg)

---

## 📑 Reports & Dashboards

- **`report.html`** — Compact reproducibility log with pipeline + config snapshots.  
- **`diagnostics_dashboard.html`** — Interactive diagnostics (symbolic overlays, SHAP, latent projections, calibration).  
- **`COMPARISON_GUIDE.md`** — Explains `comparison_overview.png` in context.  

---

## 🛠 Reproducibility & CI

- **Hydra configs** — parameterized run capture.  
- **DVC pipelines** — calibration → train → diagnose → submit.  
- **GitHub Actions** — selftest, diagnostics, mermaid export.  
- **Logs** — `logs/v50_debug_log.md` (append-only), JSONL events.  
- **Diagram tests** — `test_diagrams.py --render --strict`.

---

## 🔗 References

- [Pipeline Overview Diagram](diagrams/pipeline_overview.svg)  
- [Architecture Stack Diagram](diagrams/architecture_stack.svg)  
- [Symbolic Logic Layers Diagram](diagrams/symbolic_logic_layers.svg)  
- [Kaggle CI Pipeline Diagram](diagrams/kaggle_ci_pipeline.svg)  
- [Comparison Guide](COMPARISON_GUIDE.md)  
- [Comparison Overview Image](comparison_overview.png)  
- [Reproducibility Report](report.html)  
- [Diagnostics Dashboard](diagnostics_dashboard.html)