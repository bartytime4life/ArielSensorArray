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
  • Multi-output regression head (all λ bins at once).  
  • Stable, interpretable, consistent across spectrum.  

**Design responses in V50:**  
- Residual-style encoders with domain priors: **Mamba SSM (FGS1)** + **Graph NN (AIRS λ-graph)**.  
- **Physics-informed symbolic losses**: smoothness, FFT coherence, non-negativity, asymmetry, molecular alignment.  
- Explicit **uncertainty calibration**: temperature scaling + COREL GNN.  
- Full **reproducibility stack**: Hydra YAML configs, DVC-tracked data, GitHub Actions CI, selftest CLI.  
- **Dashboard-ready diagnostics**: SHAP, symbolic overlays, latent projections, FFT, z-score maps.

---

## 🖼 Pipeline Overview

**End-to-end flow:**  

![Pipeline Overview](diagrams/pipeline_overview.svg)

**Stages:**  
1. **Ingestion** — FGS1/AIRS frames + metadata.  
2. **Calibration** — Bias/dark/flat/CDS, trace extraction, jitter correction, normalization.  
3. **Modeling** — Encoders (Mamba SSM + GNN), latent fusion, μ/σ decoders.  
4. **Uncertainty Calibration** — Temperature scaling + COREL GNN.  
5. **Diagnostics** — Metrics, FFT, SHAP, symbolic overlays, latent projections, HTML reports.  
6. **Submission** — Validator → bundle (CSV/ZIP + reports) → Kaggle upload.  
7. **Reproducibility & Ops** — Hydra, DVC, CI workflows, structured logs.

---

## 🖼 Architecture Stack

**Layered system design:**  

![Architecture Stack](diagrams/architecture_stack.svg)

**Layers:**  
- **L0 Entry Points** — Typer CLI (`spectramind …`), console UX, optional GUI hooks (HTML/Jupyter).  
- **L1 Orchestration** — Hydra configs, Makefiles, Poetry/Docker environments.  
- **L2 Data/Versioning** — DVC pipelines/remotes, Git commit hashing.  
- **L3 Calibration** — Bias/dark/flat/CDS, photometry, jitter correction, normalization.  
- **L4 Modeling** — FGS1 Mamba SSM, AIRS GNN, latent fusion, μ/σ decoders.  
- **L5 UQ** — Temperature scaling, COREL conformal GNN.  
- **L6 Diagnostics** — Metrics, FFT/smoothness/asymmetry, SHAP/attention, symbolic logic, UMAP/t-SNE projections.  
- **L7 Submission** — Validator, bundler, Kaggle artifact.  
- **L8 Observability/CI** — Structured telemetry, audit logs, GitHub Actions.  
- **L9 Runtime/Integrations** — CUDA/cuDNN, Kaggle GPUs/TPUs, Hugging Face, Ollama.

---

## 🖼 Symbolic Logic Layers

**Constraint engine and overlays:**  

![Symbolic Logic Layers](diagrams/symbolic_logic_layers.svg)

**Rule Families:**  
- **Non-negativity** — μ(λ) ≥ 0  
- **Smoothness** — penalize large ∂μ/∂λ  
- **Asymmetry Guard** — block unphysical lobes  
- **FFT Coherence** — enforce frequency plausibility  
- **Molecular Alignment** — H₂O, CO₂, CH₄ absorption bands  
- **Optional Monotonicity** — monotone segments in specific ranges  

**Evaluation & Diagnostics:**  
- Per-bin violation maps  
- Rule scoring & symbolic loss  
- HTML overlays, violation tables, heatmaps  
- Training hooks: curriculum weights, selective backprop

---

## 🖼 Kaggle CI Pipeline

**Continuous integration + leaderboard flow:**  

![Kaggle CI Pipeline](diagrams/kaggle_ci_pipeline.svg)

**Flow:**  
1. **GitHub Actions CI** — triggers on pushes/PRs.  
2. **Selftest** — validates configs, modules, CLI integrity.  
3. **Training** — Hydra-driven, DVC-backed runs.  
4. **Diagnostics** — SHAP, symbolic overlays, metrics, HTML dashboards.  
5. **Validation** — shape/bin checks, uncertainty coverage.  
6. **Packaging** — CSV/ZIP + `report.html`.  
7. **Submission** — Kaggle artifact push.  
8. **Artifact Registry** — models, plots, diagnostics, HTML bundles.

---

## 📑 Reports & Dashboards

- **`report.html`** — Compact reproducibility log with pipeline + config snapshots.  
- **`diagnostics_dashboard.html`** — Interactive diagnostics (symbolic overlays, SHAP, latent projections, calibration).  
- Both embed **assets/diagrams/** `.svg` files directly for CI-consistent visuals.

---

## 🛠 Reproducibility & CI

- **Hydra configs** — full parameter capture per run.  
- **DVC pipelines** — calibration → train → diagnose → submit, tied to Git commits.  
- **GitHub Actions** — selftest, diagnostics, mermaid export, artifact upload.  
- **Logs** — `logs/v50_debug_log.md` (append-only), JSONL event streams.  
- **Diagram tests** — `test_diagrams.py --render --strict` ensures visuals stay reproducible.  

---

## 🔗 References

- [Pipeline Overview Diagram](diagrams/pipeline_overview.svg)  
- [Architecture Stack Diagram](diagrams/architecture_stack.svg)  
- [Symbolic Logic Layers Diagram](diagrams/symbolic_logic_layers.svg)  
- [Kaggle CI Pipeline Diagram](diagrams/kaggle_ci_pipeline.svg)  
- [Reproducibility Report](report.html)  
- [Diagnostics Dashboard](diagnostics_dashboard.html)

---