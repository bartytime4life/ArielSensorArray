# 🚀 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

<p align="center">
  <a href="https://github.com/your-org/your-repo/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/your-org/your-repo/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://github.com/your-org/your-repo/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/your-org/your-repo/actions/workflows/tests.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/your-org/your-repo">
    <img alt="Coverage" src="https://codecov.io/gh/your-org/your-repo/branch/main/graph/badge.svg">
  </a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.1-success.svg">
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra-1f6feb.svg"></a>
  <a href="https://dvc.org/"><img alt="DVC" src="https://img.shields.io/badge/Data-DVC-945dd6.svg"></a>
  <a href="https://mlflow.org/"><img alt="MLflow" src="https://img.shields.io/badge/Tracking-MLflow-0194E2.svg"></a>
  <a href="https://hub.docker.com/r/your-docker-namespace/spectramind-v50">
    <img alt="Docker pulls" src="https://img.shields.io/docker/pulls/your-docker-namespace/spectramind-v50.svg">
  </a>
  <a href="https://github.com/your-org/your-repo/pkgs/container/spectramind-v50">
    <img alt="GHCR" src="https://img.shields.io/badge/Container-GHCR-2b3137.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://your-org.github.io/your-repo/">
    <img alt="Docs" src="https://img.shields.io/badge/Docs-Website-0ea5e9.svg">
  </a>
  <a href="https://www.kaggle.com/competitions/ariel-data-challenge-2025">
    <img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-NeurIPS%202025%20Ariel-20BEFF.svg">
  </a>
  <a href="https://github.com/your-org/your-repo/releases">
    <img alt="Release" src="https://img.shields.io/github/v/release/your-org/your-repo?display_name=tag&sort=semver">
  </a>
</p>

---

## 🌌 Overview

**SpectraMind V50** is a neuro-symbolic, physics-informed AI pipeline for the  
[NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025).  

It predicts **exoplanet transmission spectra (μ, σ across 283 bins)** from raw **FGS photometry** and **AIRS spectroscopy**.  
Design principles: **NASA-grade reproducibility, CLI-first automation, Hydra configs, DVC data versioning, symbolic physics constraints.**

---

## 🧩 Features

- **CLI-first** (`spectramind …`) with Typer + Hydra — discoverable, reproducible, tab-completion.
- **Physics-informed encoders**:  
  - FGS1 → Mamba State-Space Model (long-context sequence).  
  - AIRS → Graph Neural Network with molecular/temporal edges.
- **Dual decoders** → μ and σ, with uncertainty calibration (temperature scaling, COREL, conformal prediction).
- **Symbolic logic engine** enforcing smoothness, non-negativity, molecular priors.
- **Diagnostics**: FFT, UMAP/t-SNE, SHAP overlays, symbolic violations, calibration heatmaps.
- **MLOps**: Hydra configs, DVC-tracked data & checkpoints, MLflow tracking (optional).
- **CI/CD**: GitHub Actions runs training, diagnostics, submission packaging, HTML dashboard build.

---

## ⚙️ Installation

```bash
# clone repo
git clone https://github.com/your-org/your-repo.git
cd your-repo

# setup env (Poetry recommended)
poetry install

# or pip
pip install -e .
````

Docker image is published at:
`docker pull your-docker-namespace/spectramind-v50:latest`

---

## 🚦 Quickstart

```bash
# run a pipeline self-test
spectramind test

# calibrate telescope data
spectramind calibrate configs/data/nominal.yaml

# train model
spectramind train model=v50 optimizer=adamw trainer.gpus=1

# run diagnostics + dashboard
spectramind diagnose dashboard --open

# package submission
spectramind submit --selftest
```

All configs live under `configs/` and are Hydra-composable.
Override any param on the CLI (e.g. `optimizer.lr=1e-3`).

---

## 🗺️ End-to-End Workflow

```mermaid
flowchart TD
  A[User] -->|invokes| B[spectramind CLI]
  B -->|compose + override| C[Hydra Configs<br/>(configs/*.yaml)]
  C --> D[Pipeline Orchestrator]
  D --> E[Calibration<br/>(FGS/AIRS processing)]
  E --> F[Model Training<br/>(SSM + GNN → μ, σ)]
  F --> G[Diagnostics & Explainability<br/>(GLL, FFT, SHAP, Symbolic)]
  G --> H[Submission Bundler<br/>(selftest, manifest, zip)]
  H --> I[Kaggle Leaderboard]

  %% artifacts
  C -. logs, overrides .-> J[(Artifacts<br/>outputs/YYYY-MM-DD/HH-MM-SS)]
  E -. DVC tracked data .-> J
  F -. checkpoints, metrics .-> J
  G -. HTML dashboard .-> J
  H -. submission.zip .-> J
```

---

## 📂 Repository Structure

```
configs/        # Hydra YAML configs (data, model, train, diagnose, etc.)
src/            # Core pipeline code (data, models, CLI, diagnostics)
data/           # Raw + processed data (DVC tracked, not in Git)
tests/          # Unit tests + CLI integration tests
.github/        # CI workflows
docs/           # Architecture & diagrams
artifacts/      # Generated diagnostics, dashboards, logs
```

---

## 📊 Kaggle Challenge Context

* Dataset: **ESA Ariel telescope simulation** (FGS photometer, AIRS spectrometer).
* Goal: predict mean (μ) & uncertainty (σ) spectra for \~1,100 exoplanets.
* Runtime budget: ≤ 9h end-to-end on Kaggle GPU.
* Metric: **Gaussian log-likelihood (GLL)** + leaderboard evaluation.

---

## 🤝 Contributing

Contributions welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md).
All PRs must pass `spectramind test` and CI workflows before merge.

---

## 📜 License

MIT © 2025 \[Your Org / Your Team]
See [LICENSE](LICENSE) for details.

```
