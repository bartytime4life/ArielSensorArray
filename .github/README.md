# 🚀 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

<p align="center">
  <!-- CI / QA -->
  <a href="https://github.com/your-org/your-repo/actions/workflows/ci.yml">
    <img alt="CI" src="https://github.com/your-org/your-repo/actions/workflows/ci.yml/badge.svg">
  </a>
  <a href="https://github.com/your-org/your-repo/actions/workflows/tests.yml">
    <img alt="Tests" src="https://github.com/your-org/your-repo/actions/workflows/tests.yml/badge.svg">
  </a>
  <a href="https://codecov.io/gh/your-org/your-repo">
    <img alt="Coverage" src="https://codecov.io/gh/your-org/your-repo/branch/main/graph/badge.svg">
  </a>
  <!-- Runtime / Tooling -->
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.1-success.svg">
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra-1f6feb.svg"></a>
  <a href="https://dvc.org/"><img alt="DVC" src="https://img.shields.io/badge/Data-DVC-945dd6.svg"></a>
  <a href="https://mlflow.org/"><img alt="MLflow" src="https://img.shields.io/badge/Tracking-MLflow-0194E2.svg"></a>
  <!-- Containers -->
  <a href="https://hub.docker.com/r/your-docker-namespace/spectramind-v50">
    <img alt="Docker pulls" src="https://img.shields.io/docker/pulls/your-docker-namespace/spectramind-v50.svg">
  </a>
  <a href="https://github.com/your-org/your-repo/pkgs/container/spectramind-v50">
    <img alt="GHCR" src="https://img.shields.io/badge/Container-GHCR-2b3137.svg">
  </a>
  <!-- Project -->
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

**SpectraMind V50** is a **neuro-symbolic, physics-informed** pipeline for the  
[NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/ariel-data-challenge-2025).

It predicts **exoplanet transmission spectra** — **μ** and **σ** across **283 bins** — from raw **FGS photometry** and **AIRS spectroscopy**.  
Principles: **NASA-grade reproducibility**, **CLI-first automation**, **Hydra configs**, **DVC versioning**, **symbolic physics constraints**.

---

## 🧩 Highlights

- **CLI-first** (`spectramind …`) powered by **Typer + Hydra** — discoverable, reproducible, tab-completion.
- **Physics-informed encoders**
  - **FGS1** → *Mamba* State-Space Model (long-context sequence).
  - **AIRS** → *Graph Neural Network* with molecular/temporal edges.
- **Dual decoders** → **μ** and **σ**, with **uncertainty calibration** (temperature scaling, COREL, conformal prediction).
- **Symbolic logic engine** enforcing *smoothness, non-negativity, molecular priors*.
- **Diagnostics**: GLL, FFT, UMAP/t-SNE, SHAP overlays, symbolic violations, calibration heatmaps, HTML dashboard.
- **MLOps**: Hydra configs, DVC-tracked data & checkpoints, optional MLflow tracking.
- **CI/CD**: Actions run *tests → diagnostics → submission bundling → docs build*.
- **Kaggle-aware**: ≤ **9h** end-to-end, **no-internet**, FGS1 **~58×** weighting in metric baked into loss.

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

Docker

docker pull your-docker-namespace/spectramind-v50:latest
# optional: mount configs/data
docker run --gpus all -it --rm \
  -v $PWD/configs:/app/configs \
  -v $PWD/outputs:/app/outputs \
  your-docker-namespace/spectramind-v50:latest bash

Python 3.10+ • CUDA 12.1 (GPU path) • Linux recommended

⸻

🚦 Quickstart

# 0) Run a pipeline self-test (fast sanity checks)
spectramind test

# 1) Calibrate telescope data (FGS + AIRS preprocessing)
spectramind calibrate data=ariel_nominal

# 2) Train model (Hydra overrides inline)
spectramind train model=v50 optimizer=adamw trainer.gpus=1 trainer.max_epochs=10

# 3) Run diagnostics & build HTML dashboard
spectramind diagnose dashboard --open

# 4) Package submission (runs selftest + manifest + zip)
spectramind submit --selftest

All configs live under configs/ and are Hydra-composable.
Override any param (e.g. optimizer.lr=1e-3 or data.fold=2).

⸻

🗺️ End-to-End Workflow

flowchart TD
  A["User"] -->|invokes| B["spectramind CLI"]
  B -->|compose + override| C["Hydra Configs<br/>configs/"]
  C --> D["Pipeline Orchestrator"]
  D --> E["Calibration<br/>FGS & AIRS processing"]
  E --> F["Training<br/>SSM + GNN → μ, σ"]
  F --> G["Diagnostics<br/>GLL, FFT, SHAP, Symbolic"]
  G --> H["Submission Bundler<br/>selftest, manifest, zip"]
  H --> I["Kaggle Leaderboard"]

  %% artifacts
  C -. "logs, overrides" .-> J["Artifacts<br/>outputs/YYYY-MM-DD/HH-MM-SS"]
  E -. "DVC-tracked data" .-> J
  F -. "checkpoints, metrics" .-> J
  G -. "HTML dashboard" .-> J
  H -. "submission.zip" .-> J


⸻

📂 Repository Structure

configs/        # Hydra YAML configs (data, model, train, diagnose, submit, etc.)
src/            # Core pipeline code (data, models, CLI, diagnostics, symbolic)
data/           # Raw + processed data (DVC tracked, not in Git)
tests/          # Unit tests + CLI integration tests
.github/        # CI workflows (lint, tests, diagnostics, docs, security scans)
docs/           # Architecture & diagrams (MkDocs/Material compatible)
artifacts/      # Generated diagnostics, dashboards, logs, manifests
outputs/        # Run outputs (submission, checkpoints, reports)


⸻

🔧 Configuration (Hydra)
	•	Base: configs/config_v50.yaml
	•	Groups: data/, model/, train/, diagnose/, submit/
	•	Override examples
	•	model=v50 encoder.fgs=mamba encoder.airs=gnn
	•	optimizer=adamw optimizer.lr=3e-4 scheduler=cosine
	•	train.seed=1337 trainer.gpus=1 trainer.precision=16
	•	Runtime logs: Config hash & overrides are recorded in v50_debug_log.md and run_hash_summary_v50.json.

Repro tip: Commit the exact config file (or pinned override line) alongside results.

⸻

🔬 Reproducibility & Security
	•	Determinism: seeds fixed; dry-run + self-test guardrails (spectramind selftest --deep)
	•	Data integrity: DVC + checksums; artifacts carry run hash & config snapshot
	•	No secrets: use .env, GitHub secrets, or DVC remotes (never commit tokens)
	•	Security policy: see SECURITY.md and advisory template
	•	CI scans: bandit, ruff, mypy, pip-audit, codeql, trivy pass before merge

⸻

📊 Kaggle Challenge Context
	•	Data: ESA Ariel simulation — FGS photometer + AIRS spectrometer
	•	Task: predict μ & σ for ~1,100 exoplanets
	•	Metric: Gaussian Log-Likelihood (GLL) — FGS1 weighted ~58×
	•	Constraints: no-internet, ≤ 9h GPU budget, reproducible kernels

Note: Loss weighting and the FGS1 head prioritize the white-light channel to align with GLL.

⸻

🧪 Testing Matrix

Layer	Tooling	Command(s)
Lint & Style	ruff, black (check), isort	pre-commit run --all-files
Types	mypy	make typecheck or mypy src/
Unit	pytest	pytest -q
CLI Self-test	built-in	spectramind test / spectramind selftest
Diagnostics Report	HTML dashboard	spectramind diagnose dashboard
Security	bandit, pip-audit, codeql, trivy	CI (auto)


⸻

🧭 Kaggle Usage (Notebook Tips)
	•	Start from the competition page → “New Notebook” → add dataset Ariel Data Challenge 2025
	•	Attach your weights/artifacts dataset (optional)
	•	Disable internet, enable GPU, set timeout ≤ 9h
	•	In notebook: !pip install -q spectramind-v50 (or mount src as dataset), then call CLI

⸻

🤝 Contributing

Contributions welcome!
Please read CONTRIBUTING.md and use the PR checklist:
PULL_REQUEST_REVIEW_CHECKLIST.md

PR gate: CI green + spectramind selftest + diagnostics bundle.

⸻

❓ FAQ (Quick)
	•	Why SSM (Mamba) for FGS1? Long-context, stable sequence modeling for white-light trends.
	•	Why GNN for AIRS? Captures spectral locality + molecule & temporal edges.
	•	How is uncertainty calibrated? Temperature scaling → COREL / conformal on held-out folds.
	•	Do you support CPU-only? Yes for diagnostics; training expects GPU (CUDA 12.1).
	•	Can I rerun older results? Yes via DVC + config hash pinned in the artifact manifest.

⸻

📜 License

MIT © 2025 · Your Org / Your Team
See LICENSE for details.

