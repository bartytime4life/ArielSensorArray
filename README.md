🚀 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

<p align="center">
  <a href="https://github.com/your-org/your-repo/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/your-org/your-repo/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://github.com/your-org/your-repo/actions/workflows/tests.yml"><img alt="Tests" src="https://github.com/your-org/your-repo/actions/workflows/tests.yml/badge.svg"></a>
  <a href="https://codecov.io/gh/your-org/your-repo"><img alt="Coverage" src="https://codecov.io/gh/your-org/your-repo/branch/main/graph/badge.svg"></a>
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-blue.svg">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.1-success.svg">
  <a href="https://hydra.cc/"><img alt="Hydra" src="https://img.shields.io/badge/Config-Hydra-1f6feb.svg"></a>
  <a href="https://dvc.org/"><img alt="DVC" src="https://img.shields.io/badge/Data-DVC-945dd6.svg"></a>
  <a href="https://mlflow.org/"><img alt="MLflow" src="https://img.shields.io/badge/Tracking-MLflow-0194E2.svg"></a>
  <a href="https://hub.docker.com/r/your-docker-namespace/spectramind-v50"><img alt="Docker pulls" src="https://img.shields.io/docker/pulls/your-docker-namespace/spectramind-v50.svg"></a>
  <a href="https://github.com/your-org/your-repo/pkgs/container/spectramind-v50"><img alt="GHCR" src="https://img.shields.io/badge/Container-GHCR-2b3137.svg"></a>
  <a href="https://opensource.org/licenses/MIT"><img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-yellow.svg"></a>
  <a href="https://your-org.github.io/your-repo/"><img alt="Docs" src="https://img.shields.io/badge/Docs-Website-0ea5e9.svg"></a>
  <a href="https://www.kaggle.com/competitions/ariel-data-challenge-2025"><img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-NeurIPS%202025%20Ariel-20BEFF.svg"></a>
  <a href="https://github.com/your-org/your-repo/releases"><img alt="Release" src="https://img.shields.io/github/v/release/your-org/your-repo?display_name=tag&sort=semver"></a>
</p>



⸻

🌌 Overview

SpectraMind V50 is a neuro-symbolic, physics-informed AI pipeline for the
NeurIPS 2025 Ariel Data Challenge.

It predicts exoplanet transmission spectra (μ, σ across 283 bins) from raw FGS photometry and AIRS spectroscopy.
Design principles: NASA-grade reproducibility, CLI-first automation, Hydra configs, DVC data versioning, symbolic physics constraints.

⸻

📚 Table of Contents
	•	Features
	•	Architecture
	•	Install
	•	Quickstart
	•	Hydra Configs
	•	Reproducibility & CI
	•	DVC Pipeline
	•	Docker & Compose
	•	Kaggle Notebook Path
	•	CLI Cheat-Sheet
	•	Repo Structure
	•	Security Policy
	•	FAQ / Troubleshooting
	•	Cite

⸻

🧩 Features
	•	CLI-first (spectramind …) via Typer + Hydra — discoverable, reproducible, tab-completion.
	•	Physics-informed encoders
	•	FGS1 → Mamba State-Space Model (long-context sequence).
	•	AIRS → Graph Neural Network with molecular/temporal/region edges.
	•	Dual decoders → μ and σ, with uncertainty calibration (temperature scaling, COREL, conformal prediction).
	•	Symbolic logic engine enforcing smoothness, non-negativity, molecular priors; violation heatmaps.
	•	Diagnostics: GLL, FFT, UMAP/t-SNE, SHAP overlays, symbolic violations, calibration reliability.
	•	MLOps: Hydra configs, DVC-tracked data & models, MLflow tracking (optional), run manifests + config hashes.
	•	CI/CD: GitHub Actions drive selftest → train → diagnostics → submission bundling → HTML dashboard.
	•	Containers: Dockerfile + docker-compose for GPU/CPU, FastAPI UI, JupyterLab, MkDocs, TensorBoard, Ollama.

⸻

🧠 Architecture

flowchart TD
  A["User / CI"] -->|invokes| B["spectramind CLI (Typer)"]
  B -->|compose + override| C["Hydra Configs (configs/*)"]
  C --> D["Calibration: FGS/AIRS → cubes"]
  D --> E["Encoders: FGS1=Mamba, AIRS=GNN"]
  E --> F["Decoders: μ, σ (GLL loss) + calibration"]
  F --> G["Diagnostics: GLL, FFT, SHAP, UMAP/t-SNE, Symbolic"]
  G --> H["Submission Bundle: manifest + zip"]
  H --> I["Kaggle Leaderboard"]

  C -. logs, overrides .-> J["Artifacts (outputs/*)"]
  D -. DVC .-> J
  E -. ckpts/metrics .-> J
  G -. HTML dashboard .-> J
  H -. submission.zip .-> J


⸻

🛠️ Install

# clone
git clone https://github.com/your-org/your-repo.git
cd your-repo

# Poetry (recommended)
poetry install

# or plain pip (editable)
pip install -e .

Docker image (optional):

docker pull your-docker-namespace/spectramind-v50:latest


⸻

🚦 Quickstart

Local, single-GPU:

# smoke check
spectramind selftest

# calibrate raw → calibrated cubes (paths via Hydra configs)
spectramind calibrate --config-dir config --config-name config_v50.yaml

# 1-epoch train (deterministic seed)
spectramind train +training.epochs=1 +training.seed=1337 --device cuda

# diagnostics + dashboard
spectramind diagnose dashboard --outdir outputs/diagnostics --open

# package submission (writes bundle.zip + manifest)
spectramind submit --pred outputs/predictions --bundle outputs/submission/bundle.zip --selftest

Makefile CI path (deterministic, non-interactive):

make ci           # validate → selftest → train → diagnose → analyze → artifacts
make ci-fast      # lighter path
make docs         # optional pandoc export if available


⸻

🧬 Hydra Configs

All configs live in config/ (or configs/ if you prefer). Compose/override any key on the CLI:

# example overrides
spectramind train \
  data.mode=kaggle \
  training.epochs=3 \
  model.encoder=fgs1_mamba_v2 \
  model.gnn=airs_gat_edgefeat \
  model.decoder=multi_scale_v3 \
  +loss.symbolic.smoothness_w=0.3 \
  +loss.symbolic.nonneg_w=0.1

Common groups:
	•	data.*: input paths, Kaggle mode, bins
	•	model.*: encoder/decoder selections, heads
	•	training.*: epochs, batch_size, lr, seed, device
	•	diagnostics.*: which plots/overlays, HTML options
	•	submission.*: Kaggle formatting, validation flags

⸻

♻️ Reproducibility & CI
	•	Deterministic defaults: fixed seeds, single-threaded CPU math in CI.
	•	Run manifests: outputs/manifests/ci_run_manifest_*.json with git SHA, config hash, device.
	•	Log analysis: spectramind analyze-log → outputs/log_table.{md,csv}.
	•	Security/SBOM (optional): make security (pip-audit/Bandit), make sbom (syft/grype).

GitHub Actions (suggested):
	•	ci.yml → selftest → train (1 epoch) → diagnostics → bundle artifact
	•	tests.yml → unit + CLI tests
	•	release.yml → container build/push, docs publish

⸻

📦 DVC Pipeline

dvc.yaml orchestrates: selftest → calibrate → train → predict → diagnostics → bundle → submit.

# initialize (first time)
dvc init
dvc repro             # run the full DAG
dvc exp run -S training.epochs=1 -S data.mode=kaggle
dvc plots show        # visualize CSV plots configured in dvc.yaml

Outputs:
	•	logs/calibration.json (metrics)
	•	outputs/models/metrics.json
	•	outputs/predictions/summary.json
	•	outputs/diagnostics/diagnostic_summary.json
	•	outputs/submission/bundle.zip, outputs/submission/manifest.json

⸻

🐳 Docker & Compose

Dev & services are defined in docker-compose.yml (GPU/CPU profiles, API, web, docs, Jupyter, TB, CI, Ollama).

# build base images
docker compose build

# GPU shell
docker compose --profile gpu up -d spectramind-gpu && docker compose exec spectramind-gpu bash

# API server (FastAPI on 9000)
docker compose --profile api up api

# JupyterLab (8888), Docs (8000), TB (6006)
docker compose --profile lab up jupyter
docker compose --profile docs up docs
docker compose --profile viz up tensorboard

Volumes persist: pip/poetry caches, HF cache, artifacts, logs.

⸻

🧪 Kaggle Notebook Path

Two single-file notebooks are provided (or generate via CLI scaffolding):
	1.	Pretraining (MAE + optional contrastive; FGS1/AIRS masking; AMP + cosine warmup)
	2.	Process & Submit (calibrate → infer → package submission)

Guidelines:
	•	Respect the ≤ 9h runtime budget for ~1,100 planets.
	•	Use --device auto with AMP; prefer efficient binning for FGS1.
	•	Limit external network calls; rely on shipped assets/configs.
	•	Always emit submission.csv and a small HTML diagnostics section.

⸻

🧰 CLI Cheat-Sheet

# Integrity & environment
spectramind selftest               # fast checks
spectramind selftest --deep        # deeper Hydra/DVC checks
spectramind env-capture            # snapshot env for manifests

# Core pipeline
spectramind calibrate --config-dir config --config-name config_v50.yaml
spectramind train +training.epochs=1 +training.seed=1337 --device cpu
spectramind predict --model outputs/models --out-csv outputs/predictions/submission.csv

# Diagnostics
spectramind diagnose smoothness --outdir outputs/diagnostics
spectramind diagnose dashboard --outdir outputs/diagnostics --html-out outputs/diagnostics/report.html

# Submission
spectramind submit --pred outputs/predictions --bundle outputs/submission/bundle.zip --meta outputs/submission/manifest.json --selftest

# Logs
spectramind analyze-log --md outputs/log_table.md --csv outputs/log_table.csv

Make targets of interest:

make ci          # validate → selftest → train → diagnose → analyze → artifacts
make ci-fast     # lighter path
make ci-calibration
make predict-e2e
make ablate-ci
make docs docs-clean
make kaggle-submit ALLOW_KAGGLE_SUBMIT=1 COMPETITION=neurips-2025-ariel
make sbom security


⸻

📂 Repository Structure

config/                         # Hydra YAMLs (data/model/training/diagnostics/submission)
src/
  cli/                          # Typer entrypoints & wrappers
  models/                       # FGS1 Mamba, AIRS GNN, decoders
  pipeline/                     # calibration pipeline
  predict/                      # prediction packaging
  diagnostics/                  # SHAP, UMAP/t-SNE, symbolic, HTML report
  server/                       # FastAPI app (optional)
tests/                          # unit + CLI tests
outputs/                        # artifacts (DVC-managed)
logs/                           # json logs & run manifests
.github/workflows/              # CI pipelines
docker-compose.yml              # profiles for dev/services
Makefile                        # mission-grade CI/Kaggle targets
dvc.yaml                        # pipeline stages (selftest→submit)


⸻

🔐 Security Policy
	•	Private reporting only via GitHub Security Advisory (preferred) or security email.
	•	SLA targets: Acknowledge ≤ 2 business days; Critical/High fixes ≤ 30 days; Medium/Low ≤ 90 days.
	•	Supply chain: SBOM via make sbom, audits via make security, pinned deps/actions where feasible.
	•	See SECURITY.md for Safe Harbor, scope, and mitigation playbook.

⸻

❓ FAQ / Troubleshooting

Q: CUDA mismatch / torch wheel fails?
A: Use the provided Docker images, or set TORCH_WHL_INDEX to match your CUDA (https://download.pytorch.org/whl/cu121). CPU mode: --device cpu.

Q: Kaggle path resolution?
A: Use data.mode=kaggle. The notebook auto-detects /kaggle/input/ariel-data-challenge-2025. Outputs go to /kaggle/working.

Q: Dashboard too heavy for CI?
A: Run diagnose dashboard --no-umap --no-tsne (the Makefile already tries a lightweight mode first).

Q: Submission validator failing?
A: Ensure submission.csv has required headers/order; run spectramind submit --selftest to print validator hints.

Q: Determinism?
A: In CI we pin threads and seeds. Local runs may vary with GPU nondeterminism; use CUBLAS_WORKSPACE_CONFIG=:16:8 and disable TF32 if needed.

⸻

📜 Cite

If you use SpectraMind V50 in academic work or the NeurIPS Ariel Data Challenge, please cite:

@software{spectramind_v50_2025,
  author  = {Your Team},
  title   = {SpectraMind V50: Neuro-symbolic, Physics-informed Spectroscopy for the NeurIPS 2025 Ariel Data Challenge},
  year    = {2025},
  url     = {https://github.com/your-org/your-repo},
  version = {v0.50.0}
}

—

MIT © 2025 Your Org / Your Team — see LICENSE.