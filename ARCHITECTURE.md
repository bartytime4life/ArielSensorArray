# SpectraMind V50 — ArielSensorArray

**Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw Ariel **FGS1/AIRS frames** → **calibrated light curves** → **μ/σ spectra across 283 bins** → **diagnostics & symbolic overlays** → **leaderboard-ready submission** — **fully reproducible** via CLI, Hydra configs, DVC, CI, and Kaggle integration.

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-76B900)
![Kaggle](https://img.shields.io/badge/platform-Kaggle-20BEFF)

---

## 0) Overview

**ArielSensorArray** is the root repository for **SpectraMind V50**, a **NASA-grade, mission-critical pipeline** designed for the **NeurIPS 2025 Ariel Data Challenge**.

It integrates **astrophysical calibration**, **symbolic physics-informed modeling**, and **deep learning architectures** into a fully automated, CLI-first pipeline.

Core highlights:

- **Calibration Kill Chain** — ADC, bias, dark, flat, nonlinearity, dead-pixel masking, CDS, wavelength alignment, jitter correction.  
- **Dual Encoders**:
  - **FGS1 → Mamba SSM** for long-sequence transit modeling.
  - **AIRS → Graph Neural Network** with edge definitions (wavelength adjacency, molecule priors, detector regions).  
- **Decoders**: μ (mean spectrum) and σ (uncertainty), with support for quantile/diffusion heads.  
- **Uncertainty Calibration**: temperature scaling + **COREL conformal GNN**.  
- **Diagnostics**: GLL/entropy heatmaps, SHAP overlays, symbolic violation maps, FFT/UMAP/t-SNE, HTML dashboards.  
- **Symbolic Physics Layer**: smoothness, positivity, asymmetry, FFT suppression, radiative transfer, gravitational & micro-lensing corrections.  
- **Reproducibility**: Hydra configs, DVC/lakeFS, deterministic seeds, Git SHA + config hashes, CI pipelines.  
- **Unified CLI**: `spectramind` orchestrates everything (train, predict, calibrate, diagnose, ablate, submit, selftest, analyze-log, check-cli-map).  

The system is **Kaggle-ready**, optimized for **≤9 hr runtime** on ~1,100 planets with A100 GPUs.

---

## 1) Quickstart

### Clone

```bash
git clone https://github.com/bartytime4life/ArielSensorArray.git
cd ArielSensorArray

Environment Setup

Poetry (recommended):

pipx install poetry
poetry install --no-root
poetry run pre-commit install

Pip/venv:

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

Docker (GPU-ready):

docker build -t spectramindv50:dev .
docker run --gpus all -it --rm -v "$PWD":/workspace spectramindv50:dev bash

DVC Setup

dvc init
dvc remote add -d storage <remote-url>
dvc pull

Sanity Check

python -m spectramind selftest


⸻

2) Unified CLI

python -m spectramind --help

Key Commands:
	•	selftest — pipeline integrity
	•	calibrate — run full FGS1/AIRS calibration
	•	train — train the V50 model
	•	predict — μ/σ inference + submission artifacts
	•	calibrate-temp — temperature scaling
	•	corel-train — conformal calibration
	•	diagnose — symbolic + SHAP diagnostics
	•	dashboard — generate HTML diagnostics
	•	ablate — automated ablation sweeps
	•	submit — selftest → predict → validate → ZIP
	•	analyze-log — parse CLI logs → CSV/heatmap
	•	check-cli-map — validate CLI ↔ file mapping

⸻

3) Configs (Hydra 1.3)

All parameters live in configs/:
	•	data/, model/, training/, diagnostics/, calibration/, logging/.

Example:

python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337

Hydra generates snapshots + hashes for exact reproducibility.

⸻

4) Data & Artifacts

data/
  raw/         # raw FGS1/AIRS frames
  processed/   # calibrated spectra
  meta/        # metadata

outputs/
  checkpoints/ # model weights
  predictions/ # μ/σ spectra
  diagnostics/ # HTML/PNG/JSON reports
  calibrated/  # post-calibration cubes

logs/
  v50_debug_log.md  # append-only CLI log

All artifacts tracked with DVC.

⸻

5) Scientific Background
	•	Spectroscopy: spectral “fingerprints” from molecular absorption (H₂O, CO₂, CH₄, Na, K).
	•	Radiation physics: Planck law, blackbody emission, quantized photon energy.
	•	Gravitational lensing: mass-induced deflection distorts exoplanetary transit curves.
	•	Noise/systematics: spacecraft jitter, detector nonlinearity, cosmic rays.
	•	Symbolic priors: smoothness, asymmetry, positivity, FFT suppression.

⸻

6) Kaggle Integration
	•	Competition hardware/runtime constraints enforced.
	•	Pipeline tuned for 9 hr budget on A100 GPUs.
	•	Benchmarked against Kaggle baselines:
	•	Thang Do Duc — 0.329 LB baseline
	•	V1ctorious3010 — 80-block deep residual model
	•	Fawad Awan — Spectrum Regressor

⸻

7) Reproducibility
	•	Deterministic seeds + config hashes
	•	DVC-tracked datasets + checkpoints
	•	GitHub CI pre-flight checks
	•	Poetry + Docker environment parity
	•	Hydra YAML overrides logged per run

⸻

8) Roadmap
	•	TorchScript/JIT for fast inference
	•	Expanded symbolic overlays in HTML dashboards
	•	GUI dashboard (React + FastAPI)
	•	Kaggle leaderboard automation
	•	Micro-lensing & non-Gaussian noise calibration

⸻

9) Citation

@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro-symbolic, Physics-informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team and Andy Barta},
  year    = {2025},
  url     = {https://github.com/bartytime4life/ArielSensorArray}
}


⸻

10) License

MIT — see LICENSE.

