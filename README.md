
⸻

SpectraMind V50 — ArielSensorArray

Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge

North Star: From raw Ariel FGS1/AIRS frames → calibrated light curves → μ/σ spectra across 283 bins → diagnostics & symbolic overlays → leaderboard-ready submission — fully reproducible via CLI, Hydra configs, DVC, CI, and Kaggle integration ￼ ￼.

⸻



⸻

0) What is this?

ArielSensorArray is the root repository for SpectraMind V50, our end-to-end NASA-grade pipeline for the NeurIPS 2025 Ariel Data Challenge ￼.

It provides a CLI-first, reproducible, physics-informed workflow with:
	•	Calibration Kill Chain — ADC, bias, dark, flat, nonlinearity, dead-pixel masking, CDS, wavelength alignment, jitter correction ￼.
	•	Dual-encoder modeling:
	•	FGS1 → Mamba SSM for long-sequence transit curves.
	•	AIRS → Graph Neural Network (edges = wavelength adjacency, molecules, detector regions).
	•	Decoders: μ (mean spectrum), σ (uncertainty), with quantile/diffusion options.
	•	Uncertainty calibration: temperature scaling + COREL conformal GNN ￼.
	•	Diagnostics: GLL/entropy maps, SHAP overlays, symbolic rule scoring, FFT/UMAP/t-SNE, HTML dashboards ￼.
	•	Symbolic physics layer: smoothness, positivity, FFT suppression, asymmetry, radiative transfer, gravitational/micro-lensing corrections ￼.
	•	Reproducibility: Hydra configs, DVC/lakeFS, deterministic seeds, Git SHA + config hashes, CI pipelines ￼ ￼.
	•	Unified Typer CLI: spectramind orchestrates all (train, predict, calibrate, diagnose, ablate, submit, selftest, analyze-log, check-cli-map).

The pipeline is Kaggle-ready: optimized for ≤9 hr runtime on ~1,100 planets using A100 GPUs ￼.

⸻

1) Quickstart

Clone

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

Key commands:
	•	selftest — pipeline + config integrity
	•	calibrate — full FGS1/AIRS calibration chain
	•	train — train V50 model
	•	predict — μ/σ inference + submission artifacts
	•	calibrate-temp — temperature scaling
	•	corel-train — conformal calibration
	•	diagnose — symbolic + SHAP diagnostics
	•	dashboard — generate HTML diagnostics report
	•	ablate — automated ablation sweeps
	•	submit — full selftest → predict → validate → ZIP bundle
	•	analyze-log — parse CLI logs → CSV/heatmap
	•	check-cli-map — validate CLI ↔ file mapping ￼

⸻

3) Configs (Hydra 1.3)

All parameters live in configs/ (data/, model/, training/, diagnostics/, calibration/, logging/).

Example:

python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337

Hydra snapshots + hashes ensure exact reproducibility ￼.

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

All artifacts tracked by DVC ￼.

⸻

5) Scientific Background
	•	Spectroscopy: spectral “fingerprints” from molecular absorption (H₂O, CO₂, CH₄, Na, K) ￼.
	•	Radiation physics: photon quantization, Planck law, blackbody radiation, spectral lines ￼.
	•	Gravitational lensing: mass-induced deflection distorts exoplanetary transit light curves ￼.
	•	Noise/systematics: spacecraft jitter, cosmic rays, detector nonlinearity ￼.
	•	Symbolic priors: smoothness, asymmetry, positivity, FFT suppression ￼.

⸻

6) Kaggle Integration
	•	Kaggle competitions run on restricted hardware/time ￼.
	•	Pipeline optimized for 9 hr budget on A100 GPUs ￼.
	•	Benchmarked against public Kaggle baselines:
	•	Thang Do Duc — 0.329 LB baseline ￼
	•	V1ctorious3010 — deep residual 80-block model ￼
	•	Fawad Awan — Spectrum Regressor ￼

⸻

7) Reproducibility
	•	Deterministic seeds + config hashes ￼
	•	DVC-tracked datasets and checkpoints ￼
	•	GitHub CI pre-flight checks (unit + smoke tests) ￼
	•	Poetry + Docker environment parity ￼
	•	Hydra YAML overrides logged per run ￼

⸻

8) Roadmap
	•	TorchScript/JIT for fast inference
	•	Expanded symbolic overlays in HTML
	•	GUI dashboard (React + FastAPI) ￼
	•	Kaggle leaderboard automation
	•	Micro-lensing & non-Gaussian noise calibration ￼

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

⸻
