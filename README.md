
# SpectraMind V50 — ArielSensorArray

Neuro-symbolic, physics-informed pipeline for the **NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw FGS1/AIRS frames → calibrated light curves → μ/σ per 283 bins → diagnostics → leaderboard-ready submission — **fully reproducible** via CLI, Hydra configs, DVC, and CI.

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-76B900)

---

## 0) What is this?

**ArielSensorArray** is the repository root for the **SpectraMind V50** solution targeting the NeurIPS 2025 Ariel Data Challenge. It implements a mission-grade, CLI-first pipeline:

* **Calibration kill chain** for FGS1/AIRS (bias/dark/flat, trace extraction, wavelength alignment).
* **Dual-encoder modeling** (FGS1 long-sequence + AIRS spectral graph) with decoders for **μ** and **σ**.
* **Uncertainty calibration** (temperature scaling + COREL/conformal).
* **Diagnostics & explainability** (UMAP/t-SNE, SHAP overlays, symbolic constraints, FFT & smoothness).
* **Reproducibility** (Hydra YAMLs, DVC/lakeFS-ready, MLflow optional, Docker).
* **Unified Typer CLI**: `spectramind` (train / predict / calibrate / diagnose / submit / ablate / selftest / analyze-log / corel-train / check-cli-map).

The code is engineered to pass Kaggle constraints (end-to-end ≤9 hours for ~1,100 planets), with **deterministic seeds**, **config hashes**, and **audit logging** to `v50_debug_log.md`.

---

## 1) Quickstart (5 minutes)

### Clone and enter
```bash
git clone https://github.com/bartytime4life/ArielSensorArray.git
cd ArielSensorArray

Choose environment

Option A — Poetry (recommended)

pipx install poetry   # if not already installed
poetry install --no-root
poetry run pre-commit install

Option B — pip/venv

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install

Option C — Docker (GPU)

docker build -t spectramindv50:dev .
docker run --gpus all -it --rm -v "$PWD":/workspace spectramindv50:dev bash

DVC data setup

dvc init
dvc remote add -d storage <remote-url>
dvc pull

Sanity check

python -m spectramind selftest


⸻

2) Unified CLI

Run:

python -m spectramind --help

Common subcommands
	•	selftest — integrity and wiring checks
	•	calibrate — run calibration kill chain
	•	train — train the V50 model
	•	predict — generate μ/σ predictions
	•	calibrate-temp — temperature scaling
	•	corel-train — conformal calibration
	•	diagnose — diagnostics suite
	•	dashboard → build HTML report
	•	tsne-latents, smoothness, profile, etc.
	•	ablate — automated ablation study
	•	submit — full pipeline: selftest → predict → package zip
	•	analyze-log — parse CLI logs → Markdown/CSV
	•	check-cli-map — validate CLI ↔ file mapping

⸻

3) Configs (Hydra 1.3)

All parameters live in configs/:
	•	data/, model/, training/, diagnostics/, calibration/, logging/

Example override:

python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337

Hydra writes resolved configs into run outputs; config hashes are logged.

⸻

4) Data & Artifacts

data/
  raw/          # raw frames
  processed/    # calibrated spectra
  meta/         # metadata

outputs/
  checkpoints/
  predictions/
  diagnostics/
  calibrated/

logs/
  v50_debug_log.md


⸻

5) Reproducibility
	•	Deterministic seeds
	•	Config hashes
	•	Git SHA in logs
	•	Logging: console, file, JSONL
	•	Docker (optional)
	•	CI: smoke pipeline on PRs

⸻

6) Makefile

make help       # list targets
make selftest   # run integrity checks
make train      # run training
make predict    # run inference
make diagnose   # diagnostics report
make submit     # full pipeline + zip


⸻

7) Testing & CI
	•	Pytest suite
	•	spectramind selftest in CI
	•	Pre-commit hooks: ruff, black, isort, yaml

⸻

8) Citation

@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro-symbolic, Physics-informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team},
  year    = {2025},
  url     = {https://github.com/bartytime4life/ArielSensorArray}
}


⸻

9) Roadmap
	•	Unit tests for symbolic loss & ∂L/∂μ
	•	TorchScript/JIT inference
	•	Optional GUI panel (React+FastAPI)
	•	Leaderboard automation

⸻

10) Repo Layout

ArielSensorArray/
  README.md
  ARCHITECTURE.md
  CONTRIBUTING.md
  CITATION.cff
  pyproject.toml
  requirements.txt
  VERSION
  .env.example
  .pre-commit-config.yaml
  .gitignore
  dvc.yaml
  Makefile
  spectramind.py
  v50_debug_log.md
  configs/
  src/
  outputs/
  data/
  docs/
  logs/
  .github/workflows/ci.yml


⸻

License: MIT
Contact: Please open Issues/Discussions in this repo.

---
