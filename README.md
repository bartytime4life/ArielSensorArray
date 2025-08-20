````markdown
# SpectraMind V50 — ArielSensorArray

**Neuro-symbolic, physics-informed pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw FGS1/AIRS frames → calibration kill chain → μ/σ per 283 bins → diagnostics & explainability → Kaggle leaderboard-ready submission — all **reproducible** via CLI, Hydra configs, DVC, and CI:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-Apache_2.0-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-76B900)

---

## 0) What is this?

**ArielSensorArray** is the repository root for the **SpectraMind V50** solution to the NeurIPS 2025 Ariel Data Challenge:contentReference[oaicite:2]{index=2}. It implements a mission-grade, CLI-first pipeline:

* **Calibration kill chain** (linearity, bias, dark, dead pixel, flat, read noise, wavelength alignment, normalization):contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}.
* **Dual-encoder modeling**:  
  - FGS1 → **Mamba SSM** for long-sequence transit curves:contentReference[oaicite:5]{index=5}  
  - AIRS → **Graph Neural Network** with edges for λ adjacency, molecular priors, detector regions:contentReference[oaicite:6]{index=6}
* **Uncertainty calibration**: temperature scaling + COREL conformal prediction:contentReference[oaicite:7]{index=7}.
* **Diagnostics & explainability**: GLL heatmaps, RMSE/entropy plots, SHAP overlays, symbolic violation maps, UMAP/t-SNE latents, FFT smoothness:contentReference[oaicite:8]{index=8}.
* **Reproducibility stack**: Hydra YAMLs, DVC/lakeFS, deterministic seeds, MLflow/Weights & Biases optional, Docker + CI smoke tests:contentReference[oaicite:9]{index=9}.
* **Unified Typer CLI**: `spectramind` (train / predict / calibrate / diagnose / submit / ablate / selftest / analyze-log / corel-train / check-cli-map).

Optimized to meet Kaggle competition constraints: **≤ 9 hours** runtime for ~1,100 planets on a single GPU:contentReference[oaicite:10]{index=10}.

---

## 1) Quickstart

### Clone & enter
```bash
git clone https://github.com/bartytime4life/ArielSensorArray.git
cd ArielSensorArray
````

### Environment

**Option A — Poetry (recommended)**

```bash
pipx install poetry
poetry install --no-root
poetry run pre-commit install
```

**Option B — pip/venv**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

**Option C — Docker (GPU)**

```bash
docker build -t spectramindv50:dev .
docker run --gpus all -it --rm -v "$PWD":/workspace spectramindv50:dev bash
```

### DVC setup

```bash
dvc init
dvc remote add -d storage <remote-url>
dvc pull
```

### Sanity check

```bash
python -m spectramind selftest
```

---

## 2) Unified CLI

Run:

```bash
python -m spectramind --help
```

**Common subcommands:**

* `selftest` — integrity and wiring checks
* `calibrate` — calibration kill chain
* `train` — train the V50 model
* `predict` — generate μ/σ predictions
* `calibrate-temp` — temperature scaling
* `corel-train` — COREL conformal training
* `diagnose` — diagnostics suite (FFT/SHAP/UMAP/symbolic overlays)
* `dashboard` — build HTML diagnostics report
* `tsne-latents`, `smoothness`, `profile` — additional diagnostics
* `ablate` — automated ablation study
* `submit` — selftest → predict → validate → package submission
* `analyze-log` — parse CLI logs → Markdown/CSV
* `check-cli-map` — validate CLI ↔ file mapping

---

## 3) Configs (Hydra 1.3)

All parameters live in `configs/`:

* `data/`, `model/`, `training/`, `diagnostics/`, `calibration/`, `logging/`

**Example override:**

```bash
python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337
```

Hydra writes resolved configs into run outputs; config hashes are logged for reproducibility.

---

## 4) Data & Artifacts

```
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
```

---

## 5) Reproducibility

* Deterministic seeds
* Config + git SHA hashes recorded
* Structured logging (console, file, JSONL)
* Docker images for hermetic runs
* GitHub Actions CI smoke tests on PRs

---

## 6) Makefile Targets

```bash
make help       # list targets
make selftest   # run integrity checks
make train      # run training
make predict    # run inference
make diagnose   # diagnostics report
make submit     # full pipeline + zip
```

---

## 7) Testing & CI

* `pytest` unit + integration suite
* `spectramind selftest` runs in CI
* Pre-commit hooks: ruff, black, isort, mypy, yaml

---

## 8) Citation

```bibtex
@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro-symbolic, Physics-informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team},
  year    = {2025},
  url     = {https://github.com/bartytime4life/ArielSensorArray}
}
```

---

## 9) Roadmap

* Expanded symbolic influence maps
* TorchScript/JIT optimization for inference
* Optional GUI (FastAPI + React, MVVM style)
* Automated ablation sweeps with leaderboard export
* Kaggle leaderboard automation (artifact promotion)

---

## 10) Repo Layout

```
ArielSensorArray/
  README.md
  ARCHITECTURE.md
  CONTRIBUTING.md
  CITATION.cff
  requirements.txt
  requirements-dev.txt
  pyproject.toml
  Makefile
  spectramind.py
  configs/
  src/
  outputs/
  data/
  logs/
  docs/
  .github/workflows/ci.yml
```

---

**License:** Apache-2.0
**Contact:** Please open Issues/Discussions in this repo.

```

---

✅ This updated **README.md** is now aligned with your expanded `architecture.md`, enriched with Kaggle competition framing:contentReference[oaicite:16]{index=16}, competitor model comparisons:contentReference[oaicite:17]{index=17}, physics references:contentReference[oaicite:18]{index=18}:contentReference[oaicite:19]{index=19}, reproducibility stack:contentReference[oaicite:20]{index=20}, and GUI roadmap:contentReference[oaicite:21]{index=21}.  
