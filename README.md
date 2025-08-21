Here’s a **fully updated and upgraded `README.md`** for your repo.
I’ve modernized it with all the SpectraMind V50 upgrades, Kaggle integration details, CI workflows (including the new Artifact Sweeper), and developer guidance.

````markdown
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

## 0) What is this?

**ArielSensorArray** is the root repository for **SpectraMind V50**, a **NASA-grade, reproducible pipeline** for the **NeurIPS 2025 Ariel Data Challenge**.

It provides a **CLI-first, physics-informed workflow** with:

- **Calibration Kill Chain** — ADC, bias, dark, flat, nonlinearity, dead-pixel masking, CDS, wavelength alignment, jitter correction.  
- **Dual-encoder modeling**:
  - **FGS1 → Mamba SSM** for long-sequence transit curves.  
  - **AIRS → Graph Neural Network** (edges = wavelength adjacency, molecules, detector regions).  
- **Decoders:** μ (mean spectrum), σ (uncertainty), with quantile/diffusion options.  
- **Uncertainty calibration:** temperature scaling + **SpectralCOREL conformal GNN**.  
- **Diagnostics:** GLL/entropy maps, SHAP overlays, symbolic rule scoring, FFT/UMAP/t-SNE, HTML dashboards.  
- **Symbolic physics layer:** smoothness, positivity, FFT suppression, asymmetry, radiative transfer, gravitational/micro-lensing corrections.  
- **Reproducibility:** Hydra configs, DVC/lakeFS, deterministic seeds, Git SHA + config hashes, CI pipelines.  
- **Unified Typer CLI:** `spectramind` orchestrates all (train, predict, calibrate, diagnose, ablate, submit, selftest, analyze-log, check-cli-map).  

Optimized for **≤9 hr runtime** on Kaggle A100 GPUs (~1,100 planets).

---

## 1) Quickstart

### Clone

```bash
git clone https://github.com/bartytime4life/ArielSensorArray.git
cd ArielSensorArray
````

### Environment Setup

**Poetry (recommended):**

```bash
pipx install poetry
poetry install --no-root
poetry run pre-commit install
```

**Pip/venv:**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Docker (GPU-ready):**

```bash
docker build -t spectramindv50:dev .
docker run --gpus all -it --rm -v "$PWD":/workspace spectramindv50:dev bash
```

### DVC Setup

```bash
dvc init
dvc remote add -d storage <remote-url>
dvc pull
```

### Sanity Check

```bash
python -m spectramind selftest
```

---

## 2) Unified CLI

```bash
python -m spectramind --help
```

**Key commands:**

* `selftest` — pipeline + config integrity
* `calibrate` — full FGS1/AIRS calibration chain
* `train` — train V50 model
* `predict` — μ/σ inference + submission artifacts
* `calibrate-temp` — temperature scaling
* `corel-train` — conformal calibration
* `diagnose` — symbolic + SHAP diagnostics
* `dashboard` — generate HTML diagnostics report
* `ablate` — automated ablation sweeps
* `submit` — selftest → predict → validate → ZIP bundle
* `analyze-log` — parse CLI logs → CSV/heatmap
* `check-cli-map` — validate CLI ↔ file mapping

---

## 3) Configs (Hydra 1.3)

All parameters live in `configs/` (`data/`, `model/`, `training/`, `diagnostics/`, `calibration/`, `logging/`).

**Example:**

```bash
python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337
```

Hydra snapshots + hashes guarantee reproducibility.

---

## 4) Data & Artifacts

```
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
```

All tracked by **DVC**.

---

## 5) Scientific Background

* **Spectroscopy:** molecular absorption (H₂O, CO₂, CH₄, Na, K).
* **Radiation physics:** photon quantization, Planck law, blackbody radiation, spectral lines.
* **Gravitational lensing:** mass-induced deflection of transit light curves.
* **Noise/systematics:** spacecraft jitter, cosmic rays, detector nonlinearity.
* **Symbolic priors:** smoothness, asymmetry, positivity, FFT suppression.

---

## 6) Kaggle Integration

* Pipeline hardened for Kaggle 9 hr GPU runtime.
* Benchmarked against public baselines:

  * **Thang Do Duc — 0.329 LB baseline**
  * **V1ctorious3010 — deep residual 80-block model**
  * **Fawad Awan — Spectrum Regressor**
* Kaggle notebooks + models analyzed and integrated.

---

## 7) CI Workflows

This repo uses **GitHub Actions** for CI/CD:

* **`ci.yml`** — full test + build pipeline.
* **`diagnostics.yml`** — runs symbolic + SHAP diagnostics, generates HTML dashboards.
* **`nightly-e2e.yml`** — nightly end-to-end training + inference smoke test.
* **`kaggle-submit.yml`** — automated submission packaging for Kaggle.
* **`lint.yml`** — style/linting (ruff, black, isort, mypy, yaml, markdown).
* **`artifact-sweeper.yml`** — cleans old artifacts & caches.

### 🧹 Artifact Sweeper

* Keeps artifacts newer than **14 days** (configurable).
* Preserves **open PR** and **tagged release** artifacts.
* Supports **dry-run mode** (default).
* Purges stale caches older than 14 days.

---

## 8) Reproducibility

* Deterministic seeds + config hashes.
* DVC-tracked datasets & checkpoints.
* GitHub CI preflight checks (unit + smoke tests).
* Poetry + Docker for environment parity.
* Hydra YAML overrides logged per run.

---

## 9) Roadmap

* TorchScript/JIT for fast inference.
* Expanded symbolic overlays in HTML dashboards.
* GUI dashboard (React + FastAPI).
* Kaggle leaderboard automation.
* Micro-lensing & non-Gaussian noise calibration.

---

## 10) Citation

```bibtex
@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro-symbolic, Physics-informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team and Andy Barta},
  year    = {2025},
  url     = {https://github.com/bartytime4life/ArielSensorArray}
}
```

---

## 11) License

MIT — see [LICENSE](./LICENSE).

```

Would you like me to also create a **Mermaid diagram** for the README (data → calibration → modeling → diagnostics → submission), so the architecture is visually clear at a glance?
```
