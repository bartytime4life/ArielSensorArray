
---

# SpectraMind V50 — ArielSensorArray

**Neuro-symbolic, physics-informed pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw FGS1/AIRS frames → calibrated light curves → μ/σ per 283 bins → diagnostics → leaderboard-ready submission — **fully reproducible** via CLI, Hydra configs, DVC, CI, and Kaggle integration.

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

**ArielSensorArray** is the root repository for **SpectraMind V50**, our end-to-end solution to the **NeurIPS 2025 Ariel Data Challenge**.

It implements a **mission-grade, CLI-first pipeline** with:

* **Calibration kill chain** for FGS1/AIRS (ADC, bias, dark, flat, dead-pixel map, nonlinearity, linearity correction, trace extraction, wavelength alignment, normalization).
* **Dual-encoder modeling**:

  * FGS1 → **Mamba SSM** for long-sequence transit curves.
  * AIRS → **Graph Neural Network** (edges: wavelength adjacency, molecular priors, detector regions).
* **Decoders**: μ (mean spectrum) and σ (uncertainty), with support for quantile/diffusion heads.
* **Uncertainty calibration**: temperature scaling + **COREL conformal prediction**.
* **Diagnostics & explainability**: GLL/entropy, SHAP overlays, symbolic rule violations, FFT, UMAP/t-SNE latent maps, HTML dashboard.
* **Symbolic physics layer**: smoothness, positivity, FFT suppression, asymmetry, radiative-transfer priors, gravitational lensing & micro-lensing modeling.
* **Reproducibility**: Hydra YAMLs, DVC/lakeFS, deterministic seeds, config hashing, CI pipelines.
* **Unified Typer CLI**: `spectramind` (train / predict / calibrate / calibrate-temp / corel-train / diagnose / ablate / submit / selftest / analyze-log / check-cli-map).

The pipeline is Kaggle-compliant: ≤9 hrs runtime across \~1,100 planets on Kaggle A100 GPUs.

---

## 1) Quickstart

### Clone & enter

```bash
git clone https://github.com/bartytime4life/ArielSensorArray.git
cd ArielSensorArray
```

### Environment setup

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

**Option C — Docker (GPU-ready)**

```bash
docker build -t spectramindv50:dev .
docker run --gpus all -it --rm -v "$PWD":/workspace spectramindv50:dev bash
```

### DVC data setup

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

Common subcommands:

* `selftest` — integrity checks
* `calibrate` — run calibration kill chain
* `train` — train V50 model
* `predict` — μ/σ inference + submission
* `calibrate-temp` — temperature scaling
* `corel-train` — conformal calibration
* `diagnose` — diagnostics suite
* `dashboard` — HTML diagnostics report
* `ablate` — automated ablation sweeps
* `submit` — selftest → predict → validate → bundle ZIP
* `analyze-log` — parse CLI logs → Markdown/CSV/heatmaps
* `check-cli-map` — verify CLI↔file mapping

---

## 3) Configs (Hydra 1.3)

All parameters live in `configs/`:

* `data/`, `model/`, `training/`, `diagnostics/`, `calibration/`, `logging/`

Example:

```bash
python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337
```

Hydra writes resolved configs into run outputs; config hashes logged to `v50_debug_log.md`.

---

## 4) Data & Artifacts

```
data/
  raw/         # raw frames
  processed/   # calibrated spectra
  meta/        # metadata

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

* Deterministic seeds & curriculum configs
* Config hashes + Git SHA logged
* Hydra YAMLs with overrides
* DVC/lakeFS for datasets
* Docker & Poetry for environments
* GitHub Actions CI: selftest + smoke pipelines

---

## 6) Kaggle Integration

* Pipeline optimized for **≤9 hr Kaggle runtime**.
* Compatible with Kaggle Notebooks & datasets.
* Models benchmarked against Kaggle baselines:

  * Thang Do Duc’s 0.329 LB baseline
  * V1ctorious3010’s deep residual model
  * Fawad Awan’s Spectrum Regressor

---

## 7) Scientific Priors

* **Astrophysics**: transmission spectroscopy, radiative transfer, gas absorption (H₂O, CO₂, CH₄, Na, K).
* **Systematics**: spacecraft jitter, detector dead pixels, nonlinearity, cosmic rays.
* **Gravitational lensing & micro-lensing corrections**: flux amplification patterns, time-domain distortions.
* **Symbolic constraints**: smoothness, asymmetry, FFT suppression, positivity.

---

## 8) Roadmap

* TorchScript/JIT inference
* Expanded symbolic influence maps in HTML
* GUI (React+FastAPI) dashboard
* Kaggle leaderboard automation
* Advanced calibration: micro-lensing, non-Gaussian noise modeling

---

## 9) Citation

If you use this repo:

```bibtex
@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro-symbolic, Physics-informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team and Andy Barta},
  year    = {2025},
  url     = {https://github.com/bartytime4life/ArielSensorArray}
}
```

---

## 10) License

MIT — see [LICENSE](./LICENSE).

---

