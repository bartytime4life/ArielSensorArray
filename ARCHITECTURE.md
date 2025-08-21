
# SpectraMind V50 — ArielSensorArray

**Neuro-symbolic, physics-informed AI pipeline for the NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw Ariel **FGS1/AIRS frames** → **calibrated light curves** → **μ/σ spectra across 283 bins** → **diagnostics & symbolic overlays** → **leaderboard-ready submission** — fully reproducible via CLI, Hydra configs, DVC, CI, and Kaggle integration.

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-76B900)
![Kaggle](https://img.shields.io/badge/platform-Kaggle-20BEFF)

⸻

## 0) Overview

**ArielSensorArray** is the engineering blueprint of **SpectraMind V50**: a **NASA-grade, mission-critical pipeline** for the NeurIPS 2025 Ariel Data Challenge.

It integrates **astrophysical calibration**, **symbolic physics-informed modeling**, and **deep learning architectures** into a reproducible, **CLI-first** workflow.

### 🛰️ Core Highlights
- **Calibration Kill Chain** — ADC, bias, dark, flat, nonlinearity, dead-pixel masking, CDS, wavelength alignment, jitter correction.  
- **Dual Encoders**:  
  • **FGS1 → Mamba SSM** (long-sequence transit modeling)  
  • **AIRS → Graph Neural Network** (edges = wavelength adjacency, molecule priors, detector regions)  
- **Decoders** — μ (mean spectrum), σ (uncertainty), quantile & diffusion heads.  
- **Uncertainty Calibration** — temperature scaling + **SpectralCOREL GNN**.  
- **Diagnostics** — GLL/entropy heatmaps, SHAP overlays, symbolic rule maps, FFT/UMAP/t‑SNE, HTML dashboards.  
- **Symbolic Physics Layer** — smoothness, positivity, asymmetry, FFT suppression, radiative transfer, gravitational & micro‑lensing corrections.  
- **Reproducibility** — Hydra configs, DVC/lakeFS, deterministic seeds, Git SHA + config hashes, CI pipelines.  
- **Unified CLI** — `spectramind` orchestrates train, predict, calibrate, diagnose, ablate, submit, selftest, analyze-log, check-cli-map.  

⏱ Optimized for **≤9 hr runtime** on ~1,100 planets with Kaggle A100 GPUs.

⸻

## 1) System Architecture

```mermaid
flowchart TD
    A[Raw Data: FGS1/AIRS Frames] --> B[Calibration Kill Chain]
    B --> C1[FGS1 → Mamba SSM]
    B --> C2[AIRS → Graph Neural Net]
    C1 --> D[Multi-Scale Fusion]
    C2 --> D[Multi-Scale Fusion]
    D --> E1[μ Decoder (mean spectrum)]
    D --> E2[σ Decoder (uncertainty)]
    E1 --> F[Uncertainty Calibration (Temperature Scaling)]
    E2 --> F
    F --> G[Diagnostics Suite]
    G --> H1[GLL Heatmaps]
    G --> H2[SHAP Overlays]
    G --> H3[Symbolic Rule Scoring]
    G --> H4[FFT/UMAP/t‑SNE]
    G --> I[HTML Dashboard]
    F --> J[Submission Bundle (Kaggle)]
````

⸻

## 2) Calibration Kill Chain (Detailed)

> **Goal:** Transform raw detector frames into science-ready, wavelength-registered, jitter-corrected time–wavelength cubes and light curves with reliable uncertainties.

```mermaid
flowchart LR
    A[Raw Frames] --> B[ADC / Digitizer Normalization]
    B --> C[Bias Subtraction]
    C --> D[Dark Current Correction]
    D --> E[Flat-Fielding]
    E --> F[Nonlinearity Correction]
    F --> G[Dead / Hot Pixel Masking]
    G --> H[Correlated Double Sampling (CDS)]
    H --> I[Wavelength Registration / Trace Extraction]
    I --> J[Jitter Modeling & Correction]
    J --> K[Background / Cosmic-Ray Handling]
    K --> L[Time Series Assembly (per λ)]
    L --> M[Calibrated Cubes & Light Curves]
```

### 🧪 Notes

* **CDS** reduces low-frequency drift; cosmic rays are detected via temporal outliers per pixel/λ.
* **Wavelength registration** aligns dispersion per exposure; **trace extraction** supports optimal/box extraction.
* **Jitter correction** uses FGS1-driven motion proxies to decorrelate systematics in AIRS channels.
* **All steps** are parameterized via Hydra and emit **DVC-tracked** intermediates for auditability.

⸻

## 3) Modeling & Uncertainty

* **Encoders**
  • **FGS1 → Mamba SSM** for long-range temporal dependencies and transit morphology.
  • **AIRS → GNN** with edges from wavelength adjacency, shared molecule bands, and detector-region priors.

* **Decoders**
  • **μ** (mean spectrum across 283 bins)
  • **σ** (heteroscedastic aleatoric uncertainty)

* **Calibration of Uncertainty**
  • **Temperature scaling** for over/under confidence.
  • **SpectralCOREL** (conformal GNN) to couple coverage across spectrally related bins.

⸻

## 4) Diagnostics & Symbolic Layer

* **Metrics & Maps:** GLL, entropy, per‑bin residuals, calibration coverage.
* **Explainability:** SHAP overlays (temporal & spectral), attention/attribution traces.
* **Symbolic Rule Engine:** smoothness, positivity, asymmetry, FFT‑band suppression, radiative-transfer checks.
* **Interactive Outputs:** HTML dashboard (UMAP/t‑SNE, rule matrices, heatmaps, FFT panels), CSV/JSON exports.

⸻

## 5) Reproducibility & CI

* **Hydra** (1.3): hierarchical configs, group overrides, run snapshots & hashes.
* **DVC** (3.x): versioned datasets, checkpoints, diagnostics; remote storage.
* **Poetry + Docker**: environment parity (local/CI/Kaggle).
* **GitHub Actions**:
  • `ci.yml` test/build, `diagnostics.yml` dashboards, `nightly-e2e.yml` smoke runs, `kaggle-submit.yml`, `lint.yml`
  • `artifact-sweeper.yml` tidies old artifacts & caches (dry‑run default).
* **Self‑Test:** `python -m spectramind selftest` validates config integrity, file mapping, and required assets.

⸻

## 6) Repository Layout

```
ArielSensorArray/
  configs/            # Hydra configs (data/, model/, training/, diagnostics/, calibration/, logging/)
  src/                # Pipeline modules (calibration/, encoders/, decoders/, symbolic/, cli/, utils/)
  data/               # DVC-tracked datasets (raw/, processed/, meta/)
  outputs/            # checkpoints/, predictions/, diagnostics/, calibrated/
  logs/               # v50_debug_log.md (append-only, CLI call history)
  .github/workflows/  # ci.yml, diagnostics.yml, nightly-e2e.yml, kaggle-submit.yml, lint.yml, artifact-sweeper.yml
```

⸻

## 7) Unified CLI

```bash
python -m spectramind --help
```

**Core Commands**

* `selftest` — pipeline integrity
* `calibrate` — full FGS1/AIRS calibration
* `train` — train the V50 model
* `predict` — μ/σ inference + submission artifacts
* `calibrate-temp` — temperature scaling
* `corel-train` — conformal calibration
* `diagnose` — symbolic + SHAP diagnostics
* `dashboard` — generate HTML diagnostics
* `ablate` — automated ablation sweeps
* `submit` — selftest → predict → validate → ZIP
* `analyze-log` — parse CLI logs → CSV/heatmap
* `check-cli-map` — validate CLI ↔ file mapping

⸻

## 8) Kaggle Integration

* Hardened for **9 hr** GPU budget (A100).
* **Submission path:** `submit` bundles μ/σ CSV and manifest; validates schema.
* Benchmarked against public baselines (Thang Do Duc, V1ctorious3010, Fawad Awan); diagnostics reflect gaps and improvements.

⸻

## 9) Roadmap

* TorchScript/JIT inference
* Extended symbolic overlays & rule discovery
* Web UI (React + FastAPI) with live profiles/overlays
* Leaderboard automation & run registry
* Micro‑lensing & non‑Gaussian noise calibration

⸻

## 10) Citation

```bibtex
@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro-symbolic, Physics-informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team and Andy Barta},
  year    = {2025},
  url     = {https://github.com/bartytime4life/ArielSensorArray}
}
```

⸻

## 11) License

MIT — see [LICENSE](./LICENSE).

```
::contentReference[oaicite:0]{index=0}
```
