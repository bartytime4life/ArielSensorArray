# 🧭 SpectraMind V50 — Pipeline Architecture

> Companion to [`pipeline_overview.md`](./pipeline_overview.md)
> Purpose: Describe each subsystem of the end-to-end DAG in **narrative form** with reproducibility guarantees.

---

## 0) Design Principles

SpectraMind V50 is engineered as a **CLI-first, Hydra-safe, DVC-tracked pipeline** for the NeurIPS 2025 Ariel Data Challenge.
Key principles:

* **CLI-first**: all runs via `spectramind …`, no hidden scripts.
* **Hydra-safe configs**: YAML-driven, composable, version-controlled.
* **DVC + Git**: data, calibration outputs, checkpoints, and submissions are tracked and reproducible.
* **NASA-grade reproducibility**: every run logs config hash, dataset version, and outputs for audit.

---

## 1) CLI Orchestration Layer

* Single entrypoint: `spectramind` (Typer CLI).
* Subcommands:

  * `calibrate`
  * `train`
  * `diagnose`
  * `submit`

Each CLI call composes configs with Hydra, executes the corresponding pipeline, and appends to `v50_debug_log.md`.
Interactive feedback uses **Rich console dashboards** (progress bars, live metrics).

---

## 2) Calibration Subsystem

**Inputs**: Raw telescope frames (FGS1, AIRS).
**Stages**:

* ADC correction
* Nonlinearity correction
* Dark subtraction
* Flat fielding
* CDS / trace extraction
* Normalization & phase alignment

**Outputs**: Calibrated cubes ready for feature packaging.
**Config**: `configs/data/nominal.yaml`.
**Artifacts**: DVC-tracked calibrated data, reproducible via `dvc repro`.

---

## 3) Preprocess & Packaging

**Purpose**: QC and SNR checks, assemble AIRS + FGS1 + metadata into feature bundles.
**Outputs**: Numpy arrays / parquet files (`artifacts/features/`).
**Config**: `configs/data/` group.
**CLI**: `spectramind calibrate --with-qc`.

---

## 4) Training & Inference

**Pipeline**:

* **Hydra Compose**: merges `configs/train/*` (model, optimizer, scheduler, loss).
* **Trainer Engine**:

  * FGS1 encoder → Mamba SSM
  * AIRS encoder → GNN (edge-feature aware)
  * Decoders → μ and σ predictions
* **Checkpoints**: saved + versioned with DVC.
* **Outputs**: μ, σ arrays for \~283 bins per planet.

**Configs**:

* `configs/model/v50.yaml`
* `configs/train/train.yaml`

**CLI**:

```bash
spectramind train model=v50 optimizer=adamw train.epochs=50
```

**Artifacts**:

* `checkpoints/` (DVC-tracked)
* `predictions/` (μ, σ)

---

## 5) Diagnostics Subsystem

**Purpose**: Post-run explainability, symbolic overlays, and reproducibility checks.
**Modules**:

* Metrics: GLL, RMSE, MAE
* Calibration checks: σ vs residuals
* Symbolic overlays: smoothness, non-negativity, FFT priors
* Latent projections: UMAP (`src/diagnostics/plot/umap/v50.py`), t-SNE (`tsne/interactive.py`)
* Spectral diagnostics: FFT (`fft/autocorr/mu.py`), smoothness maps (`spectral/smoothness/map.py`)

**Outputs**:

* HTML reports (`artifacts/diagnostics/…html`)
* PNG snapshots
* JSON summaries (`diagnostic_summary.json`)

**Integration**: `generate_html_report.py` bundles diagnostics into a unified dashboard.

---

## 6) Submission & Packaging

**Stages**:

1. Validate submission file shape (`validate_submission.py`).
2. Bundle into ZIP with manifest + hash.
3. Upload to Kaggle competition (Leaderboard).

**Artifacts**:

* `submission.zip`
* `manifest.json` (hashes, config snapshot, reproducibility metadata)

**CLI**:

```bash
spectramind submit --predictions artifacts/preds.npy --out submission.zip
```

**Integration with Kaggle**: ensures compatibility with leaderboard evaluation.

---

## 7) Side-Channels (Reproducibility)

* **Run Snapshots**: Hydra resolved YAML written to `runs/snapshots/`.
* **Structured Logs**: `events.jsonl` for structured logging.
* **Diagnostics Reports**: HTML/PNG in artifacts.
* **Manifest**: `manifest.json` includes config hash, Git commit, DVC pointers.

---

## 8) Future Extensions

* **Experiment tracking**: add MLflow or W\&B integration for live run dashboards.
* **Physics-informed constraints**: enforce spectral line priors, symbolic cycle-consistency.
* **GUI layer**: optional thin dashboard over CLI for teaching/demos.
* **More diagnostics**: cluster overlays, symbolic violation maps, uncertainty calibration plots.

---

✅ This `ARCHITECTURE.md` complements the **DAG view** (`pipeline_overview.md`) with detailed prose.
Together, they provide a full reproducibility-ready description of the SpectraMind V50 pipeline.

---
