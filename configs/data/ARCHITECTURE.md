# 🛰️ `/configs/data/ARCHITECTURE.md` — Data Configuration Architecture

---

## 0) Purpose & Scope

`/configs/data` defines the **data ingestion, calibration, preprocessing, splitting, and runtime dataset controls** for the **SpectraMind V50** pipeline (NeurIPS 2025 Ariel Data Challenge).

This subsystem transforms **raw telescope signals** — **FGS1** photometry (time×channels) + **AIRS** spectroscopy (time×wavelength) — into **calibrated, normalized, model-ready** tensors under strict **mission-grade reproducibility**:

- **Deterministic composition** (Hydra snapshots + seeded ops)
- **Physics-informed calibration** (ADC, nonlinearity, dark, flat, CDS, trace/phase)
- **Scenario flexibility** (full science / Kaggle / CI smoke)
- **Auditability & traceability** (DVC lineage, config hash, run manifest)

---

## 1) Design Philosophy

- **Hydra-first modularity**  
  Each dataset mode is a YAML component (`nominal.yaml`, `kaggle.yaml`, `debug.yaml`) composed by higher-level configs (`train.yaml`, `predict.yaml`, `selftest.yaml`) via Hydra defaults.
- **Zero hardcoding**  
  Paths, splits, calibration flags, and loader knobs live in YAMLs — never in Python.
- **DVC integration**  
  Raw/processed artifacts are DVC-tracked. Configs reference only DVC-managed or environment mounts (Kaggle).
- **Physics-informed realism**  
  Configs encode calibration steps & symbolic constraints (non-negativity, smoothness, molecular windows), plus optional 356→283 bin remap for decoder compatibility.
- **Environment awareness**  
  Dedicated configs for **local/HPC** (`nominal`), **Kaggle** (`kaggle`), and **CI** (`debug`).
- **Mission constraints**  
  Kaggle enforces ≤9 hr walltime, ≤16 GB GPU, **no internet**; CI smoke completes in **< 60 s**.

---

## 2) Directory Structure

```

configs/data/
├─ nominal.yaml     # Full scientific dataset (default for experiments)
├─ kaggle.yaml      # Kaggle runtime-safe dataset (offline, resource-guarded)
├─ debug.yaml       # Tiny deterministic slice (CI/self-test)
├─ README.md        # Quick how-to and usage
├─ ARCHITECTURE.md  # (this document)
└─ .gitkeep         # Keep dir tracked; inline purpose notes

````

---

## 3) Component Responsibilities

### `nominal.yaml`
- References full DVC-tracked inputs.
- Applies **complete** calibration kill-chain (ADC, nonlinearity, dark, flat, CDS, photometry, trace/phase).
- Rich preprocessing (detrend, smoothing optional), symbolic hooks, diagnostics.
- Suitable for long experiments and leaderboard training.

### `kaggle.yaml`
- IO maps to:  
  **Input** `/kaggle/input/neurips-2025-ariel/` • **Output** `/kaggle/working/` • **Cache** `/kaggle/temp/`
- Runtime guardrails: `num_workers≤2`, `batch_size≤64`, `enforce_no_internet=true`, lean diagnostics.
- Optimized for ≤9 hr submission runs on P100/T4.

### `debug.yaml`
- **5-planet** slice with identical schema to nominal; deterministic, no augmentation.
- Loader: `batch_size=2`, `num_workers=0`.  
- Completes in **seconds**; used by CI/self-test and local smoke.

---

## 4) Data Flow & Calibration Chain

```mermaid
flowchart TD
    A[Raw Inputs<br/>FGS1 + AIRS] --> B[Calibration<br/>ADC, Nonlin, Dark, Flat, CDS, Photometry, Trace/Phase]
    B --> C[Preprocessing<br/>Normalize Flux, Detrend, (Re)sample, Jitter?]
    C --> D[Hydra Dataset Object<br/>nominal | kaggle | debug]
    D --> E[Training / Prediction Pipelines]
    E --> F[Outputs + Logs<br/>μ, σ, diagnostics, submission.csv]
````

**Step guide**

1. **Calibration** — Correct detector/systematics and standardize instrument responses
2. **Preprocessing** — Normalize flux, detrend AIRS, enforce time grid; optionally apply jitter/smoothing
3. **Hydra dataset** — Exposes schema (`num_bins=283`, paths), splits, loader knobs
4. **Pipelines** — Train, Predict, Diagnose consume the composed data config
5. **Artifacts** — DVC-tracked outputs, HTML/PNG diagnostics, run manifests

---

## 5) Validation & Safety Gates

Fail-fast checks (per mode, with small variations):

* **Schema**: Ranks/shapes/dtypes for FGS1 `(N,32,32)`, AIRS `(N,32,356)`; labels columns (`target_mu[0:283]`, `target_sigma[0:283]`)
* **Instrument**: Optional **356→283** `bin_remap` (verify mapping integrity)
* **Numeric safety**: `min_sigma`, `max_sigma`, `max_abs_mu`, non-negativity post-preproc
* **Runtime**: writability, GPU memory (nominal), Kaggle time budget, CI time budget (debug)
* **Split consistency**: fractions sum≈1, stratify/group invariants

---

## 6) Integration Points

* **Hydra CLI**
  Swap datasets with `data=<nominal|kaggle|debug>`, e.g.:

  ```bash
  spectramind train data=kaggle
  spectramind diagnose data=nominal diagnostics.fft_analysis=true
  ```
* **CI**
  Self-tests run `data=debug` by default; heavy ops disabled.
* **Kaggle**
  Runs with `data=kaggle` (offline, reduced workers, capped diagnostics).
* **Dashboards/Reports**
  `generate_html_report.py` embeds data config snapshot, calibration flags, hashes.

---

## 7) Runtime Modes — Quick Reference

| Mode    | IO Roots                           | Target Hardware        | Diagnostics | Expected Time |
| ------- | ---------------------------------- | ---------------------- | ----------- | ------------- |
| nominal | DVC-tracked local/cluster paths    | Local/HPC GPU(s)       | Rich        | hours         |
| kaggle  | `/kaggle/input`, `/kaggle/working` | Kaggle P100/T4 (16 GB) | Lean        | ≤ 9 hr        |
| debug   | `data/debug`, `outputs/debug`      | CI/CPU or any          | Minimal     | **< 60 s**    |

---

## 8) Adding a New Dataset Mode (Checklist)

1. **Copy a template**
   Start from `debug.yaml` (for small) or `nominal.yaml` (for full).
2. **Set paths**
   Ensure **all** inputs/outputs resolve to DVC-tracked files or environment mounts (never ad-hoc).
3. **Calibrate/Preprocess**
   Toggle kill-chain steps; set detrending/smoothing/jitter appropriately.
4. **Schema & safety**
   Keep `interface.num_bins=283` (or provide a validated `bin_remap`), set numeric bounds.
5. **Splits/Loader**
   Define fractions, seeds, and loader knobs (`batch_size`, `num_workers`) for your environment.
6. **Diagnostics**
   Tune to your runtime budget (rich for research, lean for Kaggle/CI).
7. **Wire it in**
   Reference it via Hydra defaults in `train.yaml` / `predict.yaml`; test with `spectramind train data=<new>`.

---

## 9) Example Recipes

**Switch detrend method (nominal)**

```bash
spectramind train data=nominal preprocessing.detrend.method=savgol
```

**Accelerate diagnose by pruning heavy ops**

```bash
spectramind diagnose data=nominal diagnostics.fft_analysis=false diagnostics.shap_overlay=false
```

**Kaggle memory safety**

```bash
spectramind train data=kaggle loader.batch_size=48 runtime.reduce_heavy_ops=true
```

**Run CI smoke in seconds**

```bash
spectramind train data=debug training.epochs=1
```

---

## 10) DVC Pipeline (Config → Stages)

```mermaid
flowchart LR
  A0{{Hydra<br/>data=nominal|kaggle|debug}} --> A1[Compose cfg<br/>configs/data/*.yaml]

  subgraph S0[Versioned Sources (DVC)]
    R1[(raw_fgs1/)]:::dvc --- R2[(raw_airs/)]:::dvc
    R3[(calib_refs/)]:::dvc
  end

  subgraph P0[Data Pipeline (DVC DAG)]
    direction LR
    S1[calibrate]:::stage --> S2[preprocess]:::stage --> S3[split]:::stage --> S4[package_batches]:::stage
  end

  A1 -->|paths, flags| S1
  R1 --> S1
  R2 --> S1
  R3 --> S1

  S1 --> O1[(calibrated/)]:::dvc
  S2 --> O2[(processed/)]:::dvc
  S3 --> O3[(splits/)]:::dvc
  S4 --> O4[(batches/)]:::dvc

  O4 --> T1[[train]]:::cons
  O4 --> P1[[predict]]:::cons
  O4 --> D1[[diagnose]]:::cons

  classDef dvc fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef stage fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
  classDef cons fill:#fff3e0,stroke:#ef6c00,color:#e65100
```

**Stage Mapping**

| Stage             | Inputs (DVC)           | Hydra-bound parameters (examples)                                   | Outputs (DVC) |
| ----------------- | ---------------------- | ------------------------------------------------------------------- | ------------- |
| `calibrate`       | `raw_*`, `calib_refs/` | `calibration.*` (adc, nonlin, dark, flat, cds, photometry, phase)   | `calibrated/` |
| `preprocess`      | `calibrated/`          | `preprocessing.*` (normalize\_flux, detrend, align\_phase, jitter?) | `processed/`  |
| `split`           | `processed/`           | `splits.*` (strategy, fractions, seed, export)                      | `splits/`     |
| `package_batches` | `splits/`              | `loader.*`, `paths.*`, `interface.*`                                | `batches/`    |

> Hydra selection (`data=<mode>`) parametrizes the **same** DVC DAG with different IO roots & flags.

---

## 11) FAQ

**Q:** Why 356→283 bin remap?
**A:** Some AIRS sources provide 356 wavelengths; the model expects **283**. The `bin_remap` provides a reproducible index mapping to ensure head compatibility.

**Q:** Can I disable calibration for prototyping?
**A:** Yes (`calibration.*.enabled=false`), but physics realism and metrics will degrade; diagnostics should mark this explicitly.

**Q:** How do I create tiny packs for `debug.yaml`?
**A:** Use `configs/dat/ariel_toy_dataset.py` to generate deterministic toy/debug `.pkl`/`.npy` packs and point `paths.*` to them.

---

## 12) References

* SpectraMind V50 Technical Plan
* SpectraMind V50 Project Analysis
* Strategy for Updating & Extending V50
* Hydra for AI Projects
* Kaggle Platform Guide
* Comparison of Kaggle Models

---

### ✅ TL;DR

`/configs/data` is the **flight control module** for SpectraMind V50 datasets.
Pick `data=<nominal|kaggle|debug>` and the pipeline will execute with **mission-grade**, physics-aware, reproducible behavior across **local**, **Kaggle**, and **CI** environments.

```
```
