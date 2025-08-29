# 🛰️ `/configs/data/ARCHITECTURE.md` — Data Configuration Architecture

---

## 0. Purpose & Scope

The `/configs/data` directory defines the **data ingestion, calibration, preprocessing, and runtime dataset controls** for the SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).

This subsystem governs how **raw telescope signals** (FGS photometry + AIRS spectroscopy) are transformed into calibrated, normalized, and model-ready inputs under strict **NASA-grade reproducibility**.

It ensures:

* **Deterministic dataset composition** (Hydra config snapshots + DVC versioning)
* **Physics-informed calibration** (ADC, dark subtraction, flat-field, jitter correction)
* **Scenario flexibility** (nominal full dataset, Kaggle runtime, CI/debug slice)
* **Auditability & traceability** (every config is logged, hashed, and tied to outputs)

---

## 1. Design Philosophy

* **Hydra-first modularity**: Each dataset mode is a YAML file; `train.yaml` and `predict.yaml` compose them via Hydra defaults.
* **No hardcoding**: Data paths, splits, and calibration flags live in configs, not code.
* **DVC integration**: Raw and processed data are tracked by DVC; configs reference only DVC-managed paths.
* **Physics-informed**: Configs encode calibration steps derived from astrophysical principles (non-negativity, spectral smoothness, jitter-aware alignment).
* **Environment-aware**: Local, Kaggle, and CI runs use dedicated configs (`nominal.yaml`, `kaggle.yaml`, `debug.yaml`).
* **Mission constraints**: Kaggle configs enforce ≤9 hr runtime, ≤16 GB GPU memory, and no internet access.

---

## 2. Directory Structure

```
configs/data/
├── nominal.yaml    # Full dataset config (default for science runs)
├── kaggle.yaml     # Kaggle runtime-safe config (competition submission)
├── debug.yaml      # Tiny slice for CI/self-tests
└── ARCHITECTURE.md # This document
```

---

## 3. Component Responsibilities

### **`nominal.yaml`**

* References full DVC-tracked dataset.
* Applies **all calibrations** (ADC, dark, flat, CDS correction, jitter detrending).
* Configures realistic batch sizes and workers for HPC/local runs.
* Used in long experiments and leaderboard training.

### **`kaggle.yaml`**

* Enforces Kaggle runtime guardrails:

  * Input: `/kaggle/input/neurips-2025-ariel/`
  * Output: `/kaggle/working/`
  * Cache: `/kaggle/temp/`
* Limits workers to 2, batch size to 64 for Tesla P100/T4 GPUs.
* Disables internet and heavy diagnostics.
* Optimized for ≤9 hr full-pipeline submission.

### **`debug.yaml`**

* Loads **5-planet slice** for rapid validation (<2 min runtime).
* Batch size = 2, workers = 0 for determinism.
* Disables heavy preprocessing (no jitter injection, no FFT smoothing).
* Integrated into CI pipelines (`spectramind test`).

---

## 4. Data Flow & Calibration Chain

```mermaid
flowchart TD
    A[Raw Inputs<br/>FGS1 + AIRS] --> B[Calibration Configs<br/>(ADC, dark, flat, CDS, nonlinearity)]
    B --> C[Preprocessing Configs<br/>(normalize_flux, jitter, detrend, align_phase)]
    C --> D[Hydra Dataset Object<br/>nominal|kaggle|debug]
    D --> E[Training / Prediction Pipelines]
    E --> F[Outputs + Logs<br/>μ, σ, diagnostics, submission.csv]
```

* **Calibration**: Corrects detector systematics, applies dark/flat/ADC/CDS adjustments.
* **Preprocessing**: Normalizes flux, injects jitter (if enabled), aligns phases, detrends AIRS spectra.
* **Hydra object**: Exposes dataset properties (`num_planets`, `num_bins`, file paths) for downstream modules.
* **Outputs**: DVC-tracked logs, model-ready batches, and diagnostic metadata.

---

## 5. Integration

* **Hydra CLI**: Dataset is swapped via `data=<mode>`, e.g. `spectramind train data=kaggle`.
* **CI**: Uses `data=debug` for smoke tests.
* **Kaggle**: Uses `data=kaggle` for official submissions (safe for P100/T4 GPUs, no internet).
* **Diagnostics**: `generate_html_report.py` embeds dataset config, calibration choices, and DVC hashes.
* **Logging**: All configs are hashed and logged in `logs/v50_debug_log.md`.

---

## 6. Future Extensions

* 🔭 **Symbolic-aware data configs** (e.g. molecule-region masking, SNR-based dropout).
* 🚀 **Hybrid calibration modes**: toggle symbolic physics rules during calibration.
* 📦 **Dataset bundling**: automated YAML → DVC pipeline → Kaggle-ready `.zip`.
* 🛰️ **External simulator integration** (Ariel radiative transfer models) for forward-checks.

---

## 7. References

* **SpectraMind V50 Technical Plan**
* **SpectraMind V50 Project Analysis**
* **SpectraMind V50 Update Strategy**
* **Hydra for AI Projects**
* **Kaggle Platform Guide**
* **Comparison of Kaggle Models**

---

## 8. DVC Pipeline (Config → Stages) — Visual

```mermaid
flowchart LR
  %% Entry: select dataset mode via Hydra override
  A0{{Hydra<br/>data=nominal|kaggle|debug}} --> A1[compose cfg<br/>configs/data/*.yaml]

  %% DVC source & cache
  subgraph S0[Versioned Sources (DVC)]
    R1[(raw_fgs1/)]:::dvc --- R2[(raw_airs/)]:::dvc
    R3[(calib_refs/)]:::dvc
  end

  %% DVC stages
  subgraph P0[Data Pipeline (DVC DAG)]
    direction LR
    S1[calibrate\nadc,dark,flat,cds,nonlinearity]:::stage
    S2[preprocess\nnormalize_flux, detrend, align_phase, jitter?]:::stage
    S3[split\nplanet_holdout / folds]:::stage
    S4[package_batches\npt/npy parquet for loaders]:::stage
  end

  %% Hydra -> DVC IO binding
  A1 -->|paths, flags| S1
  R1 --> S1
  R2 --> S1
  R3 --> S1
  S1 --> S2
  S2 --> S3
  S3 --> S4

  %% Outputs & consumers
  subgraph O0[Tracked Outputs]
    O1[(calibrated/)]:::dvc
    O2[(processed/)]:::dvc
    O3[(splits/)]:::dvc
    O4[(batches/)]:::dvc
  end

  S1 --> O1
  S2 --> O2
  S3 --> O3
  S4 --> O4

  %% Downstream
  O4 --> T1[[train]]:::cons
  O4 --> P1[[predict]]:::cons
  O4 --> D1[[diagnose]]:::cons

  %% Styles
  classDef dvc fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef stage fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
  classDef cons fill:#fff3e0,stroke:#ef6c00,color:#e65100
```

### Stage Legend & Mapping

| DVC Stage         | Inputs (DVC)                            | Hydra-bound parameters (examples)                                                     | Outputs (DVC) |
| ----------------- | --------------------------------------- | ------------------------------------------------------------------------------------- | ------------- |
| `calibrate`       | `raw_fgs1/`, `raw_airs/`, `calib_refs/` | `calibration.*` (adc\_correction, dark\_subtraction, flat\_field, cds, nonlinearity)  | `calibrated/` |
| `preprocess`      | `calibrated/`                           | `preprocessing.*` (normalize\_flux, detrend\_method, align\_phase, jitter\_injection) | `processed/`  |
| `split`           | `processed/`                            | `validation.*` (split\_strategy, val\_fraction, random\_seed, deterministic)          | `splits/`     |
| `package_batches` | `splits/`                               | `dataset.*` (batch\_size, num\_workers, pin\_memory, paths)                           | `batches/`    |

> **Notes**
>
> * **Hydra selection** (`data=nominal|kaggle|debug`) composes dataset paths, calibration flags, and loader knobs that parameterize the DVC stages at runtime.
> * **DVC** guarantees immutable versioning of `raw_*`, `calib_refs`, and all stage outputs; the composed Hydra config is snapshot-logged for every run.
> * **Kaggle mode** (`data=kaggle`) maps all IO to `/kaggle/input/*`, `/kaggle/working`, enforces `num_workers<=2`, `batch_size<=64`, and disables internet to meet notebook constraints.
> * **Debug mode** (`data=debug`) truncates planets/bins and sets tiny loader limits to keep CI smoke tests sub-minute.

### Operational Flow (CLI → Hydra → DVC)

1. **CLI**: `spectramind train data=kaggle`
2. **Hydra**: composes `configs/data/kaggle.yaml` into the active config (paths, flags).
3. **DVC**: executes the DAG (`calibrate → preprocess → split → package_batches`) only where inputs changed (cache-aware), emitting tracked artifacts consumed by **train/predict/diagnose**.

---
