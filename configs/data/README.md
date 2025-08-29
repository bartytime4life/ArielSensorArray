# 🗂️ `/configs/data` — Dataset & Calibration Configurations

## 0. Purpose & Scope

The `/configs/data` directory defines **all dataset, calibration, and preprocessing parameters** for the SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).
It ensures that **data ingestion, normalization, augmentation, and calibration** are **Hydra-safe**, reproducible, and versioned with DVC.

Configs here govern:

* Dataset sources (nominal, Kaggle, debug)
* Calibration parameters (ADC correction, dark subtraction, flat-field, jitter)
* Preprocessing (spectral binning, normalization, masking, augmentation)
* Environment overrides (local paths, Kaggle data mounts)
* Diagnostics integration (FFT, autocorrelation, symbolic overlays)

---

## 1. Design Philosophy

* **Hydra-first**: All datasets are defined in modular YAMLs (nominal, kaggle, debug) and composed at runtime.
* **No hardcoding**: Paths, batch sizes, and calibrations live here, never inside Python code.
* **Physics-informed**: Data configs enforce NASA-grade calibration realism (non-negativity, spectral smoothness, molecular priors).
* **DVC-tracked**: All raw/processed data is versioned by DVC, ensuring reproducibility across Kaggle, CI, and local runs.
* **Scenario coverage**: Each YAML corresponds to an operational mode (nominal full dataset, Kaggle-safe, debug smoke test).

---

## 2. Directory Structure

```
configs/data/
├── nominal.yaml    # Full dataset config (default for experiments)
├── kaggle.yaml     # Kaggle competition runtime-safe config
├── debug.yaml      # Lightweight debug config for CI/selftest
└── README.md       # This file
```

**File Roles**:

* **`nominal.yaml`**
  Full scientific configuration. Uses DVC paths, applies all calibration (ADC, dark, flat, trace extraction, jitter correction). Default for training & diagnostics.

* **`kaggle.yaml`**
  Kaggle runtime config. Uses competition-provided `/kaggle/input` mounts, disables internet fetches, and enforces ≤9h runtime constraints.

* **`debug.yaml`**
  Minimal dataset slice (1–2 planets, truncated sequences) for **CI smoke tests** (`spectramind test`), ensuring rapid validation.

---

## 3. Usage

Select dataset configuration at runtime via Hydra overrides:

```bash
# Default (nominal dataset)
spectramind train data=nominal

# Kaggle-safe
spectramind train data=kaggle

# Lightweight debug run
spectramind train data=debug
```

These configs are referenced by higher-level configs (`train.yaml`, `predict.yaml`, `selftest.yaml`, `ablate.yaml`) through the Hydra `defaults` mechanism.

---

## 4. Best Practices

* ✅ **Keep all paths versioned in DVC** – never point directly to untracked files.
* ✅ **Use interpolation**: e.g. `num_classes: ${data.num_classes}` for linking model configs.
* ✅ **Kaggle configs** must disable internet calls, use only mounted data, and enforce safe memory/runtime.
* ✅ **Debug configs** should run in <60s for CI pipelines.
* ✅ **Snapshot every run**: Hydra auto-saves the composed data config under `outputs/` for auditability.

---

## 5. Integration

* **CLI**: All commands (`train`, `predict`, `diagnose`, `submit`) reference dataset configs through Hydra.
* **CI**: GitHub Actions runs `data=debug` in self-tests.
* **Kaggle**: Uses `data=kaggle`, ensuring compatibility with Kaggle GPU quotas, offline mode, and <9hr runtime.
* **Dashboard**: Diagnostics (`generate_html_report.py`) embeds dataset metadata, calibration choices, and config hash for transparency.

---

## 6. References

* **SpectraMind V50 Technical Plan**
* **SpectraMind V50 Project Analysis**
* **SpectraMind V50 Update Strategy**
* **Hydra for AI Projects**
* **Kaggle Platform Guide**

---

✅ With this setup, `/configs/data` is the **flight control module for all dataset handling**, ensuring **mission-grade reproducibility** across local, Kaggle, and CI environments.

---
