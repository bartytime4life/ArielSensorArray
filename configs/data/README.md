# 🗂️ `/configs/data` — Dataset & Calibration Configurations (SpectraMind V50)

> **Mission:** one place to define **what data we use**, **how we calibrate it**, **how we preprocess it**, and **how we validate it** — for local science runs, Kaggle submissions, and CI smoke tests.  
> **Guarantees:** Hydra-composable, DVC/lakeFS-traceable, Kaggle-safe, CI-fast, physics-informed.

---

## 0) Purpose & Scope

This folder defines **all dataset, calibration, preprocessing, split, and loader parameters** for the **SpectraMind V50** pipeline (NeurIPS 2025 Ariel Data Challenge).

Configs in this directory control:

- **Dataset sources** (`nominal`, `kaggle`, `debug`, `toy`)
- **Calibration kill-chain** (ADC, non-linearity, dark, flat, CDS, photometry, trace/phase)
- **Preprocessing** (detrend/savgol, time/grid standardization, bin mapping, smoothing, resampling)
- **Augmentations** (jitter, dropout/mask, SNR-based drops, noise injection)
- **Symbolic hooks** (non-negativity, FFT priors, molecular windows, region masks)
- **Diagnostics** (FFT, Z-score, symbolic overlays, SHAP overlays)
- **Runtime guardrails** (Kaggle/CI limits, integrity checks, fail-fast validation)

Everything here is **Hydra-first**, **reproducible by construction**, and **fail-fast**.

---

## 1) Design Principles

- **Hydra-first:** No hard-coded paths/hparams in Python. All knobs live in YAML selected via `data=<name>`.
- **Reproducibility:** DVC/lakeFS for data lineage, Hydra for config capture, timestamped run dirs.
- **Physics-informed:** Enforce non-negativity, spectral smoothness, molecular priors, and realistic calibration steps.
- **Scenario coverage:** Dedicated YAMLs for science runs, Kaggle runtime, CI/debug, and synthetic toy sets.
- **Fail-fast:** Early schema & path validation prevents expensive wasted GPU time; guardrails everywhere.

---

## 2) Directory Layout