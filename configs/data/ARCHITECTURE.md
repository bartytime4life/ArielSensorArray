# 🛰️ `/configs/data/ARCHITECTURE.md` — Data Configuration Architecture (SpectraMind V50)

> **Scope:** Authoritative architecture for **data ingestion → calibration → preprocessing → splitting → loading** across **local/HPC**, **Kaggle**, and **CI**.  
> **Guarantees:** Hydra-first composition, DVC/lakeFS traceability, physics-informed defaults, fail-fast validation, Kaggle/CI guardrails.

---

## 0) Purpose & Scope

`/configs/data` transforms **raw telescope signals** — **FGS1** photometry (time×channels) + **AIRS** spectroscopy (time×wavelength) — into **calibrated, normalized, model-ready** tensors under strict **mission-grade reproducibility**:

- **Deterministic composition** (Hydra snapshots, fixed seeds)
- **Physics-informed calibration** (ADC, nonlinearity, dark, flat, CDS, photometry, trace/phase)
- **Scenario flexibility** (full science / Kaggle / CI smoke / toy synthetic)
- **Auditability** (DVC lineage, config hash, generator manifest, run manifests & artifacts)

---

## 1) Design Principles

- **Hydra-first modularity**  
  Dataset modes are YAML components (`nominal.yaml`, `kaggle.yaml`, `debug.yaml`, `toy.yaml`) composed by higher-level configs (`train.yaml`, `predict.yaml`, `selftest.yaml`) via Hydra `defaults`.
- **Zero hardcoding**  
  Paths, splits, calibration flags, loaders, bin maps live in YAML — never in Python.
- **DVC/lakeFS integration**  
  Raw/processed artifacts are DVC-tracked; configs reference tracked paths or environment mounts (`/kaggle/input`).
- **Physics realism**  
  Encodes calibration and symbolic constraints (non-negativity, smoothness, molecular windows). Supports 356→283 bin remap for decoder compatibility.
- **Environment awareness**  
  Dedicated runtime profiles for **Local/HPC** (`nominal`), **Kaggle** (`kaggle`), **CI** (`debug`), **Synthetic Dev** (`toy`).
- **Mission constraints**  
  Kaggle: ≤9 hr walltime, ≤16 GB GPU, **no internet**.  
  CI smoke: **<60 s**. Toy: **<3 min**.

---

## 2) Directory Structure