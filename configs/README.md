# 🗂️ `/configs` — SpectraMind V50 Configuration System

## 0. Purpose & Scope

The **`/configs`** directory defines all **experiment parameters** for the **SpectraMind V50 pipeline** (NeurIPS 2025 Ariel Data Challenge).
It is the **single source of truth** for:

* 📡 **Data ingestion & calibration** — FGS1 photometer and AIRS spectrometer paths, preprocessing, detrending, noise models
* 🧠 **Model architectures** — FGS1 Mamba encoder, AIRS GNN, multi-scale/uncertainty decoders, COREL calibration layers
* ⚙️ **Training hyperparameters** — curriculum schedules, optimizer settings, mixed-precision, checkpointing, loss weights
* 🔬 **Symbolic/physics constraints** — smoothness, non-negativity, FFT priors, molecular fingerprints, COREL calibration&#x20;
* 📊 **Diagnostics & explainability** — SHAP overlays, symbolic violation maps, uncertainty calibration plots, dashboard exports
* 🖥️ **Runtime overrides** — local dev, Kaggle GPU (≤9 hr safe mode), CI, Docker

Every run is **Hydra-safe, DVC-versioned, and auditable**:

* The composed config snapshot is saved under `outputs/DATE_TIME/.hydra/`
* Logged with a **config hash** in `logs/v50_debug_log.md`
* Artifacts tracked in **DVC** for reproducibility

---

## 1. Design Philosophy

* **Hydra-first**: Modular YAMLs dynamically composed at runtime
* **No hard-coding**: Pipeline behavior is changed only via configs or CLI overrides, never by editing source
* **Hierarchical layering**: `defaults` compose from groups (`data/`, `model/`, `optimizer/`, etc.); overrides at any depth
* **Versioned & logged**: Each run saves its config + hash for reproducibility
* **DVC-integrated**: All data/model artifacts referenced in configs are DVC-tracked for exact reruns
* **Kaggle-safe**: Enforces ≤9 hr runtime, GPU RAM guardrails, no internet calls
* **Physics-informed**: Configs encode symbolic loss weights, physical priors, non-negativity

---

## 2. Directory Structure

```
configs/
├── train.yaml              # Main training config (composes defaults)
├── predict.yaml            # Inference / submission config
├── ablate.yaml             # Grid for ablation sweeps
├── selftest.yaml           # Lightweight smoke/self-test config
│
├── data/                   # Dataset + calibration options
│   ├── nominal.yaml
│   ├── kaggle.yaml
│   └── debug.yaml
│
├── model/                  # Model architectures
│   ├── v50.yaml
│   ├── fgs1_mamba.yaml
│   ├── airs_gnn.yaml
│   └── decoder.yaml
│
├── optimizer/              # Optimizers & schedulers
│   ├── adam.yaml
│   ├── adamw.yaml
│   └── sgd.yaml
│
├── loss/                   # Physics/symbolic loss weights
│   ├── gll.yaml
│   ├── smoothness.yaml
│   └── symbolic.yaml
│
├── trainer/                # Training loop configs
│   ├── default.yaml
│   ├── gpu.yaml
│   └── kaggle_safe.yaml
│
├── logger/                 # Experiment logging
│   ├── tensorboard.yaml
│   ├── wandb.yaml
│   └── mlflow.yaml
│
└── local/                  # Machine-specific overrides (git-ignored)
    └── default.yaml
```

---

## 3. Usage

### Run with defaults

```bash
python train_v50.py
```

### Override values

```bash
python train_v50.py optimizer=adamw training.epochs=20 model=airs_gnn
```

### Multirun sweeps

```bash
python train_v50.py -m optimizer=adam,sgd training.batch_size=32,64
```

### Kaggle-safe run

```bash
spectramind train --config-name train.yaml trainer=kaggle_safe
```

### CI self-test

```bash
spectramind test --config-name selftest.yaml --fast
```

---

## 4. Best Practices

* **Keep configs in Git**: All YAMLs except `/local/` are version-controlled
* **Use `/local/` for secrets/paths**: Cluster creds, scratch dirs, etc. are `.gitignored`
* **Leverage interpolation**: e.g. `${data.num_classes}` ensures cross-consistency across groups
* **Snapshot every run**: Hydra auto-saves configs; never run without one
* **Sync with DVC**: Ensure every path in configs is tracked in DVC
* **Layer configs**: Use `defaults` to define baselines; override for ablation/debug/Kaggle
* **Enforce Kaggle runtime safety**: batch sizes, mixed precision, checkpointing aligned with GPU limits

---

## 5. Integration

* **CLI**: All commands (`spectramind train`, `spectramind diagnose`, `spectramind submit`) load configs through Hydra
* **CI**: GitHub Actions validates configs via self-test + sample pipeline runs
* **Kaggle**: Configs guarantee ≤9 hr runtime, GPU quota compliance, offline reproducibility
* **Dashboard**: Config metadata feeds into `generate_html_report.py` for diagnostics
* **Experiment tracking**: Optional sync with MLflow/W\&B/TensorBoard via logger configs

---

## 6. References

* Hydra configuration best practices
* SpectraMind V50 Technical Plan
* Project Analysis of repo configs
* Strategy for Updating & Extending Configs
* Kaggle Platform Technical Guide
* NASA/Physics-informed modeling refs
flowchart TB
  %% Root entrypoint
  T[train.yaml]

  %% Groups composed by train.yaml
  T --> D[data/*]
  T --> M[model/*]
  T --> O[optimizer/*]
  T --> L[loss/*]
  T --> R[trainer/*]
  T --> G[logger/*]
  T --> U[uncertainty/* (optional)]
  T --> X[ablate.yaml (optional multirun)]
  T --> P[predict.yaml (inference)]

  %% Data options
  D --> Dn[nominal.yaml]
  D --> Dk[kaggle.yaml]
  D --> Dd[debug.yaml]

  %% Model options
  M --> Mv[v50.yaml]
  M --> Mm[fgs1_mamba.yaml]
  M --> Ma[airs_gnn.yaml]
  M --> Md[decoder.yaml]

  %% Optimizers
  O --> Oa[adam.yaml]
  O --> Ow[adamw.yaml]
  O --> Os[sgd.yaml]

  %% Losses / physics
  L --> Lg[gll.yaml]
  L --> Ls[smoothness.yaml]
  L --> Ly[symbolic.yaml]

  %% Trainer profiles
  R --> Rt[default.yaml]
  R --> Rg[gpu.yaml]
  R --> Rk[kaggle_safe.yaml]

  %% Loggers
  G --> Gt[tensorboard.yaml]
  G --> Gw[wandb.yaml]
  G --> Gm[mlflow.yaml]

  %% Local overrides
  T --> LCL[local/default.yaml (git-ignored)]
  LCL --- note1{{Local overrides: secrets, scratch paths, cluster queues}}

  %% Notes
  note2{{Hydra saves composed configs<br/>to outputs/DATE_TIME/.hydra/}}
  note3{{All file paths DVC-tracked<br/>for reproducibility}}
  T --- note2
  T --- note3
