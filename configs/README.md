# 🗂️ `/configs` — SpectraMind V50 Configuration System

## 0. Purpose & Scope

The **`/configs`** directory is the **flight plan** for the **SpectraMind V50 pipeline** (NeurIPS 2025 Ariel Data Challenge).
It encodes all **experiment parameters** and ensures that **every run is reproducible, physics-informed, and Kaggle-safe**.

It serves as the **single source of truth** for:

* 📡 **Data ingestion & calibration** — FGS1 photometer and AIRS spectrometer paths, detrending, jitter/noise models
* 🧠 **Model architectures** — FGS1 Mamba encoder, AIRS GNN, multi-scale decoders, uncertainty heads, COREL calibration
* ⚙️ **Training hyperparameters** — curriculum schedules, optimizer/scheduler configs, AMP, checkpointing, loss weights
* 🔬 **Symbolic/physics constraints** — smoothness, non-negativity, FFT priors, molecular fingerprints, gravitational lensing overlays
* 📊 **Diagnostics & explainability** — SHAP overlays, symbolic violation maps, UMAP/t-SNE projections, calibration heatmaps, HTML dashboards
* 🖥️ **Runtime overrides** — local dev, Kaggle GPU (≤9 hr safe mode), CI/CD, Docker

Each run is **Hydra-safe, DVC-versioned, and audit-logged**:

* Composed configs saved under `outputs/DATE_TIME/.hydra/`
* Logged with a **config hash** in `logs/v50_debug_log.md`
* Large data/model artifacts tracked via **DVC**

---

## 1. Design Philosophy

* **Hydra-first** — modular YAMLs, dynamically composed
* **No hard-coding** — all behavior comes from configs or CLI overrides, never from code edits
* **Hierarchical layering** — `defaults` compose from groups (`data/`, `model/`, `optimizer/`, etc.)
* **Versioned & logged** — every run saves config + hash
* **DVC-integrated** — datasets/models tracked for exact reruns
* **Kaggle-safe** — ≤9 hr runtime, GPU RAM guardrails, no internet
* **Physics-informed** — configs encode symbolic & astrophysical priors

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

* **Keep configs in Git**: all YAMLs except `/local/` are version-controlled
* **Use `/local/` for secrets/paths**: cluster creds, scratch dirs, `.gitignored`
* **Leverage interpolation**: `${data.num_classes}` ensures cross-consistency
* **Snapshot every run**: Hydra saves configs under `outputs/`
* **Sync with DVC**: every path tracked for reproducibility
* **Layer configs**: define baselines in `defaults`, override for ablation/debug/Kaggle
* **Enforce Kaggle safety**: configs tuned for ≤9 hr GPU limit, AMP, checkpointing

---

## 5. Integration

* **CLI** — All commands (`spectramind train`, `spectramind diagnose`, `spectramind submit`) load configs via Hydra
* **CI** — GitHub Actions auto-runs self-tests on configs
* **Kaggle** — configs guarantee ≤9 hr runtime, GPU RAM compliance, offline reproducibility
* **Dashboard** — config metadata embedded in `generate_html_report.py` diagnostics
* **Experiment tracking** — sync with MLflow/W\&B/TensorBoard via logger configs

---

## 6. References

* Hydra configuration best practices
* SpectraMind V50 Technical Plan
* SpectraMind V50 Project Analysis
* Strategy for Updating & Extending V50
* Kaggle Platform Guide
* Physics & Modeling References

---

## 7. DVC Pipeline (Execution DAG)

```mermaid
flowchart LR
  A0{{CLI (spectramind)}} -->|Hydra compose| A1[calibrate]
  classDef stage fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
  classDef data fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef cons fill:#fff3e0,stroke:#ef6c00,color:#e65100
  classDef cfg fill:#ede7f6,stroke:#5e35b1,color:#4527a0

  C1[(configs/data/*.yaml)]:::cfg --> A1
  C2[(configs/model/*.yaml)]:::cfg --> A1
  C3[(configs/trainer/*.yaml)]:::cfg --> A1
  C4[(configs/loss/*.yaml)]:::cfg --> A1

  A1[calibrate]:::stage --> A2[preprocess]:::stage --> A3[split]:::stage --> A4[package_batches]:::stage
  A4 --> A5[train]:::stage --> A6[predict]:::stage --> A7[diagnostics]:::stage --> A8[submit]:::stage

  R1[(raw_fgs1/)]:::data --> A1
  R2[(raw_airs/)]:::data --> A1
  R3[(calib_refs/)]:::data --> A1

  A1 --> O1[(calibrated/)]:::data
  A2 --> O2[(processed/)]:::data
  A3 --> O3[(splits/)]:::data
  A4 --> O4[(batches/)]:::data
  A5 --> O5[(outputs/models/)]:::data
  A6 --> O6[(outputs/predictions/)]:::data
  A7 --> O7[(outputs/diagnostics/)]:::data
  A8 --> O8[(outputs/submission/)]:::data

  N1{{DVC caches outputs<br/>reruns only on changes}}:::cons
  A0 --- N1
```

---

## 8. Quick Commands (Hydra · DVC · Kaggle)

### Hydra & CLI

```bash
spectramind train --config-name train +dry_run=true
spectramind train --config-name train optimizer=adamw training.epochs=30 model=airs_gnn
spectramind train --config-name train -m optimizer=adam,sgd training.batch_size=32,64,96
spectramind predict --config-name predict uncertainty.corel.enabled=true
spectramind ablate --config-name ablate ablate.leaderboard.top_n=5 ablate.leaderboard.export_html=true
```

### DVC

```bash
dvc repro
dvc repro train
dvc repro -S data.mode=kaggle
dvc dag
dvc status
```

### Kaggle Tips

```bash
spectramind predict --config-name predict data=kaggle \
  predict.device=auto predict.precision=bf16 \
  predict.outputs.write_submission=true
```

---

✅ With this setup, **`/configs` is not just parameters**:
It is **mission control** for every SpectraMind V50 experiment — delivering **NASA-grade reproducibility**, **physics-informed rigor**, and **Kaggle-safe deployment**.

---
