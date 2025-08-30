# 🗂️ `/configs` — SpectraMind V50 Configuration System

---

## 0. Purpose & Scope

The **`/configs`** directory is the **mission control flight plan** for the **SpectraMind V50 pipeline**  
(NeurIPS 2025 Ariel Data Challenge).

It encodes all **parameters, constraints, and overrides** that govern the pipeline — ensuring that every run is:

* 🔬 **Reproducible** — Hydra-safe YAMLs, DVC-tracked data/models, Git-logged config hashes:contentReference[oaicite:3]{index=3}  
* 🛰️ **Physics-informed** — encodes symbolic constraints (smoothness, non-negativity, FFT priors, molecular fingerprints, gravitational lensing):contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}  
* ⚡ **Kaggle-safe** — ≤9 hr runtime, offline reproducible, GPU RAM guardrails:contentReference[oaicite:6]{index=6}  
* 📊 **Diagnostics-ready** — outputs HTML dashboards, calibration heatmaps, SHAP overlays, symbolic violation maps:contentReference[oaicite:7]{index=7}  

Every run is tracked and archived:

* Hydra snapshots under `outputs/DATE_TIME/.hydra/`
* Logs & config hashes in `logs/v50_debug_log.md`
* Large artifacts (datasets/models) via DVC  

---

## 1. Design Philosophy

* **Hydra-first** — modular YAMLs composed into layered configs:contentReference[oaicite:8]{index=8}  
* **No hardcoding** — all pipeline behavior controlled via configs or CLI overrides  
* **Hierarchical layering** — defaults pull from config groups (data, model, optimizer, etc.)  
* **Versioned & logged** — all configs are Git/DVC tracked, every run is hash-logged  
* **Kaggle-safe** — defaults respect GPU time/memory constraints, offline-friendly:contentReference[oaicite:9]{index=9}  
* **Physics-aware** — symbolic/astrophysical priors (molecular bands, diffraction, radiation, turbulence) are encoded in config knobs:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}  

---

## 2. Directory Layout

```

configs/
├── train.yaml              # Main training config (composes defaults)
├── predict.yaml            # Inference / submission config
├── ablate.yaml             # Grid search & ablation sweeps
├── selftest.yaml           # Lightweight smoke-test config
│
├── data/                   # Data ingestion + calibration modes
│   ├── nominal.yaml
│   ├── kaggle.yaml
│   └── debug.yaml
│
├── model/                  # Model architectures
│   ├── v50.yaml
│   ├── fgs1\_mamba.yaml
│   ├── airs\_gnn.yaml
│   ├── fusion.yaml
│   ├── decoder.yaml
│   └── constraints.yaml
│
├── optimizer/              # Optimizer / scheduler configs
│   ├── adam.yaml
│   ├── adamw\.yaml
│   └── sgd.yaml
│
├── loss/                   # Physics + symbolic loss weights
│   ├── gll.yaml
│   ├── smoothness.yaml
│   ├── nonnegativity.yaml
│   ├── fft.yaml
│   └── symbolic.yaml
│
├── trainer/                # Training loop configs
│   ├── default.yaml
│   ├── gpu.yaml
│   ├── kaggle\_safe.yaml
│   └── ci\_fast.yaml
│
├── logger/                 # Logging / experiment tracking
│   ├── tensorboard.yaml
│   ├── wandb.yaml
│   ├── mlflow\.yaml
│   └── rich\_console.yaml
│
└── local/                  # Local dev overrides (gitignored)
└── default.yaml

````

---

## 3. Usage Patterns

### Run with defaults
```bash
spectramind train --config-name train
````

### Override single values

```bash
spectramind train model=airs_gnn optimizer=adamw trainer.epochs=20
```

### Multirun sweeps

```bash
spectramind train -m optimizer=adam,sgd trainer.batch_size=32,64
```

### Kaggle-safe leaderboard run

```bash
spectramind train --config-name train trainer=kaggle_safe
```

### CI smoke test

```bash
spectramind test --config-name selftest --fast
```

---

## 4. Best Practices

✅ **Keep configs in Git** — all except `/local/`
✅ **Use `/local/` for secrets/paths** — cluster creds, scratch dirs
✅ **Exploit interpolation** — `${data.num_classes}` ensures consistency
✅ **Snapshot every run** — Hydra auto-saves configs in outputs
✅ **Track with DVC** — all dataset/model paths are DVC-linked
✅ **Layer configs** — define baselines, override for ablations/Kaggle-safe
✅ **Respect Kaggle limits** — ≤9 hr runtime, AMP enabled, safe batch sizes

---

## 5. Integration & Automation

* **CLI:** Every `spectramind` subcommand (`train`, `predict`, `diagnose`, `submit`, `ablate`) composes configs via Hydra
* **CI:** GitHub Actions auto-run `selftest.yaml` on PRs
* **DVC:** All pipeline stages (calibrate → preprocess → train → predict → diagnose → submit) mapped to configs & DVC DAG
* **Dashboard:** Config metadata embedded in `generate_html_report.py` for diagnostics
* **Experiment tracking:** Integrated with TensorBoard/W\&B/MLflow via logger configs

---

## 6. References

* 📘 \[SpectraMind V50 Technical Plan]
* 📘 \[SpectraMind V50 Project Analysis]
* 📘 \[Strategy for Updating & Extending V50]
* 📘 \[Hydra for AI Projects: Comprehensive Guide]
* 📘 \[Mermaid Diagrams in GitHub Markdown Reference]
* 📘 \[Physics & Modeling References: Radiation, Gravitational Lensing, Spectroscopy]

---

## 7. Execution DAG (DVC + Hydra)

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

✅ With this setup, **`/configs` is not just parameters** — it is the **operational DNA** of SpectraMind V50, delivering:

* **NASA-grade reproducibility**
* **Physics-informed rigor**
* **Kaggle-ready safety**
* **Full CLI/DVC/CI integration**

---

```
