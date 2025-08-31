# ⚙️ `/configs/train` — Training Configurations (SpectraMind V50)

## 0) Purpose & Scope

The **`/configs/train`** directory defines all **training orchestration configs** for the  
SpectraMind V50 pipeline (NeurIPS 2025 Ariel Data Challenge).  

It governs **runtime execution parameters** of the training engine:  
epochs, batch sizes, checkpointing cadence, AMP, parallelization, and ablation sweeps.  
This layer ensures training is **reproducible, Hydra-driven, Kaggle-safe, and CI-compatible**:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 1) Directory Layout

```

/configs/train/
├── defaults.yaml      # Canonical trainer defaults (baseline epochs, batch, logging)
├── kaggle\_safe.yaml   # Kaggle-GPU runtime guardrails (≤ 9h, ≤ 16 GB GPU)
├── ci\_fast.yaml       # CI smoke/fast config (tiny slice, <5 min runtime)
├── ablation.yaml      # Specialized config for ablation sweeps
└── ARCHITECTURE.md    # Design doc w/ Mermaid DAG + sequence diagram

````

---

## 2) Design Principles

* **Hydra-first modularity** — configs are composable; everything is override-able from CLI:contentReference[oaicite:2]{index=2}.  
* **CLI-driven** — `spectramind train ...` always consumes these YAMLs, never code edits:contentReference[oaicite:3]{index=3}.  
* **Reproducibility** — every run’s config is saved + hashed (`run_hash_summary_v50.json`).  
* **Physics-aware ablation** — toggles symbolic rules and loss terms systematically:contentReference[oaicite:4]{index=4}.  
* **Kaggle-safe defaults** — all configs tuned to ≤9h GPU runtime and ≤16 GB mem budget:contentReference[oaicite:5]{index=5}.  
* **CI coverage** — lightweight `ci_fast.yaml` enables smoke testing in <5 minutes:contentReference[oaicite:6]{index=6}.  

---

## 3) Key Config Types

### `defaults.yaml`  
Baseline trainer setup:
- Epochs: 50  
- Batch size: 64  
- Gradient accumulation: 2  
- Checkpoint every 10 epochs  
- AMP: enabled  

---

### `kaggle_safe.yaml`  
Runtime guardrails:
- Epochs reduced (30)  
- Smaller batch (32)  
- Timeout ≤ 540 min (9h)  
- Memory-efficient dataloaders  

---

### `ci_fast.yaml`  
Continuous Integration smoke:
- Epochs = 2  
- Batch = 8  
- Tiny data split (1–2 planets)  
- Runtime <5 minutes for GitHub Actions:contentReference[oaicite:7]{index=7}  

---

### `ablation.yaml`  
Purpose: orchestrates **systematic sweeps** of symbolic/loss toggles.  
Includes:  
- Loss term weights (`smoothness`, `fft`, `nonnegativity`, `asymmetry`)  
- Symbolic rule toggles (`molecular_fingerprint`, `gravitational_lensing`)  
- Parallel run grid (`-m` Hydra multirun or `ablation.parallel_runs`)  
- Leaderboard export (Markdown + HTML):contentReference[oaicite:8]{index=8}  

---

## 4) Execution Flow (Mermaid)

```mermaid
flowchart TD
  A0[User CLI: spectramind ablate] --> A1[Typer Entrypoint]
  A1 --> A2[Hydra Compose: defaults + overrides]
  A2 --> A3[Resolved Config (train/*.yaml)]
  A3 --> A4[Trainer Engine]
  A4 --> A5[Metrics + Artifacts]
  A5 --> A6[Diagnostics → Leaderboard]
  A6 --> A7[Markdown + HTML Reports]
````

---

## 5) CLI Usage

### Standard Training

```bash
spectramind train trainer=defaults
```

### Kaggle Runtime Safe

```bash
spectramind train trainer=kaggle_safe
```

### CI Smoke Test

```bash
spectramind train trainer=ci_fast
```

### Symbolic Ablation Sweep

```bash
spectramind ablate -m loss.composite.smoothness.weight=0.0,0.05,0.1
```

---

## 6) Artifacts & Logging

* **Logs** → `logs/v50_debug_log.md`, `logs/events.jsonl`
* **Leaderboards** → `outputs/ablation_leaderboard.{md,html}`
* **Diagnostics** → `outputs/diagnostics/ablation/*`
* **Config snapshots** → saved `.yaml` per run

---

## 7) Why This Layout Works

* Keeps training **config-driven** and **audit-proof**.
* Ensures symbolic & physics-aware ablations are first-class citizens.
* CI, Kaggle, and local HPC all share one Hydra config ecosystem.
* Every experiment is **reproducible** by design.

---

```
```
