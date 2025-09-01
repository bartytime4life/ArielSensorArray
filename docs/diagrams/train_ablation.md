# 🧪 SpectraMind V50 — Training & Ablation Execution Architecture

This document illustrates the **end-to-end orchestration** of training and ablation in SpectraMind V50, mapping the flow from **CLI entrypoint → Hydra configs → Ablation engine → Parallel/Sequential runs → Diagnostics → Leaderboard export**.

The design emphasizes:
* **CLI-first control** (`spectramind ablate ...`) for all ablation workflows.  
* **Hydra configuration snapshots**: every ablation grid is a reproducible composition of configs.  
* **Parallel/sequential run engine**: supports Kaggle GPU quotas and CI/CD triggers.  
* **Diagnostics + Leaderboard**: metrics (GLL, RMSE, symbolic overlays) collated into Markdown + HTML reports.  
* **Structured logging**: `events.jsonl`, resolved configs, and artifacts tracked with DVC.  

---

## 1. High-Level DAG

```mermaid
flowchart TD
  U[User CLI Call] -->|spectramind ablate ...| E0[Typer Entrypoint]

  E0 --> H0[Hydra Compose<br/>(train/ablation.yaml + overrides)]
  H0 --> A0[Ablation Engine]

  A0 -->|Generate run grid| A1{Configs i=1..N}

  subgraph RUN[Parallel/Sequential Runs]
    direction TB
    R1[Trainer (i)] --> R2[Metrics & Artifacts (i)] --> R3[Predictions μ, σ (i)]
  end

  A1 --> RUN
  R2 --> D0[Diagnostics Collation]
  R3 --> D0
  D0 --> L0[Leaderboard Export<br/>(MD + HTML)]

  %% side outputs
  R2 --> L1[(events.jsonl)]
  H0 --> S0[(Resolved Config Snapshots)]
````

---

## 2. Stage-by-Stage Notes

### CLI → Hydra → Engine

* **Entrypoint:** `spectramind ablate ...` calls a Typer CLI subcommand.
* **Hydra Compose:** Loads `train/ablation.yaml`, merges overrides, and resolves into structured configs.
* **Ablation Engine:** Expands configs into a **grid (i=1..N)** for runs (parallel or sequential depending on resource constraints).

### Runs

* Each config instance → `Trainer(i)` executes a **training run** with consistent logging.
* Artifacts: checkpoints, metrics, predictions (μ, σ).
* Config + dataset version hashes stored alongside logs.

### Diagnostics

* Metrics collated across runs (GLL, RMSE, entropy, symbolic violation counts).
* Visual diagnostics generated (heatmaps, overlays, ablation trend plots).

### Leaderboard

* Collated metrics written to:

  * **Markdown table** (`leaderboard.md`)
  * **Interactive HTML report** (`leaderboard.html`)
* Top-N configs packaged for downstream analysis or submission.

### Logging

* **`events.jsonl`** — structured, per-run log of events, metrics, durations.
* **Resolved configs** — YAML snapshots saved per run for reproducibility.
* All tracked by DVC + Git commit hash.

---

## 3. Reproducibility Guarantees

* **Hydra-safe configs:** Every ablation run is pinned to a resolved YAML file.
* **DVC artifact tracking:** Intermediate datasets, checkpoints, metrics cached and versioned.
* **CI/CD hooks:** Ablations can be triggered in GitHub Actions or Kaggle pipelines, ensuring runs are reproducible and automated.
* **HTML/MD outputs:** Human-readable summaries for audits and reports.

---

## 4. Best Practices

* Use **Hydra overrides** to compose ablation sweeps:

  ```bash
  spectramind ablate run \
    model=resnet50,vit \
    optimizer.lr=0.001,0.01 \
    training.epochs=10
  ```

  (Expands into 2×2×1 = 4 runs.)

* Always check `leaderboard.md` for metric trends before committing to long runs.

* Ensure **DVC cache is clean** (`dvc gc --workspace`) to avoid stale artifacts.

* For Kaggle GPU quotas, prefer **sequential ablations** with checkpoint reuse.

* Export **top configs** into `ablation_results/` for downstream packaging.

---

## ✅ Audit Checklist

* [ ] Hydra config snapshots written (`configs_resolved/*.yaml`)
* [ ] Logs captured in `events.jsonl`
* [ ] Leaderboard outputs (`.md` + `.html`) generated
* [ ] Artifacts tracked by DVC
* [ ] No code changes needed between runs — all differences expressed as config overrides

---

**End of Document**

````
