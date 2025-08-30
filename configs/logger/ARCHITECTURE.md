# `configs/logger/` — Architecture & Integration

This document explains how the **logger config group** fits into SpectraMind V50’s **Hydra + Typer** pipeline, how each logger behaves in **local / Kaggle / CI** environments, and the contract they share for **NASA-grade reproducibility** and **CLI-first UX**.

---

## 0) Design Goals

- **Hydra-first composition:** Choose a logger via `defaults:` or CLI; never hard-code logging in Python. The merged config is snapshotted for every run. :contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}  
- **CLI-first UX with Rich:** Console output should be readable, progress-aware, and low-overhead, while still writing an immutable audit trail. :contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}  
- **Kaggle/CI safety:** Avoid network calls by default; throttle I/O; keep artifacts modest; respect ≤9h wall-time. :contentReference[oaicite:4]{index=4}  
- **Reproducibility:** Persist config/code/data hashes and outcomes so anyone can re-run the exact conditions. :contentReference[oaicite:5]{index=5}:contentReference[oaicite:6]{index=6}

---

## 1) Components in this group

- `rich_console.yaml` — human-friendly terminal logger (progress bars, live metrics, pretty tracebacks), plus appending to `logs/v50_debug_log.md`.
- `tensorboard.yaml` — event files for scalars/histograms; throttled in Kaggle.
- `wandb.yaml` — W&B (offline by default in Kaggle); supports later sync.
- `mlflow.yaml` — MLflow tracking; defaults to local `mlruns/`, remote optional; CI profile minimizes churn.

These profiles are **orthogonal**: you can run one or **compose** (e.g., Rich + MLflow) for multi-sink logging. Hydra composition ensures code remains clean and reproducible. :contentReference[oaicite:7]{index=7}

---

## 2) Control Flow & Composition

```mermaid
flowchart LR
  A[train.yaml defaults] -->|logger=<profile>| B[Hydra Compose]
  B --> C[Typer CLI Runtime]
  C --> D[Logger Adapter Layer]
  D --> E1[Rich Console]
  D --> E2[TensorBoard]
  D --> E3[W&B]
  D --> E4[MLflow]
  D --> F[Audit Log (v50_debug_log.md)]
````

* **Hydra Compose** builds a single config where one logger is “primary”; others may be listed under `extra_loggers`.
* **Logger Adapter Layer** in the runtime initializes the selected backends, applies cadence/flush settings, and mirrors essentials to the **audit log**.
* **Audit Log** (Markdown + JSONL if enabled) contains: CLI command, merged config hash, Git commit, DVC snapshot, key metrics, artifact pointers.&#x20;

---

## 3) Environment Modes & Guardrails

| Environment    | Net             | Disk I/O  | Recommended profile(s)                                              | Notes                                                                                     |
| -------------- | --------------- | --------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| **Local dev**  | Allowed         | Normal    | `rich_console` (+ `mlflow` or `tensorboard`)                        | Great UX with Rich; add a tracker when you want artifacts and run comparisons.            |
| **Kaggle**     | **No internet** | Throttled | `rich_console` / `tensorboard` / `mlflow`(local) / `wandb`(offline) | TB flush ↑; histograms off; W\&B **offline**; MLflow to `mlruns/`; keep artifacts small.  |
| **CI (smoke)** | Off             | Minimal   | `rich_console` or `mlflow`(reduced)                                 | Keep logs minimal; disable heavy uploads; ensure quick failure signals.                   |

All profiles expose `kaggle_safe:` and `ci_fast:` sections to centralize these guardrails.

---

## 4) Reproducibility Contract

All logger profiles must ensure:

1. **Hydra snapshot** of the merged config is saved with outputs.&#x20;
2. **CLI audit entry** appended to `logs/v50_debug_log.md` (time, git SHA, DVC hash, config hash, key metrics, artifacts).&#x20;
3. **Determinism toggles** (seeds, CuDNN deterministic when required) are logged so re-runs match within tolerance.
4. **No hidden state:** Logger choices and parameters are visible in the run config and audit entry.

---

## 5) Profile Contracts

### 5.1 `rich_console.yaml`

* **Responsibilities:** Progress bars, live metrics table, styled logs, rich tracebacks; writes a concise **audit** line per major event.
* **Limits:** No structured metrics store; pair with TB/MLflow/W\&B if you need dashboards or artifacts.
* **Kaggle/CI:** Graceful degrade to plain text; lower refresh rates to cut I/O.&#x20;

### 5.2 `tensorboard.yaml`

* **Responsibilities:** Scalars/histograms/graphs to `logs/tb_runs/`.
* **Guardrails:** Increase `flush_secs`, disable histograms and images in Kaggle to reduce churn.
* **Interplay:** Safe to run alongside Rich; TB is file-only (works offline).

### 5.3 `wandb.yaml`

* **Responsibilities:** Metrics, artifacts, code/config snapshots; collaboration.
* **Guardrails:** `offline=true` in Kaggle; `mode=offline`; postpone model artifact uploads; sync later.
* **Interplay:** Use in labs/online; keep Rich for console UX.

### 5.4 `mlflow.yaml`

* **Responsibilities:** Local/remote tracking, artifacts, experiment comparisons.
* **Guardrails:** Default `tracking_uri=mlruns` (local). CI profiles disable artifacts and reduce frequency.
* **Interplay:** Good default tracker for air-gapped labs; pair with Rich for UX.

---

## 6) Selecting & Composing in Hydra

**Default in `train.yaml`:**

```yaml
defaults:
  - logger: rich_console
```

**Compose with an extra sink:**

```bash
spectramind train \
  logger=rich_console \
  logger.rich_console.extra_loggers=[mlflow] \
  logger.mlflow.tracking_uri=mlruns \
  logger.mlflow.experiment_name="V50_local"
```

Hydra’s override syntax keeps runs **code-free** and snapshotted for later replay.&#x20;

---

## 7) Failure Modes & Recovery

* **Terminal without truecolor/TTY:** Rich auto-degrades; set `logger.rich_console.enabled=false` if needed.
* **Network-restricted runs:** Ensure `wandb.offline=true`, `mlflow.tracking_uri=mlruns`.
* **Large artifacts:** For Kaggle/CI, disable model uploads, increase flush intervals, log scalars only.
* **Audit missing:** Treat as a failure; the adapter must always append the audit footer before exit (even on exceptions).

---

## 8) Compliance & References

* **CLI-first + Rich UX + structured audit**
* **Hydra composition & snapshotting**
* **Kaggle constraints & offline patterns**
* **Reproducibility & DVC/CI practices**

---

## 9) Checklist (for new loggers)

* [ ] Expose `enabled`, `level`, cadence/flush params.
* [ ] Provide `kaggle_safe` and `ci_fast` overrides.
* [ ] Append audit record (CLI cmd, git SHA, DVC hash, config hash, key metrics, artifacts).
* [ ] Avoid network calls by default; gate them behind explicit flags.
* [ ] Document usage in `README.md` and reference here.

> With this architecture, SpectraMind V50’s logging is **discoverable**, **composable**, and **reproducible by construction** — matching the project’s mission standards.

```
```
