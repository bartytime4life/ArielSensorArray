# `configs/logger/` — Logging Profiles for SpectraMind V50

This folder defines **Hydra-selectable logger profiles** for SpectraMind V50’s CLI-first pipeline.
Pick one (or compose several) to control **where and how** metrics, artifacts, and audit trails are recorded — with **Kaggle/CI-safe** defaults and **NASA-grade reproducibility** baked in.

> Why this matters: the V50 stack is CLI-first (Typer) with Hydra-safe configs; every run must be traceable (config snapshot + code/data hashes + metrics). These logger profiles keep console UX delightful while preserving a machine-readable audit trail for post-hoc analysis.&#x20;

---

## What’s included

* **`rich_console.yaml`** — Colorful, low-overhead console UX (progress bars, live metrics, pretty tracebacks) powered by Rich. Ideal for local dev; Kaggle/CI-safe (auto falls back to plain output).
* **`tensorboard.yaml`** — File-based event logging for scalars/histograms/graphs; throttled flush in Kaggle mode.
* **`wandb.yaml`** — Weights & Biases tracking (online/offline). Forced offline in Kaggle; can sync later.
* **`mlflow.yaml`** — MLflow tracking (local `mlruns/` by default; remote server optional). CI/Kaggle profiles reduce churn.

All four can **coexist**. Select one as the default, then add others via `logger.extra_loggers` (where supported) to **multi-log** a single run.&#x20;

---

## Quick start

In your main Hydra config (e.g., `configs/train.yaml`):

```yaml
defaults:
  - logger: rich_console   # pick one: rich_console | tensorboard | wandb | mlflow
```

CLI overrides:

```bash
# Switch logger at runtime
spectramind train logger=tensorboard

# Tune a logger param on the fly
spectramind train logger=wandb logger.wandb.offline=true

# Compose (where supported) by chaining an extra logger
spectramind train logger=rich_console logger.rich_console.extra_loggers=[mlflow]
```

Hydra composition keeps orchestration code-free and reproducible; the merged config is persisted alongside outputs.&#x20;

---

## Profiles at a glance

| Logger         | Best for                          | Network  | Artifacts | Notes                                                                  |
| -------------- | --------------------------------- | -------- | --------- | ---------------------------------------------------------------------- |
| `rich_console` | Local dev, readable CLI UX        | None     | No        | Live progress/metrics; writes key run info to `logs/v50_debug_log.md`. |
| `tensorboard`  | Lightweight scalar dashboards     | None     | TB events | Throttled flush & histograms disabled by default in Kaggle.            |
| `wandb`        | Collaboration & model registry    | Optional | Yes       | `offline=true` in Kaggle; sync later.                                  |
| `mlflow`       | Lab/on-prem tracking, comparisons | Optional | Yes       | Defaults to local `mlruns/`; CI reduces uploads.                       |

Kaggle notebooks and submissions should **avoid network calls**; use local/offline backends and throttle I/O to meet the **≤ 9-hour** runtime & storage constraints.&#x20;

---

## Selecting the right logger

* **You want instant, human-friendly feedback in the terminal** → `rich_console` (add `mlflow` for artifacts).
* **You need minimalism + existing TB habits** → `tensorboard`.
* **Your team standard is W\&B** → `wandb` (remember `offline=true` on Kaggle; sync after).
* **You run an on-prem tracker & compare many runs** → `mlflow` (leave `tracking_uri=mlruns` locally; set server URI in the lab).
* **You’re on Kaggle/CI** → Prefer `rich_console` + (optional) `mlflow` with reduced cadence; or `tensorboard` with throttled flush.&#x20;

---

## Guardrails & reproducibility

All profiles adhere to the V50 reproducibility contract:

* **Hydra snapshot**: the fully merged config is saved with each run.
* **Audit log**: key CLI call + config hash + results appended to `logs/v50_debug_log.md`.
* **Data/code hashes**: runs capture Git/DVC references so anyone can re-create exact conditions.&#x20;

---

## Examples

### 1) Rich console only (local dev)

```bash
spectramind train \
  logger=rich_console \
  logger.rich_console.level=DEBUG \
  logger.rich_console.dashboard_refresh=0.25
```

### 2) Compose rich console + MLflow (local file backend)

```bash
spectramind train \
  logger=rich_console \
  logger.rich_console.extra_loggers=[mlflow] \
  logger.mlflow.tracking_uri=mlruns \
  logger.mlflow.experiment_name="V50_local"
```

### 3) Kaggle-safe TB

```bash
spectramind train \
  logger=tensorboard \
  logger.tensorboard.flush_secs=300
```

### 4) W\&B offline (Kaggle/notebook), sync later

```bash
spectramind train \
  logger=wandb \
  logger.wandb.offline=true \
  logger.wandb.mode=offline \
  logger.wandb.log_model=false
```

---

## Troubleshooting

* **No colors / garbled progress on remote terminals** → `logger.rich_console.enabled=true` but terminal may not support rich rendering; it will degrade gracefully.
* **W\&B permission or network errors** → set `logger.wandb.offline=true` (Kaggle), or export `WANDB_API_KEY` locally.
* **MLflow writes too much in CI** → set `logger.mlflow.ci_fast.enabled_override=true` or disable artifacts.
* **Large TB logs** → reduce `histogram_freq`, increase `flush_secs`, or disable images.

---

## References

* CLI-first + Rich console dashboards and audit logging
* Hydra composition & config snapshots for reproducibility
* Kaggle runtime guardrails (no internet, limited time/storage)
* V50 reproducibility and DVC/CI pipeline practices

---

**Pro tip:** keep `logger=rich_console` as your **default** for a great DX, and **add** `mlflow` or `tensorboard` when you need artifacts or dashboards.
