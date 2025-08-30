# `configs/logger/` — Architecture & Integration

This document explains how the **logger config group** plugs into SpectraMind V50’s **Hydra + Typer** pipeline, how each logger behaves in **local / Kaggle / CI** environments, and the contract they share for **NASA-grade reproducibility** and a **CLI-first UX**.

---

## 0) Design Goals

- **Hydra-first composition.** Pick a logger in `defaults:` (or via CLI). No hard-coding in Python. The merged config is **snapshotted every run** along with overrides.
- **CLI-first UX with Rich.** Console output stays readable and progress-aware (live metrics, spinners, pretty tracebacks), while an **immutable audit trail** is written to disk.
- **Kaggle/CI safety.** Default to **no network**, throttle I/O, keep artifacts modest, and honor **≤ 9 h** wall-time and storage limits.
- **Reproducibility by construction.** Persist config/code/data hashes and outcomes so anyone can re-run the exact conditions.

---

## 1) Components in this group

- **`rich_console.yaml`** — Human-friendly terminal logger (progress bars, live tables, styled tracebacks) + concise entries appended to `logs/v50_debug_log.md`.
- **`tensorboard.yaml`** — File-backed event logging (scalars/histograms/graphs). Kaggle mode throttles flush & disables heavy features.
- **`wandb.yaml`** — W&B tracking. **Offline** in Kaggle; later sync when allowed.
- **`mlflow.yaml`** — MLflow tracking. **Local `mlruns/` by default**, remote server optional. CI profile reduces uploads.
- **`many_loggers.yaml`** — Composition of **Rich + TensorBoard + W&B + MLflow** with environment profiles to auto-prune in Kaggle/CI.

These profiles are **orthogonal**. Use one, or **compose** (e.g., Rich + MLflow) for multi-sink logging. Hydra composition keeps code clean and reproducible.

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
  D --> F[Audit Log (v50_debug_log.md + optional JSONL)]
````

* **Hydra Compose** builds the final config. One logger is “primary”; others may be listed under `extra_loggers` or composed via `many_loggers`.
* **Logger Adapter Layer** (in your runtime) reads the merged config, initializes only **active** backends, applies **kaggle/ci\_fast** overrides, and mirrors essentials to the **audit log**.
* **Audit Log** stores: CLI command, merged config hash, Git SHA, DVC snapshot, key metrics, and artifact pointers (Markdown; optional JSONL events for machine parsing).

---

## 3) Environment Modes & Guardrails

| Environment    | Net             | Disk I/O  | Recommended profile(s)                                                                       | Notes                                                                                              |
| -------------- | --------------- | --------- | -------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------- |
| **Local dev**  | Allowed         | Normal    | `rich_console` (+ `mlflow` or `tensorboard`) or `many_loggers:local`                         | Great UX with Rich; add MLflow/TB for artifacts/dashboards.                                        |
| **Kaggle**     | **No internet** | Throttled | `rich_console` / `tensorboard` / `mlflow`(local) / `wandb`(offline) or `many_loggers:kaggle` | TB flush ↑, histograms/images off, W\&B **offline**, MLflow → **`mlruns/`**. Keep artifacts small. |
| **CI (smoke)** | Off             | Minimal   | `rich_console` or `many_loggers:ci_fast`                                                     | Keep logs minimal; disable heavy uploads; ensure quick, deterministic failure signals.             |

Each logger exposes `kaggle_safe:` and `ci_fast:` sections to centralize these guardrails.

---

## 4) Reproducibility Contract

Every logger must ensure:

1. **Hydra snapshot** — Persist the fully merged config with outputs.
2. **CLI audit entry** — Append (time, Git SHA, DVC hash, config hash, key metrics, artifacts) to `logs/v50_debug_log.md` (and optionally `events.jsonl`).
3. **Determinism toggles** — Log seeds and deterministic flags (e.g., CuDNN) so re-runs match within tolerance.
4. **No hidden state** — Logger choices/parameters live in the run config and appear in the audit entry.

---

## 5) Profile Contracts

### 5.1 `rich_console.yaml`

* **Responsibilities**: Progress bars, live metrics tables, styled tracebacks; concise audit lines for major events.
* **Limits**: No structured metrics store; pair with TB/MLflow/W\&B for dashboards and artifact catalogs.
* **Kaggle/CI**: Auto-degrade to plain text on non-TTY; lower refresh for minimal I/O.

### 5.2 `tensorboard.yaml`

* **Responsibilities**: Write scalars/histograms/graphs to `logs/tb_runs/`.
* **Guardrails**: Increase `flush_secs`, set `histogram_freq=0`, `write_images=false` in Kaggle/CI.
* **Interplay**: Safe to run with Rich; TB is file-only and network-free.

### 5.3 `wandb.yaml`

* **Responsibilities**: Metrics, artifacts, and code/config snapshots; collaboration & model registry.
* **Guardrails**: Always `offline=true` and `mode=offline` for Kaggle; postpone artifact uploads; sync later if allowed.
* **Interplay**: Use in lab/online; keep Rich for console UX.

### 5.4 `mlflow.yaml`

* **Responsibilities**: Local/remote tracking, artifacts, experiment comparisons.
* **Guardrails**: Default `tracking_uri=mlruns` (local). CI profiles disable artifacts and reduce cadence.
* **Interplay**: Good default tracker for air-gapped labs; pair with Rich.

### 5.5 `many_loggers.yaml`

* **Purpose**: Single-selection multi-logging (Rich + TB + W\&B + MLflow).
* **Profiles**: `local` (all enabled), `kaggle` (offline + throttled), `ci_fast` (Rich only or heavily reduced).
* **Order/Active**: Provides `many_loggers.order` and `many_loggers.active.*` for deterministic init/teardown.

---

## 6) Selecting & Composing in Hydra

**Default in `train.yaml`:**

```yaml
defaults:
  - logger: rich_console
```

**Add a side logger (single-run):**

```bash
spectramind train \
  logger=rich_console \
  logger.rich_console.extra_loggers=[mlflow] \
  logger.mlflow.tracking_uri=mlruns \
  logger.mlflow.experiment_name="V50_local"
```

**Multi-logger with environment profile:**

```bash
# Local: all backends on
spectramind train logger=many_loggers many_loggers.profile=local

# Kaggle: offline + throttled
spectramind train logger=many_loggers many_loggers.profile=kaggle

# CI smoke: minimal logging
spectramind train logger=many_loggers many_loggers.profile=ci_fast
```

---

## 7) Failure Modes & Recovery

* **Terminal without truecolor/TTY** → Rich auto-degrades; you can set `logger.rich_console.enabled=false`.
* **Network-restricted runs** → Enforce `logger.wandb.offline=true` and `logger.mlflow.tracking_uri=mlruns`.
* **Large artifacts / TB bloat** → Raise `flush_secs`, disable images/histograms; keep only scalars.
* **Audit missing** → Treat as **failure**: adapter must append the audit footer before exit (even on exceptions).
* **Conflicting multi-logger init** → Respect `many_loggers.order` and `many_loggers.active.*` to avoid double-init/close.

---

## 8) Practical Runtime Adapter (pseudocode)

```python
def init_loggers(cfg):
    # 1) Resolve profile (for many_loggers) or single logger
    profile = getattr(cfg.get("many_loggers", {}), "profile", None)

    # 2) Apply environment overrides
    if profile and cfg.many_loggers.apply_profile_overrides:
        apply_overrides(cfg, cfg.many_loggers.profiles[profile])

    # 3) Initialize active backends in deterministic order
    active = []
    if "many_loggers" in cfg:
        for key in cfg.many_loggers.order:
            if cfg.many_loggers.active.get(ALIAS[key], False):
                active.append(init_backend(key, cfg.many_loggers[key]))
    else:
        active.append(init_backend(cfg.logger, cfg[cfg.logger]))

    # 4) Register audit hook
    audit_path = (cfg.many_loggers.audit_log_path
                  if "many_loggers" in cfg else cfg[cfg.logger].get("audit_log_path"))
    register_audit(audit_path)

    return active
```

> Your adapter should be **purely config-driven** and avoid special-case logic. All behavior must be controlled by Hydra YAML + CLI overrides.

---

## 9) Compliance & Checklist (for adding a new logger)

* [ ] Expose `enabled`, verbosity level, and flush/cadence parameters.
* [ ] Provide `kaggle_safe` and `ci_fast` sections.
* [ ] Append an audit record (CLI cmd, Git SHA, DVC hash, config hash, key metrics/artifacts).
* [ ] Default to **no network**; gate network behind explicit flags.
* [ ] Document usage in `README.md` and link here.
* [ ] Add to `many_loggers.yaml` with sane defaults and a profile matrix.

---

## 10) Security & Privacy

* **Secrets**: never hard-code. Use env vars/CI secrets (e.g., `WANDB_API_KEY`).
* **PII**: do not log PII to third-party services. Prefer anonymized run names.
* **Air-gapped**: keep **MLflow local** and **W\&B offline** by default in Kaggle/CI.

---

With this architecture, SpectraMind V50 logging is **discoverable**, **composable**, and **reproducible by construction**—perfectly aligned with mission-grade engineering and competition constraints.

```
```
