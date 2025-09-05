# `configs/local/` — Architecture & Composition (Upgraded)

Mission: lightning-fast, ergonomic, **reproducible** local runs that compose cleanly with Hydra while keeping Kaggle/CI constraints separate. These profiles centralize **machine specifics** (devices, precision, paths, dev toggles) and deliberately **avoid orchestration** that belongs in `trainer/*` (strategy, logging cadence, walltime guards).

---

## 0) TL;DR

* Put **environment knobs** in `local/*` (e.g., `devices`, `precision`, `num_workers`, `paths`, `dev flags`).
* Put **execution strategy** in `trainer/*` (e.g., `ddp`, `precision wiring`, `grad_accum`, `loggers`).
* Choose your local profile via `defaults` or CLI, e.g.:

  * `spectramind train local=default`
  * `spectramind train trainer=multi_gpu local=hpc local.devices=8`

---

## 1) How Hydra Composition Works Here

```yaml
# configs/train.yaml (excerpt)
defaults:
  - trainer: defaults           # or: kaggle_safe | ci_fast | multi_gpu
  - local: default              # or: debug | hpc
  - model: v50
  - data: nominal
  - loss: composite
  - logger: rich_console
```

* Merge order is **left → right**. Later entries win on key collisions.
* Keep **strategy** (e.g., DDP, logging cadence) in `trainer/*`.
* Keep **environment** (e.g., devices, bf16 vs fp16, paths) in `local/*`.
* If a field appears in both, **`local/*` wins** because it appears to the right of `trainer/*` in `defaults`.

---

## 2) Directory Layout

```
configs/
└─ local/
   ├─ default.yaml   # everyday dev (GPU/CPU autodetect, bf16/fp16 mixed, rich logs)
   ├─ debug.yaml     # ultra-fast smoke (fast_dev_run, 1 batch caps, verbose)
   └─ hpc.yaml       # multi-GPU friendly (DDP hints, bf16-first, higher throughput)
```

> You can add machine- or site-specific overlays (e.g., `workstation_3090.yaml`, `lab_a100.yaml`) and select them with `local=workstation_3090`.

---

## 3) Profiles (Reference)

### 3.1 `local/default.yaml` — everyday development

* **Devices**: auto (GPU → fallback CPU)
* **Precision**: prefer `bf16` if supported → else `16-mixed` → else `32`
* **Throughput**: moderate dataloader workers, pinned memory on GPU
* **UX**: rich console logging, progress bars, periodic validation
* **Repro**: global seed, `deterministic: true` where feasible

**Template:**

```yaml
# configs/local/default.yaml
_target_: spectramind.config.LocalConfig
name: default

devices: "auto"            # "auto" | "cpu" | <int> (num GPUs) | "mps"
precision: "auto-mixed"    # "bf16-mixed" | "16-mixed" | "32" | "auto-mixed"
seed: 1337
deterministic: true
cudnn_benchmark: false

num_workers:
  train: 8
  val: 4
  test: 4

prefetch_factor: 2
pin_memory: true
persistent_workers: true

accumulate_grad_batches: 1
max_epochs: 60

mlflow:
  enabled: false
  tracking_uri: "file:${oc.env:SM_MLFLOW_DIR,mlruns}"
  experiment_name: "V50_local"

paths:
  cache_dir: ${oc.env:SM_CACHE_DIR,.cache/spectramind}
  output_dir: ${oc.env:SM_OUTPUT_DIR,outputs/local}
  data_root: ${oc.env:SM_DATA_ROOT,data}
  run_manifest: ${paths.output_dir}/run_manifest.json

dev:
  fast_dev_run: false
  limit_train_batches: null      # e.g., 0.1 or 10
  limit_val_batches: null
  limit_test_batches: null
  overfit_batches: 0.0
  log_every_n_steps: 25
  enable_progress_bar: true
  check_val_every_n_epoch: 1
```

---

### 3.2 `local/debug.yaml` — ultra-fast sanity/smoke

* **Goal**: fail fast, CI-like speed locally
* **Caps**: 1 batch each split; or tiny fractional limits
* **Devices**: default CPU to avoid GPU warmup unless overridden

**Template:**

```yaml
# configs/local/debug.yaml
_target_: spectramind.config.LocalConfig
name: debug

devices: "cpu"
precision: "32"
seed: 1337
deterministic: true
cudnn_benchmark: false

num_workers:
  train: 0
  val: 0
  test: 0

prefetch_factor: 2
pin_memory: false
persistent_workers: false

accumulate_grad_batches: 1
max_epochs: 1

mlflow:
  enabled: false
  tracking_uri: "file:${oc.env:SM_MLFLOW_DIR,mlruns}"
  experiment_name: "V50_debug"

paths:
  cache_dir: ${oc.env:SM_CACHE_DIR,.cache/spectramind}
  output_dir: ${oc.env:SM_OUTPUT_DIR,outputs/debug}
  data_root: ${oc.env:SM_DATA_ROOT,data}
  run_manifest: ${paths.output_dir}/run_manifest.json

dev:
  fast_dev_run: true
  limit_train_batches: 1
  limit_val_batches: 1
  limit_test_batches: 1
  overfit_batches: 0.0
  log_every_n_steps: 1
  enable_progress_bar: true
  check_val_every_n_epoch: 1
```

---

### 3.3 `local/hpc.yaml` — multi-GPU workstation / DGX

* **Pair with**: `trainer=multi_gpu`
* **Precision**: `bf16-mixed` preferred
* **Throughput**: higher workers, pinned memory, persistent workers
* **Note**: set `local.devices` to an integer to fix GPU count (e.g., 4 or 8)

**Template:**

```yaml
# configs/local/hpc.yaml
_target_: spectramind.config.LocalConfig
name: hpc

devices: 4                 # override at CLI: local.devices=8
precision: "bf16-mixed"
seed: 1337
deterministic: true
cudnn_benchmark: true

num_workers:
  train: 16
  val: 8
  test: 8

prefetch_factor: 3
pin_memory: true
persistent_workers: true

accumulate_grad_batches: 1
max_epochs: 120

mlflow:
  enabled: true
  tracking_uri: "file:${oc.env:SM_MLFLOW_DIR,mlruns}"
  experiment_name: "V50_hpc"

paths:
  cache_dir: ${oc.env:SM_CACHE_DIR,/scratch/.spectramind/cache}
  output_dir: ${oc.env:SM_OUTPUT_DIR,/scratch/outputs/hpc}
  data_root: ${oc.env:SM_DATA_ROOT,/datasets/ariel}
  run_manifest: ${paths.output_dir}/run_manifest.json

dev:
  fast_dev_run: false
  limit_train_batches: null
  limit_val_batches: null
  limit_test_batches: null
  overfit_batches: 0.0
  log_every_n_steps: 50
  enable_progress_bar: true
  check_val_every_n_epoch: 1
```

---

## 4) Field Reference (What lives in `local/*`)

| Key                            | Type      | Purpose                                     | Typical Values / Notes                                     |
| ------------------------------ | --------- | ------------------------------------------- | ---------------------------------------------------------- |
| `devices`                      | str/int   | Which accelerator(s) to use                 | `"auto"`, `"cpu"`, `1`, `4`, `"mps"`                       |
| `precision`                    | str       | AMP/precision policy                        | `"bf16-mixed"`, `"16-mixed"`, `"32"`, `"auto-mixed"`       |
| `seed`                         | int       | Global RNG seed                             | e.g., `1337`                                               |
| `deterministic`                | bool      | Enable deterministic kernels where feasible | `true` for reproducibility                                 |
| `cudnn_benchmark`              | bool      | cudnn autotuner                             | `false` default; `true` on HPC when input sizes are stable |
| `num_workers.{train,val,test}` | int       | PyTorch DataLoader workers per split        | scale with CPU cores                                       |
| `prefetch_factor`              | int       | DataLoader prefetch                         | `2`–`4`                                                    |
| `pin_memory`                   | bool      | Pin host memory for faster H2D copies       | `true` if GPU                                              |
| `persistent_workers`           | bool      | Keep workers alive across epochs            | `true` for long runs                                       |
| `accumulate_grad_batches`      | int       | Gradient accumulation                       | e.g., `1`, `2`                                             |
| `max_epochs`                   | int       | Epoch cap                                   | e.g., `60`, `120`                                          |
| `mlflow.enabled`               | bool      | Toggle MLflow client                        | `true` / `false`                                           |
| `mlflow.tracking_uri`          | str       | MLflow tracking URI                         | typically `file:mlruns` or local server                    |
| `mlflow.experiment_name`       | str       | MLflow experiment name                      | e.g., `"V50_local"`                                        |
| `paths.cache_dir`              | str       | Cache root for intermediate artifacts       | env-overridable                                            |
| `paths.output_dir`             | str       | Outputs root (checkpoints, logs, reports)   | env-overridable                                            |
| `paths.data_root`              | str       | Dataset root                                | env-overridable                                            |
| `paths.run_manifest`           | str       | Where to write run manifest JSON            | `${paths.output_dir}/run_manifest.json`                    |
| `dev.fast_dev_run`             | bool      | Lightning fast\_dev\_run toggle             | `true` in debug                                            |
| `dev.limit_*_batches`          | int/float | Cap batches for quick iteration             | `1` or `0.1`                                               |
| `dev.overfit_batches`          | float     | Overfit small subset for checks             | `0.0` default                                              |
| `dev.log_every_n_steps`        | int       | Logging cadence                             | `25` default; higher on HPC                                |
| `dev.enable_progress_bar`      | bool      | Pretty console progress                     | `true`                                                     |
| `dev.check_val_every_n_epoch`  | int       | Validation frequency                        | `1`                                                        |

---

## 5) Trainer Strategies (pair with `local/*`)

* `trainer/defaults.yaml`: single-GPU/CPU friendly defaults.
* `trainer/kaggle_safe.yaml`: ≤9h walltime, throttled I/O, offline mode.
* `trainer/ci_fast.yaml`: smoke tests in CI.
* `trainer/multi_gpu.yaml`: DDP w/ sync BN, balanced logging, elastic seeds.

> Use **`trainer=multi_gpu` + `local=hpc`** for production-grade multi-GPU dev runs:
>
> ```
> spectramind train trainer=multi_gpu local=hpc local.devices=4
> ```

---

## 6) Common CLI Recipes (copy-paste)

```bash
# 1) Single-GPU local dev, auto precision
spectramind train local=default

# 2) Force CPU for portability tests
spectramind train local=default local.devices=cpu local.precision=32

# 3) Ultra-fast sanity check (1 batch train/val/test)
spectramind train local=debug

# 4) Multi-GPU (e.g., 4×A100) with bf16, DDP
spectramind train trainer=multi_gpu local=hpc local.devices=4

# 5) Multi-GPU + custom epochs/accumulation
spectramind train trainer=multi_gpu local=hpc \
  local.devices=8 local.max_epochs=120 local.accumulate_grad_batches=2

# 6) Log to MLflow locally (mlruns/)
spectramind train local=default local.mlflow.enabled=true \
  local.mlflow.experiment_name="V50_local_exps"

# 7) Change cache/output roots
spectramind train local=default \
  local.cache_dir=/scratch/.spectramind/cache local.output_dir=/scratch/outputs
```

---

## 7) Reproducibility Guardrails (local)

All `local/*` profiles:

* Set a **global seed** and `deterministic: true` where feasible.
* Default **`cudnn_benchmark: false`** (flip to `true` on stable shapes/HPC).
* Prefer **bf16-mixed** (if available) → **16-mixed** → **32** fallback.
* Avoid hiding orchestration: **no DDP/strategy toggles** in `local/*`.
* Write a **run manifest** (paths, config hash, seed, devices) to `${paths.run_manifest}`.

---

## 8) Environment Variables (optional overrides)

| Env Var         | Overrides                        | Example                       |
| --------------- | -------------------------------- | ----------------------------- |
| `SM_CACHE_DIR`  | `paths.cache_dir`                | `/scratch/.spectramind/cache` |
| `SM_OUTPUT_DIR` | `paths.output_dir`               | `/scratch/outputs`            |
| `SM_DATA_ROOT`  | `paths.data_root`                | `/datasets/ariel`             |
| `SM_MLFLOW_DIR` | `mlflow.tracking_uri` (file URI) | `/workspace/mlruns`           |

All env overrides shown in YAML via `${oc.env:VAR,default}`.

---

## 9) Troubleshooting & Tips

* **MPS / Apple Silicon**: set `local.devices=mps` and `precision=bf16-mixed` (if supported) or `32`.
* **Worker deadlocks**: drop `persistent_workers: false`, reduce `num_workers`, and set `prefetch_factor: 2`.
* **Throughput tuning**: increase `num_workers`, enable `cudnn_benchmark: true` when input shapes are stable.
* **OOM**: raise `accumulate_grad_batches`, reduce batch size (in `data/*`), or use lower precision.

---

## 10) Minimal Logic for “auto” Precision (documentation snippet)

Your launcher or trainer glue should implement a tiny adapter like:

```python
def pick_precision(requested: str, cuda_cap_bf16: bool, cuda_cap_fp16: bool) -> str:
  if requested == "auto-mixed":
    if cuda_cap_bf16:
      return "bf16-mixed"
    if cuda_cap_fp16:
      return "16-mixed"
    return "32"
  return requested
```

This keeps YAML clean while preserving sensible fallbacks.

---

## 11) Merge Semantics (Concrete Example)

```yaml
# trainer/multi_gpu.yaml (excerpt)
strategy: ddp
sync_batchnorm: true
log_grad_norm: false
precision: "16-mixed"     # ← strategy default (can be overridden)
```

```yaml
# local/hpc.yaml (excerpt)
precision: "bf16-mixed"    # ← this wins because local/* merges after trainer/*
devices: 8
```

Resulting effective config:

* `strategy=ddp`, `sync_batchnorm=true` (from trainer)
* `precision=bf16-mixed`, `devices=8` (from local)
* All other groups (`model`, `data`, `loss`, `logger`) layer normally

---

## 12) Visual: Where `local/*` Fits

```mermaid
flowchart LR
  A[model/*] --> M[Hydra Merge]
  B[data/*]  --> M
  C[loss/*]  --> M
  D[logger/*]--> M
  E[trainer/* (strategy, cadence)] --> M
  F[local/* (devices, precision, paths)] --> M
  M --> G[Effective Runtime Config]
  G --> H[Trainer Orchestration]
  G --> I[DataLoaders]
  G --> J[Lightning Module]
```

---

## 13) Extension Patterns

* Add `local/<host>.yaml` for a named workstation; keep diffs minimal (e.g., only `devices`, `paths`, `num_workers`).
* Keep **Kaggle/CI** limits out of `local/*`; put them in `trainer/kaggle_safe.yaml` and `trainer/ci_fast.yaml`.
* When adding new fields, prefer putting **site/machine specifics** in `local/*` first.

---
