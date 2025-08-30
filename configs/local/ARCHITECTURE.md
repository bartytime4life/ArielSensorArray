# `configs/local/` — Architecture & Composition

This folder contains **local workstation profiles** that compose cleanly with the rest of the Hydra config
tree for SpectraMind V50. The goal is to give you **fast, ergonomic, and reproducible** local runs, while
keeping Kaggle/CI constraints in their own profiles.

---

## How these compose

Hydra merges configs from left → right according to your `defaults:`. A typical `train.yaml` (simplified):

```yaml
# configs/train.yaml (excerpt)
defaults:
  - trainer: defaults          # or: kaggle_safe | ci_fast | multi_gpu
  - local: default             # or: debug | hpc
  - model: v50
  - data: nominal
  - loss: composite
  - logger: rich_console
At runtime:

trainer/* controls the execution strategy (DDP, precision wiring, logging cadence).

local/* controls the machine profile (devices, precision, seeds, paths, dev toggles).

Other groups (model/data/loss/logger) layer on top as usual.

If both trainer/* and local/* define the same field, the latter in defaults: wins. Keep
orchestration knobs (e.g., strategy) primarily in trainer/*, and environment knobs in local/*.

Profiles
local/default.yaml — everyday dev (GPU/CPU autodetect, 16-mixed, rich logging).

local/debug.yaml — ultra-fast sanity checks (fast_dev_run: true, CPU default, verbose logs).

local/hpc.yaml — multi-GPU, bf16-first, DDP hints, higher dataloader throughput.

Pick one in defaults: or via CLI: spectramind train local=hpc.

Trainer strategies
trainer/defaults.yaml (not shown): single-GPU/CPU friendly defaults.

trainer/kaggle_safe.yaml: 9-hour wall, throttled I/O, no internet.

trainer/ci_fast.yaml: tiny smoke runs for CI.

trainer/multi_gpu.yaml (added in this update): DDP strategy for 2–8 GPUs, sync BN, efficient logging.

Use trainer=multi_gpu together with local=hpc for a production-grade multi-GPU dev run.

Common CLI recipes (copy-paste)
bash
Copy code
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
Reproducibility guardrails (local)
All local profiles:

set a global seed and deterministic: true where feasible,

