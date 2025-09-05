# 📂 `configs/local/devices/` — Device Profiles

This folder defines **device-specific Hydra profiles** for SpectraMind V50.  
They compose with `local/*` environment configs and `trainer/*` strategies.

## Profiles
- **cpu.yaml** — Force CPU-only runs (CI, portability tests).
- **single_gpu.yaml** — Default single-GPU dev with AMP.
- **multi_gpu.yaml** — Multi-GPU rig (2–8 GPUs, DDP).
- **kaggle_t4.yaml** — Kaggle T4 runtime (16 GB GPU).
- **kaggle_a10.yaml** — Kaggle A10 runtime (24 GB GPU).
- **kaggle_l4.yaml** — Kaggle L4 runtime (24 GB GPU, Ada Lovelace arch).
- **hpc_a100.yaml** — HPC clusters with A100s (multi-GPU DDP).

## Usage
In CLI:
```bash
spectramind train local=default local.devices=single_gpu
spectramind train trainer=multi_gpu local=hpc local.devices=hpc_a100
