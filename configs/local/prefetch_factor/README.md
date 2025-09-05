# `/configs/local/prefetch_factor/`

Prefetch factor configs for **SpectraMind V50** dataloaders.

## Purpose
Controls how many batches each DataLoader worker preloads ahead of time.  
This balances **GPU utilization** (avoid idle time) against **system memory usage**.

## Files
- **default.yaml** — safe baseline (2 batches, Kaggle-friendly).
- **fast.yaml** — more aggressive (4 batches, local SSD/HPC).
- **debug.yaml** — minimal prefetching (1 batch, low-memory/debugging).
- **heavy.yaml** — maximum prefetching (8 batches, HPC only).

## Usage
Hydra override from CLI:
```bash
spectramind train local.prefetch_factor=fast
