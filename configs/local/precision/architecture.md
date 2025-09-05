# 📂 `configs/local/precision/` — Precision Profiles

## Purpose
This group isolates **numerical precision & determinism** from device and trainer configs.
Use it to swap precision (FP32 / FP16-AMP / BF16-AMP / auto) across any device or strategy.

## Design
- `local/devices/*` ⇒ hardware (CPU/GPU count, threads, walltime)
- `trainer/*`      ⇒ orchestration (DDP, logging cadence, accumulation)
- `local/precision/*` ⇒ **precision policy** (AMP mode, determinism, cuDNN autotune)

## Usage
In your main config (e.g., `configs/train.yaml`):
```yaml
defaults:
  - trainer: multi_gpu
  - local: hpc
  - local/devices: hpc_a100
  - local/precision: bf16_mixed    # ← choose a precision preset here

Or via CLI:

spectramind train local/precision=16_mixed

Profiles

32.yaml – strict FP32 (max stability, slower)

16_mixed.yaml – FP16 AMP (good on T4/A10/L4/A100; watch for overflow)

bf16_mixed.yaml – BF16 AMP (best on A100/H100; robust range)

auto_mixed.yaml – prefers BF16 when supported, else FP16, else FP32

strict_deterministic.yaml – layer to force determinism (pair on top if needed)

Rule of Thumb

A100/H100 → bf16_mixed

T4/A10/L4 → 16_mixed

CPU or debugging → 32 (optionally add strict_deterministic)
