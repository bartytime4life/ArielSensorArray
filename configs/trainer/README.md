# Trainer Configurations (`configs/trainer/`)

This directory defines **training runtime configurations** for **SpectraMind V50** (NeurIPS 2025 Ariel Data Challenge).  
It captures all **orchestration hyperparameters** for model training and evaluation, ensuring every run is **reproducible, Hydra-safe, and Kaggle-compatible**.

---

## 🗂 Directory Map

```

configs/trainer/
├─ defaults.yaml        # Canonical baseline (epochs, logging, ckpt, determinism)
├─ kaggle.yaml          # Notebook-friendly profile (≤9h, AMP, low I/O)
├─ kaggle\_safe.yaml     # Leaderboard-safe profile (≤9h, throttled I/O, early stop)
├─ ci\_fast.yaml         # CI smoke run (<20 min; tiny batches)
├─ multi\_gpu.yaml       # Single-node multi-GPU DDP (devices=auto)
├─ gpu.yaml             # Multi-GPU / cluster-ready DDP (auto + knobs)
├─ ddp.yaml             # Explicit multi-node DDP (num\_nodes × devices)
└─ README.md            # You’re here

````

> Profiles are composable. Pick one at launch (`trainer=<name>`) and override any field via CLI.

---

## 🎯 Purpose & Design Principles

Trainer configs encode **how training runs**, decoupled from model/data specifics. They provide knobs for:

- Epochs, batch size, accumulation, gradient clipping  
- Mixed precision (AMP/BF16) and device placement  
- Checkpointing frequency & resume policy  
- Logging cadence (Rich console / TensorBoard / CSV / W&B if enabled)  
- **Kaggle runtime compliance** (GPU quota, wall-clock, memory limits)  
- **DDP best practices** (static graphs, NCCL, sync BN, bucket sizes)

All configs follow **Hydra composition**: choose one at runtime, **override fields via CLI**, and Hydra will snapshot the composed config for exact reproducibility.

---

## 🛠 Quick Usage

### Select a Trainer Profile

In `train.yaml` (or via CLI):

```yaml
defaults:
  - trainer: defaults
````

Run baseline:

```bash
spectramind train trainer=defaults
```

Kaggle-safe (≤9h, low I/O):

```bash
spectramind train trainer=kaggle_safe
```

Notebook-friendly Kaggle profile:

```bash
spectramind train trainer=kaggle
```

CI smoke:

```bash
spectramind train trainer=ci_fast
```

Single-node multi-GPU (auto-detect):

```bash
spectramind train trainer=multi_gpu
```

Cluster / multi-node DDP:

```bash
spectramind train trainer=ddp trainer.devices=8 trainer.num_nodes=2
```

---

## 🔁 Common Overrides (CLI)

```bash
# Schedule
spectramind train trainer.max_epochs=30 trainer.check_val_every_n_epoch=1

# Precision & devices
spectramind train trainer.precision=16 trainer.accelerator=gpu trainer.devices=4

# DDP toggles
spectramind train trainer.strategy.init_args.static_graph=false
spectramind train trainer.sync_batchnorm=true

# Gradient accumulation & clipping
spectramind train trainer.accumulate_grad_batches=2 trainer.gradient_clip_val=1.0

# Dataloader tuning
spectramind train trainer.dataloader.num_workers=8 trainer.dataloader.prefetch_factor=2

# Kaggle iteration speedups (if your DataModule honors them)
spectramind train +limits.train_frac=0.5 +limits.max_train_batches=300
```

> Tip: Use `+` to inject optional keys (e.g., `limits.*`) if they’re read by your datamodule.

---

## 📊 Profile Cheat-Sheet

| Profile       | Scope / Intent                                    | Default Devices               | Time Budget      | Notable Flags                                    |
| ------------- | ------------------------------------------------- | ----------------------------- | ---------------- | ------------------------------------------------ |
| `defaults`    | Local dev baseline                                | `devices=1`                   | flexible         | AMP(16), deterministic, frequent eval            |
| `kaggle`      | Notebook-friendly Kaggle runs                     | `devices=1`                   | ≤ 9h             | AMP(16), conservative I/O, optional limits       |
| `kaggle_safe` | Leaderboard-safe Kaggle profile                   | `devices=1`                   | ≤ 9h (8:30 wall) | Throttled ckpt (every\_n\_epochs), early stop    |
| `ci_fast`     | CI smoke test                                     | auto (CPU/GPU)                | < 20 min         | tiny batches, no ckpt, deterministic             |
| `multi_gpu`   | Single-node multi-GPU DDP                         | `devices=auto`                | flexible         | static\_graph, NCCL, sync BN, bucket\_cap tuning |
| `gpu`         | Multi-GPU / cluster-ready DDP (node or multinode) | `devices=auto`                | flexible         | full DDP knobs, env tips for NCCL                |
| `ddp`         | Explicit multi-node DDP                           | `devices=8,num_nodes=2` (ex.) | flexible         | `static_graph`, `timeout`, rank-zero ckpt        |

---

## ✅ Best Practices

* **Start with `defaults.yaml`** for local development & tuning.
* **Switch to `kaggle_safe.yaml`** for leaderboard runs — ≤ 9 h, throttled I/O, determinism.
* **Use `ci_fast.yaml`** in CI to catch regressions quickly (deterministic, tiny batches).
* **Log every run**: Hydra snapshots the merged config; pipe out run hashes & metadata via your logging hooks.
* **Don’t hard-edit code to change training behavior** — use YAML + CLI overrides to preserve Hydra-based reproducibility.

---

## 🧪 Troubleshooting & Tips

* **OOM / small VRAM**: increase `trainer.accumulate_grad_batches`, reduce per-GPU batch size, or flip to BF16 on Ampere/Hopper (`precision="bf16-mixed"`).
* **Slow storage**: raise `trainer.dataloader.num_workers`, enable `persistent_workers`, and cache preprocessed tensors in your datamodule.
* **DDP graph rebuilds**: if you see excessive sync overhead, try `strategy.init_args.static_graph=true` (only if the graph truly doesn’t change per step).
* **BN across ranks**: set `trainer.sync_batchnorm=true` for multi-GPU.
* **NCCL stability** (export in launcher):

  ```bash
  export NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_DEBUG=WARN
  export TORCH_DISTRIBUTED_DEBUG=OFF
  export CUDA_LAUNCH_BLOCKING=0
  ```
* **Kaggle**: throttle checkpoints (`model_checkpoint.every_n_epochs`), keep `num_workers` small (e.g., 2), and use `limits.*` for quicker iteration.

---

## 🛠 Extending Trainer Configs

1. **Add a file** in this folder (e.g., `h100_bf16.yaml`).
2. **Define orchestration** (devices, precision, DDP knobs, logging/ckpt cadence).
3. **Document** the profile here (one-liner + key differences).
4. **Test**: add unit/CLI tests under `/tests/trainer/` to validate compose, overrides, and rank-zero ckpt behavior.

---

## 🧭 Example Recipes

**Hopper BF16 multi-GPU single-node**

```bash
spectramind train trainer=multi_gpu \
  trainer.devices=8 trainer.precision=bf16-mixed \
  trainer.strategy.init_args.bucket_cap_mb=50
```

**Kaggle 50% speed run**

```bash
spectramind train trainer=kaggle_safe \
  +limits.train_frac=0.5 +limits.max_train_batches=400 \
  trainer.max_epochs=16
```

**Multi-node DDP (2×8 GPUs)**

```bash
spectramind train trainer=ddp \
  trainer.devices=8 trainer.num_nodes=2 \
  trainer.strategy.init_args.static_graph=true
```

---

## 📖 References

* Hydra modular config composition & structured overrides
* CLI-driven reproducibility & logging in SpectraMind V50
* Kaggle runtime constraints (GPU quotas, wall-clock, I/O hygiene)
* CI smoke testing & deterministic execution guidelines

---

> *“Trainer configs are the flight plan for your run — switch them like mission modes: **nominal**, **Kaggle-safe**, or **CI-fast**.”* 🚀

```
```
