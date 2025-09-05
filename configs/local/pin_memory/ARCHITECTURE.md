Here’s the **full set of upgraded files for `/configs/local/pin_memory/`**, built in the same style as your `precision/`, `num_workers/`, and `cudnn_benchmark/` configs:

---

### `configs/local/pin_memory/enabled.yaml`

```yaml
# ==============================================================================
# 📌 Pin Memory — Enabled
# ------------------------------------------------------------------------------
# Purpose:
#   • Enable page-locked (pinned) host memory in DataLoader.
#   • Speeds up host→GPU transfers by avoiding OS paging overhead.
# ------------------------------------------------------------------------------
# When to Use
#   • ✅ GPU training/inference (especially large batches).
#   • ⚠️ Not needed on CPU-only runs (ignored).
# ==============================================================================

pin_memory: true

notes: |
  Pin memory enabled:
  • Faster transfers from host (CPU) to device (GPU).
  • Recommended for most Kaggle/HPC GPU runs.
  • Slightly higher host RAM usage; avoid if memory is tight.
```

---

### `configs/local/pin_memory/disabled.yaml`

```yaml
# ==============================================================================
# 📌 Pin Memory — Disabled
# ------------------------------------------------------------------------------
# Purpose:
#   • Use regular pageable host memory.
#   • Lower RAM footprint; deterministic and safe for debugging.
# ------------------------------------------------------------------------------
# When to Use
#   • ✅ CPU-only runs (no effect).
#   • ✅ Low-RAM environments (avoid extra pinned allocations).
# ==============================================================================

pin_memory: false

notes: |
  Pin memory disabled:
  • Safer in memory-constrained or CPU-only environments.
  • Slightly slower GPU data transfers, but reproducible.
  • Good default for CI or audit/debug runs.
```

---

### `configs/local/pin_memory/ARCHITECTURE.md`

````markdown
# 📂 `configs/local/pin_memory/` — Pin Memory Profiles

## Purpose
Control whether PyTorch DataLoaders use **pinned host memory** for batch tensors.  
Pinned memory speeds up host→GPU transfer but uses more RAM.

## Profiles
- **`enabled.yaml`** — Enables pinning (best for GPU training/inference).
- **`disabled.yaml`** — Disables pinning (best for CPU, CI, or low-RAM).

## Usage
In your main config (e.g. `configs/train.yaml`):

```yaml
defaults:
  - local/pin_memory: enabled   # or disabled
````

Or via CLI override:

```bash
spectramind train local/pin_memory=enabled
```

## Rule of Thumb

| Environment                 | Recommendation                   |
| --------------------------- | -------------------------------- |
| Kaggle GPU (T4/A10/L4/A100) | `enabled` (faster transfers)     |
| HPC GPU Cluster             | `enabled` (8–16 workers benefit) |
| CPU-only                    | `disabled` (no effect)           |
| CI / Debugging              | `disabled` (safe + reproducible) |
| Memory constrained          | `disabled`                       |

## Notes

* Pinning can increase host RAM usage; disable if memory is limited.
* On GPUs, pinned memory usually improves throughput, especially with larger batch sizes.
* Combine with `num_workers` > 0 for best performance (workers prefetch into pinned buffers).

```

