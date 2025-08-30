# 🖥️ `configs/local/` — Local Development Profiles

This folder defines **local workstation profiles** for SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge).  
They provide **developer-friendly defaults** for running the pipeline on your own machine (Linux/Mac/Windows) while maintaining **Hydra-safe composition**, **CLI-first orchestration**, and **NASA-grade reproducibility**:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## Purpose

- **Local iteration**: Fast, balanced configs for debugging, prototyping, and training small/medium experiments.  
- **Hydra integration**: Always composable via `defaults:` or CLI overrides.  
- **Reproducibility**: Every run snapshots its merged config, Git commit, and DVC dataset hash to `logs/v50_debug_log.md`.  
- **Separation of concerns**: Kaggle/CI profiles handle strict quotas, while local profiles emphasize flexibility and developer ergonomics.

---

## Files

- **`default.yaml`** — Baseline local config (GPU/CPU autodetect, mixed precision, rich console logger).  
- *(future)* `debug.yaml` — Stripped-down fast_dev_run profile for debugging code.  
- *(future)* `hpc.yaml` — For high-performance workstations with multiple GPUs or large memory.  

---

## Usage

In your Hydra master config (e.g., `train.yaml`):

```yaml
defaults:
  - local: default
````

### CLI overrides

```bash
# Run with default local profile
spectramind train local=default

# Force CPU-only run
spectramind train local.devices=cpu

# Use bf16 precision if supported
spectramind train local.precision=bf16

# Change cache/output paths
spectramind train local.cache_dir=/scratch/cache local.output_dir=/scratch/outputs
```

Hydra will merge these with other config groups (e.g., `trainer`, `model`, `data`).

---

## Profile Design

| Field                            | Default Value  | Notes                                                   |
| -------------------------------- | -------------- | ------------------------------------------------------- |
| `devices`                        | `auto`         | Detects available GPUs; fallback to CPU if none.        |
| `precision`                      | `16-mixed`     | Mixed precision (speedup + stability).                  |
| `deterministic`                  | `true`         | CuDNN determinism enforced.                             |
| `seed`                           | `42`           | Global reproducibility.                                 |
| `logger`                         | `rich_console` | Human-friendly console output.                          |
| `dvc.enabled`                    | `true`         | Ensure artifacts tracked by DVC.                        |
| `mlflow.enabled`                 | `false`        | Off by default locally; toggle for experiment tracking. |
| `wandb.enabled`                  | `false`        | Default offline; opt-in for online logging.             |
| `save_checkpoint_every_n_epochs` | `5`            | Saves checkpoints every 5 epochs.                       |
| `keep_last_n_checkpoints`        | `3`            | Retain last 3 checkpoints.                              |

---

## When to use

* **Everyday dev runs**: Use `local=default` to quickly test new configs, models, or features.
* **Debugging**: Combine with `trainer.ci_fast` or future `local/debug.yaml` for fast\_dev\_run.
* **Extended experiments**: Switch to `local/hpc.yaml` (planned) when using multi-GPU nodes.
* **Leaderboard submissions**: Always switch to `trainer=kaggle_safe` instead of `local`.

---

## Guardrails

Local configs deliberately **do not** enforce strict runtime/disk quotas.
For Kaggle and CI runs, use the dedicated profiles (`configs/trainer/kaggle_safe.yaml`, `configs/trainer/ci_fast.yaml`) which enforce:

* ≤ 9 hours wall-clock (Kaggle GPU quota)
* ≤ 16 GB VRAM (Tesla P100/T4)
* Limited persistent storage and no internet by default

---

## References

* **SpectraMind V50 Project Analysis**: Hydra configs and CLI integration
* **Strategy for Updating/Extending V50**: Config layering, reproducibility, and MLOps
* **Ubuntu Science Workstation Guide**: Local HPC/GPU environment best practices
* **Kaggle Platform Guide**: Runtime constraints motivating Kaggle/CI-safe configs

---

**Pro tip 🚀**:
Keep `local=default` for daily work. For leaderboard runs, switch to `trainer=kaggle_safe`.
This separation ensures you enjoy **speed + rich logs locally** without accidentally breaking **competition safety rules**.

```
