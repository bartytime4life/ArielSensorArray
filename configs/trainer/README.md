# Trainer Configurations (`configs/trainer/`)

This directory defines **training runtime configurations** for SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge).
It captures all **orchestration hyperparameters** for model training and evaluation, ensuring every run is **reproducible, Hydra-safe, and Kaggle-compatible**.

---

## 🗂 Directory Map

* `defaults.yaml` — Canonical baseline trainer (epochs, batch size, logging, checkpointing).
* `kaggle_safe.yaml` — Kaggle-GPU runtime guardrails (≤ 9 hrs, ≤ 16 GB GPU mem).
* `ci_fast.yaml` — CI smoke/fast test config (tiny data split, < 5 min runtime).
* `README.md` — You’re here! Explains design philosophy and usage.

---

## 🎯 Purpose & Design Principles

Trainer configs encode **how training runs**, decoupled from model/data specifics.
They provide knobs for:

* Epochs, batch size, accumulation, gradient clipping
* Mixed precision (AMP) and device placement
* Checkpointing frequency and resume policy
* Logging cadence (Rich console, JSONL, MLflow/W\&B if enabled)
* Kaggle runtime compliance (GPU quota, wall-clock time, memory limits)

All configs follow **Hydra composition**: choose one at runtime, override fields via CLI.

---

## 🛠 Usage

### Select a Trainer Config

In `train.yaml` (or CLI):

```yaml
defaults:
  - trainer: defaults
```

Run training:

```bash
spectramind train trainer=defaults
```

Switch to Kaggle-safe mode:

```bash
spectramind train trainer=kaggle_safe
```

CI fast smoke test:

```bash
spectramind train trainer=ci_fast
```

---

### Override Parameters Inline

Hydra lets you override trainer params at launch:

```bash
spectramind train trainer=defaults \
  trainer.max_epochs=30 \
  trainer.gradient_clip_val=1.0
```

This composes `defaults.yaml`, then applies overrides.
Hydra snapshots are logged for reproducibility.

---

## ✅ Best Practices

* **Always start with `defaults.yaml`** for local dev.
* **Switch to `kaggle_safe.yaml`** for leaderboard runs. It enforces ≤ 9 hr runtime with conservative safety margins.
* **Use `ci_fast.yaml`** in CI or when debugging to catch regressions quickly (< 5 min runtime).
* **Log every run**: configs + overrides are written to `logs/v50_debug_log.md` and hashed via `run_hash_summary_v50.json`.
* **Do not hard-edit source**: every change flows through YAML + CLI overrides to maintain Hydra-based reproducibility.

---

## 🔧 Extending Trainer Configs

To add a new scenario:

1. Create a new YAML in this folder (e.g., `multi_gpu.yaml`).
2. Define training orchestration (devices, ddp, precision, etc.).
3. Document in this README.
4. Add tests under `/tests/trainer/` to validate integration.

---

## 📖 References

* Hydra modular config patterns
* CLI-driven reproducibility & logging in SpectraMind V50
* Kaggle runtime constraints & GPU quotas
* Continuous integration & smoke testing philosophy

---

> “Trainer configs are the flight plan for your run.
> Switch them like modes of a spacecraft: nominal, Kaggle-safe, or CI-fast.” 🚀

---
