````markdown
# 🛰️ `/configs/local/accumulate_grad_batches/ARCHITECTURE.md` — Gradient Accumulation Policies

---

## 0) Purpose & Scope

This group defines **gradient accumulation** policies for SpectraMind V50.  
It controls how many **mini-batches** are accumulated **before** a backward/optimizer step in the **train loop**.

Why it matters:
- **Simulate large batch sizes** on limited VRAM without changing the physical `data.batch_size`.
- **Stabilize updates** by averaging gradients across N micro-steps.
- **Stay Kaggle-safe (≤ 9h)** while preserving model quality.

These policies are wired into the main config (e.g., `configs/train.yaml`) as:
```yaml
training:
  gradient_accumulation_steps: ${accumulate_grad_batches}
````

…and composed via Hydra defaults:

```yaml
defaults:
  - local/accumulate_grad_batches: train
  - local/accumulate_grad_batches@val_loader: val
  - local/accumulate_grad_batches@test_loader: test
```

---

## 1) Mental Model

Let:

* `B = data.batch_size`  (per-GPU mini-batch)
* `K = training.gradient_accumulation_steps` (this policy)
* `W = world_size` (number of data-parallel workers/GPUs, 1 for single-GPU)

Then:

* **Effective batch per optimizer step**: `B_eff = B × K × W`
* **Effective learning rate** (rule-of-thumb): `lr_eff ≈ lr_base × K`
  (Adjust *either* `lr` or `K`; increasing both is usually harmful.)

**What actually happens**

1. Forward/Backward on each mini-batch → accumulate gradients.
2. After `K` micro-steps → **optimizer.step()**, **zero\_grad()**.
3. One scheduler step per optimizer step (see §4 Interactions).

---

## 2) Files & Profiles

```
configs/local/accumulate_grad_batches/
├── train.yaml  # default K for training steps (e.g., 2)
├── val.yaml    # K for validation (no-op; recorded for symmetry/logging)
└── test.yaml   # K for inference  (no-op; recorded for symmetry/logging)
```

Recommended defaults:

* `train.yaml`: `accumulate_grad_batches: 2`
  Good Kaggle baseline: doubles effective batch without extra VRAM.
* `val.yaml`:   `accumulate_grad_batches: 1`
* `test.yaml`:  `accumulate_grad_batches: 1`

You may add site-specific variants, e.g.:

* `default.yaml` (K=2), `high.yaml` (K=4/8), `debug.yaml` (K=1)

---

## 3) Environment Playbook

**Kaggle (T4/A10/L4)**

* Start with: `B=64`, `K=2` → `B_eff=128` per-GPU (single-GPU)
* If OOM: lower `B` first; if still OOM, increase `K` to 4.
* If training too slow: prefer raising `B` back (if VRAM allows) before raising `K` further.

**HPC (A100)**

* Prefer **larger `B`** + modest `K` (1–2). Bigger `B` is typically more compute-efficient than large `K`.
* Scale `lr` thoughtfully as you scale `B_eff` and/or `W` (linear or square-root scaling depending on scheduler).

**CI / Smoke Tests**

* `K=1` (no accumulation)
* Tiny `B` (`8–16`), short epochs, strict determinism for fast-fail.

---

## 4) Interactions & Gotchas

* **Schedulers**
  Most schedulers assume **one step per optimizer step**. When you increase `K`, you reduce the number of optimizer steps per epoch.

  * Cosine/OneCycle: you are effectively compressing step count; verify `total_steps`/`epochs` semantics.
  * Warmup by steps: warmup will take **longer in wall-clock** if `K` is larger (fewer steps per epoch).

* **Gradient Clipping**
  Clipping (e.g., `grad_clip_norm`) happens **after** accumulation, i.e., on the *averaged* gradients. Keep norms comparable across different `K`.

* **Mixed Precision (AMP/BF16)**
  Safe by default. Accumulation occurs on FP32 master grads via scaler. Confirm your AMP scaler updates per optimizer step (not per micro-step).

* **DDP / Data Parallel**
  With DDP, `B_eff = B × K × W`. Gradient **all-reduce** still occurs each micro-step unless you use advanced bucket/fuse tricks; default behavior is fine.

* **Checkpointing / Early Stop**
  Monitor on the same **iteration cadence** (e.g., “per optimizer step”). If you log “per batch”, expect sparser log points as `K` grows.

* **Throughput vs. Quality**
  Very large `K` (≳8) may degrade optimizer dynamics compared to equivalent `B` increase. Prefer increasing `B` if VRAM allows; treat `K` as a pressure valve.

---

## 5) Hydra Usage

**Select profile at train time**

```bash
# Baseline (K=2)
spectramind train local/accumulate_grad_batches=train

# Force no accumulation (debug/CI)
spectramind train local/accumulate_grad_batches=train@accumulate_grad_batches=1

# Heavier accumulation for tight VRAM
spectramind train local/accumulate_grad_batches=train@accumulate_grad_batches=4
```

**Per-stage wiring (already in train.yaml)**

```yaml
training:
  gradient_accumulation_steps: ${accumulate_grad_batches}
eval:
  gradient_accumulation_steps: ${val_loader.accumulate_grad_batches}   # informational
test:
  gradient_accumulation_steps: ${test_loader.accumulate_grad_batches}  # informational
```

---

## 6) Tuning Recipes

* **OOM without losing stability**

  * Drop `B` by 2×, raise `K` by 2× → preserve `B_eff` and optimizer behavior.
  * Verify wall-clock; `K` increases latency per update.

* **Underfitting at same wall-clock**

  * Try `B` up (if VRAM OK) **before** `K` up.
  * If `K` up: consider `lr = lr_base × (K_new / K_old)`; confirm with a short LR range test.

* **Noisy loss curves**

  * Moderate `K` (2–4) can smooth updates. Re-tune `lr`/`wd` slightly.

* **Scheduler mismatch**

  * For step-based warmup: keep warmup **in epochs** or recompute steps with the new `K`.

---

## 7) Validation & Logging

* Log **effective batch**: `B`, `K`, `W`, and `B_eff` per run.
* Persist **final composed config** (Hydra) to the run dir.
* Record **lr schedule** vs steps *after* accumulation to ensure reproducibility.

---

## 8) FAQ

**Q: Does accumulation change the math of gradient averaging?**
A: In standard implementations, grads across `K` micro-batches are averaged (sum/`K`) before `optimizer.step()`. This closely approximates a larger batch.

**Q: Should I always scale LR linearly with `K`?**
A: Not always. It’s a helpful starting heuristic; confirm with short learning-rate sweeps.

**Q: Why set K=1 for val/test if they don’t backprop?**
A: For **clarity and logging symmetry**. It also prevents accidental non-default behavior in wrappers that might inspect this field.

---

## 9) Summary

* Use **`K=2`** as a **Kaggle-safe** default.
* Grow `B` before `K` when VRAM allows.
* Keep an eye on **scheduler semantics** and **effective LR** when changing `K`.
* Document & log all changes for **MCP-grade reproducibility**.

```
```
