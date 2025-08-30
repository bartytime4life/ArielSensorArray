# 🧩 Architecture: `configs/trainer/`

This document provides a **bird’s-eye view** of the trainer configuration layer in **SpectraMind V50**.  
Use it to **navigate, extend, and integrate** training runtime profiles confidently under Hydra/CLI control with strict, NASA-grade reproducibility:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 1) Purpose & Scope

- **Goal:** Centralize *how* training runs (devices, precision, epochs, logging, checkpoints) in modular, Hydra-composable YAMLs — never hard-coding orchestration in Python:contentReference[oaicite:2]{index=2}.  
- **Scope:** All trainer profiles in `configs/trainer/` selected via Hydra defaults/CLI and recorded in auditable logs for end-to-end reproducibility:contentReference[oaicite:3]{index=3}:contentReference[oaicite:4]{index=4}.  
- **When to edit:** Only for structural changes (new profile or orchestration contract). Parameter tweaks belong in run-specific overrides.

---

## 2) Directory File Map

- **`defaults.yaml`** — Canonical baseline for local development (sane epochs, logging, checkpoints).  
- **`kaggle_safe.yaml`** — Kaggle GPU profile with ≤ 9-hour wall safety, AMP, throttled I/O:contentReference[oaicite:5]{index=5}.  
- **`ci_fast.yaml`** — CI smoke run (tiny batch limits, ≤ ~20 min; no checkpoints).  
- **`multi_gpu.yaml`** — Single-node DDP for throughput (auto devices, AMP).  
- **`ddp.yaml`** — Multi-node DDP (devices×nodes explicit, sync BN).  
- **`README.md`** — Developer-level usage & quick picks (see repo).  
- **`architecture.md`** — You’re here — stable codemap for the trainer layer.

All profiles are selected through Hydra composition (e.g., `spectramind train trainer=kaggle_safe`) and **snapshotted** to logs with the merged config for traceability:contentReference[oaicite:6]{index=6}:contentReference[oaicite:7]{index=7}.

---

## 3) Configuration Flow & Boundaries

```mermaid
flowchart TD
  A[train.yaml] -->|defaults: { trainer: <profile> }| B[trainer/<profile>.yaml]
  B --> C[Hydra Compose]
  C --> D[Runtime Engine]
  D --> E[Logging & Checkpointing]
  E --> F[Audit Trail (logs/, v50_debug_log.md, config hash)]
````

**Pathway**

1. `train.yaml` selects a trainer profile via Hydra defaults.
2. Hydra composes `trainer/<profile>.yaml` + CLI overrides.
3. Runtime consumes the merged config (no Python edits).
4. Structured logs and checkpoints are written; the exact config & run hash are recorded for reproducibility.

**Boundaries**

* *Trainer profiles* define orchestration only (devices, precision, epochs, logging, ckpt).
* *Model/Data/Loss* are configured in their own groups and must not leak concerns into trainer YAMLs (separation of concerns).

---

## 4) Architectural Invariants

* **Hydra-first, code-free orchestration:** Every run is driven by composed YAML + CLI overrides; the merged config is persisted.
* **CLI-first execution:** All flows are invoked through `spectramind` Typer CLI; no ad-hoc scripts.
* **Reproducibility by construction:** Git/DVC and config/hash logging allow exact reruns; CI enforces integrity with smoke runs.
* **Determinism defaults:** Deterministic flags on; CuDNN benchmark off unless shape-stable and explicitly enabled.
* **I/O discipline:** Checkpoint/log cadence tuned per profile to respect time/storage budgets (esp. Kaggle).

---

## 5) Trainer Profiles (Rationale & Fit)

### A) `defaults.yaml` — **Local Dev Baseline**

* Balanced epochs, AMP optional, frequent enough validation & safe gradient clipping.
* Use for everyday iteration; switch to `multi_gpu.yaml` only when scaling.

### B) `kaggle_safe.yaml` — **Leaderboard/Notebook Safety**

* Wall-time guardrails (≤ 9 h), AMP(16), conservative dataloader workers, throttled checkpoints/logging to avoid I/O stalls and space overuse.
* Prefer for Kaggle GPU notebooks/submissions; override only small knobs (e.g., `max_epochs`) to stay within the budget.

### C) `ci_fast.yaml` — **CI Smoke / Rapid Sanity**

* `max_epochs=1`, tight `limit_*_batches`, disable checkpoints; CSV/TB minimal logging.
* Catches schema/shape/NaN issues in minutes; used by CI gates.

### D) `multi_gpu.yaml` — **Single-Node DDP Throughput**

* Auto devices (override with `trainer.devices=N`), `DDPStrategy(find_unused_parameters=false)`, AMP, sync-safe defaults.
* For workstation/one node servers; tune `accumulate_grad_batches` with VRAM.

### E) `ddp.yaml` — **Multi-Node Scale-Out**

* Explicit `num_nodes` × `devices`, `sync_batchnorm=true` for BN-heavy nets; static graph where applicable to reduce overhead.
* Use on clusters with homogeneous nodes; ensure shared storage for checkpoints.

---

## 6) Cross-Cutting Concerns

* **Reproducible runs:** Every invocation logs the merged Hydra config, CLI, git commit, and data/model hashes to the audit trail (`logs/`, `v50_debug_log.md`).
* **AMP & Stability:** FP16 by default on GPUs for throughput; pair with gradient clipping to avoid overflow; escalate to BF16 if supported and numerically stable.
* **Checkpoint Strategy:** Rank-zero saving; throttle frequency in Kaggle-safe to reduce I/O overhead; enable resume in long runs.
* **Logging:** Rich console + TB/CSV minimal footprints; integrate external trackers behind toggles to keep Kaggle/CI clean.
* **Kaggle constraints:** Respect internet-off, GPU quotas, and storage/time limits; prefer conservative workers and deterministic seeds.

---

## 7) Extension Points

1. **New profile:** Add `trainer/<name>.yaml`, document here, and include in `README.md`.
2. **Specialized hardware:** Create `h100.yaml`/`a100.yaml` with BF16 defaults and NCCL knobs.
3. **Research modes:** Add `ablation.yaml` with shorter eval cadence + forced checkpoints to snapshot branches.
4. **Eval-only profile:** Minimal runtime to score artifacts without training.

---

## 8) Quick Start (Hydra + CLI)

```bash
# Local baseline
spectramind train trainer=defaults

# Kaggle-safe leaderboard run (guardrails on)
spectramind train trainer=kaggle_safe trainer.max_epochs=30

# CI smoke (fast)
spectramind train trainer=ci_fast

# Single-node multi-GPU (override devices)
spectramind train trainer=multi_gpu trainer.devices=4

# Multi-node DDP
spectramind train trainer=ddp trainer.devices=8 trainer.num_nodes=2
```

All merged configs are captured for reproducibility and audit.

---

## 9) References

* Hydra modular configuration & overrides
* CLI-first automation & reproducibility in SpectraMind V50
* Audit logging / hash-based traceability
* Kaggle runtime constraints & best practices

> *“Profiles are mission modes. Choose the right one, stay within the envelope, and every run is flight-ready.”* 🚀

```
```
