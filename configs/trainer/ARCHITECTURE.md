# 🧩 Architecture: `configs/trainer/`

This document provides a **bird’s-eye view** of the trainer configuration layer in **SpectraMind V50** (NeurIPS 2025 Ariel Data Challenge).
Use it to **navigate, extend, and integrate** training runtime profiles confidently under Hydra/CLI control with strict, NASA-grade reproducibility.

---

## 1) Purpose & Scope

* **Goal:** Centralize *how* training runs (devices, precision, epochs, logging, checkpoints) in modular, Hydra-composable YAMLs — never hard-coding orchestration in Python.
* **Scope:** All trainer profiles in `configs/trainer/` are selectable via Hydra defaults or CLI flags, and recorded in structured logs for end-to-end reproducibility.
* **When to edit:** Add new profiles or orchestration contracts. Run-specific tweaks should use Hydra overrides (`trainer.max_epochs=10`) instead of editing YAMLs directly.

---

## 2) Directory File Map

* `defaults.yaml` — Canonical local baseline (balanced epochs, logging, checkpoints).
* `kaggle_safe.yaml` — Kaggle GPU profile with ≤ 9h wall-time, AMP, throttled I/O.
* `ci_fast.yaml` — CI smoke run (≤ 20 min, no ckpts).
* `multi_gpu.yaml` — Single-node DDP (auto device discovery, AMP).
* `ddp.yaml` — Multi-node DDP (explicit devices×nodes, sync BN).
* `README.md` — Quickstart for developers.
* `architecture.md` — (you are here).

Profiles are always selected via Hydra composition:

```bash
spectramind train trainer=kaggle_safe
```

All merged configs are **snapshotted** to logs (`logs/`, `v50_debug_log.md`) with hashes for traceability.

---

## 3) Configuration Flow & Boundaries

```mermaid
flowchart TD
  A[train.yaml] -->|defaults: { trainer: <profile> }| B[trainer/<profile>.yaml]
  B --> C[Hydra Compose]
  C --> D[Runtime Engine]
  D --> E[Logging & Checkpointing]
  E --> F[Audit Trail: logs/, v50_debug_log.md, config hash]
```

**Boundaries**

* *Trainer profiles* = orchestration only (devices, precision, epochs, logging, ckpt).
* *Model/Data/Loss* configs are separate (Hydra groups) to ensure **separation of concerns**.
* Every run must be CLI/Hydra-driven — no hidden Python defaults.

---

## 4) Architectural Invariants

* **Hydra-first orchestration:** YAML + CLI overrides only; merged configs persisted.
* **CLI-first execution:** All flows (`spectramind train`, `spectramind calibrate`) run through Typer CLI.
* **Reproducibility:** Config snapshots, Git/DVC hashes.
* **Determinism defaults:** Seeds fixed; CuDNN deterministic on (unless explicitly benchmarked).
* **I/O discipline:** Checkpoint/log cadence tuned per profile (critical for Kaggle 9h jobs).

---

## 5) Trainer Profiles (Rationale & Fit)

### `defaults.yaml` — Local Dev Baseline

* Balanced epochs, gradient clipping, optional AMP.
* Frequent validation/checkpoints for iteration.

### `kaggle_safe.yaml` — Kaggle Runtime Safety

* ≤ 9h wall clock, AMP16, conservative workers, throttled ckpts/logs.
* Internet-off safe, GPU mem ≤ 16 GB.

### `ci_fast.yaml` — CI Smoke Run

* `max_epochs=1`, `limit_*_batches`, checkpoints disabled.
* Runs <20 min; detects schema/NaN errors in CI.

### `multi_gpu.yaml` — Single-Node DDP

* Auto device count (`trainer.devices=N`).
* `DDPStrategy(find_unused_parameters=false)`, AMP, sync-safe.
* Scale-up on workstation servers.

### `ddp.yaml` — Multi-Node Distributed

* Explicit `num_nodes × devices`, `sync_batchnorm=true`.
* For HPC clusters with shared storage.

---

## 6) Cross-Cutting Concerns

* **Reproducibility:** Hydra config + CLI overrides + Git commit + DVC hash recorded.
* **Precision & Stability:** FP16 (AMP) default, BF16 optional, gradient clipping on.
* **Checkpointing:** Rank-0 save only; throttled in Kaggle-safe. Resume supported.
* **Logging:** Rich console UI + JSONL logs. MLflow/W\&B optional.
* **Kaggle guardrails:** ≤ 9h wall-time, ≤ 16 GB GPU, no internet.

---

## 7) Extension Points

1. **New profile:** Add `trainer/<name>.yaml` + doc here + entry in `README.md`.
2. **Hardware-specific:** e.g. `h100.yaml` (BF16, NCCL knobs).
3. **Research modes:** `ablation.yaml` with extra ckpts.
4. **Eval-only:** minimal runtime scoring profile.
5. **Hybrid orchestration:** Combine `lookahead` optimizer with `multi_gpu` trainer .

---

## 8) Quick Start

```bash
# Local baseline
spectramind train trainer=defaults

# Kaggle-safe leaderboard run
spectramind train trainer=kaggle_safe trainer.max_epochs=30

# CI smoke run
spectramind train trainer=ci_fast

# Single-node multi-GPU
spectramind train trainer=multi_gpu trainer.devices=4

# Multi-node DDP (cluster)
spectramind train trainer=ddp trainer.devices=8 trainer.num_nodes=2
```

All runs are **captured with config snapshots + hashes** for reproducibility.

---

## 9) References

* Hydra modular configuration
* CLI-first automation & reproducibility
* Kaggle GPU constraints & leaderboard rules
* NASA-grade modeling & verification standards
* SpectraMind V50 design & reproducibility plan

> *“Profiles are mission modes. Choose wisely, stay within the envelope, and every run is flight-ready.”* 🚀

---
