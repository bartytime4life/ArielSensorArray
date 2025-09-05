````markdown
# `/configs/local/prefetch_factor/ARCHITECTURE.md`

---

## 0) Purpose & Scope

This directory defines the **prefetching strategies** for PyTorch DataLoaders in **SpectraMind V50**.  
Prefetching controls how many batches each worker process loads ahead, impacting:

- **GPU utilization** (reduce idle time due to I/O stalls)  
- **Host RAM usage** (higher prefetch ⇒ higher memory)  
- **I/O balance** (SSD/NVMe vs. HDD vs. Kaggle ephemeral storage)

The profiles here are **Hydra modules** you can compose per-environment for **portability, throughput, and reproducibility**.

---

## 1) Design Philosophy

- **Hydra-first modularity:** one file per profile (`debug`, `default`, `fast`, `heavy`) selected via `local/prefetch_factor=*`.
- **Mission-grade clarity:** each file documents intent/risks (memory, Kaggle constraints).
- **Reproducibility:** DataLoader knobs (`prefetch_factor`, `persistent_workers`, `pin_memory`) are explicit, not implicit defaults.

---

## 2) Profiles Overview

| Profile     | `prefetch_factor` | `persistent_workers` | `pin_memory` | Primary Use Case |
|-------------|-------------------:|:--------------------:|:------------:|------------------|
| `debug`     | 1 | false | true  | CI smoke, step-through debugging, low-RAM |
| `default`   | 2 | true  | true  | Safe baseline for Kaggle & local dev |
| `fast`      | 4 | true  | true  | SSD/NVMe workstations, mid-scale HPC |
| `heavy`     | 8 | true  | true  | Multi-GPU HPC, NVMe RAID, ≥128 GB RAM |

> **Rule of thumb:** Increase prefetch only if (1) storage is fast enough and (2) system RAM is abundant.

---

## 3) Mermaid: Prefetch in the Data Pipeline

```mermaid
flowchart LR
  subgraph Host[Host System]
    C[CPU Workers<br/>num_workers=N]:::cpu
    PF[[prefetch_factor k]]:::cfg
    MEM[(Host RAM)]:::mem
    IO[(SSD/NVMe / Ephemeral)]:::io
  end

  subgraph DL[PyTorch DataLoader]
    Q{{Worker Queues<br/>k batches each}}:::q
    PW[(persistent_workers)]:::cfg
    PM[(pin_memory)]:::cfg
  end

  subgraph GPU[GPU]
    HB[Host→GPU Copy<br/>(pinned mem if PM=true)]:::gpu
    TRT[Training Step]:::gpu
  end

  IO -->|read batches| C --> Q
  PF --> Q
  PW --> DL
  PM --> HB
  Q --> MEM
  MEM --> HB --> TRT

  classDef cpu fill:#1f2937,stroke:#0ea5e9,color:#e5e7eb;
  classDef gpu fill:#0b3b5e,stroke:#22d3ee,color:#e5e7eb;
  classDef cfg fill:#0f172a,stroke:#22c55e,color:#a7f3d0,stroke-width:2px;
  classDef q fill:#111827,stroke:#f59e0b,color:#fde68a,stroke-width:2px;
  classDef io fill:#111827,stroke:#f97316,color:#fff7ed;
  classDef mem fill:#111827,stroke:#a78bfa,color:#ede9fe;

  click PF "https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader" "_blank"
````

**Interpretation:**

* **Higher `prefetch_factor (k)`** ⇒ each worker queues more batches ⇒ **fewer GPU stalls** *if* I/O is fast *and* RAM is sufficient.
* **`persistent_workers=true`** ⇒ reuse workers across epochs ⇒ lower startup overhead.
* **`pin_memory=true`** ⇒ faster H2D copies (HB), improving step cadence.

**Trade-offs:**

* `k ↑` ⇒ **RAM usage ↑** (≈ `num_workers × k × batch_bytes`)
* On Kaggle/small nodes, prefer `debug`/`default` to avoid OOM and stay ≤ 9h.

---

## 4) Usage (Hydra)

`configs/train.yaml` composes the group:

```yaml
defaults:
  - local/prefetch_factor: default
```

Override on the CLI:

```bash
# Low-memory / CI
spectramind train local/prefetch_factor=debug data.num_workers=0

# Balanced (Kaggle/local default)
spectramind train local/prefetch_factor=default

# Fast SSD/NVMe
spectramind train local/prefetch_factor=fast

# HPC / Multi-GPU nodes (ensure RAM is abundant)
spectramind train local/prefetch_factor=heavy
```

---

## 5) Integration Snippet

```python
train_loader = DataLoader(
    dataset=train_ds,
    batch_size=cfg.data.batch_size,
    num_workers=cfg.data.num_workers,
    prefetch_factor=cfg.prefetch_factor,          # from local/prefetch_factor/*
    persistent_workers=cfg.persistent_workers,    # from profile
    pin_memory=cfg.data.pin_memory,               # typically true for GPU
    shuffle=True,
)
```

> Ensure `cfg.data.pin_memory=true` for CUDA to accelerate H2D copies; keep `persistent_workers=true` for non-debug runs.

---

## 6) Environment Guidance

* **Kaggle / Small GPU (T4/A10/L4):** `default` (2) or `debug` (1).
* **Local SSD 32–64 GB RAM:** `fast` (4).
* **HPC ≥128 GB RAM / NVMe RAID / Multi-GPU:** `heavy` (8).
* **When unsure:** start with `default`, monitor RAM & GPU idle, then scale up.

---

## 7) Monitoring & Guardrails

* Track **GPU idle time** (Profiler / step time variance) and **RAM headroom**.
* If OOM or cache thrashing occurs: **reduce** `prefetch_factor` or `num_workers`.
* Log the active profile in `v50_debug_log.md` (config hash) for reproducibility.

---

## 8) Future Extensions

* **Adaptive prefetching:** auto-tune `k` based on observed I/O wait.
* **Per-stage profiles:** e.g., lighter prefetch for calibration/validation, heavier for training.
* **Cluster-aware presets:** auto-resolve by node SKU (Kaggle vs. A100 vs. L4).

---

```
```
