# 🧩 Architecture: `configs/loss/`

This document is the **high-level codemap** for the loss-configuration system in **SpectraMind V50** (NeurIPS 2025 Ariel Data Challenge).  
Use it to **navigate, extend, and integrate** the physics-informed loss components with confidence. It’s designed to be **stable** (rarely changing), while individual YAMLs and code can evolve beneath it.

---

## 1) Purpose & Scope

- **Goal**: Centralize all loss hyper-parameters and control logic in **modular, Hydra-friendly YAML** files so experiments are configurable from the CLI without code edits.
- **Scope**: Covers standalone loss modules:
  - `gll.yaml` — Gaussian Log-Likelihood (primary leaderboard loss).
  - `smoothness.yaml` — curvature/continuity penalties across wavelengths.
  - `nonnegativity.yaml` — soft physical constraint μ ≥ 0 (hinge/softplus/exponential barrier).
  - `fft.yaml` — Fourier-domain suppression of high-frequency artifacts.
  - `symbolic.yaml` — molecular/astrophysical physics priors & rule constraints.
  - `composite.yaml` — orchestration layer that aggregates/toggles/weights the above.
- **Update cadence**: Revisit this **architecture doc** when the **structure** changes (new layers, naming, or orchestration semantics). For minor parameter tweaks, update the YAMLs and their README.

---

## 2) Directory File Map

- **`gll.yaml`**  
  Config for the **Gaussian Log-Likelihood** objective. Exposes options like reduction (`mean|sum`), weighting mode, and σ handling flags.

- **`smoothness.yaml`**  
  Config for **spectral continuity** (first/second-order differences, curvature power, robust penalties). Tunable order, neighborhood, and weight.

- **`nonnegativity.yaml`**  
  Config for **μ ≥ 0** with barrier types (`hinge|softplus|exp`), ramp schedules, and max penalty.

- **`fft.yaml`**  
  Config for **frequency-domain** regularization: cutoff frequency, roll-off, windowing, and normalization.

- **`symbolic.yaml`**  
  Config for **symbolic physics constraints**: molecule fingerprints, band masks, rule sets, soft vs. hard enforcement, violation caps, trace/diagnostics.

- **`composite.yaml`**  
  **Master orchestrator**:  
  - Global enable/disable per term  
  - Per-term weight and local overrides (forwarded into each base config)  
  - Safety limits (e.g., max global penalty)  
  - Evaluation order and combination scheme (Σ wᵢ · Lᵢ)

- **`ARCHITECTURE.md`** *(this file)*

---

## 3) Configuration Flow & Boundaries

```mermaid
flowchart TD
    A[train.yaml] --> B[defaults: { loss: composite }]
    B --> C[composite.yaml]
    C --> D[gll.yaml]
    C --> E[smoothness.yaml]
    C --> F[nonnegativity.yaml]
    C --> G[fft.yaml]
    C --> H[symbolic.yaml]
    B --> I[CLI Overrides (Hydra)]
    I --> C
    C --> J[Loss Builder (runtime)]
    J --> K[Enabled Modules]
    K --> L[Weighted Sum  Σ wᵢ Lᵢ  (with safety caps)]
````

**Flow narrative**

1. `train.yaml` points to `loss: composite`.
2. `composite.yaml` **imports** the base group configs and **exposes** simplified switches/weights.
3. CLI **overrides** (Hydra) can target both `loss.composite.*` and deep `loss.groups.*` fields.
4. The **Loss Builder** (Python side) **reads** the merged config, **instantiates** enabled sub-losses, and **combines** them with runtime safety guards (e.g., max penalty, NaN guards).

**Boundaries**

* **Leaf layer**: semantics of each loss (math + parameters) live in its own YAML and code module.
* **Orchestration layer**: `composite.yaml` controls **what** is on and **how much** it counts.
* **Execution layer**: model/optimizer step reads **only** the merged runtime config; no constants hard-coded.

---

## 4) Architectural Invariants & Principles

* **Modularity**
  Each loss has its **own config** and **unit tests**. You can evolve one term without side-effects on others.

* **Isolation of defaults**
  Base YAMLs are **pristine defaults**. Experiments change behavior via:

  * `composite.yaml` switches/weights, or
  * **CLI overrides** (`spectramind train loss.composite.fft.enabled=false …`).

* **Layered abstraction**
  *Leaf → Orchestration → Execution* ensures clarity of roles and simplifies debugging & docs.

* **Human navigability**
  Names mirror scientific intent (`smoothness`, `fft`, `symbolic`) so domain users can reason about side-effects.

* **Reproducibility by construction**
  Every override lives in the Hydra config log; outputs are written to unique run dirs; seeds are wired in the trainer.

---

## 5) Cross-Cutting Considerations

### 5.1 Reproducibility

* Treat every CLI change as a **versioned spec** (Hydra output + run hash in logs).
* Keep Kaggle/notebook runs **internet-off** and **deterministic** (seed flags + CUDA determinism).
* DVC/Git snapshot data and checkpoints so the loss mix maps to exact artifacts.

### 5.2 Safety & Numerical Health

* Loss builder applies **caps** to any single penalty (e.g., `max_component_penalty`) to prevent instabilities.
* NaN/Inf guards: if any component returns NaN, reduce its contribution to zero and surface a warning.

### 5.3 Testing

* `/tests/loss/` includes:

  * **Schema checks** (all expected fields with defaults)
  * **Override checks** (CLI → merged config = expected)
  * **Smoke compute** (random μ/σ gives finite loss, gradients exist)
  * **Boundary cases** (empty masks, zero weights, extreme cutoffs)

### 5.4 Extensibility (adding a new loss)

1. Create `configs/loss/yourloss.yaml` with explicit fields and sane defaults.
2. Add an entry under `composite.yaml`:

   * `enabled`, `weight`, and `overrides` mapping into your leaf config.
3. Implement the compute in the loss builder (module + factory registration).
4. Add unit tests + docs short note in this file or the `README.md`.
5. Optionally add a diagnostics hook (e.g., save per-bin penalty or rule violations).

---

## 6) Quick Start (Common Patterns)

### 6.1 Single-loss run (sanity)

```yaml
# train.yaml (excerpt)
defaults:
  - loss: smoothness
```

CLI:

```bash
spectramind train loss.smoothness.weight=0.10
```

### 6.2 Composite physics stack

```yaml
# train.yaml (excerpt)
defaults:
  - loss: composite
```

CLI:

```bash
spectramind train \
  loss=composite \
  loss.composite.gll.enabled=true \
  loss.composite.smoothness.enabled=true loss.composite.smoothness.weight=0.10 \
  loss.composite.fft.enabled=true        loss.composite.fft.weight=0.05 loss.composite.fft.cutoff_freq=40 \
  loss.composite.nonnegativity.enabled=true loss.composite.nonnegativity.weight=0.02 loss.composite.nonnegativity.barrier=softplus
```

### 6.3 Symbolic probes (soft first)

```bash
spectramind train \
  loss=composite \
  loss.composite.symbolic.enabled=true \
  loss.composite.symbolic.weight=0.03 \
  loss.composite.symbolic.program=defaults/molecules+lensing \
  loss.composite.symbolic.overrides.hard=false
```

---

## 7) Design Notes per Term (TL;DR)

* **GLL**
  Aligns with competition metric; ensure σ handling is consistent with your calibration stage.
  *Knobs*: reduction (`mean|sum`), entropy/coverage weighting, σ floor/clip.

* **Smoothness**
  Penalizes large adjacent differences (1st/2nd-order).
  *Knobs*: order (1|2), neighborhood width, robust penalty (Huber/Charbonnier), normalization by spectrum scale.

* **Nonnegativity**
  Keeps μ ≥ 0 with soft barriers.
  *Knobs*: barrier type (`hinge|softplus|exp`), weight schedules, tolerance for near-zero regions.

* **FFT**
  Suppresses power beyond a cutoff; useful when spectra get “spiky.”
  *Knobs*: cutoff frequency, roll-off (linear|cosine), window (Hann|Hamming|none), per-bin scaling.

* **Symbolic**
  Enforces interpretable, physics-motivated rules (molecule masks, line families, astrophysical relations).
  *Knobs*: rule program, soft vs. hard, per-rule weights, violation caps, tracing/export for dashboards.

---

## 8) Example Kaggle CLI Playbook (Loss-Focused)

All examples assume the unified CLI `spectramind` and Hydra defaults include `configs/loss/`. They’re **internet-off**, **deterministic**, and **leaderboard-safe** patterns.

### A) Baseline (GLL-only) — fast sanity

```bash
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export CUDA_LAUNCH_BLOCKING=1

spectramind train \
  loss=gll \
  trainer.seed=1234 \
  data.num_workers=2 \
  +runtime.kaggle_mode=true \
  hydra.run.dir=outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}_gll_only
```

### B) Composite physics stack — smoothness + FFT + nonnegativity

```bash
spectramind train \
  loss=composite \
  loss.composite.gll.enabled=true \
  loss.composite.smoothness.enabled=true \
  loss.composite.smoothness.weight=0.10 \
  loss.composite.smoothness.overrides.order=2 \
  loss.composite.fft.enabled=true \
  loss.composite.fft.weight=0.05 \
  loss.composite.fft.cutoff_freq=40 \
  loss.composite.nonnegativity.enabled=true \
  loss.composite.nonnegativity.weight=0.02 \
  loss.composite.nonnegativity.barrier=softplus \
  loss.composite.symbolic.enabled=false \
  trainer.seed=1234 \
  +runtime.kaggle_mode=true
```

### C) Add symbolic constraints (soft mode first)

```bash
spectramind train \
  loss=composite \
  loss.composite.symbolic.enabled=true \
  loss.composite.symbolic.weight=0.05 \
  loss.composite.symbolic.program=defaults/molecules+lensing \
  loss.composite.symbolic.overrides.hard=false \
  trainer.seed=2025 \
  +runtime.kaggle_mode=true
```

### D) Align with σ calibration (e.g., temperature scaling)

```bash
spectramind train \
  loss=composite \
  loss.composite.gll.enabled=true \
  +calibration.temperature.enabled=true \
  +calibration.temperature.value=${oc.env:T_STAR,1.0} \
  loss.composite.symbolic.enabled=true \
  loss.composite.symbolic.weight=0.03 \
  trainer.seed=7 \
  +runtime.kaggle_mode=true
```

### E) Mini-ablation grid (weights & toggles)

```bash
spectramind ablate \
  'loss=composite' \
  'loss.composite.smoothness.weight=[0.05,0.10,0.20]' \
  'loss.composite.fft.enabled=[true,false]' \
  trainer.max_time="00:30:00" \
  +runtime.kaggle_mode=true \
  +ablation.report.html=true \
  +ablation.report.md=true
```

### F) Low-budget symbolic probe

```bash
spectramind train \
  loss=composite \
  loss.composite.symbolic.enabled=true \
  loss.composite.symbolic.weight=0.02 \
  trainer.max_epochs=2 \
  trainer.limit_train_batches=0.15 \
  trainer.limit_val_batches=0.15 \
  +runtime.kaggle_mode=true
```

Follow with:

```bash
spectramind diagnose symbolic-rank --export-html
```

### G) Barrier experiments (nonnegativity)

```bash
# Hinge
spectramind train \
  loss=composite \
  loss.composite.nonnegativity.enabled=true \
  loss.composite.nonnegativity.barrier=hinge \
  loss.composite.nonnegativity.weight=0.03 \
  +runtime.kaggle_mode=true

# Softplus
spectramind train \
  loss=composite \
  loss.composite.nonnegativity.enabled=true \
  loss.composite.nonnegativity.barrier=softplus \
  loss.composite.nonnegativity.weight=0.02 \
  +runtime.kaggle_mode=true
```

**Guardrails (always useful)**

* Internet **OFF**; seeds + CUDA determinism **ON**.
* Keep `data.num_workers` conservative.
* Always set `hydra.run.dir` to separate artifacts and ensure traceability.

---

## 9) Troubleshooting & FAQs

**Q: Composite weights look “right” but training becomes unstable.**
A: Lower symbolic/FFT weights first; set `max_component_penalty` in `composite.yaml`; confirm σ calibration alignment; check for NaN guards.

**Q: My spectra get “flatlined.”**
A: Reduce nonnegativity weight or switch to `softplus` barrier; decrease smoothness order/weight; verify FFT cutoff isn’t suppressing real features.

**Q: Symbolic rules increase validation loss.**
A: Start `hard=false` and lower weight. Inspect `diagnose symbolic-rank` to identify which rules over-penalize; re-weight or disable noisy rules.

**Q: Where should I set σ calibration?**
A: Keep it **consistent** across runs (e.g., temperature scaling) and ensure GLL parameters don’t contradict your calibrated σ.

---

## 10) Summary

* `configs/loss/` defines a **structured, modular, and physics-informed** loss stack.
* `composite.yaml` is the **single orchestration point** for everyday experimentation.
* This **ARCHITECTURE.md** is your **codemap**: concise, high-level, and intentionally stable as the system grows.

> *“A codemap is a map of a country, not an atlas.”*
> This file aims to be that codemap—rarely invalidated as the code evolves.

```
```
