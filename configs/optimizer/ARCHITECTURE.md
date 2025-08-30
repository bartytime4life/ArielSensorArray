# `/configs/optimizer/architecture.md`

> Design notes and integration contract for SpectraMind V50 optimizer configs.
> Scope: how optimizer YAMLs are composed, overridden, logged, and extended (e.g., Lookahead, schedulers, AMP).

---

## 0) Goals & Design Principles

* **Hydra-first composition**: every optimizer is a small, self-contained YAML module selectable from `train.yaml` or via CLI.
* **Zero code edits**: switch optimizers, change LR, toggle Lookahead, or alter weight decay entirely from the CLI/configs.
* **Kaggle/CI safety**: sensible defaults, AMP-safe, and scheduler-aware for ≤9h jobs.
* **Reproducible by construction**: every optimizer config is persisted (Markdown + JSON), with the exact values used per run.
* **Extensible**: add new optimizers in minutes; shared schema + hooks (schedulers, meta-optimizers, logging).

---

## 1) Directory Map

```
/configs/optimizer/
├─ adam.yaml      # torch.optim.Adam  (baseline adaptive)
├─ adamw.yaml     # torch.optim.AdamW (recommended baseline; decoupled weight decay)
└─ sgd.yaml       # torch.optim.SGD   (momentum + Nesterov; ablations/stability)
```

Each file exposes a uniform schema under the `optimizer:` key and optional subtrees for wrappers (e.g., `lookahead:`), diagnostics, and hooks.

---

## 2) Config Schema (contract)

All optimizer configs adhere to this minimal contract:

```yaml
optimizer:
  name: <string>             # friendly identifier (e.g., "adamw")
  type: <string>             # torch optimizer class name (e.g., "AdamW")

  # primary hyperparameters (vary by optimizer)
  lr: <float>
  # e.g., betas, eps, weight_decay, momentum, nesterov, dampening, amsgrad, ...

  # wrappers / meta-optimizers (optional)
  lookahead:
    enabled: <bool>
    alpha: <float>
    k: <int>

  # hooks / integration
  scheduler_hook: <bool>     # allow train.yaml to attach scheduler
  precision_safe: <bool>     # AMP-safe flag (fp16/bf16)
  fused: <bool>              # enable fused kernels if available (opt-in)

  # diagnostics
  log_config: <bool>
  export_json: <path>
```

### Required keys

* `name`, `type`, and the optimizer’s **minimum viable hyperparameters** (e.g., `lr`, `betas` for Adam/AdamW; `momentum` for SGD).

### Optional keys

* `lookahead.*` (wrapping any base optimizer)
* `scheduler_hook` (delegates scheduling to `configs/train.yaml`)
* `precision_safe`, `fused` (deployment/runtime hints)
* `log_config`, `export_json` (reproducibility logging)

> **Note:** No Python code references live in YAML; resolution is handled in the training layer by instantiating `torch.optim.<type>` with the provided kwargs.

---

## 3) Composition Patterns

### A) Select the optimizer (defaults in `configs/train.yaml`)

```yaml
defaults:
  - optimizer: adamw   # swap to adam or sgd here or via CLI
```

### B) CLI override (no code edits)

```bash
# Switch optimizer
spectramind train optimizer=adamw

# Tweak hyperparameters
spectramind train optimizer=adamw optimizer.lr=5e-4 optimizer.weight_decay=0.02
```

### C) Enable Lookahead wrapper

```bash
spectramind train optimizer=adamw \
  optimizer.lookahead.enabled=true optimizer.lookahead.alpha=0.5 optimizer.lookahead.k=6
```

### D) Scheduler delegation

Schedulers are defined in `configs/train.yaml` and only attach if `optimizer.scheduler_hook: true`.

```yaml
# configs/train.yaml (snippet)
scheduler:
  name: cosine          # "cosine" | "onecycle" | "none"
  warmup_steps: 500
  min_lr: 1.0e-6
```

---

## 4) Provided Optimizers (defaults & rationale)

* **`adamw.yaml` — Recommended baseline**

  * `lr=3e-4`, `weight_decay=1e-2`, `betas=[0.9, 0.999]`, `eps=1e-8`, `amsgrad=false`
  * Decoupled weight decay → better generalization. Works well with Cosine LR + warmup.

* **`adam.yaml` — Classic Adam**

  * Same default `lr=3e-4`; L2 weight decay is *coupled* (use when you need Adam semantics).

* **`sgd.yaml` — Momentum + Nesterov**

  * `lr=1e-2`, `momentum=0.9`, `nesterov=true`
  * Slower to converge but great for ablations and stability. Try Cosine or OneCycle.

All three include: `scheduler_hook: true`, `precision_safe: true`, optional `lookahead` wrapper, and diagnostics.

---

## 5) Logging & Reproducibility

Each optimizer config activates two complementary sinks:

* **Markdown**: append a structured summary to `logs/v50_debug_log.md` (run timestamp, optimizer name, LR, decay, etc.)
* **JSON**: write machine-readable hyperparameters under `outputs/diagnostics/optimizer_*.json`

These sit alongside Hydra’s **full config snapshot** for the run, making every experiment replayable (same code + same YAML → same result, modulo randomness).

---

## 6) AMP & Precision

* Set `precision_safe: true` for optimizers that are compatible with fp16/bf16 (Adam/AdamW/SGD are).
* **bf16** is preferred where supported (e.g., A100/H100); otherwise use AMP fp16.
* The optimizer YAML does not select precision; that belongs to trainer/device config, but it documents compatibility.

---

## 7) Schedulers (separation of concerns)

Optimizers do not hard-code schedules. Instead:

* Keep `optimizer.scheduler_hook: true` in YAML.
* Define the schedule policy in `configs/train.yaml` (`scheduler.name` etc.).
* This separation lets you A/B test both optimizer and schedule orthogonally.

**Heuristics**

* AdamW/Adam → Cosine with warmup (`warmup_steps≈1–3% total steps`)
* SGD → OneCycle or Cosine; start with a slightly higher LR, ensure proper warmup.

---

## 8) Examples

### A) Quick sweep (Hydra multirun)

```bash
spectramind -m train optimizer=adamw \
  optimizer.lr=1e-4,3e-4,1e-3 \
  optimizer.weight_decay=0.0,0.01,0.02
```

### B) Compare optimizers at fixed schedule

```bash
spectramind -m train optimizer=adamw,adam,sgd \
  scheduler.name=cosine scheduler.warmup_steps=500 scheduler.min_lr=1e-6
```

### C) Stability run with Lookahead

```bash
spectramind train optimizer=sgd optimizer.lookahead.enabled=true \
  optimizer.lookahead.alpha=0.5 optimizer.lookahead.k=6 \
  optimizer.lr=1e-2 optimizer.momentum=0.9
```

---

## 9) Extension Guide (add a new optimizer)

1. **Create** `/configs/optimizer/<name>.yaml` with:

   * `optimizer.name`, `optimizer.type` (torch class), and required kwargs.
   * Optional `lookahead`, `scheduler_hook`, `precision_safe`, `fused`, `log_config`, `export_json`.
2. **Document** defaults & rationale in this file + update `README.md` table.
3. **Smoke test**:

   ```bash
   spectramind train optimizer=<name> training.epochs=1
   ```
4. **Ablate** against AdamW to validate expected behavior (convergence curves, final metric, calibration effects).

> **Tip:** If your optimizer needs extra kwargs (e.g., Adafactor’s `relative_step`), place them under `optimizer.*` and they’ll be passed through.

---

## 10) Troubleshooting

* **No scheduler effect**: ensure `optimizer.scheduler_hook=true` and that `configs/train.yaml` defines a non-`none` scheduler.
* **AMP overflow or instabilities**: reduce LR, enable warmup, try bf16, or disable fused kernels (`fused: false`).
* **Unexpected regularization**: verify whether you’re using Adam (coupled L2) vs AdamW (decoupled). Switch YAML accordingly.
* **Lookahead too sluggish**: reduce `alpha` (e.g., `0.3`) or `k` (e.g., `5`); if still slow, disable for final leaderboard runs.

---

## 11) Roadmap

* **`lookahead.yaml` wrapper** (optional) that composes over any base optimizer via `base=<opt>` group.
* **Additional optimizers**: `rmsprop.yaml`, `lion.yaml`, `adafactor.yaml`.
* **Policy packs**: presets for “fast-train”, “best-val”, and “calibration-friendly” (weight decay/LR bundles).
* **Auto-tuner**: Hydra multirun recipes under `/configs/experiment/optimizer_sweeps/*.yaml`.

---

## 12) Appendix — Minimal YAML Templates

**AdamW template**

```yaml
optimizer:
  name: "adamw"
  type: "AdamW"
  lr: 3.0e-4
  betas: [0.9, 0.999]
  eps: 1.0e-8
  weight_decay: 0.01
  amsgrad: false
  lookahead: {enabled: false, alpha: 0.5, k: 6}
  scheduler_hook: true
  precision_safe: true
  fused: false
  log_config: true
  export_json: "outputs/diagnostics/optimizer_adamw.json"
```

**SGD template**

```yaml
optimizer:
  name: "sgd"
  type: "SGD"
  lr: 1.0e-2
  momentum: 0.9
  dampening: 0.0
  weight_decay: 0.0
  nesterov: true
  lookahead: {enabled: false, alpha: 0.5, k: 6}
  scheduler_hook: true
  precision_safe: true
  fused: false
  log_config: true
  export_json: "outputs/diagnostics/optimizer_sgd.json"
```

---

**Verdict**: This architecture makes optimizers **modular**, **traceable**, and **composable**.
Swap, tune, and sweep optimizers with one-liners — no code edits, all runs reproducible.
