# `/configs/optimizer/architecture.md`

> Design notes and integration contract for SpectraMind V50 optimizer configs.
> Scope: how optimizer YAMLs are composed, overridden, logged, and extended (e.g., Lookahead, schedulers, AMP).

---

## 0) Goals & Design Principles

* **Hydra-first composition** — every optimizer is a small, self-contained YAML module selectable from `train.yaml` or via CLI.
* **Zero code edits** — switch optimizers, change LR, toggle Lookahead, or alter weight decay entirely from the CLI/configs.
* **Kaggle/CI safety** — sensible defaults, AMP/bf16-safe, scheduler-aware for ≤9h jobs.
* **Reproducible by construction** — each optimizer config is persisted (Markdown + JSON), with the exact values used per run.
* **Extensible** — add new optimizers in minutes; shared schema + hooks (schedulers, meta-optimizers, logging).

---

## 1) Directory Map

```
/configs/optimizer/
├─ adam.yaml      # torch.optim.Adam   (baseline adaptive)
├─ adamw.yaml     # torch.optim.AdamW  (recommended; decoupled weight decay)
└─ sgd.yaml       # torch.optim.SGD    (momentum + Nesterov; ablations/stability)
```

Each file exposes a uniform schema under the `optimizer:` key and optional subtrees for wrappers (e.g., `lookahead:`), diagnostics, and hooks.

---

## 2) Config Schema (contract)

All optimizer configs adhere to this minimal contract:

```yaml
optimizer:
  # identity
  name: <string>             # "adamw"
  type: <string>             # "AdamW" (torch class name)

  # primary hyperparameters (vary by optimizer)
  lr: <float>
  # e.g., betas, eps, weight_decay, momentum, nesterov, dampening, amsgrad, ...

  # optional param-group helpers (when supported by trainer bridge)
  auto_param_groups:
    enabled: <bool>          # if true, biases/norms get weight_decay=0.0
    no_decay_patterns: [ "bias", "LayerNorm.weight", "layer_norm.weight", "ln", "norm", "bn", "BatchNorm.weight" ]
    override_no_decay_weight: 0.0

  # wrappers / meta-optimizers
  lookahead:
    enabled: <bool>
    alpha: <float>
    k: <int>

  # hooks / integration
  scheduler_hook: <bool>     # allow train.yaml to attach scheduler
  warmup_steps: <int>        # (hint consumed by scheduler)
  clip_grad_norm: <float|0>  # 0/None disables clipping
  accumulate_steps: <int>    # gradient accumulation for larger effective batch

  # runtime/precision hints
  precision_safe: <bool>     # AMP/bf16-safe flag
  bf16_preferred: <bool>     # prefer bf16 where supported
  fused: <bool>              # enable fused kernels if available
  detect_anomaly: <bool>     # enable autograd anomaly detection (debug)

  # diagnostics
  log_config: <bool>
  rich_console: <bool>       # pretty CLI table
  hash_to_debuglog: <bool>   # append config hash → logs/v50_debug_log.md
  export_json: <path>        # machine-readable dump
```

### Required keys

* `name`, `type`, and the optimizer’s **minimum viable hyperparameters** (e.g., `lr`, `betas` for Adam/AdamW; `momentum` for SGD).

### Optional keys

* `auto_param_groups.*` (zero-decay biases/normalization parameters).
* `lookahead.*` (wrap any base optimizer).
* `scheduler_hook`, `warmup_steps`, `clip_grad_norm`, `accumulate_steps`.
* `precision_safe`, `bf16_preferred`, `fused`, `detect_anomaly`.
* `log_config`, `rich_console`, `hash_to_debuglog`, `export_json`.

> **Note:** YAMLs never contain Python import paths; the training layer instantiates `torch.optim.<type>` with these kwargs.

---

## 3) Composition Patterns

### A) Select the optimizer (defaults in `configs/train.yaml`)

```yaml
defaults:
  - optimizer: adamw   # swap to adam or sgd here or via CLI
```

### B) Switch & override from CLI (no code edits)

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
  `lr=3e-4`, `weight_decay=1e-2`, `betas=[0.9, 0.999]`, `eps=1e-8`, `amsgrad=false`
  Decoupled weight decay → better generalization. Strong default with Cosine + warmup.

* **`adam.yaml` — Classic Adam**
  Same default `lr=3e-4`; L2 weight decay is **coupled** (use when you need Adam semantics).

* **`sgd.yaml` — Momentum + Nesterov**
  `lr=1e-2`, `momentum=0.9`, `nesterov=true`
  Slower to converge but helpful for ablations & stability. Try OneCycle or Cosine.

All three include: `scheduler_hook: true`, `precision_safe: true`, optional `lookahead`, and diagnostics exports.

---

## 5) Logging & Reproducibility

Each optimizer config activates two sinks:

* **Markdown audit** — append a structured summary to `logs/v50_debug_log.md` (timestamp, optimizer, LR, weight decay, momentum/betas, etc.; plus a config hash).
* **JSON export** — write machine-readable hyperparameters under `outputs/diagnostics/optimizer_*.json`.

These sit alongside the run’s **Hydra config snapshot** and trainer logs so every experiment is replayable.

---

## 6) AMP & Precision

* All optimizers here set `precision_safe: true`.
* Prefer **bf16** on A100/H100 (`bf16_preferred: true`) where applicable; otherwise use AMP fp16.
* Precision is selected in trainer/device configs; optimizer YAMLs only document compatibility and recommended preferences.

---

## 7) Schedulers (separation of concerns)

* Optimizers **don’t** hard-code schedules.
* Keep `optimizer.scheduler_hook: true` in YAML; define the schedule in `configs/train.yaml`.
* This lets you A/B test optimizer vs schedule orthogonally.

**Heuristics**

* AdamW/Adam → **Cosine** with warmup (`warmup_steps ≈ 1–3%` of total steps).
* SGD → **OneCycle** or Cosine; start with slightly higher LR; ensure warmup/clipping as needed.

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
* Optional: `auto_param_groups`, `lookahead`, `scheduler_hook`, `warmup_steps`, `clip_grad_norm`, `accumulate_steps`, `precision_safe`, `bf16_preferred`, `fused`, `log_config`, `export_json`.

2. **Document** defaults & rationale in `/configs/optimizer/README.md` (table + brief tuning recipe).

3. **Smoke test**

```bash
spectramind train optimizer=<name> training.epochs=1
```

4. **Ablate** vs AdamW (convergence curves, GLL score, calibration behavior).

> **Tip:** If an optimizer needs special kwargs (e.g., Adafactor’s `relative_step`), surface them as `optimizer.*` leaves; the trainer passes them through to `torch.optim`.

---

## 10) Troubleshooting

* **No scheduler effect** → ensure `optimizer.scheduler_hook=true` and a non-`none` scheduler in `configs/train.yaml`.
* **AMP overflow/instability** → reduce LR, add warmup, try bf16, enable grad-clip (`clip_grad_norm`) or disable fused kernels (`fused: false`).
* **Regularization surprises** → verify Adam (coupled L2) vs AdamW (decoupled).
* **Lookahead sluggish** → reduce `alpha` (e.g., `0.3`) or `k` (e.g., `5`); for final runs, consider disabling if it slows convergence.

---

## 11) Roadmap

* **`lookahead.yaml` wrapper** that composes over any base optimizer via `base=<opt>` group.
* Additional optimizers: `rmsprop.yaml`, `lion.yaml`, `adafactor.yaml`.
* **Policy packs**: presets for “fast-train”, “best-val”, “calibration-friendly” (LR/decay/schedule bundles).
* **Auto-tuner**: multirun recipes under `/configs/experiment/optimizer_sweeps/*.yaml`.

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
  warmup_steps: 0
  clip_grad_norm: 1.0
  accumulate_steps: 1
  precision_safe: true
  bf16_preferred: true
  fused: false
  detect_anomaly: false
  log_config: true
  rich_console: true
  hash_to_debuglog: true
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
  warmup_steps: 0
  clip_grad_norm: 1.0
  accumulate_steps: 1
  precision_safe: true
  bf16_preferred: true
  fused: false
  detect_anomaly: false
  log_config: true
  rich_console: true
  hash_to_debuglog: true
  export_json: "outputs/diagnostics/optimizer_sgd.json"
```

---

**Summary.** This contract makes optimizers **modular**, **traceable**, and **composable**.
Swap, tune, and sweep with one-liners — no code edits; all runs stay Kaggle-safe and reproducible.
