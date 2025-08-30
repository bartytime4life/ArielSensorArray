# `/configs/optimizer` — Optimizers for SpectraMind V50

This folder contains **Hydra-composable** YAML configs for training optimizers used by the SpectraMind V50 pipeline.
Each config is drop-in selectable from `train.yaml` and fully overrideable from the CLI.

> TL;DR
>
> * **Recommended baseline:** `adamw.yaml`
> * Alternatives for ablations/diagnostics: `adam.yaml`, `sgd.yaml`
> * All optimizers support schedulers, AMP/mixed-precision, and optional Lookahead wrapping.

---

## Available optimizers

| File         | Optimizer           | Default LR | Weight Decay | Notes                                                         |
| ------------ | ------------------- | ---------: | -----------: | ------------------------------------------------------------- |
| `adamw.yaml` | `torch.optim.AdamW` |     `3e-4` |       `1e-2` | **Recommended**. Decoupled weight decay; best generalization. |
| `adam.yaml`  | `torch.optim.Adam`  |     `3e-4` |       `1e-2` | Classic Adam. Use if you specifically want coupled L2.        |
| `sgd.yaml`   | `torch.optim.SGD`   |     `1e-2` |        `0.0` | Momentum + Nesterov. Good for ablations/stability checks.     |

All three configs include:

* `scheduler_hook`: enable schedulers configured in `configs/train.yaml` (e.g., cosine/onecycle/none).
* `precision_safe`: compatible with fp32/fp16/bf16 (AMP).
* Optional **Lookahead** wrapper: `lookahead.enabled=true` (with `alpha`, `k`).

---

## How they are composed

`configs/train.yaml` (excerpt):

```yaml
defaults:
  - model: v50
  - model/decoder: decoder
  - optimizer: adamw        # ← switch this to adam or sgd to change optimizer
  - loss: gll
  - loss: smoothness
  - loss: symbolic
  - trainer: kaggle_safe
  - logger: tensorboard
  - _self_
```

Switch optimizers via CLI without editing code:

```bash
# Use AdamW (baseline)
spectramind train optimizer=adamw

# Try Adam
spectramind train optimizer=adam

# Try SGD with momentum
spectramind train optimizer=sgd
```

---

## Common CLI overrides

Hydra lets you override any leaf in the config. Examples:

```bash
# Tune AdamW
spectramind train optimizer=adamw \
  optimizer.lr=5e-4 optimizer.weight_decay=0.02 optimizer.betas=[0.9,0.98]

# Try Lookahead on top of AdamW
spectramind train optimizer=adamw optimizer.lookahead.enabled=true \
  optimizer.lookahead.alpha=0.5 optimizer.lookahead.k=6

# SGD ablation with cosine schedule (scheduler set in train.yaml)
spectramind train optimizer=sgd optimizer.lr=1e-2 optimizer.momentum=0.9 optimizer.nesterov=true
```

> Tip: You can combine these with your scheduler options from `configs/train.yaml` (e.g., `scheduler.name=cosine`, `scheduler.warmup_steps=500`, `scheduler.min_lr=1e-6`).

---

## Multirun sweeps (Hydra)

Use Hydra’s `-m` multirun for quick sweeps:

```bash
# Sweep LR and weight decay for AdamW
spectramind -m train optimizer=adamw \
  optimizer.lr=1e-4,3e-4,1e-3 \
  optimizer.weight_decay=0.0,0.01,0.02

# Compare AdamW vs Adam @ fixed hyperparams
spectramind -m train optimizer=adamw,adam optimizer.lr=3e-4
```

Hydra will create separate runs per combo (with isolated output dirs). Logs and configs are persisted for full reproducibility.

---

## Scheduler & precision guidance

* **Schedulers:** Prefer **Cosine** with warmup for AdamW/Adam; **OneCycle** can work well with SGD.
  Configure in `configs/train.yaml` under `scheduler:` and keep `optimizer.scheduler_hook: true`.
* **Precision:** Use **bf16** where supported; otherwise AMP fp16 is fine. All optimizers here are AMP-safe.

---

## Tuning recipes (practical defaults)

* **AdamW (recommended)**

  * Start: `lr=3e-4`, `weight_decay=1e-2`, `betas=[0.9, 0.999]`, Cosine schedule + warmup.
  * If overfitting: raise `weight_decay` to `2e-2`.
  * If underfitting early: try `lr=5e-4` (or longer warmup).

* **Adam**

  * Start: `lr=3e-4`, `weight_decay=1e-2`.
  * Consider switching to AdamW unless you specifically want Adam’s coupled L2 behavior.

* **SGD (with Nesterov)**

  * Start: `lr=1e-2`, `momentum=0.9`, `nesterov=true`, Cosine/OneCycle.
  * Expect slower convergence; good for stability/ablations.

* **Lookahead** (optional wrapper for any of the above)

  * Enable for stability in exploratory runs: `lookahead.enabled=true`, `alpha=0.5`, `k=6`.

---

## Diagnostics & reproducibility

Each optimizer config provides:

* `log_config: true` — optimizer hyperparameters are recorded to `logs/v50_debug_log.md`.
* `export_json` — a JSON dump under `outputs/diagnostics/optimizer_*.json`.

This ensures your runs are transparent and reproducible alongside the full Hydra config snapshot.

---

## File structure

```
/configs/optimizer/
├─ adam.yaml     # torch.optim.Adam
├─ adamw.yaml    # torch.optim.AdamW  (baseline)
└─ sgd.yaml      # torch.optim.SGD
```

> Future extension: we can add a dedicated `lookahead.yaml` wrapper to compose on top of any base optimizer (e.g., `optimizer=lookahead base=adamw`). For now, each optimizer supports `lookahead.*` inline.

---

## FAQ

**Q: How do I switch optimizers without editing any code?**
A: Use `optimizer=<name>` on the CLI (e.g., `optimizer=adamw`) — Hydra composes the right YAML.

**Q: Where do I set the scheduler?**
A: In `configs/train.yaml` under `scheduler:`. Keep `optimizer.scheduler_hook: true` in the optimizer config.

**Q: Does this work with mixed precision?**
A: Yes. All optimizers here set `precision_safe: true`. Choose fp16/bf16 in your trainer config.

**Q: Where are hyperparams logged?**
A: `logs/v50_debug_log.md` and `outputs/diagnostics/optimizer_*.json`, together with the run’s full Hydra config snapshot.

---

## Contributing new optimizers

1. Add a new YAML under `/configs/optimizer/<name>.yaml`.
2. Include keys: `name`, `type`, primary hyperparams, `scheduler_hook`, `precision_safe`, `log_config`, and `export_json`.
3. Update this README with a short row in the table and tuning recipe.
4. Verify with a smoke run:

   ```bash
   spectramind train optimizer=<name> training.epochs=1
   ```

Happy training 🚀
