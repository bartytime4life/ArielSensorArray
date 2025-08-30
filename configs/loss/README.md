# Loss Configurations (`configs/loss/`)

This directory houses all **Hydra YAML configurations** for loss functions used in the **SpectraMind V50** pipeline (NeurIPS 2025 Ariel Data Challenge).  
Each file defines **standalone physics-aware loss terms** or a **composite controller**, ensuring they are:

* 🛰 **Reproducible** — every parameter tracked via Hydra + DVC:contentReference[oaicite:0]{index=0}  
* 🔬 **Physics-informed** — enforcing smoothness, non-negativity, symbolic constraints:contentReference[oaicite:1]{index=1}:contentReference[oaicite:2]{index=2}  
* 🖥 **CLI-driven** — fully overridable from `spectramind train …`:contentReference[oaicite:3]{index=3}  

---

## 🗂 Directory Overview

- `gll.yaml` — Gaussian Log-Likelihood (GLL), the **primary metric loss** of the challenge:contentReference[oaicite:4]{index=4}  
- `smoothness.yaml` — Spectral curvature penalty; promotes **continuity and differentiability**:contentReference[oaicite:5]{index=5}  
- `nonnegativity.yaml` — Soft μ ≥ 0 constraint to enforce physical realism:contentReference[oaicite:6]{index=6}  
- `fft.yaml` — FFT-domain regularizer; suppresses high-frequency artifacts:contentReference[oaicite:7]{index=7}  
- `symbolic.yaml` — Encodes **symbolic physics rules** (molecular patterns, lensing, alignment):contentReference[oaicite:8]{index=8}  
- `composite.yaml` — Unified Hydra config to toggle/weight all above losses together  
- `README.md` — You’re here! Maintainers’ and contributors’ guide.

---

## 🎯 Purpose & Design Philosophy

Losses here combine **machine learning flexibility** with **astrophysical constraints**:

* **Domain knowledge baked in** — spectral smoothness, symbolic priors, non-negative μ, FFT suppression.  
* **Modular + composable** — swap terms on/off or adjust weights without code edits:contentReference[oaicite:9]{index=9}.  
* **CLI discoverability** — every flag accessible via `--help` and tab-completion:contentReference[oaicite:10]{index=10}:contentReference[oaicite:11]{index=11}.  
* **Reproducibility** — configs tracked via Hydra + Git + DVC; all overrides logged:contentReference[oaicite:12]{index=12}.  

This structure mirrors **NASA-grade scientific modeling** where each regularizer has a **traceable rationale**:contentReference[oaicite:13]{index=13}.

---

## ⚙️ How to Use

### Base Loss Configs

Use a standalone loss in `train.yaml`:

```yaml
defaults:
  - loss: gll
````

CLI override example:

```bash
spectramind train loss.gll.reduction=sum loss.gll.weighting=entropy
```

### Composite Config

Enable multiple physics-informed losses at once:

```yaml
defaults:
  - loss: composite
```

Example CLI overrides:

* Disable FFT term:

  ```bash
  spectramind train loss.composite.fft.enabled=false
  ```

* Increase smoothness weight & change penalty order:

  ```bash
  spectramind train \
    loss.composite.smoothness.weight=0.2 \
    loss.composite.smoothness.overrides.order=2
  ```

---

## ✅ Best Practices

* **Start simple**: Train with GLL (`weight=1.0`) as baseline.
* **Add regularizers gradually**: Smoothness first, then FFT, symbolic last.
* **Use composite mode** for ablations: `spectramind ablate loss.composite.* …`.
* **Always log overrides**: Kaggle shake-ups happen if losses aren’t reproducible.
* **Calibrate σ with COREL**: Loss configs must align with uncertainty calibration.

---

## 🔧 Contribution Guidelines

Want to add a new loss?

1. Create a YAML file here with its parameters (e.g., `gravitational.yaml`).
2. Update `composite.yaml` to expose toggles/weights.
3. Add documentation here (purpose, usage).
4. Write unit tests under `/tests/loss/` to validate structure.

Loss configs must follow **SpectraMind reproducibility standards**:

* Deterministic seeds
* Hydra override compatibility
* Physics-aware justification

---

## 📚 References

* **NeurIPS 2025 Ariel Challenge metric** — GLL as primary leaderboard score
* **Physics constraints** — smoothness, non-negativity, FFT priors
* **Symbolic astrophysics** — molecular fingerprints, gravitational lensing
* **Hydra/YAML composition** — modular, CLI-overridable config design
* **CLI UX** — discoverability, progress feedback, error clarity

---

> “Every loss term is not just a number — it encodes physics.”
> — SpectraMind V50 Documentation Team

```
