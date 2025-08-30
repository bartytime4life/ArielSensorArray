# Loss Configurations (`configs/loss/`)

This directory houses all standalone and composite Hydra YAML configurations used to define and tune the loss function components for **SpectraMind V50**. Each config aligns with our mission-critical design ethos: **reproducible**, **physics-informed**, and **fully configurable via CLI**.

---

## 🗂 Directory Overview

- `gll.yaml` — Gaussian Log-Likelihood (primary metric-based loss)
- `smoothness.yaml` — Penalizes curvature / promotes spectral smoothness
- `nonnegativity.yaml` — Soft constraint enforcing μ ≥ 0
- `fft.yaml` — FFT-domain high-frequency suppression
- `symbolic.yaml` — Encodes symbolic/physics-first regularizers
- `composite.yaml` — Unified master config incorporating all above
- `README.md` — You're here! Guides maintainers and contributors.

---

##  Purpose & Design Philosophy

These configurations enable modular, composable, and transparent loss definitions that:

- Reflect domain knowledge: spectral smoothness, physical constraints, molecular patterns.
- Retain full CLI manipulation: toggle modules, adjust weights, override parameters via Hydra.
- Support reproducible experiments by linking all hyperparameters to version control.

---

##  How to Use

### Base Config Usage

To employ one of the individual loss configs:

```yaml
# In your training YAML
defaults:
  - loss: gll
# CLI example overrides:
# spectramind train loss.gll.weighting=entropy loss.gll.reduction=sum
````

### Composite Config Usage

Integrate multiple loss terms via `composite.yaml`:

```yaml
defaults:
  - loss: composite
```

Then manipulate via CLI:

* Disable FFT regularization:

  ```
  spectramind train loss.composite.fft.enabled=false
  ```
* Increase smoothness influence:

  ```
  spectramind train loss.composite.smoothness.weight=0.2 \
    loss.composite.smoothness.overrides.order=2
  ```

---

## Best Practices

* **Start with GLL** (weight ≈ 1.0); add regularizers incrementally.
* Use composite mode for rapid experimentation and ablations.
* Keep your config usage reproducible by noting CLI overrides in your experiment logs.
* Document new sub-configs here to maintain transparency.

---

## Want to Contribute?

* Add new physics-informed losses? Create a matching config YAML plus unit tests.
* Enhancements to existing configs? Update the YAML and add notes/license as needed.
* Think something’s missing? Let’s evolve it collaboratively—open an issue or PR!

---

## References & Further Reading

* Clear README structure and readability ideas([Medium][1]).
* Value of README for onboarding new developers and guiding contributions([blogs.incyclesoftware.com][2]).

---

> “A README file should be your new team member’s best friend.”
> — Internal documentation best-practice, emphasizing clarity and onboarding ease([blogs.incyclesoftware.com][2])

```

