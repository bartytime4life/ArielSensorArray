# Architecture: `configs/loss/`

This document provides a bird’s-eye view of the architecture and operational pathways for loss configuration in **SpectraMind V50**. Use it to navigate, extend, and integrate physics-informed loss components confidently.

---

## 1. Purpose & Scope

- **Goal**: Organize all loss-related hyper-parameters/control logic in modular, Hydra-friendly YAML files.
- **Scope**: Covers standalone loss modules (`gll.yaml`, `smoothness.yaml`, `nonnegativity.yaml`, `fft.yaml`, `symbolic.yaml`) and their unified orchestrator (`composite.yaml`).
- Should be revisited only occasionally as the core structure evolves, not for minor tweaks :contentReference[oaicite:2]{index=2}.

---

## 2. Directory File Map

- **`gll.yaml`**  
  Primary Gaussian log-likelihood loss—captures calibration and uncertainty.

- **`smoothness.yaml`**  
  Encourages spectral continuity via curvature penalties.

- **`nonnegativity.yaml`**  
  Enforces μ ≥ 0 softly via hinge, softplus, or exponential barriers.

- **`fft.yaml`**  
  Penalizes high-frequency spectral artifacts via Fourier-domain loss.

- **`symbolic.yaml`**  
  Encapsulates advanced physics-informed symbolic constraints (e.g., molecular fingerprints, lensing, smoothness, nonnegativity, FFT).

- **`composite.yaml`**  
  Master config aggregating all sub-losses, allowing joint enabling, weighting, and per-component overrides.

- **`ARCHITECTURE.md`**  
  This architecture overview.

---

## 3. Configuration Flow & Boundaries

```text
train.yaml
  └── defaults: { loss: composite }
         ↳ loads composite.yaml
               ↳ imports base configs (gll, smoothness, nonnegativity, fft)
               ↳ exposes composite.{gll, smoothness, nonnegativity, fft} for toggles/weights
               ↳ integrates symbolic.yaml if physics-aware mode is desired
  ↳ CLI overrides modify both composite.* and groups.* fields directly
  ↳ Loss builder:
      • Reads composite config
      • Applies any overrides to loaded group configs
      • Instantiates loss modules for each enabled subterm
      • Combines (sum of weight_i * loss_i)
````

---

## 4. Architectural Invariants & Principles

* **Modularity**: Each loss term evolves independently, yet composes cleanly.
* **Isolation**: Base configs remain unchanged; overrides flow through Hydra’s CLI or `composite.yaml`.
* **Layered abstraction**:

  * *Leaf*: individual loss behavior (definitions)
  * *Orchestration layer*: composite config managing how components interact
  * *Execution layer*: loss builder reading configs and executing logic
* **Human navigability**: Names and structure reflect domain concepts (e.g., smoothness, FFT, etc.) for quick lookup ([makeareadme.com][2], [matklad.github.io][1], [stackoverflow.com][3]).

---

## 5. Cross-Cutting Considerations

* **Reproducibility**: All flags/configs tied directly to YAML/CLI ensure full traceability of experiments.
* **Testing**: Each module has associated unit tests to validate behavior under config changes.
* **Extensibility**: To add a new loss:

  1. Create `yourname.yaml`
  2. Optionally include it in `composite.yaml`
  3. Draft unit tests
  4. Document in this `ARCHITECTURE.md` if structural, or README if incremental

---

## 6. Get Started Guide

1. **Need a single loss?**
   Add to `train.yaml`:

   ```yaml
   defaults:
     - loss: smoothness
   ```
2. **Mix and match?**
   Use:

   ```yaml
   defaults:
     - loss: composite
   ```

   Then override via CLI:

   ```
   spectramind train \
     loss.composite.smoothness.weight=0.2 \
     loss.composite.fft.enabled=false
   ```
3. **Understanding config interplay?**
   Trace:

   * `composite.yaml` →
   * imported groups.\* →
   * your overrides →
   * loss builder → instantiated modules

---

## 7. Summary

* **`configs/loss/`** defines structured, modular configurations for SpectraMind’s loss stack.
* **`composite.yaml`** enables centralized control and experimentation.
* **This architecture documentation** serves as a stable, developer-centric guide to navigate and evolve the loss configuration architecture.

---

> *“A codemap is a map of a country, not an atlas.”*
> This file aims to be that codemap—concise, high-level, and rarely invalidated as code evolves ([stackoverflow.com][3], [matklad.github.io][1]).

```
