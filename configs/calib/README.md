# 🔬 Calibration Configs — SpectraMind V50

**Location:** `configs/calib/`  
**Project:** SpectraMind V50 · NeurIPS 2025 Ariel Data Challenge

Calibration is the **first stage** of the pipeline, transforming raw Ariel-simulated frames into **physically corrected, science-ready light curves**. This folder contains **profile-level chain configs** (e.g., `nominal.yaml`, `fast.yaml`, `strict.yaml`) that compose the **method-level** tools in `configs/calib/method/` (ADC, dark, flat, CDS, photometry, trace, phase, etc.).

---

## 🧭 What calibration does

**Kill chain (ordered):**

1. **ADC correction** — offset/gain(PRNU), reference pixels/overscan, linearization  
2. **Nonlinearity** — (if not included in ADC) LUT/poly linearization  
3. **Dark subtraction** — master dark, exposure/temperature scaling, hot-pixel repair  
4. **Flat fielding** — PRNU/illumination normalization (master or spectral flats)  
5. **CDS** — RESET/SIGNAL differencing, robust pairing/outlier suppression  
6. **Photometric extraction** — `aperture` / `psf` / `optimal` extraction  
7. **Trace normalization** — continuum flattening + temporal detrend  
8. **Phase alignment** — ephemeris fold, in/out-of-transit windows, binning

> Each stage is configured by **one YAML** in `configs/calib/method/` and chained here.

---

## 📦 Files in this folder

| File | Purpose | Typical use |
|------|---------|-------------|
| **`nominal.yaml`** | Default **leaderboard-safe** calibration chain (≤ 9h). All key steps enabled; saves light diagnostics. | Daily development and Kaggle submissions. |
| **`fast.yaml`** | **Lightweight** chain for CI/smoke. Skips heavy steps or uses cheaper approximations. | PR checks, pipeline smoke tests, quick repro. |
| **`strict.yaml`** | **Full science-grade** chain with extra diagnostics and stricter QC. | Offline validation, audits; may exceed Kaggle limits. |

> Chains are Hydra configs that **compose** the method files under `configs/calib/method/`.

---

## 🗂️ Layout

```

configs/
└─ calib/
├─ nominal.yaml
├─ fast.yaml
├─ strict.yaml
└─ method/
├─ adc.yaml
├─ dark.yaml
├─ flat.yaml
├─ cds.yaml
├─ photometry.yaml
├─ trace.yaml
├─ phase.yaml
└─ corel.yaml        # optional post-inference calibration (μ/σ graph)

````

A chain file typically includes:

```yaml
# (Example) configs/calib/nominal.yaml
defaults:
  - method/adc
  - method/dark
  - method/flat
  - method/cds
  - method/photometry
  - method/trace
  - method/phase
# optional post-inference
# - method/corel
````

---

## 🚀 How to run

### Run default (nominal) calibration

```bash
spectramind calibrate --config-name calib/nominal
```

### Switch to fast (CI smoke)

```bash
spectramind calibrate --config-name calib/fast
```

### Run strict (offline science validation)

```bash
spectramind calibrate --config-name calib/strict
```

### Hydra overrides (examples)

**Aperture photometry with radius=8 px**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.photometry.photometry.method=aperture \
  calib.method.photometry.aperture.radius_px=8
```

**Spline trace normalization (12 knots)**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.trace.along_wavelength.method=spline \
  calib.method.trace.along_wavelength.spline.knots=12
```

**CDS pairing by nearest timestamps within 3s**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.cds.pairing.order_by=timestamp \
  calib.method.cds.pairing.policy=nearest \
  calib.method.cds.pairing.window_s=3.0
```

> Use `--cfg job --resolve` to print the composed config for auditing.

---

## 🔩 I/O contract (common defaults)

| Stage      | input\_key       | output\_key      | Shape(s)                          |
| ---------- | ---------------- | ---------------- | --------------------------------- |
| ADC        | `raw_frames`     | `adc_corrected`  | `[B,H,W]` or `[B,C,H,W]`          |
| Dark       | `adc_corrected`  | `dark_corrected` | `[B,H,W]` or `[B,C,H,W]`          |
| Flat       | `dark_corrected` | `flat_corrected` | `[B,H,W]` or `[B,C,H,W]`          |
| CDS        | `flat_corrected` | `cds_corrected`  | `[B,H,W]` or `[B,C,H,W]`          |
| Photometry | `cds_corrected`  | `photometry`     | Flux `[B]` or `[B,C]` (+ meta)    |
| Trace      | `photometry`     | `trace_norm`     | Flux `[B]` or `[B,C]`             |
| Phase      | `trace_norm`     | `phase_fold`     | Phase `[B]`; binned `[M]`/`[M,C]` |

Method files define these keys precisely in their `io` blocks.

---

## 📊 DVC integration

Calibration is a DVC-tracked stage for **reproducibility**:

* **Inputs:** raw FGS1/AIRS frames + calibration config(s) + method artifacts (masters/LUTs/maps)
* **Outputs:** calibrated cubes/series under `${RUN_DIR}` (and optionally `data/processed/<profile>/calibrated`)
* **Behavior:** if **inputs + config hashes** are unchanged, DVC **skips re-run** using the cache.

> Ensure large artifacts (master darks, flats, masks, LUTs) are **DVC dependencies** with stable paths.

---

## ⚙️ Performance tips

* Use `calib/fast` in CI & quick iteration.
* Disable heavy diagnostics: `*.diagnostics.save_images=false`, reduce `save_samples`.
* Lower QC sampling: `quality_checks.sampling_fraction=0.05`.
* Prefer median-based statistics for robustness; avoid deep spline fits on smoke runs.

---

## 🧪 QC & validation

* Enable `validation.require_existing_paths=true` in development.
* Use `validation.check_shape_compatibility=true` to catch map/LUT mismatches early.
* Assert finite outputs: `assert_finite_output=true`.
* Watch histograms for zero/saturation pile-ups after ADC/flat.

---

## 🧰 Quick recipes

**CI-friendly run**

```bash
spectramind calibrate --config-name calib/fast \
  calib.method.photometry.photometry.method=aperture \
  calib.method.photometry.aperture.radius_px=6 \
  calib.method.trace.along_time.enabled=false \
  calib.method.phase.binning.nbins=120
```

**Instrument switch (FGS1)**

```bash
spectramind calibrate --config-name calib/nominal \
  calib.method.photometry.instrument.name=FGS1 \
  calib.method.trace.instrument.name=FGS1 \
  calib.method.photometry.instrument.per_channel=false
```

**Post-inference COREL calibration (optional)**

```bash
spectramind calibrate --config-name calib/nominal \
  calib.method.corel.method=corel \
  calib.method.corel.model.arch=gat \
  calib.method.corel.train.epochs=50
```

---

## 🧷 Notes

* All configs are **Hydra-composable** and **override-friendly**.
* **Nominal** is Kaggle-safe; **fast** is for CI; **strict** is for audits & high-rigor studies.
* Keep **units** consistent (DN↔e⁻ conversions via `conversion_gain_e_per_adu`).
* Prefer `float32` during calibration; quantize to integers only when required downstream.

---

✅ With this setup, calibration configs are the **launchpad** for every SpectraMind V50 run — delivering **reproducibility**, **physics-informed rigor**, and **Kaggle-safe** deployment.

```
```
