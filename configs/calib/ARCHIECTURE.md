# 🧭 `configs/calib/architecture.md`

**SpectraMind V50** · **NeurIPS 2025 Ariel Data Challenge**  
**Document type:** Architecture & schema reference for *profile-level* calibration chains

---

## 0) Purpose

This document defines the **architecture, composition rules, and contracts** for the *profile-level* calibration
configs in `configs/calib/` (e.g., `nominal.yaml`, `fast.yaml`, `strict.yaml`).  
These profiles **compose** method YAMLs from `configs/calib/method/` (ADC, dark, flat, CDS, photometry, trace, phase, corel)
into a reproducible **kill chain** that turns raw frames into science-ready light curves.

---

## 1) Scope & placement

```

configs/
└─ calib/
├─ nominal.yaml           # default leaderboard-safe chain (≤ 9h)
├─ fast.yaml              # CI/smoke-friendly chain
├─ strict.yaml            # full diagnostics/audit chain
├─ architecture.md        # (this file)
└─ method/                # atomic method configs (one per stage)
├─ adc.yaml
├─ dark.yaml
├─ flat.yaml
├─ cds.yaml
├─ photometry.yaml
├─ trace.yaml
├─ phase.yaml
└─ corel.yaml          # optional post-inference calibration (μ/σ graph)

````

- **Profile configs** (this folder) orchestrate a *sequence* of **method configs** (`calib/method/*`).
- A profile must be **Hydra-composable**, **override-friendly**, and **self-validating**.

---

## 2) Kill-chain overview

```mermaid
flowchart LR
  A[ADC] --> B[Dark]
  B --> C[Flat]
  C --> D[CDS]
  D --> E[Photometry]
  E --> F[Trace normalization]
  F --> G[Phase alignment]
  G -. optional .-> H[COREL (graph μ/σ)]
````

* **ADC**: offset/gain(PRNU), ref pixels/overscan, linearization, clamps
* **Dark**: master selection/build, exposure/temperature scaling, hot-pixel repair
* **Flat**: master/spectral flats, PRNU renormalization, optional illumination field
* **CDS**: RESET/SIGNAL pairing, robust differencing, optional temporal filters
* **Photometry**: `aperture | psf | optimal` extraction; background & centroiding
* **Trace**: along-wavelength continuum + along-time detrend; masks & QC
* **Phase**: ephemeris fold, windows, binning; flags & coverage checks
* **COREL** (*optional post-inference*): GNN over bins to calibrate μ/σ (scale or delta head)

---

## 3) Profile schema

Every `configs/calib/*.yaml` profile should follow this structure:

```yaml
# configs/calib/<profile>.yaml
defaults:
  - method/adc
  - method/dark
  - method/flat
  - method/cds
  - method/photometry
  - method/trace
  - method/phase
  # - method/corel      # optional post-inference step

# Optional: central toggles/overrides for the chain
calib:
  method:
    adc.enabled: true
    dark.enabled: true
    flat.enabled: true
    cds.enabled: true
    photometry.enabled: true
    trace.enabled: true
    phase.enabled: true
    # corel.enabled: false
```

> **Rule:** method files are **atomic** (one stage per file), and profiles compose them **in order**.

---

## 4) I/O contracts (stage interfaces)

| Stage      | input\_key       | output\_key      | Typical shapes                      |
| ---------- | ---------------- | ---------------- | ----------------------------------- |
| ADC        | `raw_frames`     | `adc_corrected`  | `[B,H,W]` or `[B,C,H,W]`            |
| Dark       | `adc_corrected`  | `dark_corrected` | `[B,H,W]` or `[B,C,H,W]`            |
| Flat       | `dark_corrected` | `flat_corrected` | `[B,H,W]` or `[B,C,H,W]`            |
| CDS        | `flat_corrected` | `cds_corrected`  | `[B,H,W]` or `[B,C,H,W]`            |
| Photometry | `cds_corrected`  | `photometry`     | Flux `[B]` or `[B,C]` (+ ancillary) |
| Trace      | `photometry`     | `trace_norm`     | Flux `[B]` or `[B,C]`               |
| Phase      | `trace_norm`     | `phase_fold`     | Phase `[B]`; binned `[M]` / `[M,C]` |
| COREL\*    | `inference`      | `corel`          | μ/σ `[B,C]`; metrics dict           |

**Conventions (examples)**

* Photometry fields: `photometry_flux`, `photometry_flux_var`, `photometry_sky`, `photometry_flags`, centroid XY
* Trace fields: `trace_flux_norm`, `trace_flux_scale`, `trace_flags`, `channel_grid`
* Phase fields: `phase_series`, `binned_phase`, `binned_flux`, `binned_flux_err`, `windows`, `phase_flags`

> These keys are bound in the **method** YAMLs (`<method>.io.*`) and must remain consistent across the chain.

---

## 5) Profiles matrix

| Aspect      | **fast**                         | **nominal**                            | **strict**                          |
| ----------- | -------------------------------- | -------------------------------------- | ----------------------------------- |
| Purpose     | CI/smoke & dev                   | Kaggle leaderboard                     | Full science audit                  |
| Diagnostics | minimal                          | light                                  | extensive                           |
| ADC         | basic offset/gain, no heavy refs | ref-pixel/overscan as needed           | full per-amp, LUT/linearization     |
| Dark        | single master, no plane fit      | master + scaling, hot-pixel repair     | select/build master, residual plane |
| Flat        | optional or master only          | master/spectral + illum corr as needed | spectral flats + illum correction   |
| CDS         | off or simple diff               | full pairing (timestamp/index)         | full pairing + temporal filters     |
| Photometry  | aperture small radius            | method chosen (ap/psf/opt)             | optimal/psf + full variance model   |
| Trace       | light norm (poly low-order)      | poly/spline as data require            | spline/lowess + robust masks        |
| Phase       | coarse bins                      | balanced bins                          | fine bins + strict QC               |
| Runtime     | shortest                         | ≤ 9h safe                              | longest                             |

---

## 6) Composition patterns

### 6.1 Nominal (example skeleton)

```yaml
# configs/calib/nominal.yaml
defaults:
  - method/adc
  - method/dark
  - method/flat
  - method/cds
  - method/photometry
  - method/trace
  - method/phase

calib:
  method:
    photometry.photometry.method: aperture
    photometry.aperture.radius_px: 8
    trace.along_wavelength.method: poly
    phase.binning.nbins: 200
```

### 6.2 Fast (example skeleton)

```yaml
# configs/calib/fast.yaml
defaults:
  - method/adc
  - method/photometry
  - method/trace
  - method/phase

calib:
  method:
    adc.diagnostics.save_images: false
    photometry.photometry.method: aperture
    photometry.aperture.radius_px: 6
    trace.along_time.enabled: false
    phase.binning.nbins: 120
```

### 6.3 Strict (example skeleton)

```yaml
# configs/calib/strict.yaml
defaults:
  - method/adc
  - method/dark
  - method/flat
  - method/cds
  - method/photometry
  - method/trace
  - method/phase
  # - method/corel

calib:
  method:
    adc.reference_pixel.use: true
    adc.overscan.enabled: true
    dark.residual_plane.enabled: true
    flat.illumination_correction.enabled: true
    cds.temporal_filter.enabled: true
    photometry.photometry.method: optimal
    trace.along_wavelength.method: spline
    trace.along_wavelength.spline.knots: 16
    phase.binning.nbins: 400
```

---

## 7) Hydra overrides — quick recipes

**Nearest-timestamp CDS within 3s**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.cds.pairing.order_by=timestamp \
  calib.method.cds.pairing.policy=nearest \
  calib.method.cds.pairing.window_s=3.0
```

**Switch photometry to PSF fit**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.photometry.photometry.method=psf \
  calib.method.photometry.psf.model=gaussian_moffat
```

**Spline continuum with 12 knots**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.trace.along_wavelength.method=spline \
  calib.method.trace.along_wavelength.spline.knots=12
```

**Enable COREL calibration (GAT, 50 epochs)**

```bash
spectramind calibrate \
  --config-name calib/nominal \
  calib.method.corel.method=corel \
  calib.method.corel.model.arch=gat \
  calib.method.corel.train.epochs=50
```

> Use `--cfg job --resolve` to print the composed config for auditing.

---

## 8) Validation strategy

* **Files**: `validation.require_existing_paths=true` for methods using LUTs/maps/masters
* **Shapes**: `validation.check_shape_compatibility=true` (map/LUT vs batch)
* **Metadata**: enforce keys when required (e.g., CDS timestamps/flags; Phase ephemeris)
* **Numerics**: `assert_finite_output=true` to block NaN/Inf propagation

**Common failure hints**

* ADC negative flood → relax `post_subtraction_clamp.min_adu` slightly or verify offset order
* Dark residual banding → enable `residual_plane` or check master scaling units
* Flat explosion → ensure `renormalize_to_mean_one=true` and verify PRNU epsilon
* CDS unpaired → adjust `pairing.policy/window_s` or supply `is_reset` flags
* Phase mis-center → confirm period/t0 and time units (BJD\_TDB vs MJD/UNIX)

---

## 9) DVC & caching

* Each stage can emit intermediates to `${RUN_DIR}`; large artifacts (masters, flats, masks) should be **DVC deps**.
* Profiles may set `cache.write_intermediate=true` per method or in chain-level overrides.
* When inputs + config hashes are unchanged, DVC **skips re-run**.

**Recommended out dirs (per method)**

* ADC → `${RUN_DIR}/calib/adc`
* Dark → `${RUN_DIR}/calib/dark`
* Flat → `${RUN_DIR}/calib/flat`
* CDS → `${RUN_DIR}/calib/cds`
* Photometry → `${RUN_DIR}/calib/photometry`
* Trace → `${RUN_DIR}/calib/trace`
* Phase → `${RUN_DIR}/calib/phase`

---

## 10) Diagnostics & QC

Enable diagnostics selectively to remain Kaggle-safe:

* Images/plots: `diagnostics.save_images/plots=false` for speed in **fast** and **nominal**
* Keep `save_samples` small (e.g., 2)
* Use QC gates as **soft guards** (`on_fail: "warn"`) during sweep, **strict** for audits

---

## 11) Performance guidance

* Prefer `float32` during calibration; quantize only if required.
* Use **median** over **mean** for robustness (dark combine, mesh background).
* Avoid expensive spline grids in **fast**; reduce `knots/grid`, disable heavy robust loops.
* Control worker count via `runtime.num_workers` per method.

---

## 12) Reproducibility

* Log the **composed config** and a **run hash** (e.g., `${RUN_DIR}/run_hash_summary_v50.json`).
* Store *method artifact versions* (master filenames, LUT checksums) in the run log.
* Prefer DVC-tracked references for masters & flats with stable paths.

---

## 13) Testing checklist

* [ ] Files exist (maps/LUTs/masters)
* [ ] Shape compatibility across stages
* [ ] Required metadata present (CDS pairing, Phase ephemeris, etc.)
* [ ] No NaN/Inf after each stage
* [ ] QC gates pass or warn with clear diagnostics
* [ ] CI smoke (`calib/fast`) runs within tight budget

---

## 14) Glossary

* **ADC** — Analog-to-Digital Conversion stage (bias/PRNU/linearization)
* **PRNU** — Photo-Response Non-Uniformity (gain map / flat)
* **CDS** — Correlated Double Sampling (RESET/SIGNAL differencing)
* **SNR** — Signal-to-Noise Ratio, often μ/σ in this context
* **OOT** — Out-Of-Transit (used for normalization and QC)

---

## 15) Changelog (profiles)

* **v1.3** — Strict profile adds per-amp ADC LUTs, dark residual plane, spline-trace; Phase coverage checks
* **v1.2** — Nominal refines CDS pairing & Photometry defaults; Fast trims trace/diagnostics further
* **v1.1** — Initial profile set (fast/nominal/strict), base chain & Hydra overrides

---

## 16) Quick-start

```bash
# Nominal (Kaggle-ready)
spectramind calibrate --config-name calib/nominal

# Fast (CI/smoke)
spectramind calibrate --config-name calib/fast

# Strict (science audit)
spectramind calibrate --config-name calib/strict
```

**With these conventions, `configs/calib/` defines robust, Hydra-composable calibration chains that
remain reproducible, physics-informed, and Kaggle-safe.**

```
```
