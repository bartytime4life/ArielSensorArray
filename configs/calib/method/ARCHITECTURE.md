# 🧭 `configs/calib/method/architecture.md`

**SpectraMind V50** · **NeurIPS 2025 Ariel Data Challenge**  
**Document type:** Architecture & schema reference for method-level calibration configs

---

## 0) Purpose

This document defines the **architecture, schema, and integration contracts** for method-level calibration files in  
`configs/calib/method/`. It ensures every method YAML (ADC, dark, flat, CDS, photometry, trace, phase, corel) is:

- **Composable** via Hydra defaults/overrides  
- **Deterministic & auditable** with strong validation, logging, and diagnostics  
- **Interoperable** across pipeline stages with consistent I/O keys  
- **Kaggle-safe** (no internet, no interactive prompts; fast failure on missing deps)  

This file is the **single source of truth** for writing, reviewing, and extending calibration method configs.

---

## 1) Scope & placement

```

configs/
└── calib/
├── chain.yaml               # Orchestration of stages (nominal/fast/heavy)
├── method/
│   ├── adc.yaml
│   ├── dark.yaml
│   ├── flat.yaml
│   ├── cds.yaml
│   ├── photometry.yaml
│   ├── trace.yaml
│   ├── phase.yaml
│   └── corel.yaml           # (optional, post-inference graph calibration)
└── method/
└── architecture.md      # (this file)

````

- Each **method** YAML configures **one** stage of the kill chain.
- A chain file (e.g. `calib/nominal.yaml`) composes the methods in order.

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

* **ADC**: offset/gain/PRNU, reference pixels/overscan, linearization, (optional) re-quantization
* **Dark**: master/metadata selection, exposure/temperature scaling, hot-pixel handling
* **Flat**: master/spectral flats, PRNU normalization, (optional) illumination field
* **CDS**: pair RESET/SIGNAL reads (flags/index/timestamp), robust differencing
* **Photometry**: aperture/psf/optimal extraction, centroiding & background, variance propagation
* **Trace**: along-wavelength continuum + along-time detrending, masks & QC
* **Phase**: ephemeris fold, windows, binning, coverage flags
* **COREL** (optional): graph GNN over bins to calibrate μ/σ

---

## 3) Method schema: common sections

> **Every method file** uses this structural contract (top-level name matches the method key).

* `_meta`: `{ schema_version, last_updated }`
* `<method>.enabled`: stage enable/disable
* `<method>.io`: I/O binding (Hydra contracts to the in-memory batch dict)

  * `input_key`: where the stage reads input tensor(s)/dict
  * `output_key`: where the stage writes outputs
  * `debug_keys` *(optional)*: diagnostic field names under `output_key`
* `<method>.validation`: fail-fast rules (files, shapes, metadata keys)
* `<method>.diagnostics`: where to store plots/images/summaries (`${RUN_DIR}`)
* `<method>.logging` / `<method>.runtime`: verbosity, seed, workers, deterministic mode
* Stage-specific configuration (e.g., `offset`, `gain`, `pairing`, `background`, `centroid`, etc.)

**YAML example (abbreviated):**

```yaml
_meta:
  schema_version: "1.3.0"
  last_updated: "${now:%Y-%m-%d}"

adc:
  enabled: true
  io:
    input_key: "raw_frames"
    output_key: "adc_corrected"
  validation:
    require_existing_paths: true
    check_shape_compatibility: true
```

---

## 4) I/O contracts (keys & shapes)

All stages operate on **in-memory dicts** carried by the executor. Consistent **keys** and **shapes** are required.

| Stage      | Input key        | Output key       | Typical shape(s)                             |
| ---------- | ---------------- | ---------------- | -------------------------------------------- |
| ADC        | `raw_frames`     | `adc_corrected`  | `[B,H,W]` or `[B,C,H,W]`                     |
| Dark       | `adc_corrected`  | `dark_corrected` | `[B,H,W]` or `[B,C,H,W]`                     |
| Flat       | `dark_corrected` | `flat_corrected` | `[B,H,W]` or `[B,C,H,W]`                     |
| CDS        | `flat_corrected` | `cds_corrected`  | `[B,H,W]` or `[B,C,H,W]`                     |
| Photometry | `cds_corrected`  | `photometry`     | Flux: `[B]` or `[B,C]` (+ ancillary scalars) |
| Trace      | `photometry`     | `trace_norm`     | Flux: `[B]` or `[B,C]`                       |
| Phase      | `trace_norm`     | `phase_fold`     | Phase: `[B]`; binned flux: `[M]` or `[M,C]`  |
| COREL\*    | `inference`      | `corel`          | μ/σ: `[B,C]`; metrics: dict                  |

> **Channels (C)** are spectral bins (e.g., 283). **B** is time/frames/samples; **H/W** are image dimensions.

**Default upstream keys · examples**

* Photometry outputs:
  `photometry_flux`, `photometry_flux_var`, `photometry_centroid_x`, `photometry_centroid_y`, `photometry_flags`
* Trace outputs:
  `trace_flux_norm`, `trace_flux_scale`, `trace_flags`
* Phase outputs:
  `phase_series`, `binned_phase`, `binned_flux`, `binned_flux_err`, `windows`, `phase_flags`

---

## 5) Validation strategy (fail fast)

Each method should declare **strict** validation rules:

* **Files**: `validation.require_existing_paths=true` if method uses LUTs/maps/masters
* **Shapes**: `validation.check_shape_compatibility=true` checks loaded arrays vs first batch
* **Metadata**: e.g. CDS pairing by timestamp requires `acq_timestamp_s`; phase requires `{period_days, t0_bjd}`
* **Numerics**: `assert_finite_output=true` (no NaN/Inf propagation)

**Example (CDS)**

```yaml
validation:
  require_metadata:
    timestamp_for_ts_mode: true
    index_for_index_mode: true
    is_reset_for_explicit_flags: true
  check_shape_compatibility: true
  assert_finite_output: true
```

---

## 6) Diagnostics & QC (soft guards)

* **Diagnostics**: each method may emit plots/masks/images under
  `diagnostics.output_dir: "${RUN_DIR}/diag/<stage>"`
* **QC**: non-mutating checks (histogram/percentiles/stats, coverage/flags) with `on_fail: "warn"|"raise"|"skip"`

**Example (flat)**

```yaml
quality_checks:
  enabled: true
  flat_stats:
    expected_mean_range: [0.8, 1.2]
    max_zero_frac: 0.02
  action_on_fail: "warn"
```

---

## 7) DVC & caching

If you maintain a DVC DAG:

* Each stage can set `cache.write_intermediate` and a **DVC stage name** (e.g., `dvc_stage_name: "calib_flat"`)
* Unchanged dependencies skip re-run automatically
* Large refs (masters, maps, LUTs) should be DVC-tracked artifacts

---

## 8) Hydra composition patterns

**Chain composition (simplified):**

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
```

**Inline overrides:**

```bash
# switch ADC to per-pixel PRNU map
spectramind calibrate \
  calib.method.adc.gain.method=per_pixel \
  calib.method.adc.gain.prnu_map_path=data/calib/adc/prnu_map.npy
```

**Instrument-specific toggles:**

```bash
# FGS1: single channel, smaller aperture
spectramind calibrate \
  calib.method.photometry.instrument.name=FGS1 \
  calib.method.photometry.instrument.per_channel=false \
  calib.method.photometry.aperture.radius_px=6
```

---

## 9) Method-specific highlights

### 9.1 ADC (`method/adc.yaml`)

* Offset → Gain (PRNU) → Linearization → (Optional) smear/bloom → Clamp/Quantize
* Reference pixels/overscan, per-amp offsets/gains, per-amp LUTs (optional)
* QC: zero/saturated fractions, per-amp parity checks

### 9.2 Dark (`method/dark.yaml`)

* Master mode: `single | select_by_metadata | build_from_raw`
* Exposure & temperature scaling; hot-pixels, bad columns; optional residual plane fit
* QC: max mean dark, hot pixel fraction, residual std ratio

### 9.3 Flat (`method/flat.yaml`)

* Master vs spectral flat; PRNU renormalization; optional illumination correction
* Post-flat bad pixel cleanup; QC on flat stats and residuals

### 9.4 CDS (`method/cds.yaml`)

* Pairing policy: `n_minus_1 | nearest | exact | interpolate` (`index`/`timestamp`)
* Reset classification: `explicit_flags | static | pattern`
* Robust sigma-clip & outlier replacement; optional temporal filters

### 9.5 Photometry (`method/photometry.yaml`)

* `aperture | psf | optimal` extraction; centroiding ROI; background model (`annulus|mesh|poly2d|spline2d`)
* Variance propagation: read/shot noise model or `variance_from_input_key`
* Flags: saturation, off-center, excessive background, too-many-replaced

### 9.6 Trace (`method/trace.yaml`)

* Along-wavelength continuum (`poly | spline | lowess`), masks for exclude windows
* Along-time detrend (`poly | spline | highpass`), target=`divide|subtract`
* Robust weighting from variance; clamps & invalid handling

### 9.7 Phase (`method/phase.yaml`)

* Ephemeris source: metadata/file; time scale conversion; wrap to interval \[-0.5,0.5]
* Windows: in/out-of-transit; OOT normalization; binning (median/mean; variance-weighted)
* QC: phase center tolerance, binning coverage

### 9.8 COREL (`method/corel.yaml`) *(optional)*

* Graph over bins (knn/dense/threshold), edge features, positional encodings
* Backends: `gcn|gat|mpnn`; heads: `scale` (T\_b) or `delta` (Δμ, Δlogσ)
* Conformal coverage metrics; Laplacian smoothness prior

---

## 10) Adding a new method

1. **Create** `configs/calib/method/<new>.yaml` with:

   * `_meta.schema_version` & `last_updated`
   * `<new>.enabled`, `<new>.io`, `<new>.validation`, `<new>.diagnostics`, `<new>.logging`, `<new>.runtime`
   * Stage-specific blocks (clear names, comments, sane defaults)
2. **Wire** it in a chain (e.g., `calib/nominal.yaml`) under `defaults`.
3. **Declare** I/O keys and ensure shape & metadata validation.
4. **Provide** a minimal test in `/tests/calibration/test_<new>_method.py`.
5. **Document** quick overrides and examples in the YAML comments.

---

## 11) Data, units, dtype, and shapes

* **Units**: keep DN↔e⁻ conversions explicit via `conversion_gain_e_per_adu`.
* **Dtype**: prefer `float32` through calibration; only quantize to `uint16` when required downline.
* **Shapes**: image stages accept `[B,H,W]` or `[B,C,H,W]`; spectral stages accept `[B,C]`.

---

## 12) Performance & CI notes

* Turn off heavy diagnostics and robust fits (sigma-clips, spline grids) for CI/Kaggle smoke runs.
* Use `num_workers` and small sample subsets for QC (`sampling_fraction`).
* Prefer **median** over **mean** when resilience to outliers is needed.

---

## 13) Security & provenance

* No network calls; all artifacts must be local or DVC-managed.
* Avoid silently **creating** large artifacts during inference; fail fast if missing.
* Log config hashes and dataset IDs in the run log for reproducibility.

---

## 14) Testing checklist

* [ ] Files exist (`require_existing_paths`)
* [ ] Shapes match (`check_shape_compatibility`)
* [ ] Metadata keys present for chosen modes (pairing, ephemeris)
* [ ] No NaN/Inf in outputs (`assert_finite_output`)
* [ ] QC gates pass (or warn)
* [ ] Minimal smoke in CI passes within time budget

---

## 15) Changelog (excerpt)

* **v1.3** — Per-amp ADC LUTs; CDS `interpolate` mode; trace dual-stage norm; phase coverage checks; COREL conformal metrics
* **v1.2** — Expanded QC & diagnostics; stricter validation; improved background/centroid options
* **v1.1** — Initial full-stack method schema, I/O contracts, and Hydra override recipes

---

## 16) Quick-reference commands

```bash
# Full nominal chain
spectramind calibrate --config-name calib/nominal

# Fast path (CI/Kaggle-friendly)
spectramind calibrate --config-name calib/nominal \
  calib.method.photometry.photometry.method=aperture \
  calib.method.photometry.aperture.radius_px=6 \
  calib.method.trace.along_time.enabled=false \
  calib.method.phase.binning.nbins=120
```

---

**With these conventions, `configs/calib/method/` is a robust, modular toolbox for the calibration pipeline — each YAML is a well-typed, overrideable “unit” that composes cleanly into the kill chain.**

```
```
