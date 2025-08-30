# ⚙️ Calibration Methods — SpectraMind V50

This folder defines **method-level calibration profiles** for the SpectraMind V50 pipeline in the  
**NeurIPS 2025 Ariel Data Challenge**. Each file here specifies how an **individual calibration step**  
(ADC, dark, flat, CDS, photometry, trace, phase, etc.) is parametrized and can be composed into the  
main **calibration kill chain** via Hydra.

---

## 📌 Purpose

Calibration in SpectraMind V50 follows a reproducible, swappable **kill chain** of corrections:

1) **ADC correction** — analog-to-digital normalization (offset/gain/PRNU, reference pixels, overscan).  
2) **Nonlinearity** — response linearization (LUT/polynomial).  
3) **Dark subtraction** — remove dark current & bias structure.  
4) **Flat fielding** — correct pixel-to-pixel sensitivity variations (PRNU/illumination field).  
5) **Correlated double sampling (CDS)** — suppress reset & low-frequency noise by differencing reads.  
6) **Photometric extraction** — aperture / PSF / optimal profile extraction of flux.  
7) **Trace normalization** — normalize spectral traces to continuum; phase-agnostic detrending.  
8) **Phase alignment** — fold to orbital phase; define in/out-of-transit; (optionally) bin.  
9) *(Optional post-inference)* **COREL-style calibration** — graph GNN over bins to calibrate μ/σ.

Each **method** can be overridden independently without changing pipeline code.

---

## 📁 Files in this folder

| File | Stage | What it configures |
|---|---|---|
| **adc.yaml** | ADC | Offsets, per-channel/per-pixel gains (PRNU), reference pixels, overscan, linearization, re-quantization. |
| **dark.yaml** | Dark | Master dark selection/build, exposure/temperature scaling, hot pixel/bad column repair, residual plane fit. |
| **flat.yaml** | Flat | Master/spectral flats, PRNU normalization, illumination (blaze/vignetting) correction. |
| **cds.yaml** | CDS | Pairing policy (index/timestamp), explicit RESET flags, static reset fallback, robust outlier handling. |
| **photometry.yaml** | Photometry | `aperture`/`psf`/`optimal`, centroiding, background model, noise/variance propagation, flags. |
| **trace.yaml** | Trace normalization | Along-wavelength continuum fit (`poly`/`spline`/`lowess`), along-time detrend, masks & QC. |
| **phase.yaml** | Phase alignment | Ephemeris source, time scale, in/out-of-transit masks, binning & outlier policy. |
| **corel.yaml** *(optional)* | Post-inference | COREL-style graph calibration over spectral bins (GCN/GAT/MPNN; “scale” or “delta” head). |

> **Convention:** Every file uses a top-level key matching its method name (e.g. `adc:`, `dark:` …)  
> plus standard sections like `io`, `qc/quality_checks`, `diagnostics`, `runtime`, and `validation`.

---

## 🧩 Hydra composition & overrides

These configs live under the group `calib/method`. They are **atomic** (one stage per file) and are usually  
composed by a chain like `configs/calib/chain.yaml` or `configs/calib/nominal.yaml`.

### Quick overrides

**Aperture photometry with radius 8 px**
```bash
spectramind calibrate \
  calib.method.photometry.photometry.method=aperture \
  calib.method.photometry.aperture.radius_px=8
````

**Switch trace normalization to spline with 12 knots**

```bash
spectramind calibrate \
  calib.method.trace.along_wavelength.method=spline \
  calib.method.trace.along_wavelength.spline.knots=12
```

**Use per-channel offsets and LUT linearization in ADC**

```bash
spectramind calibrate \
  calib.method.adc.offset.method=per_channel \
  calib.method.adc.offset.table_path=data/calib/adc/offsets_per_channel.npy \
  calib.method.adc.linearization.method=lut \
  calib.method.adc.linearization.lut_path=data/calib/adc/linearization_lut.npy
```

**CDS pairing by nearest timestamp within 3 s**

```bash
spectramind calibrate \
  calib.method.cds.pairing.order_by=timestamp \
  calib.method.cds.pairing.policy=nearest \
  calib.method.cds.pairing.window_s=3.0
```

**Enable COREL calibration with GAT(4 heads) for 50 epochs**

```bash
spectramind calibrate \
  calib.method.corel.method=corel \
  calib.method.corel.model.arch=gat \
  calib.method.corel.model.heads=4 \
  calib.method.corel.train.epochs=50
```

> **Tip:** Use `--cfg job` or `--cfg hydra` to view the composed config; `--resolve` to evaluate interpolations.

---

## 🔗 How methods are chained

Your chain file (e.g., `configs/calib/nominal.yaml`) sets the order and enables the methods:

```yaml
# configs/calib/nominal.yaml (example skeleton)
defaults:
  - method/adc
  - method/dark
  - method/flat
  - method/cds
  - method/photometry
  - method/trace
  - method/phase
  # - method/corel   # optional post-inference step

# Per-stage enables can be toggled here as well:
calib:
  method:
    adc.enabled: true
    dark.enabled: true
    flat.enabled: true
    cds.enabled: true
    photometry.enabled: true
    trace.enabled: true
    phase.enabled: true
```

---

## 🧠 Standard schema & contracts

Every method file follows these conventions:

* `_<meta>`: schema version and timestamp.
* `<method>.enabled`: gate the stage on/off.
* `<method>.io`: `input_key`, `output_key`, and any auxiliary fields (e.g., debug masks).
* `<method>.validation`: fail-fast rules (shape/file/metadata keys).
* `<method>.diagnostics`: where to save debug artifacts (`${RUN_DIR}` by default).
* `<method>.logging` / `runtime`: verbosity, seed, workers, determinism.

**Typical I/O keys (examples)**

* ADC: `"raw_frames"` → `"adc_corrected"`
* Dark: `"adc_corrected"` → `"dark_corrected"`
* Flat: `"dark_corrected"` → `"flat_corrected"`
* CDS: `"flat_corrected"` → `"cds_corrected"`
* Photometry: `"cds_corrected"` → `"photometry"` (flux, variance, flags, centroid)
* Trace: `"photometry"` → `"trace_norm"` (normalized flux series, flags)
* Phase: `"trace_norm"` → `"phase_fold"` (phase, binned flux, masks, flags)
* COREL: `"inference"` → `"corel"` (calibrated μ/σ, coverage metrics)

> If a method uses files (maps/LUTs/masters), set `validation.require_existing_paths=true` to fail fast.

---

## 🧪 DVC & caching

* Each method can write intermediates to `${RUN_DIR}` (see `diagnostics`/`cache` blocks).
* If your repo includes a DVC DAG, bind `dvc_stage_name` per method so unmodified inputs skip re-run.
* Large artifacts (masters, flats, masks) should be versioned as DVC dependencies for full reproducibility.

---

## 🧭 CLI quickstart recipes

**Run the full nominal chain with safe defaults**

```bash
spectramind calibrate --config-name calib/nominal
```

**Fast CI/Kaggle smoke**

```bash
spectramind calibrate --config-name calib/nominal \
  calib.method.photometry.photometry.method=aperture \
  calib.method.photometry.aperture.radius_px=6 \
  calib.method.trace.along_time.enabled=false \
  calib.method.phase.binning.nbins=120
```

**Instrument-specific tweaks (FGS1 vs AIRS)**

```bash
spectramind calibrate \
  calib.method.photometry.instrument.name=FGS1 \
  calib.method.trace.instrument.name=FGS1 \
  calib.method.photometry.instrument.per_channel=false
```

---

## 🗺️ Kill-chain diagram

```mermaid
flowchart LR
  A[ADC] --> B[Dark]
  B --> C[Flat]
  C --> D[CDS]
  D --> E[Photometry]
  E --> F[Trace normalization]
  F --> G[Phase alignment]
  G -. optional .-> H[COREL (graph μ/σ)]
```

---

## 🧷 Notes & best practices

* **Keep units consistent.** If converting DN↔e⁻, use `conversion_gain_e_per_adu` consistently across methods.
* **Renormalize PRNU/gain maps** to mean 1 to avoid global scale shifts.
* **Validate early.** Enable `validation.require_existing_paths` and `check_shape_compatibility` in development.
* **Metadata keys.** Pairing/ephemeris often needs keys like `frame_index`, `acq_timestamp_s`, `period_days`, `t0_bjd`.
* **Diagnostics.** Turn on `diagnostics.save_images` or `save_debug_plots` when tuning; disable for speed in CI/Kaggle.

---

✅ With this structure, `/configs/calib/method/` acts as the **toolbox** for the calibration pipeline — each YAML corresponds to **one correction method**, enabling **reproducibility, modularity, and Hydra/Kaggle-safe overrides**.

```
```
