# 🗂️ `/configs/data` — Dataset & Calibration Configurations

## 0) Purpose & Scope

This folder defines **all dataset, calibration, preprocessing, split, and loader parameters** for the **SpectraMind V50** pipeline (NeurIPS 2025 Ariel Data Challenge).  
Everything here is **Hydra-composable**, **DVC-traceable**, **Kaggle-safe**, and suitable for **CI smoke tests**.

Configs in this directory control:

- **Dataset sources** (nominal, kaggle, debug)
- **Calibration kill-chain** (ADC, nonlinearity, dark, flat, CDS, photometry, trace/phase normalization)
- **Preprocessing** (detrend/savgol, time/grid standardization, bin mapping, smoothing, resampling)
- **Augmentations** (jitter, dropout/mask, SNR-based drops, noise injection)
- **Symbolic physics hooks** (non-negativity, FFT priors, molecular windows, region masks)
- **Diagnostics** (FFT, Z-score, symbolic overlays, SHAP overlays)
- **Runtime guardrails** (Kaggle/CI limits, integrity checks, fail-fast validation)

---

## 1) Design Philosophy

- **Hydra-first**: No hard-coded paths/hparams in Python. All data knobs live in YAML and are selected via `data=<name>`.
- **Reproducibility by construction**: DVC/lakeFS for data lineage, Hydra for config capture, consistent run directories.
- **Physics-informed**: Enforce non-negativity, spectral smoothness, molecular priors, and realistic calibration steps.
- **Scenario coverage**: Distinct YAMLs for full science runs, Kaggle offline runs, and CI/debug smoke tests.
- **Fail-fast**: Early schema & path validation prevents expensive wasted GPU time.

---

## 2) Directory Layout

```

configs/data/
├─ nominal.yaml   # Full scientific dataset (default for experiments)
├─ kaggle.yaml    # Kaggle competition runtime-safe dataset
├─ debug.yaml     # Tiny, deterministic smoke slice for CI/self-test
├─ .gitkeep       # Keeps this folder tracked + inline notes
└─ README.md      # You are here

````

**File roles**

- **`nominal.yaml`** — Mission-grade configuration. Full calibration kill-chain, preprocessing, symbolic hooks, rich diagnostics. DVC paths and integrity checks enabled.
- **`kaggle.yaml`** — Competition-safe runtime: uses `/kaggle/input` mounts, **no internet**, ≤9 hr runtime hints, lean diagnostics, conservative loaders.
- **`debug.yaml`** — **Fast** (seconds) CI/self-test dataset. Deterministic, minimal I/O, zero augmentation; exercises the same codepaths as nominal.

---

## 3) Quick Usage

Select the dataset at runtime with Hydra:

```bash
# Full science runs (local/cluster)
spectramind train data=nominal

# Kaggle offline notebook
spectramind train data=kaggle

# CI/self-test / local smoke
spectramind train data=debug training.epochs=1
````

Other common commands:

```bash
# Run calibration-only diagnostics on nominal data
spectramind calibrate data=nominal calibration.save_intermediate=true

# Disable heavy FFT in a quick debug run
spectramind diagnose data=debug diagnostics.fft_analysis=false
```

Higher-level configs (`train.yaml`, `predict.yaml`, `selftest.yaml`, `ablate.yaml`) include `data=<...>` via Hydra `defaults`.

---

## 4) Calibration Kill-Chain (Order Matters)

| Step                         | Key knobs (see YAML)                              | Purpose                              |
| ---------------------------- | ------------------------------------------------- | ------------------------------------ |
| `adc_correction`             | `bit_depth`, `gain_map`, `offset_map`             | Remove ADC offsets & gain structure  |
| `nonlinearity_correction`    | `lut_path`, `saturation_dn`                       | Correct sensor response nonlinearity |
| `dark_subtraction`           | `master_dark_path`, exposure/temperature scaling  | Remove dark current                  |
| `flat_fielding`              | `master_flat_path`, `epsilon`                     | Correct pixel sensitivity variations |
| `correlated_double_sampling` | `strategy`, `noise_threshold_dn`                  | Reduce kTC & 1/f via CDS             |
| `photometric_extraction`     | `aperture`, `radius_px`, `bkg_annulus_px`, method | Extract photometry                   |
| `trace_normalization`        | `reference_window`, `epsilon`                     | Normalize per-trace                  |
| `phase_alignment`            | `method`, `max_shift`                             | Align transit phases                 |

> The same composite appears (leaner) in `kaggle.yaml`, and ultra-lean in `debug.yaml`.

---

## 5) Schema, Safety & Integrity Gates

Each config enforces **fail-fast** checks:

* **Schema**: expected ranks/shapes/dtypes for FGS1 `(N,32,32)`, AIRS `(N,32,356)`, labels columns (`target_mu[0:283]`, `target_sigma[0:283]`).
* **Instrument remap**: optional **356→283** bin mapping (nominal/kaggle/debug) to keep model heads consistent.
* **Safety bounds**: numeric guards (`min_sigma`, `max_sigma`, `max_abs_mu`) and non-negativity.
* **Runtime consistency**: DVC remote/path writability, GPU memory (nominal), Kaggle time limit (kaggle), CI time budget (debug).

---

## 6) Diagnostics & Reports

All configs support a lightweight diagnostics suite (richer in `nominal.yaml`):

* **FFT**/**Z-Score** analyses, **symbolic overlays**, **SHAP overlays**
* Per-step calibration previews (strided to keep I/O modest)
* JSON/PNG/HTML artifacts under `${paths.artifacts_dir}` / `${hydra.run.dir}`

Example (nominal):

```bash
spectramind diagnose data=nominal diagnostics.save_plots=true
```

---

## 7) Best Practices

* ✅ **No hardcoding** — never bake absolute paths into code; use config interpolation.
* ✅ **Keep raw/processed under DVC** — reference data via tracked paths or Kaggle mounts.
* ✅ **Kaggle** — **no internet**, rely only on `/kaggle/input`, store outputs in `/kaggle/working`.
* ✅ **Debug** — smoke runs should complete in **< 60 s**; disable heavy ops.
* ✅ **Commit the composed config** — Hydra already snapshots under `outputs/…`; attach in reports/manifests.

---

## 8) Common Recipes

**Switch detrending method on-the-fly**

```bash
spectramind train data=nominal preprocessing.detrend.method=savgol
```

**Tighten outlier clipping & disable smoothing**

```bash
spectramind train data=nominal preprocessing.outlier_sigma=4 \
  preprocessing.smoothing.enabled=false
```

**Reduce diagnostics to accelerate turnaround**

```bash
spectramind diagnose data=nominal diagnostics.fft_analysis=false \
  diagnostics.shap_overlay=false
```

**Kaggle-safe training**

```bash
spectramind train data=kaggle runtime.reduce_heavy_ops=true loader.batch_size=48
```

---

## 9) FAQ

**Q: Why do AIRS arrays have 356 but the model expects 283 bins?**
A: Use the provided **356→283** `bin_remap` (enabled by default) to keep outputs compatible with the 283-head decoders.

**Q: Can I run without the full calibration kill-chain?**
A: Yes. Disable steps via `calibration.method.<step>.enabled=false`, but physics performance will degrade.

**Q: How do I create a tiny dataset for a new smoke test?**
A: Use `configs/dat/ariel_toy_dataset.py` to generate deterministic toy/debug packs, then point `debug.yaml` paths to them.

---

## 10) References

* SpectraMind V50 Technical Plan
* SpectraMind V50 Project Analysis
* Strategy for Updating & Extending V50
* Hydra for AI Projects
* Kaggle Platform Guide

---

### ✅ TL;DR

`/configs/data` is the **flight control module** for all dataset handling: pick `data=<nominal|kaggle|debug>`, and the pipeline will execute with **mission-grade**, reproducible data behavior across local, Kaggle, and CI environments.

```
```
