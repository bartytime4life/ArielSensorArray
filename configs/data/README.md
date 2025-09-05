# 🗂️ `/configs/data` — Dataset & Calibration Configurations (SpectraMind V50)

> **Mission:** one place to define **what data we use**, **how we calibrate it**, **how we preprocess it**, and **how we validate it** — for local science runs, Kaggle submissions, and CI smoke tests.  
> **Guarantees:** Hydra-composable, DVC/lakeFS-traceable, Kaggle-safe, CI-fast, physics-informed.

---

## 0) Purpose & Scope

This folder defines **all dataset, calibration, preprocessing, split, and loader parameters** for the **SpectraMind V50** pipeline (NeurIPS 2025 Ariel Data Challenge).

Configs in this directory control:

- **Dataset sources** (`nominal`, `kaggle`, `debug`, `toy`)
- **Calibration kill-chain** (ADC, non-linearity, dark, flat, CDS, photometry, trace/phase)
- **Preprocessing** (detrend/savgol, time/grid standardization, bin mapping, smoothing, resampling)
- **Augmentations** (jitter, dropout/mask, SNR-based drops, noise injection)
- **Symbolic hooks** (non-negativity, FFT priors, molecular windows, region masks)
- **Diagnostics** (FFT, Z-score, symbolic overlays, SHAP overlays)
- **Runtime guardrails** (Kaggle/CI limits, integrity checks, fail-fast validation)

Everything here is **Hydra-first**, **reproducible by construction**, and **fail-fast**.

---

## 1) Design Principles

- **Hydra-first:** No hard-coded paths/hparams in Python. All knobs live in YAML selected via `data=<name>`.
- **Reproducibility:** DVC/lakeFS for data lineage, Hydra for config capture, timestamped run dirs.
- **Physics-informed:** Enforce non-negativity, spectral smoothness, molecular priors, and realistic calibration steps.
- **Scenario coverage:** Dedicated YAMLs for science runs, Kaggle runtime, CI/debug, and synthetic toy sets.
- **Fail-fast:** Early schema & path validation prevents expensive wasted GPU time; guardrails everywhere.

---

## 2) Directory Layout

configs/data/
├─ nominal.yaml     # Full scientific dataset (default for experiments)
├─ kaggle.yaml      # Kaggle competition runtime-safe dataset
├─ debug.yaml       # Tiny, deterministic smoke slice for CI/self-test
├─ toy.yaml         # Synthetic toy dataset (fast dev/CI with generator support)
├─ method/
│  ├─ top_level.yaml  # Umbrella method schema (ADC→…→phase; overridable)
│  └─ nominal.yaml    # Nominal method profile (balanced accuracy/runtime)
├─ README.md        # You are here
└─ .gitkeep

**File roles**

- **`nominal.yaml`** — Mission-grade configuration. Full calibration kill-chain, preprocessing, symbolic hooks, rich diagnostics. DVC paths and integrity checks enabled.
- **`kaggle.yaml`** — Competition-safe runtime: uses `/kaggle/input` mounts, **no internet**, ≤9 hr runtime hints, lean diagnostics, conservative loaders.
- **`debug.yaml`** — **Fast** (seconds) CI/self-test dataset. Deterministic, minimal I/O, zero augmentation; exercises the same codepaths as nominal.
- **`toy.yaml`** — Synthetic dataset for quick dev & CI; compatible with `configs/dat/ariel_toy_dataset.py`.
- **`method/top_level.yaml`** — Hydra-overridable schema for the calibration chain (ADC→…→phase), remap, contracts, artifacts.
- **`method/nominal.yaml`** — Balanced, leaderboard-safe calibration settings (≈200 bins, Savitzky–Golay detrending, FGS1-guided alignment).

---

## 3) Quick Usage

**Pick the dataset at runtime:**

```bash
# Full science runs (local/cluster)
spectramind train data=nominal

# Kaggle offline notebook
spectramind train data=kaggle

# CI/self-test / local smoke
spectramind train data=debug training.epochs=1

# Synthetic toy set (fast dev; generator-compatible)
spectramind train data=toy

More:

# Run calibration-only diagnostics on nominal data
spectramind calibrate data=nominal calibration.save_intermediate=true

# Disable heavy FFT in a quick debug run
spectramind diagnose data=debug diagnostics.fft_analysis=false

# Switch detrending method on-the-fly
spectramind train data=nominal preprocessing.detrend.method=savgol

# Tighten outlier clipping & disable smoothing
spectramind train data=nominal preprocessing.outlier_sigma=4 preprocessing.smoothing.enabled=false

# Kaggle-safe training
spectramind train data=kaggle runtime.reduce_heavy_ops=true loader.batch_size=48

Higher-level configs (train.yaml, predict.yaml, selftest.yaml, ablate.yaml) include data=<...> via Hydra defaults.

⸻

4) Calibration Kill-Chain (Order Matters)

Step	Key knobs (see YAML)	Purpose
adc_correction	bit_depth, gain_map, offset_map	Remove ADC offsets & gain structure
nonlinearity_correction	lut_path, saturation_dn	Correct sensor response nonlinearity
dark_subtraction	master_dark_path, exposure/temperature scaling	Remove dark current
flat_fielding	master_flat_path, epsilon	Correct pixel sensitivity variations
correlated_double_sampling	strategy, noise_threshold_dn	Reduce kTC & 1/f via CDS
photometric_extraction	aperture, radius_px, bkg_annulus_px, method	Extract photometry
trace_normalization	reference_window, epsilon	Normalize per-trace
phase_alignment	method, max_shift	Align transit phases

The same composite appears leaner in kaggle.yaml, and ultra-lean in debug.yaml/toy.yaml.

⸻

5) Schema, Safety & Integrity Gates

All configs enforce fail-fast checks:
	•	Schema: expected ranks/shapes/dtypes for FGS1 (N, T_fgs1, 32) (legacy accepted: (N,32,32)), AIRS (N, T_airs, 283|356), label columns (target_mu[0:283], target_sigma[0:283]).
	•	Instrument remap: optional 356→283 bin mapping (nominal/kaggle/debug/toy) to keep model heads consistent.
	•	Safety bounds: numeric guards (min_sigma, max_sigma, max_abs_mu) and non-negativity.
	•	Runtime consistency: DVC remote/path writability, GPU memory (nominal), Kaggle time limit (kaggle), CI time budget (debug/toy).
	•	Split integrity: exportable indices; acceptance of generator-provided splits; deterministic seeds.

⸻

6) Diagnostics & Reports

All configs support a lightweight diagnostics suite (richest in nominal.yaml):
	•	FFT/Z-Score analyses, symbolic overlays, SHAP overlays
	•	Per-step calibration previews (strided to keep I/O modest)
	•	JSON/PNG/HTML artifacts under ${paths.artifacts_dir} / ${hydra.run.dir}

Example:

spectramind diagnose data=nominal diagnostics.save_plots=true


⸻

7) Mermaid Overview (Data → Calibration → Preprocess → Splits → Loader)

flowchart LR
  A[Select data=<nominal|kaggle|debug|toy>] --> B[Paths/Schema/Guards]
  B --> C{Calibration enabled?}
  C -- yes --> C1[ADC→Nlin→Dark→Flat→CDS→Photometry→Trace→Phase]
  C -- no  --> D[Bypass/Preview-only]
  C1 --> E[Preprocess: detrend/savgol, normalize, remap 356→283, clip]
  D  --> E
  E --> F[Splits: load or derive (export *.npy)]
  F --> G[Loader: batch/workers/prefetch/shuffle]
  G --> H[Diagnostics: FFT/Z-score/overlays]
  H --> I[Hydra run dir + artifacts]


⸻

8) Common Recipes

Switch detrending method on-the-fly

spectramind train data=nominal preprocessing.detrend.method=savgol

Accelerate turnaround by reducing diagnostics

spectramind diagnose data=nominal diagnostics.fft_analysis=false diagnostics.shap_overlay=false

Kaggle-safe training

spectramind train data=kaggle runtime.reduce_heavy_ops=true loader.batch_size=48

Use toy/debug generator outputs (.npz/.npy/.pkl) with splits & manifest

spectramind train data=toy validate.checks=[paths_exist,schema_match,no_nan_inf,manifest_consistency]


⸻

9) Troubleshooting
	•	“Missing file” on CI/Kaggle: Confirm io.fail_on_missing=true and the actual path exists. For Kaggle, inputs must be under /kaggle/input/....
	•	Shape mismatch: Check schema.accepted_shapes — legacy exports (N,32,32) are auto-transposed if supported; AIRS 356 vs 283 needs instrument.bin_remap.enabled=true.
	•	Long runtimes on Kaggle: Lower loader.batch_size, enable runtime.reduce_heavy_ops, and disable heavy diagnostics. Ensure no internet calls.
	•	Out-of-memory: Reduce batch_size, disable memmap for tiny sets; for large sets, keep memmap=true and increase num_workers only if stable.
	•	Split drift: Prefer generator-provided splits/*.npy. Otherwise, fix splits.seed and export indices to lock comparisons.

⸻

10) Versioning & Lineage
	•	Hydra snapshots are stored under the run dir; attach to reports.
	•	DVC/lakeFS should track data/raw/** and data/processed/**.
	•	Generator manifest (toy_manifest.json) records hashes, shapes, and split sizes for reproducibility.
	•	Config versions are bumped on schema/guard changes (see dataset.version fields).

⸻

11) FAQ

Q: Why do AIRS arrays have 356 but the model expects 283 bins?
A: Use the 356→283 bin_remap (enabled by default) to keep outputs compatible with the 283-head decoders.

Q: Can I run without the full calibration chain?
A: Yes. Toggle steps via calibration.method.<step>.enabled=false. Expect degraded physics performance; fine for debug/toy.

Q: How do I create a tiny dataset for a new smoke test?
A: Use configs/dat/ariel_toy_dataset.py (deterministic; .npz/.npy/.pkl + splits + manifest), then select data=toy or data=debug.

⸻

12) Change Log (excerpt)
	•	2025-09-05: Added toy.yaml, array-first loader paths (.npz>.npy>.pkl), generator manifest checks, split source/export symmetry.
	•	2025-09-04: Upgraded kaggle.yaml to accept legacy shapes, bin remap guard, and learned diagnostics caps.
	•	2025-09-03: Hardened nominal.yaml with richer validation gates, symbolic overlays, and calibration stride previews.

⸻

✅ TL;DR

/configs/data is the flight control module for all dataset handling: pick data=<nominal|kaggle|debug|toy>, and the pipeline will execute with mission-grade, reproducible data behavior across local, Kaggle, and CI environments.

