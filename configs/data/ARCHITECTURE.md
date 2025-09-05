# 🛰️ `/configs/data/ARCHITECTURE.md` — Data Configuration Architecture (SpectraMind V50)

> **Scope:** Authoritative architecture for **data ingestion → calibration → preprocessing → splitting → loading** across **local/HPC**, **Kaggle**, and **CI**.  
> **Guarantees:** Hydra-first composition, DVC/lakeFS traceability, physics-informed defaults, fail-fast validation, Kaggle/CI guardrails.

---

## 0) Purpose & Scope

`/configs/data` transforms **raw telescope signals** — **FGS1** photometry (time×channels) + **AIRS** spectroscopy (time×wavelength) — into **calibrated, normalized, model-ready** tensors under strict **mission-grade reproducibility**:

- **Deterministic composition** (Hydra snapshots, fixed seeds)
- **Physics-informed calibration** (ADC, nonlinearity, dark, flat, CDS, photometry, trace/phase)
- **Scenario flexibility** (full science / Kaggle / CI smoke / toy synthetic)
- **Auditability** (DVC lineage, config hash, generator manifest, run manifests & artifacts)

---

## 1) Design Principles

- **Hydra-first modularity**  
  Dataset modes are YAML components (`nominal.yaml`, `kaggle.yaml`, `debug.yaml`, `toy.yaml`) composed by higher-level configs (`train.yaml`, `predict.yaml`, `selftest.yaml`) via Hydra `defaults`.
- **Zero hardcoding**  
  Paths, splits, calibration flags, loaders, bin maps live in YAML — never in Python.
- **DVC/lakeFS integration**  
  Raw/processed artifacts are DVC-tracked; configs reference tracked paths or environment mounts (`/kaggle/input`).
- **Physics realism**  
  Encodes calibration and symbolic constraints (non-negativity, smoothness, molecular windows). Supports 356→283 bin remap for decoder compatibility.
- **Environment awareness**  
  Dedicated runtime profiles for **Local/HPC** (`nominal`), **Kaggle** (`kaggle`), **CI** (`debug`), **Synthetic Dev** (`toy`).
- **Mission constraints**  
  Kaggle: ≤9 hr walltime, ≤16 GB GPU, **no internet**.  
  CI smoke: **<60 s**. Toy: **<3 min**.

---

## 2) Directory Structure

configs/data/
├─ nominal.yaml       # Full scientific dataset (default for experiments)
├─ kaggle.yaml        # Kaggle runtime-safe dataset (offline, resource-guarded)
├─ debug.yaml         # Tiny deterministic slice (CI/self-test)
├─ toy.yaml           # Synthetic toy dataset (generator-compatible)
├─ method/
│  ├─ top_level.yaml  # Umbrella method schema (ADC→…→phase; contracts/artifacts)
│  └─ nominal.yaml    # Balanced method profile (binning, detrending, jitter align)
├─ README.md          # Quick how-to and usage
└─ ARCHITECTURE.md    # (this document)

**Cross-refs**
- Generator: `configs/dat/ariel_toy_dataset.py` (deterministic `.npz/.npy/.pkl` + `splits/` + `toy_manifest.json`)
- Self-test: `selftest.yaml` calls data=debug and runs validation/preview.

---

## 3) Component Responsibilities

### `nominal.yaml` (Science)
- Full DVC-tracked inputs.
- **Complete** calibration kill-chain (ADC→Nonlin→Dark→Flat→CDS→Photometry→Trace/Phase).
- Preprocessing (normalize, detrend polyfit/savgol), symbolic hooks, rich diagnostics.
- Fail-fast gates: schema, remap integrity, split sums, non-negativity, GPU memory.
- Use for experiments & leaderboard‐grade training.

### `kaggle.yaml` (Competition)
- IO roots: `/kaggle/input/neurips-2025-ariel/`, `/kaggle/working/`, `/kaggle/temp/`.
- Guardrails: `num_workers≤2`, `batch_size≤64`, **no internet**, lean diagnostics, memmap support.
- Accepts legacy shapes; optional bin remap 356→283.
- Optimized to finish within **≤9 hr**.

### `debug.yaml` (CI/Smoke)
- **5-planet** slice mirroring nominal schema; **deterministic**, minimal I/O, zero augmentation.
- Loader: `batch_size=2`, `num_workers=0`.
- Completes in **seconds**; used by CI/self-test and local smoke.

### `toy.yaml` (Synthetic Dev/CI)
- 64-planet synthetic with `.npz/.npy/.pkl` compatibility, `splits/`, and manifest checks.
- Designed to smoke a **calibrate→train→diagnose→submit** loop in **<3 min**.

---

## 4) Data Flow & Calibration Chain

```mermaid
flowchart TD
    A[Raw Inputs<br/>FGS1 (T×C) + AIRS (T×λ)] --> B[Calibration<br/>ADC • Nonlin • Dark • Flat • CDS • Photometry • Trace/Phase]
    B --> C[Preprocessing<br/>Normalize • Detrend • (Re)sample • Remap 356→283 • Clip]
    C --> D[Hydra Dataset Object<br/>data=nominal|kaggle|debug|toy]
    D --> E[Train / Predict / Diagnose]
    E --> F[Artifacts<br/>μ, σ, submission.csv, JSON/PNG/HTML diagnostics]

Calibration steps (method/top_level.yaml → method/nominal.yaml overrides)

Step	Key knobs (YAML)	Purpose
adc_correction	bit_depth, gain_map, offset_map	Remove ADC offsets & gain structure
nonlinearity_correction	lut_path, saturation_dn	Correct sensor response nonlinearity
dark_subtraction	master_dark_path, exposure/temperature scaling	Remove dark current
flat_fielding	master_flat_path, epsilon	Correct pixel sensitivity variations
correlated_double_sampling	strategy, noise_threshold_dn, reset_frame_path	Reduce kTC & 1/f noise
photometric_extraction	aperture, radius_px, bkg_annulus_px, method=sum	Extract photometry
trace_normalization	reference_window, epsilon	Normalize per trace
phase_alignment	`method=xcorr	template, max_shift`


⸻

5) Schema, Safety & Integrity Gates

Fail-fast checks:
	•	Schema
FGS1: accept (N, T_fgs1, 32) (legacy (N,32,32) auto-transpose).
AIRS: accept (N, T_airs, 283) or (N, T_airs, 356) → remap 356→283 if instrument.bin_remap.enabled.
	•	Instrument remap
Validates map shape (283,), indices monotonic within source range, no dupes when required.
	•	Numeric safety
Post-preproc ensure σ∈[min_sigma, max_sigma], μ within conservative flux bounds, non-negativity where applicable.
	•	Runtime consistency
DVC remote & writable paths (nominal), GPU VRAM floor, Kaggle time budget (kaggle), CI time (debug/toy).
	•	Splits
Fractions sum≈1, source splits/*.npy preferred; export mirrors under ${paths.processed_dir}/splits.

⸻

6) Integration Points
	•	Hydra CLI
Swap datasets with data=<nominal|kaggle|debug|toy>:

spectramind train data=kaggle
spectramind diagnose data=nominal diagnostics.fft_analysis=true
spectramind train data=debug training.epochs=1
spectramind train data=toy loader.batch_size=8


	•	Dashboards/Reports
generate_html_report.py embeds data snapshot (paths, method flags, bin map hash, config hash).
	•	Self-Test
selftest.py asserts: paths exist, schema match, remap integrity, non-negativity after preproc, split indices ok.

⸻

7) Runtime Modes — Quick Reference

Mode	IO Roots	Target Hardware	Diagnostics	Expected Time
nominal	DVC-tracked local/cluster paths	Local/HPC GPU(s)	Rich	hours
kaggle	/kaggle/input, /kaggle/working	Kaggle P100/T4 (16 GB)	Lean	≤ 9 hr
debug	data/debug, outputs/debug	CI/CPU or any	Minimal	< 60 s
toy	data/raw/toy, data/processed	CI/CPU/GPU	Light	< 3 min


⸻

8) Adding a New Dataset Mode (Checklist)
	1.	Copy a template (debug.yaml for small; nominal.yaml for full).
	2.	Set paths to DVC or environment mounts (Kaggle). No ad-hoc absolutes.
	3.	Calibrate/Preprocess: toggle kill-chain steps; pick detrend/smoothing/jitter.
	4.	Schema/Safety: interface.num_bins=283 or provide validated bin_remap.
	5.	Splits/Loader: fractions, seed, batch_size, num_workers per environment.
	6.	Diagnostics: tune to budget (rich → research; lean → Kaggle/CI).
	7.	Register via Hydra defaults in train.yaml/predict.yaml & verify with:

spectramind train data=<new> validate.fail_fast=true



⸻

9) Common Recipes

Switch detrend method (nominal)

spectramind train data=nominal preprocessing.detrend.method=savgol

Accelerate diagnostics

spectramind diagnose data=nominal diagnostics.fft_analysis=false diagnostics.shap_overlay=false

Kaggle memory safety

spectramind train data=kaggle loader.batch_size=48 runtime.reduce_heavy_ops=true

Run CI smoke in seconds

spectramind train data=debug training.epochs=1

Toy with generator manifest checks

spectramind train data=toy validate.checks=[paths_exist,schema_match,no_nan_inf,manifest_consistency]


⸻

10) DVC Pipeline (Config → Stages)

flowchart LR
  A0{{Hydra<br/>data=nominal|kaggle|debug|toy}} --> A1[Compose cfg<br/>configs/data/*.yaml]

  subgraph S0[Versioned Sources (DVC)]
    R1[(raw_fgs1/)]:::dvc --- R2[(raw_airs/)]:::dvc
    R3[(calib_refs/)]:::dvc
  end

  subgraph P0[Data Pipeline (DVC DAG)]
    direction LR
    S1[calibrate]:::stage --> S2[preprocess]:::stage --> S3[split]:::stage --> S4[package_batches]:::stage
  end

  A1 -->|paths, flags| S1
  R1 --> S1
  R2 --> S1
  R3 --> S1

  S1 --> O1[(calibrated/)]:::dvc
  S2 --> O2[(processed/)]:::dvc
  S3 --> O3[(splits/)]:::dvc
  S4 --> O4[(batches/)]:::dvc

  O4 --> T1[[train]]:::cons
  O4 --> P1[[predict]]:::cons
  O4 --> D1[[diagnose]]:::cons

  classDef dvc fill:#e8f5e9,stroke:#2e7d32,color:#1b5e20
  classDef stage fill:#e3f2fd,stroke:#1565c0,color:#0d47a1
  classDef cons fill:#fff3e0,stroke:#ef6c00,color:#e65100

Stage Mapping

Stage	Inputs (DVC)	Hydra-bound parameters (examples)	Outputs (DVC)
calibrate	raw_*, calib_refs/	calibration.* (adc, nonlin, dark, flat, cds, photometry, phase)	calibrated/
preprocess	calibrated/	preprocessing.* (normalize, detrend, align, remap, clip)	processed/
split	processed/	splits.* (strategy, fractions, seed, export)	splits/
package_batches	splits/	loader.*, paths.*, interface.*	batches/

Hydra selection (data=<mode>) parametrizes the same DVC DAG with different IO roots & flags.

⸻

11) Validation Matrix

Check	nominal	kaggle	debug	toy
paths_exist	✅	✅	✅	✅
schema_match	✅	✅	✅	✅
bin_remap_ok (356→283)	✅	✅	✅	✅
no_nan_inf	✅	✅	✅	✅
split_indices_ok	✅	✅	✅	✅
manifest_consistency	⚙️	⚙️	⚙️	✅
gpu_memory_floor	✅	✅	–	–
kaggle_time_budget	–	✅	–	–
ci_time_budget	–	–	✅	✅

Legend: ✅ enforced • ⚙️ optional (if manifest present) • – not applicable

⸻

12) Troubleshooting
	•	Missing file on CI/Kaggle → Verify io.fail_on_missing=true and path under expected root (Kaggle: /kaggle/input/...).
	•	Shape mismatch → Check schema.accepted_shapes; enable instrument.bin_remap for AIRS 356; auto-transpose legacy FGS1 (N,32,32) if supported.
	•	OOM or slow on Kaggle → Lower loader.batch_size, set runtime.reduce_heavy_ops=true, disable heavy diagnostics.
	•	Split drift → Prefer generator’s splits/*.npy; fix splits.seed; export indices for reproducibility.
	•	Non-negativity violation → Confirm detrend and normalization order; inspect diagnostics/*_preview.png.

⸻

13) References
	•	SpectraMind V50 Technical Plan • Project Analysis • Strategy for Updating & Extending V50
	•	Hydra for AI Projects • Kaggle Platform Guide • Comparison of Kaggle Models

⸻

✅ TL;DR

/configs/data is the flight control module for SpectraMind V50 datasets.
Pick data=<nominal|kaggle|debug|toy> and the pipeline runs with mission-grade, physics-aware, reproducible behavior across local, Kaggle, and CI environments.

