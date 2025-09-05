🔬 Calibration Configs — SpectraMind V50 (upgraded)

Location: configs/calib/
Project: SpectraMind V50 · NeurIPS 2025 Ariel Data Challenge

Calibration is the first stage of the SpectraMind V50 pipeline, transforming raw Ariel-simulated frames into physically corrected, science-ready light curves. This folder contains profile-level chain configs (e.g., nominal.yaml, fast.yaml, strict.yaml) that compose the method-level tools in this directory (adc.yaml, dark.yaml, flat.yaml, cds.yaml, photometry.yaml, trace.yaml, phase.yaml).

Note: Some docs refer to a method/ subfolder. In V50 we flatten these to configs/calib/*.yaml and compose via Hydra defaults.

⸻

🧭 What calibration does

Kill chain (ordered):
	1.	ADC correction — offset & gain(PRNU), reference pixels/overscan, linearization
	2.	Nonlinearity — LUT/poly linearization (if not done in ADC)
	3.	Dark subtraction — master or per-pixel map, exposure/temperature scaling, hot-pixel repair
	4.	Flat fielding — PRNU & illumination normalization (master or spectral flats)
	5.	CDS — RESET/SIGNAL differencing, robust pairing, outlier suppression
	6.	Photometric extraction — aperture / psf / optimal profile extraction
	7.	Trace normalization — continuum flattening + temporal detrend
	8.	Phase alignment — ephemeris fold, in/out-of-transit windows, binning

Each stage is configured by one YAML in this folder and chained by the profile files.

flowchart LR
  A[raw_frames] --> B[ADC<br/>adc.yaml]
  B --> C[Dark<br/>dark.yaml]
  C --> D[Flat<br/>flat.yaml]
  D --> E[CDS<br/>cds.yaml]
  E --> F[Photometry<br/>photometry.yaml]
  F --> G[Trace Norm<br/>trace.yaml]
  G --> H[Phase Align<br/>phase.yaml]
  H --> I[μ(t), σ(t) & binned products]


⸻

📦 Files in this folder

File	Purpose	Typical use
nominal.yaml	Default leaderboard-safe calibration chain (≤ 9 h). All key steps enabled; saves light diagnostics.	Daily development & Kaggle submissions.
fast.yaml	Lightweight chain for CI/smoke. Skips heavy steps or uses cheaper approximations.	PR checks, pipeline smoke tests, quick repro.
strict.yaml	Full science-grade chain with extra diagnostics & stricter QC.	Offline validation, audits; may exceed Kaggle limits.
Stage YAMLs	adc.yaml, dark.yaml, flat.yaml, cds.yaml, photometry.yaml, trace.yaml, phase.yaml	Tunable method configs; composed by profiles.
(optional) top_level_calib.yaml	Uncertainty calibration (temperature/COREL) for σ/μ after inference	Post-inference calibration workflows.

A profile composes stages via Hydra:

# (Example) configs/calib/nominal.yaml
defaults:
  - adc
  - dark
  - flat
  - cds
  - photometry
  - trace
  - phase
# optional post-inference (uncertainty calibration)
# - top_level_calib


⸻

🚀 How to run

Run default (nominal) calibration

spectramind calibrate --config-name calib/nominal

Switch to fast (CI smoke)

spectramind calibrate --config-name calib/fast

Run strict (offline science validation)

spectramind calibrate --config-name calib/strict

Inspect the fully composed config (audit)

spectramind calibrate --config-name calib/nominal --cfg job --resolve


⸻

🧩 Override cheat-sheet (Hydra)

Aperture photometry with radius = 8 px

spectramind calibrate \
  --config-name calib/nominal \
  calib.photometry.photometry.method=aperture \
  calib.photometry.aperture.radius_px=8

Spline trace normalization (12 knots)

spectramind calibrate \
  --config-name calib/nominal \
  calib.trace.along_wavelength.method=spline \
  calib.trace.along_wavelength.spline.knots=12

CDS pairing by nearest timestamps within 3 s

spectramind calibrate \
  --config-name calib/nominal \
  calib.cds.pairing.order_by=timestamp \
  calib.cds.pairing.policy=nearest \
  calib.cds.pairing.window_s=3.0

Switch instruments (FGS1 broadband)

spectramind calibrate \
  --config-name calib/nominal \
  calib.photometry.instrument.name=FGS1 \
  calib.photometry.instrument.per_channel=false \
  calib.trace.instrument.name=FGS1


⸻

🔩 I/O contract (common defaults)

Stage	input_key	output_key	Shape(s)
ADC	raw_frames	adc_corrected	[B,H,W] or [B,C,H,W]
Dark	adc_corrected	dark_corrected	[B,H,W] or [B,C,H,W]
Flat	dark_corrected	flat_corrected	[B,H,W] or [B,C,H,W]
CDS	flat_corrected	cds_corrected	[B,H,W] or [B,C,H,W]
Photometry	cds_corrected	photometry	Flux [B] or [B,C] (+ meta)
Trace	photometry	trace_norm	Flux [B] or [B,C]
Phase	trace_norm	phase_fold	Phase [B]; binned [M]/[M,C]

Method files define these keys precisely in their io blocks.

⸻

✅ QC & validation (what to enable where)
	•	validation.require_existing_paths=true — development & CI; fail fast if artifacts absent.
	•	validation.check_shape_compatibility=true — catch map/LUT mismatches early.
	•	quality_checks.* — per-stage soft guards (histograms, stats bounds, pairing integrity).
	•	assert_finite_output=true (ADC/flat/dark/CDS) — ensure no NaN/Inf propagate.
	•	Watch histograms for zero/saturation pile-ups after ADC/flat; tune clip & gains.

⸻

📊 DVC integration

Calibration is a DVC-tracked stage for reproducibility:
	•	Inputs: raw FGS1/AIRS frames + profile config + method artifacts (masters/LUTs/maps).
	•	Outputs: calibrated cubes/series under ${RUN_DIR} (and optionally data/processed/<profile>/calibrated).
	•	Behavior: if inputs + config hashes are unchanged, DVC skips re-run via cache.

Ensure large artifacts (master darks, flats, masks, LUTs) are DVC dependencies with stable paths.

⸻

⚙️ Performance tips
	•	CI & quick iteration: calib/fast.
	•	Disable heavy diagnostics: *.diagnostics.enabled=false, save_debug_images=false.
	•	Reduce QC sampling: quality_checks.sampling_fraction=0.05.
	•	Prefer median-based stats for robustness; avoid deep spline fits in smoke runs.
	•	For Kaggle: keep mixed precision where appropriate downstream; calibration steps are CPU-tolerant.

⸻

🧪 Profile presets (suggested)

Profile	Target	Key toggles
fast	CI/pr-smoke	simple ADC; master dark; master flat; cds.mode=frame_diff; aperture photometry; minimal diagnostics
nominal	Leaderboard	full ADC; exposure-scaled dark; illumination-normalized flat; tuned CDS; aperture/psf per instrument; moderate diagnostics
strict	Offline science	full ADC + linearization; temperature scaling + per-pixel dark map; spectral flat; robust CDS; optimal extraction; deep diagnostics


⸻

🧰 Quick recipes

CI-friendly run

spectramind calibrate --config-name calib/fast \
  calib.photometry.photometry.method=aperture \
  calib.photometry.aperture.radius_px=6 \
  calib.trace.along_time.enabled=false \
  calib.phase.binning.nbins=120

Post-inference uncertainty calibration (optional)

spectramind calibrate --config-name calib/nominal \
  calib.top_level_calib.method=temperature \
  calib.top_level_calib.temperature.per_bin=true
# or:
# calib.top_level_calib.method=corel corel.model.backend=gatconv corel.fit.epochs=50


⸻

🧷 Notes
	•	All configs are Hydra-composable and override-friendly.
	•	Nominal is Kaggle-safe; fast is for CI; strict is for audits & high-rigor studies.
	•	Keep units consistent (DN↔e⁻ conversions via conversion_gain_e_per_adu).
	•	Prefer float32 during calibration; quantize to integers only when required downstream.
	•	Use runtime.num_workers conservatively on Kaggle to avoid CPU contention.

⸻

With this setup, calibration configs are the launchpad for every SpectraMind V50 run — delivering reproducibility, physics-informed rigor, and Kaggle-safe deployment.