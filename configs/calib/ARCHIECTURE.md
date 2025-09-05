🧭 configs/calib/architecture.md

SpectraMind V50 · NeurIPS 2025 Ariel Data Challenge
Document type: Architecture & schema reference for profile-level calibration chains
Version: v1.4 (upgraded)

⸻

0) Purpose

This document defines the architecture, composition rules, schema, and I/O contracts for the profile-level calibration configs in configs/calib/ (e.g., nominal.yaml, fast.yaml, strict.yaml).
Profiles compose method YAMLs (ADC, dark, flat, CDS, photometry, trace, phase, optional COREL) into a reproducible kill chain that turns raw frames into science-ready light curves.
All configs are Hydra-composable, override-friendly, and DVC-tracked for reproducibility.

⸻

1) Scope & placement

configs/
└─ calib/
   ├─ nominal.yaml           # default leaderboard-safe chain (≤ 9h)
   ├─ fast.yaml              # CI/smoke-friendly chain
   ├─ strict.yaml            # full diagnostics/audit chain
   ├─ architecture.md        # (this file)
   ├─ adc.yaml               # method configs (atomic; one per stage)
   ├─ dark.yaml
   ├─ flat.yaml
   ├─ cds.yaml
   ├─ photometry.yaml
   ├─ trace.yaml
   ├─ phase.yaml
   └─ top_level_calib.yaml   # optional post-inference σ-calibration (temperature/COREL)

Rules
	•	Each method file is atomic (one stage per file) and owns its io, validation, quality_checks, logging, and cache sections.
	•	A profile is a Hydra config that lists method files in execution order and may add overrides.

⸻

2) Kill-chain overview

flowchart LR
  A[ADC] --> B[Dark]
  B --> C[Flat]
  C --> D[CDS]
  D --> E[Photometry]
  E --> F[Trace normalization]
  F --> G[Phase alignment]
  G -. optional .-> H[Top-level σ calibration<br/>(Temperature / COREL)]

	•	ADC: offset/gain(PRNU), ref pixels/overscan cleanup, optional linearization, clamps
	•	Dark: master or per-pixel map, exposure/temperature scaling, hot-pixel repair
	•	Flat: master or spectral flats, PRNU renormalization, optional illumination field
	•	CDS: RESET/SIGNAL pairing, robust differencing, optional temporal filters
	•	Photometry: aperture | psf | optimal extraction; background & centroiding
	•	Trace: along-wavelength continuum removal + along-time detrend; robust masks & QC
	•	Phase: ephemeris fold, windows, binning; flags & coverage checks
	•	Top-level calibration (optional): temperature scaling or COREL GNN for σ (and optional μ post-scale)

⸻

3) Profile schema

Every configs/calib/<profile>.yaml follows:

# configs/calib/<profile>.yaml
defaults:
  - adc
  - dark
  - flat
  - cds
  - photometry
  - trace
  - phase
  # - top_level_calib   # optional (post-inference σ calibration)

# Optional: centralized overrides/toggles
calib:
  method:
    adc.enabled: true
    dark.enabled: true
    flat.enabled: true
    cds.enabled: true
    photometry.enabled: true
    trace.enabled: true
    phase.enabled: true
    # top_level_calib.method: temperature  # or: corel

Composition rules
	•	Order in defaults == execution order.
	•	Stage I/O keys must chain: each stage’s io.output_key becomes the next stage’s io.input_key.
	•	Profiles may override any nested key via Hydra dot-paths (see §7).

⸻

4) Stage interfaces (I/O contracts)

Stage	input_key	output_key	Typical shapes
ADC	raw_frames	adc_corrected	[B,H,W] or [B,C,H,W]
Dark	adc_corrected	dark_corrected	[B,H,W] or [B,C,H,W]
Flat	dark_corrected	flat_corrected	[B,H,W] or [B,C,H,W]
CDS	flat_corrected	cds_corrected	[B,H,W] or [B,C,H,W]
Photometry	cds_corrected	photometry	Flux [B] or [B,C] (+ ancillary)
Trace	photometry	trace_norm	Flux [B] or [B,C]
Phase	trace_norm	phase_fold	Phase [B]; binned [M] / [M,C]
σ-Calib*	preds_pt	preds_calibrated.pt	μ/σ [N,B]; metrics JSON

Conventions (key fields)
	•	Photometry: photometry_flux, photometry_flux_var, photometry_sky, photometry_centroid_x|y, photometry_flags
	•	Trace: trace_flux_norm, trace_flux_scale, trace_flags, channel_grid
	•	Phase: phase_series, binned_phase, binned_flux, binned_flux_err, windows, phase_flags

Keys & shapes are defined in each method YAML’s io block and enforced by validation.

⸻

5) Profiles matrix

Aspect	fast	nominal	strict
Purpose	CI/smoke & dev	Kaggle leaderboard	Full science audit
Diagnostics	minimal	light	extensive
ADC	basic offset/gain	ref-pixel/overscan as needed	full + LUT/linearization
Dark	single master	master + exposure scaling + hot repair	per-pixel map + temp scaling
Flat	master only (optional illum)	master/spectral + illum corr	spectral + per-channel illum
CDS	simple frame diff	pairing (timestamp/index), robust	full pairing + temporal filters
Photometry	aperture small radius	aperture/psf per instrument	optimal/psf + full variance model
Trace	low-order poly	poly/spline as data require	spline/LOWESS + masks
Phase	coarse bins	balanced bins	fine bins + strict QC
Runtime	shortest	≤ 9 h safe	longest


⸻

6) Composition patterns (skeletons)

Nominal

defaults:
  - adc
  - dark
  - flat
  - cds
  - photometry
  - trace
  - phase
calib.method:
  photometry.photometry.method: aperture
  photometry.aperture.radius_px: 8
  trace.along_wavelength.method: poly
  phase.binning.nbins: 200

Fast

defaults:
  - adc
  - photometry
  - trace
  - phase
calib.method:
  adc.diagnostics.enabled: false
  photometry.photometry.method: aperture
  photometry.aperture.radius_px: 6
  trace.along_time.enabled: false
  phase.binning.nbins: 120

Strict

defaults:
  - adc
  - dark
  - flat
  - cds
  - photometry
  - trace
  - phase
calib.method:
  adc.reference_pixel.use: true
  dark.per_pixel_map.convert_units.enabled: true
  flat.illumination_correction.enabled: true
  cds.temporal_filter.enabled: true
  photometry.photometry.method: optimal
  trace.along_wavelength.method: spline
  trace.along_wavelength.spline.knots: 16
  phase.binning.nbins: 400


⸻

7) Hydra overrides — quick recipes

Nearest-timestamp CDS within 3 s

spectramind calibrate --config-name calib/nominal \
  calib.cds.pairing.order_by=timestamp \
  calib.cds.pairing.policy=nearest \
  calib.cds.pairing.window_s=3.0

Switch photometry to PSF fit

spectramind calibrate --config-name calib/nominal \
  calib.photometry.photometry.method=psf \
  calib.photometry.psf.model=gaussian_moffat

Spline continuum with 12 knots

spectramind calibrate --config-name calib/nominal \
  calib.trace.along_wavelength.method=spline \
  calib.trace.along_wavelength.spline.knots=12

Enable temperature σ-calibration per bin

spectramind calibrate --config-name calib/nominal \
  calib.top_level_calib.method=temperature \
  calib.top_level_calib.temperature.per_bin=true


⸻

8) Validation strategy (per stage)
	•	Files: validation.require_existing_paths=true when a stage references LUTs/maps/masters.
	•	Shapes: validation.check_shape_compatibility=true (enforce (C,H,W) alignment).
	•	Metadata: enforce keys where needed (CDS: timestamps/reset flags; Phase: ephemeris).
	•	Numerics: require_finite_inputs=true at each stage; clamp/epsilon guards enabled by default.

Common failure hints
	•	ADC negative flood → verify offset_correction order and post_subtraction_clamp (dark stage).
	•	Dark residual banding → enable per_pixel_map or check exposure/temperature units.
	•	Flat explosion → ensure renormalize_to_mean_one=true and PRNU epsilon sane.
	•	CDS unpaired → adjust pairing.policy/window_s or provide is_reset flags.
	•	Phase mis-center → confirm period/t0 and time units (BJD_TDB vs MJD/UNIX).

⸻

9) DVC & caching
	•	Each stage can emit intermediates to ${RUN_DIR}/calib/<stage> for provenance.
	•	Large artifacts (masters, flats, masks, LUTs) should be DVC dependencies.
	•	Inputs + config hashes unchanged ⇒ DVC cache hit (skip compute).

Recommended outputs

${RUN_DIR}/calib/adc
${RUN_DIR}/calib/dark
${RUN_DIR}/calib/flat
${RUN_DIR}/calib/cds
${RUN_DIR}/calib/photometry
${RUN_DIR}/calib/trace
${RUN_DIR}/calib/phase


⸻

10) Diagnostics & QC

Enable selectively to remain Kaggle-safe:
	•	diagnostics.enabled=false for images; keep sample sizes tiny in fast/nominal.
	•	Use QC gates as soft guards in sweeps (action_on_fail: "warn") and make strict for audits.

Suggested quick QC set (nominal)
	•	ADC/Flat histograms: zero/saturation pile-ups
	•	CDS pairing integrity: unpaired fraction < 10%
	•	Photometry flux stats: robust std sanity
	•	Trace stats: post-norm median ~1, invalid frac < 1%

⸻

11) Performance guidance
	•	Prefer float32 within calibration; quantize to integers only if required downstream.
	•	Use median over mean for robustness (dark combine/background).
	•	Avoid heavy spline grids in fast profile (reduce knots/grid, disable deep robust loops).
	•	Tune runtime.num_workers conservatively on Kaggle.

⸻

12) Reproducibility & logging
	•	Log the composed config (--cfg job --resolve) and a run hash (e.g., ${RUN_DIR}/run_hash_summary_v50.json).
	•	Record method artifact paths & checksums (e.g., master dark/flat, LUTs, masks).
	•	Append a one-line summary to logs/v50_debug_log.md where possible.

⸻

13) Testing checklist
	•	All required files exist (maps/LUTs/masters).
	•	Shapes compatible across stages.
	•	Required metadata present (CDS pairing, Phase ephemeris).
	•	No NaN/Inf after each stage.
	•	QC gates pass or warn with clear diagnostics.
	•	CI smoke (calib/fast) finishes within budget.

⸻

14) Glossary

ADC — Analog-to-Digital Conversion stage (bias/PRNU/linearization)
PRNU — Photo-Response Non-Uniformity (pixel/channel gain variations)
CDS — Correlated Double Sampling (RESET/SIGNAL differencing)
SNR — Signal-to-Noise Ratio (often μ/σ)
OOT — Out-Of-Transit (used for normalization & QC)

⸻

15) Changelog (profiles)
	•	v1.4 — Unified top-level σ-calibration entry; expanded CDS pairing & diagnostics; revised I/O contracts.
	•	v1.3 — Strict adds per-amp ADC LUTs, dark residual plane, spline-trace; Phase coverage checks.
	•	v1.2 — Nominal refines CDS pairing & Photometry defaults; Fast trims trace/diagnostics further.
	•	v1.1 — Initial profile set (fast/nominal/strict), base chain & Hydra overrides.

⸻

16) Quick-start

# Nominal (Kaggle-ready)
spectramind calibrate --config-name calib/nominal

# Fast (CI/smoke)
spectramind calibrate --config-name calib/fast

# Strict (science audit)
spectramind calibrate --config-name calib/strict

With these conventions, configs/calib/ defines robust, Hydra-composable calibration chains that remain reproducible, physics-informed, and Kaggle-safe.