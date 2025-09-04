🚀 SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge

ARCHITECTURE.md (root, upgraded)

Neuro-symbolic, physics-informed pipeline for ESA Ariel’s simulated data.
Design pillars: NASA-grade reproducibility, CLI-first automation, Hydra configs, DVC-tracked data, symbolic physics constraints, diagnostics by default, Kaggle-runtime discipline.

⸻

0) TL;DR (5 commands)

spectramind test
spectramind calibrate data=nominal
spectramind train model=v50 optimizer=adamw trainer.gpus=1
spectramind diagnose dashboard --open
spectramind submit --selftest

Artifacts are stamped with config hash, git commit, DVC rev, and an env SBOM.

⸻

1) System Overview

SpectraMind V50 predicts exoplanet transmission spectra — mean μ and uncertainty σ over 283 wavelength bins — from raw FGS1 photometry and AIRS spectroscopy. The end-to-end pipeline is automated via Typer CLI (spectramind …), configured with Hydra (YAML groups), and emits deterministic logs, manifests, and dashboards.

1.1 High-Level Dataflow

flowchart TD
  U[User] -->|spectramind ...| CLI[Typer CLI]
  CLI --> CFG["Hydra Compose<br/>configs/* + overrides"]
  CFG --> ORCH["Pipeline Orchestrator"]
  ORCH --> CAL["Calibration<br/>FGS1/AIRS preprocessing (DVC)"]
  CAL --> ENC["Encoders<br/>FGS1→Mamba SSM · AIRS→GNN"]
  ENC --> DEC["Decoders → μ & σ"]
  DEC --> CALIB["Uncertainty Calibration<br/>Temp scaling · COREL · Conformal"]
  CALIB --> DIAG["Diagnostics & Explainability<br/>GLL · FFT · UMAP · t-SNE · SHAP · Symbolic"]
  DIAG --> BUNDLE["Submission Bundler<br/>selftest · manifest · zip"]
  BUNDLE --> KAG["Kaggle Leaderboard"]

  CFG -. "resolved_config.yaml + hash" .-> ART["Artifacts<br/>outputs/YYYY-MM-DD/HH-MM-SS"]
  CAL -. "DVC stages" .-> ART
  ENC -. "checkpoints / metrics" .-> ART
  DIAG -. "HTML / PNG / JSON" .-> ART
  BUNDLE -. "submission.zip + manifest" .-> ART


⸻

2) Core Components (Contract-first)

2.1 CLI (Typer)

Subcommands: train, predict, calibrate, diagnose, submit, test, ablate, analyze-log, corel-train.
Global flags: --dry-run, --debug, --config-path, --version, --seed, --multirun.
Telemetry: appends human log to v50_debug_log.md and machine-log to events.jsonl.

2.2 Config (Hydra)

Groups: data/, model/, train/, diagnostics/, submit/, profiles/, runtime/.
Guarantees: every param has a default; the resolved config is persisted and hashed.

Minimal example:

# configs/model/v50.yaml
defaults:
  - _self_
  - data: nominal
  - train: standard
  - diagnostics: default
  - submit: kaggle

model:
  encoder:
    fgs1:
      name: mamba
      d_model: 256
      d_state: 64
      n_layers: 8
      dropout: 0.1
    airs:
      name: gnn_gat
      d_node: 192
      n_layers: 6
      heads: 4
      edge_features: [Δλ, molecule_region, detector_segment]
  decoder:
    mu:
      hidden: [256, 256]
    sigma:
      hidden: [256]
      min_log_sigma: -4.0
loss:
  gll_weight: 1.0
  fft_smooth_weight: 0.1
  asymmetry_weight: 0.05
  nonneg_weight: 0.5

2.3 Data Management (DVC + optional lakeFS)
	•	Raw → calibrated cubes not committed to Git.
	•	DVC tracks inputs/intermediates/ckpts; remotes: S3/GCS/SSH supported.
	•	Every dashboard prints DVC rev + remote URL hint.

2.4 Modeling (FGS1 + AIRS + decoders)
	•	FGS1 encoder: Mamba SSM for long photometric sequences; jitter augmentation; optional transit-shape conditioning.
	•	AIRS encoder: GNN (e.g., GAT/NNConv) with edges by wavelength proximity, molecular region, detector segment; positional encodings in λ; optional edge features.
	•	Decoders: dual heads for μ and σ; physics-aware regularization; symbolic fusion.
	•	Loss: GLL + FFT smoothness + asymmetry + non-negativity + symbolic logic. Curriculum toggles by stage.

2.5 Uncertainty Calibration

Temperature scaling → COREL bin-wise conformalization → instance-level tuning.
Outputs: coverage curves, per-bin heatmaps, Δcoverage CSV.

2.6 Diagnostics & Explainability

GLL heatmaps; FFT/autocorr; UMAP/t-SNE latents; SHAP overlays; symbolic rule tables; calibration checks; unified HTML dashboard (diagnostic_report_vN.html).

2.7 CI/CD (GitHub Actions)

Lint + type check, unit & CLI tests, security scans, SBOM, docs/diagrams, container build, release assets (optional Kaggle packaging).

⸻

3) Explicit I/O Contracts

3.1 Inputs
	•	FGS1: float32 time series, shape [T, 1] (pre-bin); after binning → [T_binned, 1], normalized.
	•	AIRS: float32 spectral frames, shape [time, rows=32, cols=356]; pipeline reduces to wavelength features per planet.

3.2 Outputs
	•	μ: float32[283] predicted mean spectrum.
	•	σ: float32[283] predictive std (post-calibration).
	•	Submission CSV: columns: planet_id,int, bin_id,int (0..282), mu,float, sigma,float.

Submission schema (JSON Schema gist):

{
  "type":"object",
  "properties":{
    "planet_id":{"type":"integer","minimum":0},
    "bin_id":{"type":"integer","minimum":0,"maximum":282},
    "mu":{"type":"number"},
    "sigma":{"type":"number","exclusiveMinimum":0}
  },
  "required":["planet_id","bin_id","mu","sigma"]
}


⸻

4) Reproducibility Stack

flowchart TD
  H[Hydra YAMLs]-->R1[resolved_config.yaml + hash]
  D[DVC]-->R2[data/model lineage]
  M[MLflow opt.]-->R3[metrics/artifacts]
  L[CLI logs]-->R4[v50_debug_log.md + events.jsonl]
  E[Env locks]-->R5[Poetry/Docker SBOM]
  C[CI workflows]-->R6[status badges]
  R1-->G[Guaranteed Replay]
  R2-->G
  R4-->G
  R5-->G

Run hash bundle: run_hash_summary_v50.json includes {git_commit, config_hash, dvc_rev, docker_image, poetry_lock_hash, cuda/torch_versions}.

⸻

5) Calibration & DVC Stages

Stage sketch (optional):

# dvc.yaml
stages:
  calibrate_fgs1:
    cmd: spectramind calibrate stage=fgs1
    deps: [data/raw/fgs1, configs/data/fgs1.yaml]
    outs: [data/cal/fgs1]
  calibrate_airs:
    cmd: spectramind calibrate stage=airs
    deps: [data/raw/airs, configs/data/airs.yaml]
    outs: [data/cal/airs]
  features:
    cmd: spectramind calibrate stage=features
    deps: [data/cal/fgs1, data/cal/airs, configs/data/features.yaml]
    outs: [data/features]

Kaggle constraint: ≤ 9 hours across ~1,100 planets on the provided GPU; stages degrade gracefully on CPU for CI smoke.

⸻

6) Training & Ablation

flowchart TD
  A0["spectramind ablate"] --> A1["Hydra Multirun"]
  A1 --> A2["Ablation Engine (grid · random · smart)"]
  A2 --> A3["Runner: train → diagnose"]
  A3 --> A4["Collector: metrics.json"]
  A4 --> A5["Leaderboard md/html/csv"]
  A5 -->|Top-N| A6["Bundle zip (+manifests)"]

Primary metrics: GLL (leaderboard), RMSE/MAE (dev), entropy, violation norms, FFT power, coverage deltas, runtime.

⸻

7) Symbolic Logic & Physics Constraints
	•	Smoothness / TV / FFT in wavelength space
	•	Non-negativity on μ and σ bounds
	•	Molecular coherence per region (CH₄, H₂O, CO₂ bands, etc.)
	•	Photonic alignment: FGS1 transit shape guides AIRS alignment + augmentation
Violations are rendered as heatmaps/tables and fed into targeted re-training or calibration.

⸻

8) Directory Layout (Top Level)

.
├─ configs/              # Hydra groups: data/, model/, train/, diagnostics/, submit/, profiles/, runtime/
├─ src/                  # Pipeline code: data/, models/, calibrate/, cli/, diagnostics/, utils/
├─ tests/                # Unit + CLI tests (fast/deep); submission/shape/selftest
├─ docs/                 # This file, READMEs, diagrams, GUI docs
├─ artifacts/            # Versioned outputs: html/png/json, logs, leaderboards
├─ .github/workflows/    # CI (lint/test/sbom/build/docs/release)
├─ pyproject.toml        # Poetry; pinned deps; tool configs (ruff/mypy/pytest/etc.)
├─ dvc.yaml              # Optional DVC stages for calibration/cache
└─ v50_debug_log.md      # CLI call log (human-readable), plus events.jsonl


⸻

9) Command Matrix

Purpose	Command	Notes
Smoke check	spectramind test	Fast/Deep modes; validates paths, shapes, config, submission template
Calibration	spectramind calibrate data=nominal	Writes DVC-tracked calibrated cubes and logs
Train	spectramind train model=v50 trainer.gpus=1	AMP, cosine LR, curriculum stages
Predict	spectramind predict ckpt=… outputs.dir=…	μ, σ tensors + CSV/NPZ export
Diagnose	spectramind diagnose dashboard --open	UMAP/t-SNE/SHAP/FFT/calibration HTML dashboard
COREL	spectramind corel-train	Bin-wise conformal σ calibration and coverage plots
Ablate	spectramind ablate +grid=…	Multirun w/ leaderboard (HTML/MD/CSV)
Submit	spectramind submit --selftest	Validates, packs submission.zip, writes manifest
Log Analysis	spectramind analyze-log --md --csv	Summarizes v50_debug_log.md with hash groups


⸻

10) Artifacts & Manifests

Per-run: outputs/<date>/<time>/
	•	resolved_config.yaml, run_hash_summary_v50.json
	•	metrics.json, gll_heatmap.png, fft_*, umap.html, tsne.html
	•	shap_overlay/*.png|json, symbolic/*.json|html
	•	calibration/*.png|json, corel/*.csv
	•	diagnostic_report_vN.html
	•	submission/manifest.json, submission.zip

Global: v50_debug_log.md, events.jsonl

Example submission/manifest.json:

{
  "version": "v50.0.0",
  "git_commit": "abcd1234",
  "config_hash": "e3b0c44298...",
  "dvc_rev": "2f7c9a...",
  "docker_image": "ghcr.io/.../spectramind:v50",
  "generated_at": "2025-09-04T12:00:00Z",
  "files": ["submission.csv"],
  "metrics": {"gll_dev": -1.234, "coverage@95": 0.948}
}


⸻

11) Runtime & Performance Targets
	•	Kaggle GPU runtime: ≤ 9h (full pipeline, ~1,100 planets).
	•	CPU CI smoke: ≤ 20 min (toy subset, reduced dims).
	•	Memory: keep per-stage < 10 GB on GPU; streaming loaders for AIRS.
	•	Determinism: PYTHONHASHSEED=0, single-threaded BLAS in CI (OMP/MKL/OPENBLAS=1), seeded dataloaders.

⸻

12) Security, Ethics, Governance
	•	No private data; source licensing respected.
	•	SBOM generated in CI; pip-audit/trivy/CodeQL scans.
	•	Reproducibility: all runs have config/env/data lineage.
	•	Optional role-based API gateway for dashboards (no analytics logic server-side).

⸻

13) Troubleshooting (Quick)
	•	Hydra composition fails: check defaults: order; debug with --cfg job --package _global_.
	•	DVC missing: dvc pull w/ remote creds; verify .dvc/config.
	•	Submission invalid: spectramind submit --selftest to emit diff vs schema.
	•	Empty dashboard: enable diagnostics; ensure generate_html_report.py paths.
	•	CUDA wheels mismatch: set TORCH_WHL_INDEX or pin torch/vision/audio.

⸻

14) Extensibility Hooks
	•	Encoders: swap mamba/transformer for FGS1; gat/nnconv/rgcn for AIRS.
	•	Symbolic engine: rule weights, soft/hard modes, per-rule loss maps, entropy overlays.
	•	Calibration: plug new conformal/quantile methods; toggle molecule-region weighting.
	•	Diagnostics: inject new plots into the HTML dashboard via generate_html_report.py.

⸻

15) Compose & Container Notes

Compose (GPU):

version: "3.9"
services:
  smv50:
    image: spectramind:gpu
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: ["gpu"]
    volumes:
      - ./:/workspace
    command: ["spectramind","train","--config-name=config_v50.yaml"]

Docker build examples:

# GPU
docker build -t spectramind:gpu .
# CPU
docker build -t spectramind:cpu --build-arg BASE_IMAGE=python:3.11-slim .


⸻

16) Glossary

FGS1: Fine Guidance Sensor photometry (time series).
AIRS: Ariel IR Spectrometer (wavelength × time frames).
μ, σ: Predicted mean spectrum & predictive uncertainty.
GLL: Gaussian Log-Likelihood (leaderboard metric).
COREL: Conformal calibration for bin-wise σ coverage.

⸻

17) Roadmap (select)
	•	💡 AIRS edge-feature GNN (distance/molecule/segment) with attention export
	•	💡 Symbolic influence maps (∂L/∂μ per rule) integrated into dashboard
	•	💡 Ablation leaderboard: HTML + Markdown with top-N ZIP export
	•	💡 UMAP/t-SNE 3D with confidence-weighted opacity and planet links
	•	💡 Config hash heatmaps from CLI logs for usage analytics

⸻

End of ARCHITECTURE.md
