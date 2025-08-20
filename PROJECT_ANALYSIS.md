
# SpectraMind V50 — Project Analysis  
*(NeurIPS 2025 Ariel Data Challenge)*

> **Purpose**: This file is a *living audit* of the ArielSensorArray / SpectraMind V50 repository.  
> It compares actual repo contents against the **engineering plan** and **external references** (Kaggle platform mechanics, competitor models), identifying what is implemented, validated, or pending.

---

## 0) Philosophy

- **CLI-first**: all operations exposed via Typer CLI, no hidden notebook state.  
- **Reproducibility**: Hydra configs, DVC data/artifacts, config + dataset hash logging.  
- **Scientific rigor**: NASA-grade calibration and physics-informed modeling.  
- **Automation**: CI/CD with self-tests and smoke pipelines on every push.  
- **Competitive fit**: designed to respect Kaggle’s runtime envelope (~9h GPU limit, quotas) [oai_citation:2‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Adaptability**: learns from competitor model archetypes (MLP baselines, deep residual nets, spectrum regressors) [oai_citation:3‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy).

---

## 1) Repository Structure

| Directory       | Status  | Notes                                                                 |
|-----------------|---------|-----------------------------------------------------------------------|
| `src/`          | ✅      | Encoders (Mamba SSM, GNN), decoders, calibration modules, CLI.        |
| `configs/`      | ✅      | Hydra group configs (`data/`, `model/`, `training/`, `diagnostics/`). |
| `data/`         | ⚠️ DVC  | Versioned via DVC, placeholders present (`.gitkeep`).                 |
| `outputs/`      | ✅      | Model checkpoints, predictions, diagnostics, logs.                    |
| `logs/`         | ✅      | `v50_debug_log.md`, JSONL streams, pytest logs.                       |
| `docs/`         | ✅      | Markdown docs, MkDocs config, diagrams (Mermaid + SVG export).        |
| `.github/`      | ✅      | CI workflow with smoke pipeline + tests.                              |

---

## 2) Configuration & Reproducibility

- Hydra v1.3 used for all runs.  
- Overrides supported via CLI:  
  ```bash
  spectramind train data=kaggle model=v50 training=default

	•	DVC v3.x integrated: dvc.yaml defines calibrate→train→predict→diagnose stages.
	•	Config + dataset + git SHA appended to v50_debug_log.md for every run.
	•	Dockerfile + Poetry lock guarantee environment reproducibility.

✅ Implemented fully.

⸻

3) CLI Design
	•	Unified entrypoint: spectramind --help.
	•	Subcommands include: selftest, calibrate, train, predict, calibrate-temp, corel-train, diagnose, submit, analyze-log, check-cli-map.
	•	Rich UX: tables, progress bars, CI-friendly error visibility.
	•	Append-only log: logs/v50_debug_log.md.

✅ Strong, production-grade CLI layer.

⸻

4) Calibration Chain

Implements full kill chain:
	•	Bias/dark subtraction
	•	Flat-fielding
	•	Nonlinearity & ADC corrections
	•	Wavelength alignment
	•	Normalization

Artifacts persisted in outputs/calibrated/.

✅ Physics-grade calibration present.

⸻

5) Modeling Architecture
	•	FGS1: Mamba SSM encoder for long lightcurve sequences.
	•	AIRS: Graph Neural Network with edge types: λ-adjacency, molecule groups, detector regions.
	•	Fusion: latent concatenation.
	•	Decoders:
	•	μ: MLP with smoothness + FFT penalties.
	•	σ: heteroscedastic head, calibrated via Temp Scaling + COREL.

✅ Implemented per design.

⸻

6) Uncertainty Quantification
	•	Aleatoric: σ predictions via GLL.
	•	Epistemic: ensemble & dropout-ready.
	•	Calibration: Temp scaling; COREL graph-based conformal.
	•	Coverage logs: JSON + plots.

⚠️ COREL symbolic weighting and temporal edges not yet fully integrated.

⸻

7) Diagnostics & Explainability
	•	UMAP & t-SNE latent visualizations.
	•	SHAP overlays (FGS1 temporal, AIRS spectral).
	•	FFT of residuals.
	•	Symbolic constraints: smoothness, nonnegativity, asymmetry, alignment.
	•	HTML dashboard: aggregates plots, overlays, logs.

✅ Implemented; symbolic overlays expanding.

⸻

8) Kaggle Platform Integration
	•	Runtime: 9h GPU/CPU sessions, ~30 GPU hrs/week ￼.
	•	Notebooks: Kaggle-API compatibility (datasets, predictions, submissions).
	•	Leaderboards: public LB (partial test split) vs private LB (final eval). Risk of “shake-up” mitigated by symbolic guardrails ￼.
	•	Submission: spectramind submit packages CSV + ZIP for Kaggle competition rules.

✅ Aligned to Kaggle infra.

⸻

9) Competitive Benchmarking (vs Kaggle Models)

Model	Strengths	Weaknesses	Lessons for V50
Thang Do Duc — 0.329 LB ￼	Simple residual MLP, fast, reproducible baseline	No uncertainty, weak domain priors	Good reference baseline; we exceed by adding physics/symbolics.
V1ctorious3010 — 80bl-128hd ￼	Very deep (~80-layer) residual MLP, high capacity	Overfit risk, heavy compute	Our Mamba/GNN fusion is leaner, physics-aligned, avoids brute-force depth.
Fawad Awan — Spectrum Regressor ￼	Multi-output PyTorch regressor, structured outputs	Limited explainability, no physics priors	V50 adds symbolic physics and uncertainty to surpass.

✅ V50 goes beyond Kaggle baselines by integrating symbolic constraints, physics priors, and calibrated uncertainty.

⸻

10) Automation & CI/CD
	•	GitHub Actions runs selftest, toy smoke pipeline.
	•	Pre-commit: ruff, black, isort, yaml, whitespace.
	•	Artifacts hashed and logged for audit.

✅ Robust CI/CD pipeline.

⸻

11) Pending / Roadmap
	•	GUI dashboard (React/FastAPI).
	•	Expanded symbolic overlays + violation heatmaps.
	•	COREL calibration expansion (temporal edges, symbolic weighting).
	•	Coverage heatmaps per-bin.
	•	Kaggle leaderboard automation with artifact gates.

⸻

12) Status Matrix

Area	Status	Notes
Repo structure	✅ Solid	Hydra/DVC clean.
CLI	✅ Complete	Typer unified, Rich UX.
Calibration	✅ Strong	Kill chain implemented.
Modeling	✅ Physics	Mamba SSM + GNN fusion.
Uncertainty	⚠️ Partial	COREL symbolic/temporal pending.
Diagnostics	✅ Active	SHAP, FFT, UMAP, symbolic overlays.
CI/CD	✅ Robust	Selftest + smoke pipelines.
Kaggle fit	✅ Aligned	Runtime & submission ready.
GUI	🚧 Planned	Thin dashboard mirror.


⸻

13) Action Items
	•	Harden COREL with symbolic priors + temporal edges.
	•	Expand symbolic overlays & violation heatmaps.
	•	Build GUI dashboard.
	•	Add Kaggle leaderboard automation job.
	•	Deepen calibration validation (coverage heatmaps).

⸻

Maintainers: SpectraMind Team
Contact: GitHub Issues
