
⸻

assets/

SpectraMind V50 — Ariel Data Challenge 2025
Central repository assets: diagrams, dashboards, reports, and reproducibility visuals

All assets are generated and consumed through the Typer CLI + Hydra configs (no hidden notebook state), so every figure is reproducible and traceable to code, data, and config.

⸻

📌 Purpose

This directory houses the visual artifacts used across SpectraMind V50:
	•	Architecture & pipeline Mermaid diagrams (source .mmd + rendered .svg/.png)
	•	HTML reports and the diagnostics dashboard
	•	Example/static images used in docs and Kaggle hand‑offs

Everything here is source‑tracked (Mermaid is the canonical source), auto‑exported in CI, and consumed by docs & dashboards.

⸻

📂 Layout

assets/
├─ diagrams/                 # Mermaid sources + renders
│  ├─ architecture_stack.mmd / .svg / .png
│  ├─ pipeline_overview.mmd  / .svg / .png
│  ├─ symbolic_logic_layers.mmd / .svg / .png
│  ├─ kaggle_ci_pipeline.mmd / .svg / .png
│  ├─ test_diagrams.py       # render/validate all .mmd
│  └─ README.md
├─ diagnostics_dashboard.html
├─ report.html
└─ sample_plots/             # optional PNGs (e.g., spectrum, SHAP overlay)

Why this structure: it keeps source→render close, participates in CI, and feeds docs & HTML outputs without manual steps.

⸻

🧭 What to edit vs. what not to edit
	•	Edit: *.mmd (Mermaid sources).
	•	Do not edit: generated .svg/.png. Regenerate them via the commands below or CI.

⸻

🛠 Rendering diagrams

Option A — Local (Mermaid CLI)

# 1) Install mermaid-cli
npm i -g @mermaid-js/mermaid-cli

# 2) Render one diagram
mmdc -i assets/diagrams/pipeline_overview.mmd -o assets/diagrams/pipeline_overview.svg
mmdc -i assets/diagrams/pipeline_overview.mmd -o assets/diagrams/pipeline_overview.png

# 3) Render all .mmd in the folder
for f in assets/diagrams/*.mmd; do
  base="${f%.mmd}"
  mmdc -i "$f" -o "${base}.svg"
  mmdc -i "$f" -o "${base}.png"
done

Option B — Python test harness

# Render & validate all Mermaid sources
python assets/diagrams/test_diagrams.py --render --strict
# Only some files
python assets/diagrams/test_diagrams.py --render --only pipeline_overview.mmd,symbolic_logic_layers.mmd

The harness is designed to fail fast in CI if any source can’t export cleanly.

Option C — GitHub Actions (recommended)

On pushes/PRs, the Mermaid export job runs and attaches the artifacts; renders are committed/packaged per repo policy. This keeps reviewers in the loop without asking them to install toolchains.

⸻

🧩 Diagrams that must exist (and what they show)
	•	pipeline_overview — FGS1/AIRS → Calibration → Modeling (μ/σ) → UQ → Diagnostics → Submission → Ops. Aligns with our calibrated, physics‑aware pipeline.
	•	architecture_stack — CLI (Typer) → Hydra configs → DVC/Git → Calibration → Encoders/Decoders → UQ → Diagnostics → Packaging → CI/Runtime. Mirrors the CLI‑first, Hydra‑composed system.
	•	symbolic_logic_layers — constraint families (smoothness, FFT coherence, asymmetry, molecular alignment) used as overlays and diagnostics.
	•	kaggle_ci_pipeline — GitHub Actions → Selftest → Train → Diagnose → Validate → Package → Kaggle Submit to ensure portable, leaderboard‑ready assets.

All four are embedded in docs and HTML dashboards (SVG preferred) and are part of CI checks.

⸻

📑 Embedding (docs, dashboards, Kaggle)
	•	Markdown:
![Pipeline Overview](assets/diagrams/pipeline_overview.svg)
	•	HTML:
<img src="assets/diagrams/pipeline_overview.svg" alt="Pipeline Overview" />
	•	Prefer SVG for clarity; keep PNG only for environments that can’t inline SVG (some viewers/Kaggle).

⸻

🔁 Reproducibility hooks (how assets stay auditable)
	•	CLI‑first: all generation is done through Typer subcommands + Hydra configs — no hidden notebook state.
	•	Hydra configs captured with each run; config + data hash land in logs so any figure can be traced to code+config+data.
	•	DVC ties large artifacts (data/models) to Git commits to reproduce figures exactly from historical states.
	•	CI renders diagrams and runs a pipeline sanity pass on sample data to ensure nothing regresses before merge.

⸻

🧪 Quick‑start commands (developer workstation)

Render diagrams + run diagnostics end‑to‑end:

# Render all diagrams locally
python assets/diagrams/test_diagrams.py --render --strict

# End-to-end smoke on a tiny slice (example; actual CLI shown here)
spectramind calibrate --sample 5
spectramind train --epochs 1 --fast_dev_run
spectramind diagnose

The CLI uses Hydra overrides (e.g., trainer=ddp or training.epochs=20) to keep experiments code‑free and repeatable.

⸻

🧪 Diagram test (assets/diagrams/test_diagrams.py)
	•	--render: build .svg/.png for each .mmd.
	•	--strict: fail the run on any Mermaid warnings/errors.
	•	--only: comma‑separated subset.
Use this both locally and in CI to catch broken sources early.

⸻

📊 Kaggle model insights (why diagrams look the way they do)

Our diagram flow reflects lessons from public Kaggle baselines:
	•	Residual MLP baselines (~0.329 LB) offer reproducible starting points.
	•	Very deep residual MLPs (80 blocks) squeeze extra signal but risk overfitting without strong detrending.
	•	Multi‑output regressors provide stable full‑spectrum predictions and are easy to operationalize.

These observations are “baked” into the architecture_stack and pipeline_overview diagrams to document why our pipeline emphasizes calibration, physics‑informed features, μ/σ prediction, and downstream validation.

⸻

🧪 Style guide (Mermaid)
	•	Graph direction: top‑down (TD) unless inherently left‑to‑right.
	•	Subgraphs: group stages (Calibration, Modeling, Diagnostics).
	•	Node text: concise; longer prose belongs in docs.
	•	No external links inside nodes.
	•	Stable IDs so SVG diffs are readable.
	•	Use defaults for theme/colors to keep CI consistent.

Keep arrows readable; prefer subgraphs and labels over spaghetti connectors.

⸻

🧷 Maintenance checklist (per PR)
	•	Only .mmd edited; no manual edits to .svg/.png.
	•	python assets/diagrams/test_diagrams.py --render --strict passes locally.
	•	Renders are readable at 100% and wired into docs/HTML.
	•	Large diffs? Consider splitting a diagram.

⸻

🔐 Source of truth & traceability
	•	Configs: configs/**.yaml (Hydra), committed.
	•	Data/Models: tracked by DVC remotes and tied to Git commits.
	•	Logs: console + structured logs with config & dataset hashes.
	•	CI: GitHub Actions runs unit/self‑tests and diagram export before merge.

⸻

📦 Packaging & handoff
	•	Reports (report.html, diagnostics_dashboard.html) bundle diagrams (SVG preferred) and JSON metrics; these are exported as artifacts in CI and included in Kaggle submissions when required.
	•	The kaggle_ci_pipeline diagram documents the path from GitHub Actions to a reproducible Kaggle artifact.

⸻

🔎 References
	•	CLI + Hydra + DVC + CI reproducibility loop.
	•	Diagram policy and rendering harness.
	•	Kaggle platform practices & constraints.

⸻

End of assets/README.md