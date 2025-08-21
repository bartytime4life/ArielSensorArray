SpectraMind V50 — assets/diagrams/ README

Mission-grade, reproducible diagrams for the ArielSensorArray repository.
This folder contains source diagrams (Mermaid .mmd) and rendered assets (.svg/.png) used across:
	•	assets/report.html and assets/diagnostics_dashboard.html
	•	README.md, ARCHITECTURE.md, and docs
	•	CI artifacts and Kaggle submission dashboards

⸻

Why this directory exists
	•	Single source of truth for all project diagrams (kept under version control).
	•	Reproducible builds: every diagram renders the same in CI and locally.
	•	CI coverage: diagrams auto-render on PRs/merges; rendering failures break CI early.
	•	Ready-to-embed assets: SVG (preferred) and PNG for environments that don’t allow inline SVG.

⸻

What’s inside (expected)
	•	pipeline_overview.mmd / .svg / .png
High-level pipeline: FGS1/AIRS Frames → Calibration → Training → Prediction → Diagnostics → Submission.
	•	architecture_stack.mmd / .svg / .png
Layers: CLI (Typer/Hydra) → Config → Data (DVC) → Encoders/Decoders → Symbolic Logic → Diagnostics/HTML.
	•	exoplanet_observation_flow.mmd / .svg / .png
Physical context: Star → Transit → Telescope (Ariel) → Detector (FGS1/AIRS) → Frames → Pipeline.
	•	symbolic_logic_layers.mmd / .svg / .png
Symbolic constraints and overlays: smoothness, non‑negativity, asymmetry, FFT coherence, molecular alignment, diagnostics hooks.
	•	gui_integration.mmd / .svg / .png
“UI‑light” CLI + optional GUI/dashboard wiring (React/Qt/Electron options), fed by the same back end.
	•	kaggle_ci_pipeline.mmd / .svg / .png
End‑to‑end CI & Kaggle path: GitHub Actions → Selftest → Train → Predict → Validate → Package → Kaggle Submit.
	•	test_diagrams.py
Sanity test that renders every .mmd to ensure the sources are valid (can be invoked locally or in CI).

NOTE: If a given .svg/.png file is missing, render it using the instructions below. SVG is the canonical asset for docs. PNG is provided for compatibility only.

⸻

How these files are used
	•	Documentation: README.md, ARCHITECTURE.md, and docs embed the .svg versions directly.
	•	HTML dashboards: assets/report.html and assets/diagnostics_dashboard.html reference these .svg files.
	•	CI artifacts: GitHub Actions renders .mmd to .svg/.png so reviewers can see diagrams in PRs.
	•	Kaggle: When bundles include docs, these assets ride along and render offline.

⸻

Render diagrams (three supported workflows)

1) GitHub Actions (recommended, zero local setup)

If you’re on the main branch or a PR, the Mermaid Export workflow auto-renders all .mmd to .svg/.png.
Just commit your changes; the CI will attach rendered assets as artifacts and update the repo if auto‑commit is enabled.

2) Local rendering via Mermaid CLI (Node)
	1.	Install Node and Mermaid CLI:

npm i -g @mermaid-js/mermaid-cli


	2.	Render a single diagram:

mmdc -i assets/diagrams/pipeline_overview.mmd -o assets/diagrams/pipeline_overview.svg
mmdc -i assets/diagrams/pipeline_overview.mmd -o assets/diagrams/pipeline_overview.png


	3.	Render all .mmd in this folder (bash):

for f in assets/diagrams/*.mmd; do
  base="${f%.mmd}"
  mmdc -i "$f" -o "${base}.svg"
  mmdc -i "$f" -o "${base}.png"
done



3) Local rendering via Python test harness

This repo ships a helper test that can render and validate all Mermaid sources:

# Optional: create a venv and install playwright dependencies if prompted by mermaid-cli wrapper
python assets/diagrams/test_diagrams.py --render

Flags:
	•	--render — Render .svg/.png for every .mmd.
	•	--strict — Fail on any warning (useful in CI).
	•	--only pipeline_overview.mmd,symbolic_logic_layers.mmd — Limit to specific files.

The test uses @mermaid-js/mermaid-cli if available; otherwise it will instruct you how to install it.

⸻

Embedding guidelines
	•	Prefer SVG in Markdown and HTML:

![Pipeline Overview](assets/diagrams/pipeline_overview.svg)

<img src="assets/diagrams/pipeline_overview.svg" alt="Pipeline Overview" />


	•	Use PNG fallback only where SVG is not supported.
	•	Keep embeds relative paths so they render on GitHub, Kaggle, and in exported bundles.

⸻

Contribution rules (must‑follow)
	1.	Edit .mmd only. Never hand‑edit generated .svg/.png.
	2.	One concern per diagram. If a diagram grows too big, split it (e.g., pipeline_overview_part2.mmd).
	3.	Keep arrows readable. In Mermaid, avoid spaghetti connectors; use subgraphs and labels.
	4.	No external links in diagram nodes (docs will carry links; diagrams keep structure).
	5.	Run the test before pushing:

python assets/diagrams/test_diagrams.py --render --strict


	6.	Commit message should describe what changed and why, e.g.:

docs(diagrams): refine symbolic_logic_layers with FFT coherence path and COREL overlay



⸻

Style guide (Mermaid)
	•	Graph direction: top‑down TD unless the flow is inherently left‑to‑right.
	•	Subgraphs: use to group pipeline stages (e.g., Calibration, Training, Diagnostics).
	•	Node text: concise nouns/verbs; leave long prose to docs.
	•	Colors/Styling: rely on Mermaid defaults for CI consistency; avoid custom themes unless standardized.
	•	IDs: stable IDs to keep diffs readable (don’t rename IDs casually).

Example snippet:

flowchart TD
  subgraph Calibration
    A[FGS1 / AIRS Frames] --> B[Bias/Dark/Flat/CDS]
    B --> C[Trace Extraction + Normalization]
  end

  subgraph Modeling
    C --> D[Encoders (FGS1:Mamba, AIRS:GNN)]
    D --> E[Decoders (μ/σ)]
  end

  subgraph Diagnostics
    E --> F[Symbolic Overlays]
    E --> G[SHAP / UMAP / t-SNE]
  end

  Diagnostics --> H[Submission Packaging]


⸻

Troubleshooting
	•	“mmdc command not found”
Install Mermaid CLI: npm i -g @mermaid-js/mermaid-cli.
	•	Playwright/Chromium issues (first‑time setup)
Run once: npx playwright install chromium (some environments require this for mmdc).
	•	CI didn’t render diagrams
Ensure the Mermaid export workflow is enabled and the paths match assets/diagrams/**/*.mmd.
	•	SVG not visible in GitHub preview
GitHub generally supports inline SVG; if a specific viewer blocks it, use PNG fallback.

⸻

Maintenance checklist (per PR)
	•	Updated relevant .mmd only (no manual edits to outputs).
	•	python assets/diagrams/test_diagrams.py --render --strict passes locally.
	•	.svg renders are readable at 100% zoom in GitHub and in HTML dashboards.
	•	Large diffs? Split diagram or simplify edges; keep complexity navigable.

⸻

Quick start (for iPhone/Codex users)

After you paste new/updated files here:

git add assets/diagrams && git commit -m "docs(diagrams): add/update Mermaid sources and renders" && git push origin main

This will trigger CI to render and validate the diagrams automatically.

⸻

Roadmap (optional diagrams)
	•	corel_calibration_flow.mmd — Visualize COREL conformal calibration and uncertainty coverage.
	•	logic_graph_overlay.mmd — End‑to‑end symbolic rule graph embedded into diagnostics HTML.
	•	umap_tsne_fusion.mmd — Interactive projection integration map for the dashboard generator.

⸻

License & attribution

Diagrams are part of the SpectraMind V50 repository and inherit the project’s license. When exporting or reusing images outside this repo, please include a short attribution:
“SpectraMind V50 (NeurIPS 2025 Ariel Data Challenge) — Diagram generated from Mermaid sources.”

⸻

End of assets/diagrams/README.md