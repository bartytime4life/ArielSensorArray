# 📓 SpectraMind V50 — Notebooks

> Lightweight, **CLI-first orchestration notebooks** for quick experiments and demos.  
> The pipeline is designed to be reproducible via the Typer CLI and Hydra configs;  
> these notebooks simply wrap the CLI so you can poke the system interactively.

---

## 📂 Contents

- `00_quickstart.ipynb`  
  Environment check, `spectramind selftest`, project tour, and config hash snapshot.

- `01_pipeline_calibrate_train_predict.ipynb`  
  Run a tiny end-to-end pipeline: **calibrate → train → predict**.  
  Uses sample data for a ≤10-minute smoke test.

- `02_diagnostics_explainability.ipynb`  
  Generate diagnostics (FFT, SHAP, symbolic overlays) and render the  
  **HTML dashboard inline** in Jupyter.

---

## 🛠️ Usage Tips

- Use the **Makefile** for daily work (`make e2e`, `make diagnose`, `make benchmark`).  
- If the `spectramind` console script is not on PATH, try:
  ```bash
  poetry run spectramind ...
  # or
  python -m spectramind ...

	•	Results land in outputs/ (diagnostics, submissions, checkpoints).
These are ignored by Git and can be recreated on demand.

⸻

⚡ Minimal Setup

# install dependencies
poetry install

# set up pre-commit hooks (lint, format, hash checks)
poetry run pre-commit install -t pre-commit -t pre-push

# verify environment, configs, CLI, and hashes
poetry run spectramind selftest --fast


⸻

🔬 Principles
	•	CLI-first — Notebooks never implement core pipeline logic; they call spectramind.
	•	Hydra-safe configs — Parameters are controlled via configs/*.yaml with overrides.
	•	Reproducibility — Every run logs config/env/git hash to logs/v50_debug_log.md.
	•	DVC & Kaggle integration — Large artifacts live under DVC/lakeFS; Kaggle notebooks
are thin wrappers around the same CLI (spectramind).

⸻

🛰️ Notes
	•	These notebooks are optional: all functionality is available via CLI.
	•	Kaggle integration: export a notebook that simply runs spectramind with the right configs.
	•	GUI/dashboard: diagnostics (UMAP/t-SNE, SHAP overlays, symbolic rule tables)
render inline via generate_html_report.py.

⸻

✅ Next Steps
	•	Run 00_quickstart.ipynb to confirm setup.
	•	Try 01_pipeline_calibrate_train_predict.ipynb for a mini E2E.
	•	Use 02_diagnostics_explainability.ipynb to explore symbolic overlays & FFT.

⸻


