# SpectraMind V50 — Notebooks

> Lightweight, **CLI‑first** notebooks for quick experiments and demos.  
> The pipeline is designed to be reproducible via the Typer CLI and Hydra configs;
> these notebooks simply wrap the CLI so you can poke the system interactively.

## Contents

- `00_quickstart.ipynb` — environment & self‑test, project tour, config hash.
- `01_pipeline_calibrate_train_predict.ipynb` — run a tiny end‑to‑end (calibrate → train → predict).
- `02_diagnostics_explainability.ipynb` — generate diagnostics and render the HTML dashboard inline.

### Tips

- Use the **Makefile** for daily work (`make e2e`, `make diagnose`, `make benchmark`).
- If the `spectramind` console script is not on PATH, try `poetry run spectramind` or `python -m spectramind`.
- Results land in `outputs/` (diagnostics, submissions, checkpoints). These are **ignored by Git** and can be **recreated**.

---

## Minimal Setup

```bash
poetry install
poetry run pre-commit install -t pre-commit -t pre-push
poetry run spectramind selftest --fast