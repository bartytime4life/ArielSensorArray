# SpectraMind V50 — ArielSensorArray

Neuro‑symbolic, physics‑informed pipeline for the **NeurIPS 2025 Ariel Data Challenge**

> **North Star:** From raw FGS1/AIRS frames → calibrated light curves → μ/σ per 283 bins → diagnostics → leaderboard‑ready submission — **fully reproducible** via CLI, Hydra configs, DVC, and CI.

---

[![Build](https://img.shields.io/badge/CI-GitHub_Actions-blue.svg)](./.github/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-3776AB)
![License](https://img.shields.io/badge/license-MIT-green)
![Hydra](https://img.shields.io/badge/config-Hydra_1.3-blueviolet)
![DVC](https://img.shields.io/badge/data-DVC_3.x-945DD6)
![GPU](https://img.shields.io/badge/CUDA-12.x-76B900)

---

## 0) What is this?

**ArielSensorArray** is the repository root for the **SpectraMind V50** solution targeting the NeurIPS 2025 Ariel Data Challenge. It implements a mission‑grade, CLI‑first pipeline:

* **Calibration kill chain** for FGS1/AIRS (bias/dark/flat, trace extraction, wavelength alignment).
* **Dual‑encoder modeling** (FGS1 long‑sequence + AIRS spectral graph) with decoders for **μ** and **σ**.
* **Uncertainty calibration** (temperature scaling + COREL/conformal).
* **Diagnostics & explainability** (UMAP/t‑SNE, SHAP overlays, symbolic constraints, FFT & smoothness).
* **Reproducibility** (Hydra YAMLs, DVC/lakeFS‑ready, MLflow optional, Docker).
* **Unified Typer CLI**: `spectramind` (train / predict / calibrate / diagnose / submit / ablate / selftest / analyze-log / corel-train / check-cli-map).

The code is engineered to pass Kaggle constraints (end‑to‑end ≤9 hours for \~1,100 planets), with **deterministic seeds**, **config hashes**, and **audit logging** to `v50_debug_log.md`.

---

## 1) Quickstart (5 minutes)

### 1.1 Clone and enter

```bash
git clone https://github.com/<your-username>/ArielSensorArray.git
cd ArielSensorArray
```

### 1.2 Choose environment

#### Option A — Poetry (recommended)

```bash
# Install poetry if needed: pipx install poetry
poetry install --no-root
poetry run pre-commit install
```

#### Option B — pip/venv

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pre-commit install
```

#### Option C — Docker (GPU)

```bash
# Example image tag; see Dockerfile at repo root if you add one later
docker build -t spectramindv50:dev .
docker run --gpus all -it --rm -v "$PWD":/workspace spectramindv50:dev bash
```

### 1.3 DVC data setup (optional now; required for full runs)

```bash
dvc init
# Configure a remote of your choice (S3, GCS, SSH, or local):
dvc remote add -d storage <your-remote-url>
# Fetch tracked datasets/checkpoints when available:
dvc pull
```

### 1.4 Sanity check

```bash
# Run the integrity self-test (no heavy compute):
python -m spectramind selftest
```

---

## 2) Unified CLI

All operations are exposed via a single Typer app: **`spectramind`**.

```bash
python -m spectramind --help
```

### Common subcommands

* `selftest` — fast integrity and wiring checks (files, configs, CLI registration).
* `calibrate` — run the calibration kill chain on raw frames (subset or batch).
* `train` — train the V50 model (Hydra configs for data/model/training).
* `predict` — generate μ/σ predictions and intermediate artifacts.
* `calibrate-temp` — temperature scaling for predictive uncertainty.
* `corel-train` — COREL conformal/graph calibration training.
* `diagnose` — diagnostics suite (GLL heatmap, UMAP/t‑SNE, SHAP overlays, symbolic).

  * `diagnose dashboard` — build an interactive HTML report.
  * `diagnose tsne-latents`, `diagnose smoothness`, `diagnose profile`, etc.
* `ablate` — automated ablation with HTML/Markdown leaderboards.
* `submit` — full pipeline: selftest → predict → validate → package submission zip.
* `analyze-log` — parse `v50_debug_log.md` into Markdown/CSV, heatmaps, trends.
* `check-cli-map` — map CLI commands to files for integrity.

#### Examples

```bash
# Self-test and environment summary
python -m spectramind selftest --deep

# Train on toy data (fast loop for CI/dev)
python -m spectramind train data=toy training=fast

# Predict with a given checkpoint and config
python -m spectramind predict model.checkpoint=outputs/checkpoints/best.ckpt \
  --out-csv outputs/submission.csv

# Temperature scaling on validation split
python -m spectramind calibrate-temp data=val model.checkpoint=... ts.max_iter=100

# Diagnostics dashboard with UMAP+t‑SNE and symbolic overlays
python -m spectramind diagnose dashboard --html-out outputs/diagnostics/report_v1.html

# End-to-end submission bundle
python -m spectramind submit --open-html --zip-out outputs/submission_bundle.zip

# Analyze CLI usage logs, group by config hash, export CSV & Markdown
python -m spectramind analyze-log --csv outputs/logs/log_table.csv --md outputs/logs/log_table.md
```

> TIP: Every CLI call appends an entry to `v50_debug_log.md` (command, config hash, git SHA, timestamps).

---

## 3) Configuration (Hydra 1.3)

All run parameters live in **`configs/`** (created in subsequent steps). Hydra composes a base config with group overrides:

* `data/` — dataset source (kaggle, toy, local), loaders, batch sizes, seeds.
* `model/` — encoder/decoder definitions, latent dims, σ‑head, loss weights.
* `training/` — optimizer/scheduler, AMP, grad accumulation, checkpoints.
* `diagnostics/` — projections (UMAP/t‑SNE), SHAP, symbolic overlays, dashboard options.
* `calibration/` — temperature scaling, COREL conformalization, plots and coverage targets.
* `logging/` — console/file/JSONL/MLflow logging backends and levels.

**Override on CLI**:

```bash
python -m spectramind train data=kaggle model=v50 training=default +training.seed=1337
```

Hydra writes the **resolved config** into run outputs; a **config hash** is appended to CLI logs and artifacts.

---

## 4) Data & Artifacts

### 4.1 Directory conventions

```
data/
  raw/             # raw frames (FGS1/AIRS)
  processed/       # calibrated lightcurves/spectra
  meta/            # per-planet metadata

outputs/
  checkpoints/     # model checkpoints
  predictions/     # μ/σ arrays & CSVs
  diagnostics/     # HTML, PNG, JSON summaries
  calibrated/      # calibrated data snapshots
logs/
  v50_debug_log.md # audit log (append-only)
```

### 4.2 DVC stages (see `dvc.yaml`)

* `calibrate`: raw → calibrated (persisted snapshot).
* `train`: uses calibrated data to train.
* `predict`: emits submission CSV/NPZ and artifacts.
* `diagnose`: computes diagnostics dashboard.

> You may attach a remote via `dvc remote add -d storage ...` and `dvc push/pull` to sync artifacts.

---

## 5) Reproducibility & Audit

* **Seeds**: deterministic where feasible (`training.seed`, dataloader ordering for the toy/CI path).
* **Config Hashes**: recorded for every run; included in artifact names and logs.
* **Git SHA**: embedded in logs and optional manifests.
* **Logging**: console, file (`logs/`), JSONL event stream; optional **MLflow/W\&B**.
* **Docker**: optional container for OS‑level reproducibility; Poetry pins Python deps.
* **CI**: GitHub Actions executes a smoke pipeline (selftest + tiny train/predict) on PRs.

`v50_debug_log.md` acts as the **append‑only** mission log for traceability. Use:

```bash
python -m spectramind analyze-log --md outputs/logs/log_table.md --csv outputs/logs/log_table.csv
```

---

## 6) Makefile targets

```makefile
make help           # list targets
make init           # install hooks, ensure dirs, optional DVC init
make selftest       # fast integrity checks
make train          # hydra-driven train (override via VARS)
make predict        # run inference
make diagnose       # build diagnostics report
make submit         # full pipeline + bundle zip
make clean          # remove caches and temp outputs
```

> You can pass variables, e.g. `make train DATA=kaggle CFG=model=v50`.

---

## 7) Testing & CI

* **Pytest** with logging to `logs/test.log`, markers for fast/deep modes.
* **`spectramind selftest`** is included in CI preflight.
* **Pre‑commit**: ruff, black, isort, YAML checks, EOF, trailing whitespace.

Run locally:

```bash
pytest -q
pre-commit run --all-files
```

CI workflow lives at `.github/workflows/ci.yml`.

---

## 8) Contribution Guidelines

See `CONTRIBUTING.md` for coding style, commit hygiene, branch naming, PR checklist, and review rules.

* Type‑hints, doctrings (NumPy/SciPy style).
* No silent catches; log at **INFO** or **WARNING** with context.
* Keep configs declarative; do not hard‑code experiment params in code.
* Update `CITATION.cff` and `VERSION` on tagged releases.
* Ensure `selftest` and CI pass before merging.

---

## 9) Citation

If this work contributes to your research:

```
@software{spectramind_v50_2025,
  title   = {SpectraMind V50 — Neuro‑symbolic, Physics‑informed Exoplanet Spectroscopy},
  author  = {SpectraMind Team},
  year    = {2025},
  url     = {https://github.com/<your-username>/ArielSensorArray}
}
```

Also see `CITATION.cff` at the repository root.

---

## 10) Troubleshooting

* **CUDA not found**: verify `nvidia-smi` and driver/CUDA 12.x; on WSL2 enable GPU passthrough.
* **Out of memory (OOM)**: reduce batch size (`training.batch_size`), enable gradient accumulation.
* **Slow I/O**: use DVC remote close to compute; pre‑stage calibrated data.
* **Hydra override errors**: ensure correct group keys (e.g., `data=toy`, `model=v50`, `training=fast`).
* **CI failure**: check pre‑commit formatting and `selftest` artifact paths.

---

## 11) Roadmap (abridged)

* [ ] Expand unit tests for symbolic loss decomposition & ∂L/∂μ maps.
* [ ] TorchScript/JIT path for fast inference.
* [ ] Optional GUI control panel (React + FastAPI) mirroring CLI diagnostics.
* [ ] Leaderboard automation job with artifact promotion gates.

---

## 12) Repository Layout (top level)

```
ArielSensorArray/
  README.md                 # this file
  ARCHITECTURE.md           # master architecture & physics rationale
  CONTRIBUTING.md           # how to contribute
  CITATION.cff              # citation metadata
  pyproject.toml            # package metadata & deps
  poetry.lock               # pinned dependency lock
  requirements.txt          # pip alternative
  VERSION                   # semantic version
  .env.example              # environment template
  .pre-commit-config.yaml   # code quality hooks
  .gitignore                # ignores (Python, DVC, outputs, etc.)
  dvc.yaml                  # pipeline stages
  Makefile                  # convenience targets
  spectramind.py            # root Typer CLI entrypoint
  v50_debug_log.md          # append-only audit log
  pytest.ini                # pytest config
  mkdocs.yml                # site config (if docs published)
  docs/
    index.md                # docs landing page
  logs/
    .gitkeep
  .github/
    workflows/
      ci.yml                # CI: selftest + smoke pipeline
  configs/                  # (added next) hydra groups: data/model/training/...
  src/                      # (added next) package: loaders, models, cli modules...
  outputs/                  # artifacts (gitignored)
  data/                     # data directories (gitignored/DVC-tracked)
```

---

## 13) iPhone/Codex one‑liner (optional)

Once the full set of root files is in place, you can commit/push with a single command on mobile shells:

```bash
git add -A && git commit -m "feat(root): add SpectraMind V50 root files (CLI/Hydra/DVC/CI)" && git push
```

---

**License:** MIT (see `LICENSE` if present; otherwise add one before public release).
**Contact:** Open an Issue or Discussion in the repository for support requests.

---
