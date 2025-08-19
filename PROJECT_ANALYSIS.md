# SpectraMind V50 — Project Analysis  
*(NeurIPS 2025 Ariel Data Challenge)*

> **Purpose**: This file is a *living audit* of the ArielSensorArray / SpectraMind V50 repository.  
> It compares actual repo contents against the **engineering plan** and **external references**, identifying what is implemented, validated, or pending.

---

## 0) Philosophy

- **CLI-first**: all operations exposed via Typer CLI, no hidden notebook state.  
- **Reproducibility**: Hydra configs, DVC data/artifacts, config + dataset hash logging.  
- **Scientific rigor**: NASA-grade calibration and physics-informed modeling.  
- **Automation**: CI/CD with self-tests and smoke pipelines on every push.  

---

## 1) Repository Structure

| Directory       | Status  | Notes                                                                 |
|-----------------|---------|-----------------------------------------------------------------------|
| `src/`          | ✅      | Encoders (Mamba SSM, GNN), decoders, calibration modules, CLI.        |
| `configs/`      | ✅      | Hydra group configs (`data/`, `model/`, `training/`, `diagnostics/`). |
| `data/`         | ⚠️ DVC  | Versioned via DVC, placeholders present (`.gitkeep`).                 |
| `outputs/`      | ✅      | Model checkpoints, predictions, diagnostics, logs.                    |
| `logs/`         | ✅      | `v50_debug_log.md`, JSONL streams, pytest logs.                       |
| `docs/`         | ✅      | Markdown docs, MkDocs config.                                         |
| `.github/`      | ✅      | CI workflow with smoke pipeline + tests.                              |

---

## 2) Configuration & Reproducibility

- Hydra v1.3 used for all runs.  
- Overrides supported via CLI:  
  ```bash
  python -m spectramind train data=kaggle model=v50 training=default
````

* DVC v3.x integrated: `dvc.yaml` defines calibrate→train→predict→diagnose stages.
* Config + dataset + git SHA appended to `v50_debug_log.md` for every run.
* Dockerfile + Poetry lock guarantee environment reproducibility.

✅ **Implemented fully**.

---

## 3) CLI Design

* Unified entrypoint:

  ```bash
  spectramind --help
  ```

* Subcommands:

  * `selftest` — fast wiring checks.
  * `calibrate` — full calibration kill chain.
  * `train` — train V50 model.
  * `predict` — output μ/σ predictions.
  * `calibrate-temp` — uncertainty temperature scaling.
  * `corel-train` — conformal calibration with graph priors.
  * `diagnose` — diagnostics suite (FFT, SHAP, UMAP/t-SNE, symbolic).
  * `submit` — end-to-end submission bundle.
  * `analyze-log` — parse CLI logs.
  * `check-cli-map` — map CLI commands to files.

* **UX**: Rich formatting for tables, progress bars, error visibility.

* **Logging**: every invocation appended to `logs/v50_debug_log.md`.

✅ **Strong, production-grade CLI layer.**

---

## 4) Calibration Chain

Implements full **kill chain**:

* Bias/dark subtraction.
* Flat-fielding.
* Nonlinearity & ADC corrections.
* Wavelength alignment.
* Normalization.

Artifacts persisted in `outputs/calibrated/`.

✅ **Physics-grade calibration present.**

---

## 5) Modeling Architecture

* **FGS1**: Structured State-Space Model (Mamba SSM) for long sequences.
* **AIRS**: Graph Neural Network (GNN with edge types: adjacency, molecule groups, detector regions).
* **Fusion**: latent concatenation.
* **Decoders**:

  * μ: MLP with spectral regularizers (smoothness, FFT).
  * σ: parallel heteroscedastic head.

Losses: Gaussian Log-Likelihood + optional symbolic constraints.

✅ **Implemented as per design.**

---

## 6) Uncertainty Quantification

* **Aleatoric**: σ predictions trained via GLL.
* **Epistemic**: ensembles / MC dropout.
* **Calibration**: temperature scaling, conformal COREL GNN.
* **Coverage logs** exported to JSON + plots.

⚠️ **COREL partially integrated; expand symbolic-weighted calibration.**

---

## 7) Diagnostics & Explainability

* **Latent visualization**: UMAP, t-SNE.
* **Attributions**: SHAP overlays, GNNExplainer, FGS1 integrated gradients.
* **Spectral checks**: FFT of residuals, smoothness penalties.
* **Symbolic rules**: smoothness, non-negativity, asymmetry, alignment.
* **HTML Dashboard**: unify plots, overlays, and logs.

✅ **Implemented; symbolic overlays still expanding.**

---

## 8) Automation & CI/CD

* GitHub Actions workflow:

  * Installs env via Poetry.
  * Runs `spectramind selftest`.
  * Executes smoke pipeline (toy data: calibrate→train→predict→diagnose).
* Fails fast on config/hash mismatches.
* Pre-commit hooks (ruff, black, isort, yaml, whitespace).

✅ **CI/CD strong.**

---

## 9) Pending / Roadmap

* GUI dashboard (Qt/Electron/Flutter) mirroring CLI diagnostics.
* Expanded symbolic overlay diagnostics (more rule classes, rule violation ranking).
* Deeper calibration validation (per-bin coverage heatmaps, symbolic-aware calibration).
* COREL calibration expansion (temporal bin correlations, edge feature integration).
* Leaderboard automation job with artifact promotion gates.

---

## 10) Status Matrix

| Area           | Status     | Notes                                |
| -------------- | ---------- | ------------------------------------ |
| Repo structure | ✅ Solid    | Hydra/DVC integrated cleanly.        |
| CLI            | ✅ Complete | Typer unified, Rich UX.              |
| Calibration    | ✅ Strong   | Kill chain implemented.              |
| Modeling       | ✅ Physics  | Mamba SSM + GNN.                     |
| Uncertainty    | ⚠️ Partial | COREL/symbolic extensions pending.   |
| Diagnostics    | ✅ Active   | SHAP, UMAP, FFT; symbolic expanding. |
| CI/CD          | ✅ Robust   | Selftest + smoke pipeline.           |
| GUI            | 🚧 Planned | Thin mirror, not yet built.          |

---

## 11) Action Items

* [ ] Finalize COREL calibration integration (symbolic + temporal edges).
* [ ] Expand symbolic rule overlays + violation heatmaps.
* [ ] Implement GUI dashboard (thin, CLI-backed).
* [ ] Add leaderboard job to CI/CD.
* [ ] Harden uncertainty calibration with symbolic priors.

---

**Maintainers**: SpectraMind Team
**Contact**: open an Issue on [GitHub](https://github.com/bartytime4life/ArielSensorArray/issues)

```
