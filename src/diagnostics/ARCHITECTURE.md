# 🧪 `crc/diagnostics/ARCHITECTURE.md` — Diagnostics & Reproducibility Core

## 0) Purpose & Scope

The **diagnostics subsystem** in SpectraMind V50 provides the **auditable bridge** between:
* raw telescope data (FGS1 + AIRS),
* calibrated μ/σ spectra,
* and **scientific validation** (GLL, FFT, symbolic overlays, calibration checks).

It ensures **NASA-grade reproducibility**:
* Every diagnostic run is driven via the `spectramind diagnose …` CLI:contentReference[oaicite:0]{index=0}.
* Configs are Hydra-safe, logged, and DVC-tracked:contentReference[oaicite:1]{index=1}.
* Outputs are stored as versioned artifacts (`diagnostic_summary.json`, plots, HTML dashboards).

> Golden rule: **Diagnostics never mutate core science data.** They *only read, analyze, and log*.

---

## 1) High-Level Flow

```mermaid
flowchart TD
    subgraph CLI
        A["spectramind diagnose"]
        A1["--dashboard"]
        A2["--symbolic-rank"]
        A3["--fft"]
        A4["--calibration"]
    end

    subgraph Configs
        C1["configs/diagnostics/*.yaml"]
        C2["Hydra overrides"]
    end

    subgraph Pipeline
        P1["FGS1/AIRS μ, σ predictions"]
        P2["Symbolic Logic Engine"]
        P3["SHAP overlays"]
        P4["FFT & smoothness analyzers"]
        P5["Calibration checker"]
    end

    subgraph Artifacts
        R1["diagnostic_summary.json"]
        R2["plots/*.png"]
        R3["umap.html / tsne.html"]
        R4["report_vN.html"]
        R5["v50_debug_log.md"]
    end

    A --> C1
    A --> C2
    C1 --> P1
    C2 --> P1
    P1 --> P2 --> R1
    P1 --> P3 --> R2
    P1 --> P4 --> R2
    P1 --> P5 --> R2
    R1 --> R4
    R2 --> R4
    A --> R5
````

---

## 2) Design Principles

* **CLI-first, GUI-optional** — all runs must be reproducible from CLI.
* **Hydra configs** govern diagnostics: symbolic weights, FFT cutoffs, calibration bins.
* **DVC lineage**: diagnostic outputs tracked alongside model/data commits.
* **Separation of concerns**: no analytics inside GUI — only renders artifacts.
* **Mission-grade rigor**: symbolic + physics checks (smoothness, non-negativity, radiation priors, lensing overlays) validate scientific plausibility.

---

## 3) Diagnostic Components

| Module                               | Role                                                 | Config Keys                                | Outputs                                 |
| ------------------------------------ | ---------------------------------------------------- | ------------------------------------------ | --------------------------------------- |
| **GLL Evaluator**                    | Compute Gaussian log-likelihood, per-bin & global    | `loss.gll.*`                               | gll\_heatmap.png, gll\_scores.json      |
| **FFT Analyzer**                     | Spectral smoothness, autocorr, molecule fingerprints | `fft.cutoff`, `fft.window`                 | fft\_power.png, fft\_clusters.json      |
| **Symbolic Engine**                  | Rule-based violations, influence maps                | `symbolic.*`                               | symbolic\_masks.npy, rule\_rank.json    |
| **Calibration Checker**              | σ vs residuals, quantile coverage                    | `calibration.*`                            | calibration\_plots/, coverage.csv       |
| **Explainability (SHAP/UMAP/t-SNE)** | Latent projection + feature importances              | `explain.*`                                | umap.html, tsne.html, shap\_overlay.png |
| **Radiation/Lensing Overlays**       | Physical priors from astrophysics                    | `physics.lensing.*`, `physics.radiation.*` | overlay\_plots/, anomaly\_flags.json    |

---

## 4) Example CLI Calls

```bash
# Run full dashboard (UMAP + GLL + symbolic overlays)
spectramind diagnose dashboard \
    diagnostics=gll_fft_symbolic \
    fft.cutoff=40 \
    symbolic.ruleset=default

# Rank symbolic rule violations per planet
spectramind diagnose symbolic-rank top_k=5

# Check calibration
spectramind diagnose calibration mode=quantile

# Generate FFT + autocorr + molecular overlays
spectramind diagnose fft molecules="[H2O, CH4, CO2]"
```

All runs:

1. Compose Hydra config (`configs/diagnostics/*.yaml`).
2. Hash + log run (`v50_debug_log.md`).
3. Save artifacts into `/artifacts/diagnostics/<timestamp>/`.

---

## 5) Reproducibility & Audit Trail

* **Hydra snapshot** of config saved with every run.
* **DVC lock** ensures outputs are tied to code+data commits.
* **Logs**: `v50_debug_log.md` records CLI call + config hash.
* **HTML dashboard**: embeds UMAP, SHAP, symbolic tables, FFT plots for visual audit.
* **CI integration**: diagnostics run as part of GitHub Actions pre-merge check.

---

## 6) Future Extensions

* **Cycle consistency**: forward simulate spectra & compare against predictions.
* **Expanded symbolic libraries**: add gravitational lensing and radiative priors.
* **MLflow/experiment tracking**: log diagnostics metrics centrally.
* **GUI hooks**: thin React layer rendering dashboard if desired (never bypass CLI).

---

✅ In summary, `crc/diagnostics/` is the **scientific conscience** of SpectraMind V50.
It enforces symbolic + physical validity, guarantees reproducibility, and produces artifacts auditable from CLI, GUI, or Kaggle.

```
```
