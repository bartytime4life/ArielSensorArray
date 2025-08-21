# assets/

**SpectraMind V50 — Ariel Data Challenge 2025**  
*Central repository assets: diagrams, dashboards, reports, and reproducibility visuals*

> See the companion architecture document: **[ARCHITECTURE.md](ARCHITECTURE.md)**

---

## 📌 Purpose

This directory consolidates all **visual artifacts** used across SpectraMind V50.  
It ensures the system is **self-documented, reproducible, and leaderboard-ready**:

- **Source-tracked** (Mermaid `.mmd` is canonical)  
- **Auto-exported** (`.svg`, `.png`, `.pdf` via CI or `make diagrams`)  
- **CI-validated** (diagram tests + mermaid-export workflow)  
- **Dashboard-ready** (`report.html`, `diagnostics_dashboard.html`)  

---

## 📂 Contents

- **`diagrams/`** — Mermaid diagrams (`.mmd` sources) + rendered `.svg`/`.png`.  
  Used in **[ARCHITECTURE.md](ARCHITECTURE.md)**, reports, and dashboards.  
- **`report.html`** — Compact reproducibility log (pipeline + configs).  
- **`diagnostics_dashboard.html`** — Interactive dashboard (UMAP/t-SNE, SHAP overlays, symbolic rule tables, calibration).  
- **`sample_plots/`** *(optional)* — Example PNGs for testing (`sample_spectrum.png`, `umap_clusters.png`, `shap_overlay.png`).  

---

## 📊 Kaggle Model Insights (Why the diagrams look this way)

SpectraMind V50 integrates lessons from Kaggle baselines in the NeurIPS 2025 Ariel Data Challenge:

- **Thang Do Duc “0.329 LB”**  
  • Residual MLP; simple preprocessing; no σ estimation.  
  • Robust, reproducible reference design.  

- **V1ctorious3010 “80bl-128hd-impact”**  
  • Extremely deep (80 residual blocks, 128 hidden).  
  • Captures subtle features; higher variance/overfitting risk; less interpretable.  

- **Fawad Awan “Spectrum Regressor”**  
  • Multi-output head predicts the entire spectrum.  
  • Stable, interpretable; consistent across bins.  

**Takeaways embedded in V50 (reflected in diagrams & docs):**  
- Residual-style encoders (**Mamba SSM** for FGS1, **GNN** for AIRS) instead of brute-force deep MLPs.  
- Physics-informed detrending & jitter correction in calibration.  
- Explicit uncertainty (σ) with **Temperature Scaling + COREL GNN**.  
- Ensembles that fuse shallow + deep + symbolic overlays.

> Narrative + diagrams: **[ARCHITECTURE.md](ARCHITECTURE.md)**

---

## 📐 Diagrams (maintained in `assets/diagrams/`)

- **Pipeline Overview** — `diagrams/pipeline_overview.svg`  
  *FGS1/AIRS → Calibration → Modeling (μ/σ) → UQ → Diagnostics → Submission → Reproducibility & Ops*

- **Architecture Stack** — `diagrams/architecture_stack.svg`  
  *Layers: CLI → Configs → DVC/Git → Calibration → Encoders/Decoders → UQ → Diagnostics → Packaging → CI → Runtime*

- **Symbolic Logic Layers** — `diagrams/symbolic_logic_layers.svg`  
  *Rule families: non-negativity, smoothness, asymmetry, FFT coherence, molecular alignment; evaluation & diagnostics*

- **Kaggle CI Pipeline** — `diagrams/kaggle_ci_pipeline.svg`  
  *GitHub Actions → Selftest → Training → Diagnostics → Validation → Packaging → Kaggle Submission → Artifact Registry*

Rendered `.svg` and `.png` files are committed for portability.  
All four are embedded in **[ARCHITECTURE.md](ARCHITECTURE.md)**.

---

## 📑 Reports

- **`report.html`** — Compact reproducibility report (pipeline + config snapshots).  
- **`diagnostics_dashboard.html`** — Rich interactive diagnostics (symbolic overlays, SHAP, latent projections, calibration checks).

---

## 🛠 Reproducibility & CI

- **Configs:** Hydra YAMLs in `configs/`.  
- **Data:** DVC-tracked datasets/models (hash-bound to runs).  
- **CI:** GitHub Actions (selftest, diagnostics, mermaid-export).  
- **Logs:** `logs/v50_debug_log.md` (append-only), JSONL event streams.  
- **Validation:** diagram tests ensure sources render and are embedded in docs.

Every artifact here is **versioned, CI-tested, and leaderboard-safe**.

---

## 🔁 How to regenerate diagrams (local)

From repo root:

```bash
# Render all diagrams from .mmd → .svg/.png
make diagrams

# Or render individual files
npx @mermaid-js/mermaid-cli \
  -i assets/diagrams/architecture_stack.mmd \
  -o assets/diagrams/architecture_stack.svg

# Run diagram tests
pytest assets/diagrams/test_diagrams.py