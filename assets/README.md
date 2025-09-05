# assets/

**SpectraMind V50 — Ariel Data Challenge 2025**  
*Central repository assets: diagrams, dashboards, reports, comparisons, and reproducibility visuals*

> See companion docs: **[ARCHITECTURE.md](../ARCHITECTURE.md)** • **[KAGGLE_GUIDE.md](../KAGGLE_GUIDE.md)** • **[COMPARISON_GUIDE.md](../COMPARISON_GUIDE.md)**

---

## 📌 Purpose

This directory consolidates all **visual + documentation artifacts** used across SpectraMind V50 and makes them **self-documented, reproducible, and leaderboard-ready**.

- **Source-tracked**: Mermaid `.mmd` files are canonical.
- **Auto-exported**: `.svg` / `.png` / `.pdf` via CI (`make diagrams`).
- **CI-validated**: Diagram tests + mermaid export job + HTML lint.
- **Dashboard-ready**: `report.html`, `diagnostics_dashboard.html` with optional `dashboards/dashboard_data.json` hydration.
- **Manifested**: Every file appears in `assets-manifest.json` with MIME, size, integrity, and version.

---

## 📂 Contents

assets/
├─ diagrams/                # Mermaid sources and rendered artifacts
│  ├─ architecture_stack.mmd
│  ├─ pipeline_overview.mmd
│  ├─ symbolic_logic_layers.mmd
│  ├─ kaggle_ci_pipeline.mmd
│  ├─ .svg .png           # CI-rendered outputs
│  └─ test_diagrams.py      # fast rendering/smoke tests
│
├─ dashboards/
│  ├─ diagnostics_dashboard.html
│  ├─ dashboard_data.json   # optional: generator hydration (KPIs, asset paths, CLI table)
│  └─ report.html
│
├─ gui/                     # optional GUI/dashboard visuals & styles
│  ├─ color_palette.json
│  └─ widget_styles.css
│
├─ logos/                   # brand + symbols (spectrum/black/white; mark/full)
│  ├─ spectrum/mark/.png .svg
│  ├─ spectrum/full/.png .svg
│  ├─ black/  white/**
│  └─ favicon/*.ico
│
├─ sample_plots/            # optional smoke assets used by dashboards and tests
│  ├─ sample_spectrum.png
│  ├─ umap_clusters.png
│  └─ shap_overlay.png
│
├─ comparison_overview.png
├─ assets-manifest.json     # integrity, size, mime, version for every asset
└─ (root docs live one level up)

---

## 📊 Kaggle Model Insights (why the diagrams look this way)

SpectraMind V50 integrates lessons from Kaggle baselines in the NeurIPS 2025 Ariel Data Challenge:

- **Thang Do Duc “0.329 LB”** — residual MLP; simple preprocessing; no σ estimation; robust and reproducible.
- **V1ctorious3010 “80bl-128hd-impact”** — very deep MLP (80 residual blocks/128 hidden); captures subtle features but higher variance/overfit risk.
- **Fawad Awan “Spectrum Regressor”** — multi-output spectrum head; stable and interpretable across bins.

**Embedded in V50 (reflected in diagrams & docs):**

- Residual-style encoders (**Mamba SSM** for FGS1; **GNN** for AIRS).
- Physics-informed detrending and jitter correction during calibration.
- Explicit uncertainty (σ) with **Temperature Scaling + COREL GNN**.
- Ensembles that fuse shallow + deep + symbolic overlays.

See **[COMPARISON_GUIDE.md](../COMPARISON_GUIDE.md)** for the full narrative and the comparison graphic.

---

## 📐 Diagrams (maintained in `assets/diagrams/`)

- **Pipeline Overview** — `diagrams/pipeline_overview.mmd` → `pipeline_overview.svg`  
  *FGS1/AIRS → Calibration → Modeling (μ/σ) → UQ → Diagnostics → Submission → Reproducibility & Ops*

- **Architecture Stack** — `diagrams/architecture_stack.mmd` → `architecture_stack.svg`  
  *CLI → Configs → DVC/Git → Calibration → Encoders/Decoders → UQ → Diagnostics → Packaging → CI → Runtime*

- **Symbolic Logic Layers** — `diagrams/symbolic_logic_layers.mmd` → `symbolic_logic_layers.svg`  
  *Families: non-negativity, smoothness, asymmetry, FFT coherence, molecular alignment; evaluation & diagnostics*

- **Kaggle CI Pipeline** — `diagrams/kaggle_ci_pipeline.mmd` → `kaggle_ci_pipeline.svg`  
  *GitHub Actions → Selftest → Training → Diagnostics → Validation → Packaging → Kaggle Submission → Artifact Registry*

> Rendered `.svg`/`.png` are committed for portability and embedded from `../ARCHITECTURE.md`.

---

## 📑 Dashboards & Reports

- **`dashboards/diagnostics_dashboard.html`** — Interactive diagnostics (UMAP/t-SNE, SHAP overlays, symbolic rules, calibration).  
  Optional data hydration file: `dashboards/dashboard_data.json` (build meta, KPIs, asset map, recent CLI calls).

- **`dashboards/report.html`** — Compact reproducibility report (pipeline + config snapshots).

**Embed mode:** append `?embed=1` (images → iframes).  
**Theme:** `?theme=dark|light|auto` (persists).  
**Keyboard:** `?` help • `r` refresh images • `t` theme • `e` embed • `g` jump to overview.

---

## 🧾 Manifest & Integrity

All assets are enumerated in **`assets-manifest.json`** with:

- **`path`** (repo-relative), **`type`**, **`mime`**, **`bytes`**, **`version`**
- **`hash`**: `sha256-…` for integrity
- **Aliases** for legacy import paths (`branding/logo.svg`, `favicon.ico`, …)
- **Deprecated** mapping with removal windows (e.g., `logos/logo.svg` → new spectrum path)

> The manifest is validated in CI and used by the dashboards to cache-bust (`?v=`) and verify integrity.

---

## 🛠 Reproducibility, CI & Tests

- **Configs:** Hydra YAMLs in `configs/` (hash-bound to runs).
- **Data/Models:** DVC-tracked; artifacts link back to commit + config hash.
- **CI:** GitHub Actions (`selftest`, `diagnose`, `mermaid-export`, `lint-html`, `manifest-verify`).
- **Logs:** `logs/v50_debug_log.md` (append-only), `events/*.jsonl`.
- **Diagram tests:** `assets/diagrams/test_diagrams.py` confirms sources render and are embedded in docs.
- **HTML lint:** `npm run lint:html` (or `html-validate`) checks dashboards & reports.

Every artifact here is **versioned, CI-tested, and leaderboard-safe**.

---

## 🔁 Regenerate diagrams locally

From repo root:

```bash
# Render all .mmd → .svg/.png
make diagrams

# Render a single diagram
npx @mermaid-js/mermaid-cli \
  -i assets/diagrams/architecture_stack.mmd \
  -o assets/diagrams/architecture_stack.svg

# Run diagram tests
pytest assets/diagrams/test_diagrams.py -q


⸻

✅ Checklist for contributions
	•	Add/modify .mmd in assets/diagrams/, run make diagrams.
	•	Update assets-manifest.json (hash/bytes/version/mime).
	•	If new dashboard assets: place under assets/ and list them in manifest.
	•	CI green: diagram tests, manifest verify, HTML lint.
	•	Link new visuals from ARCHITECTURE.md/COMPARISON_GUIDE.md.

⸻

🔒 Notes on licensing & branding

Logos are © SpectraMind and subject to the project license noted in the manifest.
Third-party icons or fonts (if any) must include attribution and compatible licenses.

⸻
