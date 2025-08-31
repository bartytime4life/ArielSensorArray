# 📂 `src/gui/app/` — GUI App Layer

## 🎯 Purpose & Scope

This folder contains the **React-based GUI routes and layout** for **SpectraMind V50**.  
It is a **thin, optional visualization layer** that sits **on top of the CLI-first pipeline**.  
All analytics, training, and diagnostics are executed by the `spectramind …` CLI and FastAPI server;  
the GUI simply loads those **CLI-generated artifacts** (`.json`, `.html`, `.png`) and presents them interactively:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

> Golden rule: **No hidden logic here** — the GUI must only reflect outputs produced by the CLI,  
> ensuring **NASA-grade reproducibility** and avoiding divergence between GUI and CLI workflows:contentReference[oaicite:2]{index=2}.

---

## 📁 Files

- **`diagnostics.tsx`**  
  Renders diagnostic summary data from `/api/diagnostics/summary`.  
  Loads artifacts like UMAP/t-SNE HTML, SHAP overlays, FFT plots, calibration heatmaps.  
  Provides panels and tabs for interactive exploration.

- **`reports.tsx`**  
  Renders saved HTML reports from `/artifacts` (e.g., `diagnostic_report_v1.html`).  
  Enables browsing multiple report versions and comparing runs.

- **`readme.md`** (this file)  
  Documentation for the `app/` folder, design principles, and usage.

---

## 🖼️ Design Principles

- **CLI-first, GUI-optional**  
  Every GUI action corresponds to a CLI command (e.g., `spectramind diagnose dashboard`).  
  GUI never bypasses Hydra configs, run hashes, or reproducibility checks:contentReference[oaicite:3]{index=3}.

- **Reproducibility by construction**  
  GUI renders artifacts under `/artifacts/`, produced by versioned CLI runs (tracked in Hydra + DVC):contentReference[oaicite:4]{index=4}.

- **Minimal coupling**  
  Components here consume REST/WS APIs (`/api/diagnostics/*`) and static mounts (`/artifacts/*`).  
  They do not perform ML, calibration, or training logic directly.

- **Cross-platform & lightweight**  
  Built with **React + Vite + Tailwind + shadcn/ui**, aiming for fast builds and minimal complexity.  
  Optional dashboards, not required for Kaggle/CI workflows:contentReference[oaicite:5]{index=5}.

---

## 🔗 API Integration

The GUI consumes the following APIs:

- `GET /api/diagnostics/summary` → `diagnostics.tsx` panel tables/charts.  
- `GET /api/diagnostics/health` → connection checks.  
- `POST /api/diagnostics/run` → optional CLI trigger (`spectramind diagnose dashboard`).  
- Static mount `/artifacts/*` → diagnostic HTML, PNG, JSON, and plots.

Auth (if enabled) is handled via headers (`X-API-Key` or `Authorization: Bearer …`):contentReference[oaicite:6]{index=6}.

---

## 📊 Example Workflow

```mermaid
flowchart LR
  GUI[GUI: diagnostics.tsx] -->|fetch| API[/api/diagnostics/summary/]
  GUI -->|open| ART[Static /artifacts/umap.html]
  GUI -->|trigger| CLI[spectramind diagnose dashboard]
  CLI --> ART
  CLI --> LOG[v50_debug_log.md]
````

1. User clicks **"Run Diagnostics"** in GUI → triggers `spectramind diagnose dashboard`.
2. CLI runs under Hydra/DVC → produces `diagnostic_summary.json` + HTML/plots.
3. GUI fetches `/api/diagnostics/summary` + loads HTML from `/artifacts/`.
4. Everything is logged to `v50_debug_log.md`, ensuring traceability.

---

## 🛠️ Development Notes

* Use **Vite** dev server for hot reload:

  ```bash
  cd src/gui
  npm install
  npm run dev
  ```
* All configs (API URL, ports, auth headers) come from `.env` (never hardcode).
* Ensure `spectramind …` CLI is installed and FastAPI server (`src/server/main.py`) is running.

---

## 📌 Future Extensions

* Add **interactive explainability views** (e.g., SHAP overlays, symbolic rule tables).
* Versioned report explorer with **run hash metadata**.
* Optional **Kaggle mode**: GUI reflects Kaggle notebook outputs in read-only dashboards.

---

✅ In short: **`src/gui/app/` is a presentation layer only.**
It visualizes CLI results, never generates them — keeping **SpectraMind V50 reproducible, auditable, and Kaggle-safe**.

```
```
