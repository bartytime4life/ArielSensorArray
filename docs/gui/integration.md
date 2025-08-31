# 🔗 GUI Integration — SpectraMind V50

## 0) Purpose

This document explains how the **optional GUI layer** integrates with the **SpectraMind V50 pipeline**.  
The integration design ensures that GUIs are **thin shells around the CLI**:  
- All computation is done by the CLI (`spectramind …`).  
- GUI simply **calls CLI subcommands** and visualizes resulting **configs, logs, and artifacts**.  
- Reproducibility, automation, and NASA-grade auditability remain intact:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

---

## 1) Integration Philosophy

- **CLI-first, GUI-thin**: GUI actions never bypass the CLI — they always resolve into Hydra configs and Typer CLI calls.  
- **Artifact-driven visualization**: GUI reads **diagnostic_summary.json**, plots (`.png`, `.html`), and logs.  
- **Two-way binding**: GUI controls map to Hydra config fields (`configs/*.yaml`), ensuring parity between CLI and GUI runs.  
- **Non-invasive**: GUI is optional and modular — the pipeline runs headless in Kaggle/CI without GUI dependencies.  

---

## 2) Data Flow Integration

```mermaid
flowchart LR
  U[User] -->|Clicks Run| G[GUI Controller]
  G -->|Invoke CLI| C[spectramind …]
  C -->|Hydra Configs| Y[configs/*.yaml]
  C -->|Outputs| O[diagnostic_summary.json + plots + HTML]
  O --> G
  G -->|Render| V[Dashboard / Charts]
  C --> L[logs/v50_debug_log.md]
  L --> G
````

* **Inputs**: Hydra configs (`configs/`) selected/overridden by GUI.
* **Pipeline Execution**: CLI orchestrates calibration, training, diagnostics.
* **Outputs**: JSON summaries, HTML dashboards, logs.
* **GUI Rendering**: GUI embeds plots, tables, UMAP/t-SNE dashboards, SHAP overlays.

---

## 3) Integration Points

### 3.1 Configs

* GUI allows browsing & editing of `configs/` YAML files.
* Overrides → CLI flags (e.g., `trainer.epochs=50`).
* GUI **never edits code**; only YAML/CLI flags.

### 3.2 CLI

* GUI backend invokes `spectramind …` subcommands:

  * `spectramind calibrate`
  * `spectramind train`
  * `spectramind diagnose dashboard`
  * `spectramind submit`
* Ensures identical behavior across GUI/CLI.

### 3.3 Artifacts

* GUI reads from:

  * `outputs/diagnostic_summary.json`
  * `outputs/plots/*.png`
  * `outputs/diagnostic_report_vX.html`
  * `logs/v50_debug_log.md`
* Supports **live streaming** of logs via WebSocket or subprocess pipe.

---

## 4) Framework Integration Modes

* **Streamlit / Gradio** → Lightweight dashboards (local/Kaggle).
* **React + FastAPI** → Full-featured web dashboards (remote, team usage).
* **PyQt / Qt** → Native mission control apps (offline).
* **Notebook Widgets** → Inline GUIs for research/teaching.

Each framework integrates at the **same boundary**: CLI calls + artifact visualization.

---

## 5) Reproducibility Safeguards

* **Config hashing**: GUI runs still generate `run_hash_summary_v50.json`.
* **Logging**: All GUI actions append to `logs/v50_debug_log.md` with timestamp and CLI equivalent.
* **No hidden state**: GUI never introduces parameters outside Hydra configs.
* **Version capture**: GUI banner records frontend hash + backend CLI version.

---

## 6) Example: End-to-End Integration Flow

1. User selects **UMAP + Symbolic Overlay** in GUI.
2. GUI builds CLI call:

   ```bash
   spectramind diagnose dashboard diagnostics.umap.enabled=true symbolic.show=true
   ```
3. CLI runs → Hydra composes configs → pipeline executes.
4. Outputs saved:

   * `outputs/diag_vX/diagnostic_summary.json`
   * `outputs/diag_vX/diagnostic_report_vX.html`
5. GUI embeds plots & HTML in dashboard panel.
6. Run logged in `v50_debug_log.md`.

---

## 7) Roadmap for Integration

* ✅ Phase 1: Streamlit wrapper (prototype GUI → CLI integration).
* ✅ Phase 2: React/FastAPI dashboard with real-time log streaming.
* 🔄 Phase 3: Mission-grade PyQt native desktop GUI.
* 🔄 Phase 4: Kaggle integration (Gradio inline apps, CI dashboards).

---

## ✅ Summary

GUI integration in SpectraMind V50 is **non-intrusive, reproducible, and artifact-driven**:

* **Thin wrappers** over CLI subcommands.
* **Artifacts & configs** as the source of truth.
* **Multiple frameworks** possible, all bound to the same integration points.
* **Audit-safe**: no GUI action occurs without logging and CLI serialization.

```
