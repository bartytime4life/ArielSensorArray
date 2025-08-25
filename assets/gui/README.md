# assets/gui/README.md
# SpectraMind V50 — GUI Assets

This folder contains the **visual design assets, style tokens, and GUI-facing configuration** used by the SpectraMind V50 diagnostics dashboard and any optional GUI front‑ends (e.g., React/Electron, Qt). It is **GUI‑agnostic** by design: everything here can be consumed by the CLI‑generated HTML dashboard, a web frontend, or a native desktop shell.

> V50 philosophy recap: the **CLI is the source of truth**. GUIs (including the HTML diagnostics dashboard) are **thin, optional layers** on top of the CLI and FastAPI endpoints. All assets in this directory must remain **portable, reproducible, and themeable**.

---

## 📦 Contents

- `color_palette.json` — Design tokens (brand colors, semantic roles, data overlays, UMAP/cluster ramps).
- `widget_styles.css` — Minimal, framework‑neutral CSS for shared widgets (cards, chips, tables, badges).
- `icons/` *(recommended)* — SVG/PNG symbols for dashboards (status, warning, success, filters, rules).
- `images/` *(recommended)* — Diagrams/illustrations used by the HTML dashboard / docs.
- `README.md` — This file.
- `ARCHITECTURE.md` — Integration diagram and data‑flow between CLI ↔ API ↔ GUI.

> If `icons/` and `images/` do not exist yet, create them when adding the first asset. Keep vector (SVG) as the canonical source; export raster sizes (1x/2x/3x) only as needed.

---

## 🔌 Where these assets are used

- **CLI‑generated HTML dashboard** (e.g., `outputs/diagnostics/report_v*.html`):  
  Reads `color_palette.json` and `widget_styles.css` to style UMAP/t‑SNE plots, rule tables, FFT/entropy heatmaps, COREL coverage, etc.

- **Web GUI (React/Electron)** *(optional)*:  
  Imports the same tokens for consistent theming across pages (Diagnostics, Profiles, Symbolic Rules, CLI Log Explorer).

- **Native GUI (Qt/PySide)** *(optional)*:  
  Loads tokens and applies palette at runtime; CSS variables are mapped to Qt palette/QSS.

> The dashboard/GUI **must not** hard‑code colors or fonts. Consume `color_palette.json` and layer local overrides via CSS variables.

---

## 🎨 Theming & Tokens

### `color_palette.json` schema

```json
{
  "$schema": "https://spectramind.dev/schemas/v50/color_palette.schema.json",
  "meta": {
    "name": "SpectraMind V50 Default",
    "version": "1.0.0",
    "created": "2025-08-24T00:00:00Z"
  },
  "brand": {
    "primary": "#2151FF",
    "primary-contrast": "#FFFFFF",
    "surface": "#0B0F14",
    "surface-contrast": "#E8EEF7",
    "muted": "#93A0B5",
    "accent": "#55E6C1",
    "warning": "#FFB020",
    "error": "#FF5D5D",
    "success": "#2ECC71"
  },
  "data": {
    "umap": ["#2151FF", "#55E6C1", "#FFB020", "#FF5D5D", "#C886FF", "#50B5FF", "#F56B8D"],
    "tsne": ["#0D99FF", "#00D5A0", "#FFC857", "#F25F5C", "#A27BFF", "#4DD0E1", "#F78FB3"],
    "heatmap-low": "#0B1020",
    "heatmap-mid": "#3C6E71",
    "heatmap-high": "#FFD166"
  },
  "semantic": {
    "text": "#E8EEF7",
    "text-muted": "#A7B2C5",
    "border": "#1E2633",
    "card": "#111827",
    "badge": {
      "info": "#2151FF",
      "ok": "#2ECC71",
      "warn": "#FFB020",
      "fail": "#FF5D5D"
    }
  }
}