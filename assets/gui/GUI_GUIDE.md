# 🎨 SpectraMind V50 — GUI Guide

This guide documents how **SpectraMind V50** manages all **graphical user interface (GUI) assets**, including style tokens, color palettes, and integration layers between the **CLI-first pipeline** and optional GUI dashboards.

---

## 🧭 Principles

1. **CLI is the Source of Truth**  
   All GUIs (HTML dashboard, Qt, Electron/React) are **thin shells** over the CLI + FastAPI endpoints.  
   Assets here must be **portable, reproducible, and themeable**.

2. **GUI-Agnostic Design**  
   JSON palettes, CSS variables, and icon sets can be applied in:
   - CLI-generated static dashboards
   - Web GUIs (React/Electron)
   - Native GUIs (Qt/PySide)

3. **Accessibility & Security**  
   - WCAG AA contrast verified  
   - ARIA roles + keyboard navigation  
   - No embedded secrets or inline scripts  

---

## 📂 Directory Contents

- `color_palette.json` — Design tokens (brand + semantic + data ramps).  
- `widget_styles.css` — Shared widget styles (framework-agnostic).  
- `icons/` — SVG/PNG for dashboards (status, warning, molecules).  
- `images/` — Diagrams + illustrations for docs and dashboards.  
- `assets-manifest.json` — Integrity hashes + versioned mapping.  
- `GUI_GUIDE.md` — This file.  
- `ARCHITECTURE.md` — End-to-end GUI ↔ CLI integration diagram.  

---

## 🔌 Usage

- **CLI HTML Dashboard**  
  `spectramind diagnose dashboard --palette assets/gui/color_palette.json`  
  Hydrates CSS vars, colors UMAP/t-SNE plots, symbolic overlays, and tables.

- **React/Electron GUI**  
  Imports tokens (`color_palette.json`, `widget_styles.css`) for consistent theming.  

- **Qt (PySide/PyQt)**  
  Loads palette JSON → maps to QSS (Qt Style Sheets).  

---

## 🎨 Theming & Tokens

### JSON Palette Example

```json
{
  "brand": {
    "primary": "#2151FF",
    "accent": "#55E6C1",
    "warning": "#FFB020",
    "error": "#FF5D5D",
    "success": "#2ECC71"
  },
  "semantic": {
    "surface": "#0B0F14",
    "surface-contrast": "#E8EEF7",
    "text": "#E8EEF7",
    "text-muted": "#A7B2C5"
  },
  "data": {
    "umap": ["#2151FF", "#55E6C1", "#FFB020", "#FF5D5D"],
    "heatmap-low": "#0B1020",
    "heatmap-high": "#FFD166"
  },
  "meta": {
    "version": "1.2.0",
    "updated": "2025-08-25",
    "hash": "sha256-abc123..."
  }
}

CSS Contract

:root {
  --sm-color-primary: #2151FF;
  --sm-color-accent: #55E6C1;
  --sm-color-warning: #FFB020;
  --sm-color-error: #FF5D5D;
  --sm-color-success: #2ECC71;

  --sm-surface: #0B0F14;
  --sm-surface-contrast: #E8EEF7;
  --sm-text: #E8EEF7;
  --sm-text-muted: #A7B2C5;

  --sm-heat-low: #0B1020;
  --sm-heat-high: #FFD166;
}


⸻

♿ Accessibility
	•	Contrast: WCAG AA+ compliance
	•	Navigation: Full keyboard support
	•	Motion: Respects prefers-reduced-motion
	•	Icons: Provide aria-label + tooltips

⸻

🔐 Security
	•	All assets = static only
	•	Manifest: assets-manifest.json with sha256 hashes
	•	Cache-busting via hash-based filenames
	•	Sanitized SVGs only

⸻

🧪 Validation
	1.	Palette → run tools/validate_palette.py (schema + WCAG).
	2.	CSS → lint via Stylelint.
	3.	Smoke Test →

poetry run spectramind diagnose dashboard --open



⸻

🔄 Versioning
	•	Each palette change = bump meta.version.
	•	Dashboard footer shows palette version + hash.
	•	All runs log to logs/v50_debug_log.md.

⸻

❓ FAQ

Q: Can I use a different palette per report?
A: Yes → --palette path/to/custom.json.

Q: Does it support light mode?
A: Yes → keep color_palette.light.json + color_palette.dark.json, select at runtime.

⸻

📊 GUI Integration Diagram

flowchart TD
  CLI["Typer CLI\n`spectramind`"] --> DIAG["Diagnostics Engine\nFFT • UMAP • SHAP"]
  DIAG --> JSON["diagnostic_summary.json"]
  DIAG --> PNG["plots/*.png, *.svg"]

  PALETTE["color_palette.json"] --> HTML["HTML Dashboard"]
  CSS["widget_styles.css"] --> HTML
  ICONS["icons/*"] --> HTML
  MANIFEST["assets-manifest.json"] -.-> HTML

  HTML --> REACT["Optional Web GUI (React/Electron)"]
  HTML --> QT["Optional Desktop GUI (Qt/PySide)"]


⸻

✅ Bottom line: /assets/gui is the theme + integration layer ensuring SpectraMind V50 diagnostics look the same across CLI dashboards, React GUIs, and Qt shells.