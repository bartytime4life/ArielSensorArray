# SpectraMind V50 — GUI Assets

This folder contains the **visual design assets, style tokens, and GUI-facing configuration** used by the SpectraMind V50 diagnostics dashboard and any optional GUI front-ends (Qt, Electron/React). It is **GUI-agnostic** by design: everything here can be consumed by the CLI-generated HTML dashboard, a web frontend, or a native desktop shell.

> **Principle**: the **CLI is the source of truth**. GUIs (including the HTML diagnostics dashboard) are thin, optional layers on top of the CLI and FastAPI endpoints. All assets here must remain portable, reproducible, and themeable.

---

## 📦 Contents

- `color_palette.json` — Design tokens (brand colors, semantic roles, data overlays, UMAP/cluster ramps).
- `widget_styles.css` — Minimal, framework-neutral CSS for shared widgets.
- `icons/` — SVG/PNG symbols for dashboards (status, warning, success, filters, rules).
- `images/` — Diagrams/illustrations used by the HTML dashboard / docs.
- `README.md` — This file.
- `ARCHITECTURE.md` — GUI ↔ CLI integration diagram.

---

## 🔌 Usage

- **HTML dashboard** (`spectramind diagnose dashboard`): reads `color_palette.json` and `widget_styles.css` to style UMAP/t-SNE plots, symbolic overlays, and tables.
- **Web GUI (React/Electron)**: imports tokens for consistent theming across pages.
- **Native GUI (Qt/PySide)**: maps palette values to QSS/Qt themes.

---

## 🎨 Theming & Tokens

### JSON palette example

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
  }
}

CSS variable contract

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
	•	WCAG AA contrast for text.
	•	Keyboard-navigable components.
	•	ARIA roles and tooltips.
	•	Animations respect prefers-reduced-motion.

⸻

🔐 Security
	•	Static assets only (no inline scripts).
	•	Hash assets in assets-manifest.json for cache-busting.
	•	No secrets embedded in GUI assets.

⸻

🧪 Validation
	•	Run tools/validate_palette.py to check JSON schema + WCAG contrast.
	•	Lint widget_styles.css (Stylelint).
	•	Smoke test: poetry run spectramind diagnose dashboard --open.

⸻

🔄 Versioning

Every palette update bumps meta.version in JSON. Dashboard embeds palette version + hash in its footer and logs to logs/v50_debug_log.md.

⸻

❓ FAQ

Q: Can I change the palette per report?
Yes. Pass --palette path/to/json to spectramind diagnose dashboard.

Q: Dark/light mode?
Maintain separate JSONs (e.g. color_palette.light.json). The loader hydrates the correct one at runtime.

---

Now here’s the second one — **`assets/gui/ARCHITECTURE.md`** in a separate box:

```markdown
# GUI Integration Architecture — SpectraMind V50

This document explains how GUI layers (HTML dashboard, React/Electron, or Qt) consume assets from `assets/gui/` and integrate with the CLI-first V50 pipeline.

---

## 🧭 High-level Flow

+———————––+
|   Typer CLI (spectramind)|
+———–+———––+
|
v
Diagnostics & Reports
|
+——+——+
|             |
HTML Dashboard   FastAPI (optional)
|             |
+— uses assets/gui —+
|
┌────────────────────────────┐
│ Optional GUIs              │
│  • Web (React/Electron)    │
│  • Desktop (Qt/PySide)     │
│  Consume tokens + CSS vars │
└────────────────────────────┘

---

## 🧱 Layers

1. **CLI (Typer)** — runs diagnostics, writes JSON + PNG + HTML, logs palette version.  
2. **HTML Report Generator** — hydrates CSS variables from JSON, embeds plots + overlays.  
3. **FastAPI (optional)** — read-only API for GUIs (`/api/diagnostics/summary`, `/api/assets/palette`).  
4. **GUI shells** — React/Electron or Qt front-ends, all import same tokens.

---

## 🎛 Token Hydration Example

React:

```ts
const tokens = await fetch('/assets/gui/color_palette.json').then(r => r.json());
document.documentElement.style.setProperty('--sm-color-primary', tokens.brand.primary);

Qt (PySide):

palette = load_json("assets/gui/color_palette.json")
app.setStyleSheet(f":root {{ --sm-color-primary: {palette['brand']['primary']}; }}")


⸻

📊 Data Contracts
	•	diagnostic_summary.json — metrics (GLL, entropy, violations, FFT) + latent coords.
	•	Images — static PNG/SVG, GUIs may re-render with Plotly/Qt but must follow palette colors.

⸻

🛡 Security
	•	HTML report = static, no dynamic code.
	•	FastAPI = read-only, CORS-restricted.
	•	Electron = disable nodeIntegration, enforce CSP.
	•	Use only sanitized SVGs for icons.

⸻

⚡ Performance
	•	Lazy-load heavy scatter plots.
	•	Cache /assets/gui/* with hash-based filenames.
	•	WebGL for large point clouds.

⸻

📸 Testing
	•	Palette validator (schema + WCAG).
	•	Snapshot tests: render mini dashboard, compare output.
	•	API contract tests: ensure required keys exist in JSON.

⸻

🔄 Traceability

Every dashboard must log:
	•	Palette version + hash.
	•	Config + data hashes.
	•	Timestamp of generation.

This ensures byte-for-byte reproducibility across GUI and CLI runs.

---