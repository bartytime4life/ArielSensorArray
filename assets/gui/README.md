# SpectraMind V50 — GUI Assets

This folder contains the **visual design assets, style tokens, and configuration** for the SpectraMind V50 diagnostics dashboard and optional GUI front-ends (Qt, Electron/React).  
All assets here are portable and themeable — the **CLI is the source of truth**, and GUIs are optional thin layers.

---

## 📦 Contents
- `color_palette.json` — Design tokens (brand colors, semantic roles, data overlays).  
- `widget_styles.css` — CSS variables for widgets and plots.  
- `icons/` — SVG/PNG symbols (status, filters, rules).  
- `images/` — Diagrams for docs/dashboards.  
- `README.md` — This file.  
- `ARCHITECTURE.md` — Integration diagram (CLI ↔ GUI).  

---

## 🔌 Usage
- **Dashboard** (`spectramind diagnose dashboard`) → hydrates `color_palette.json` + `widget_styles.css`.  
- **Web GUI (React/Electron)** → imports same tokens.  
- **Native GUI (Qt/PySide)** → maps palette values to QSS.  

---

## 🎨 Theming
Example palette:

```json
{
  "brand": { "primary": "#2151FF", "accent": "#55E6C1" },
  "semantic": { "surface": "#0B0F14", "text": "#E8EEF7" },
  "data": { "umap": ["#2151FF", "#55E6C1"], "heatmap-high": "#FFD166" }
}

CSS variables:

:root {
  --sm-color-primary: #2151FF;
  --sm-surface: #0B0F14;
  --sm-text: #E8EEF7;
  --sm-heat-high: #FFD166;
}


⸻

♿ Accessibility
	•	WCAG AA contrast
	•	Keyboard navigable
	•	ARIA roles & tooltips
	•	Respects prefers-reduced-motion

⸻

🔐 Security
	•	Static assets only
	•	Cache-bust via assets-manifest.json
	•	No secrets in assets

⸻

🧪 Validation
	•	tools/validate_palette.py → schema + WCAG checks
	•	Lint widget_styles.css
	•	Smoke test: poetry run spectramind diagnose dashboard --open

⸻

🔄 Versioning

Each palette update bumps meta.version. Dashboard logs palette version + hash to logs/v50_debug_log.md.

⸻

❓ FAQ

Change palette per report?
Yes → --palette path/to/json.

Dark/light mode?
Use alternate JSON (e.g. color_palette.light.json).

---

### **`assets/gui/ARCHITECTURE.md`**
```markdown
# GUI Integration Architecture — SpectraMind V50

How GUI layers (HTML dashboard, React/Electron, or Qt) consume assets from `assets/gui/` and integrate with the CLI-first pipeline.

---

## 🧭 Flow

+------------------+
|  CLI (spectramind)|
+---------+--------+
          v
 Diagnostics + Reports
          |
  +-------+-------+
  |               |
HTML Dashboard   FastAPI (optional)
  |               |
  +-- uses assets/gui --+
          |
   ┌─────────────────────────────┐
   │ Optional GUIs               │
   │ • Web (React/Electron)      │
   │ • Desktop (Qt/PySide)       │
   │ • All import tokens & CSS   │
   └─────────────────────────────┘

---

## 🧱 Layers
1. **CLI** → generates diagnostics, logs palette version.  
2. **HTML Report** → hydrates CSS vars from JSON.  
3. **FastAPI (optional)** → read-only endpoints (`/api/diagnostics/summary`, `/api/assets/palette`).  
4. **GUI shells** → import tokens (React/Electron/Qt).  

---

## 🎛 Token Hydration
React:
```ts
const tokens = await fetch('/assets/gui/color_palette.json').then(r => r.json());
document.documentElement.style.setProperty('--sm-color-primary', tokens.brand.primary);

Qt (PySide):

palette = load_json("assets/gui/color_palette.json")
app.setStyleSheet(f":root {{ --sm-color-primary: {palette['brand']['primary']}; }}")


⸻

📊 Data Contracts
	•	diagnostic_summary.json → metrics, latent coords, violations.
	•	Images (PNG/SVG) → must follow palette colors.

⸻

🛡 Security
	•	HTML report = static
	•	FastAPI = read-only
	•	Electron = disable nodeIntegration, enforce CSP
	•	Sanitize SVG icons

⸻

⚡ Performance
	•	Lazy-load heavy scatter plots
	•	Cache /assets/gui/* with hashed filenames
	•	Use WebGL for large point clouds

⸻

📸 Testing
	•	Palette validator (schema + WCAG)
	•	Snapshot: mini dashboard render
	•	API contract: JSON keys

⸻

🔄 Traceability

Dashboards must log:
	•	Palette version + hash
	•	Config + data hashes
	•	Timestamp

Ensures byte-for-byte reproducibility across GUI and CLI.

---