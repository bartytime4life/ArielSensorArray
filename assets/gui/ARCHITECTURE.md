# GUI Integration Architecture — SpectraMind V50

This document explains how GUI layers (HTML dashboard, React/Electron, or Qt) consume assets from `assets/gui/` and integrate with the CLI-first V50 pipeline.

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
**React**
```ts
const tokens = await fetch('/assets/gui/color_palette.json').then(r => r.json());
document.documentElement.style.setProperty('--sm-color-primary', tokens.brand.primary);

Qt (PySide)

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

This ensures byte-for-byte reproducibility across GUI and CLI.

