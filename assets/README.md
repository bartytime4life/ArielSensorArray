
# assets/ — Repository‑wide static assets

This directory holds **non-code, static artifacts** used across the SpectraMind V50 project
(icons, logos, figures, color palettes, fonts, SVG UI parts, sample spectra images, etc.).
Everything here is safe to package or reference from docs, reports, or the CLI.

> **Do not put datasets or model checkpoints here.** Those belong under `data/` and are versioned with DVC.

---

## Directory layout

assets/
├─ brand/                # Logos, lockups, wordmarks (SVG/PNG), usage specs
├─ colors/               # Palettes (JSON/YAML), Matplotlib colormaps (.cmap/.json)
├─ fonts/                # Open-license fonts only (ttf/otf/woff2) + LICENSE files
├─ figures/              # Publication-ready figures (PNG/SVG/PDF), 300–600 DPI
├─ icons/                # Small UI icons (SVG preferred), single-color where possible
├─ ui/                   # Reusable UI parts: buttons/badges/diagrams as SVG
├─ spectra/              # Example spectra images (PNG) & tiny CSVs for docs/demos
├─ papers/               # One-page PDFs, posters, diagrams used in READMEs
└─ README.md             # You are here

You can add empty `.gitkeep` files to keep a subfolder tracked when it contains no assets yet.

---

## Naming conventions

- **kebab-case** filenames: `ariel-logo-dark.svg`, `h2o-band-1p4um.png`
- **No spaces**; use `-`.
- **Semantic suffixes**:
  - `-light` / `-dark` for theme variants
  - `@2x`, `@3x` for raster scale variants (if needed)
  - `-mono` for single‑color icons
- Prefer **vector first** (`.svg`). Fall back to PNG only when raster is necessary.

---

## Formats & quality

- **Logos/icons:** SVG (tiny, scalable). Keep strokes and text converted to outlines where licensing permits.
- **Figures:** SVG or PDF for print; PNG (≥300 DPI) for README/Kaggle notebooks.
- **Spectra images:** PNG, transparent background, axis‑aligned, text set in project font.
- **Colormaps:** Provide both a **Matplotlib** JSON and a plain JSON with hex stops.

### Example: `colors/magma-extended.json`
```json
{
  "name": "magma-extended",
  "type": "sequential",
  "stops": ["#000004","#1b0c41","#4a0c6b","#7e2482","#b5367a","#e55964","#fb8761","#fecf6b","#fcfdbf"]
}

Example: Matplotlib colormap (load in Python)

import json, matplotlib.pyplot as plt, matplotlib.colors as mcolors
from pathlib import Path

stops = json.loads(Path("assets/colors/magma-extended.json").read_text())["stops"]
cmap = mcolors.LinearSegmentedColormap.from_list("magma_extended", stops)

plt.register_cmap("magma_extended", cmap)
# usage: plt.imshow(img, cmap="magma_extended")


⸻

Licensing & attribution
	•	Only include open‑licensed assets you are allowed to redistribute.
	•	Every subfolder must contain a LICENSE or ATTRIBUTION.txt when 3rd‑party assets are present.
	•	For fonts, keep the upstream license alongside the files.
	•	For figures derived from published works, add a short caption file: figure-name.caption.md
with source and citation info.

⸻

Performance & repo hygiene
	•	Vectors over rasters — smaller diffs, crisp at any size.
	•	Compress rasters (PNGQuant/oxipng):
	•	PNG: target ≤ 300–600 DPI, strip metadata, sRGB profile.
	•	Keep individual assets ≤ 2 MB when possible (figures may exceed for print PDF).
	•	Avoid duplicate variants; prefer CSS/filters for color variants in docs where feasible.

If you must version large binaries (e.g., a poster PDF > 10 MB), store it in papers/ and consider Git LFS at the repository root (.gitattributes) instead of committing directly here. Large artifacts for experiments belong to DVC, not assets/.

⸻

How other modules consume assets/
	•	Docs/README: reference with relative paths: ![Logo](assets/brand/ariel-logo-dark.svg)
	•	CLI output (Rich/ASCII): embed small SVGs only when exporting HTML reports; terminal stays text‑only.
	•	Matplotlib styling: load colormap JSONs from assets/colors/ (see example above).
	•	Papers & posters: link PDFs from assets/papers/ in READMEs.

⸻

Checks & CI (optional but recommended)

If your repo uses CI, consider a lightweight check:
	•	Validate SVGs are minified and contain no external HTTP refs.
	•	Enforce file size limits (e.g., fail on >10 MB unless in papers/).
	•	Lint JSON/YAML palettes.

⸻

Quick checklist (before committing)
	•	File uses kebab-case, no spaces
	•	Vector preferred; rasters compressed
	•	License/attribution present if 3rd‑party
	•	Correct subfolder (brand/colors/fonts/figures/icons/ui/spectra/papers)
	•	Under size guidance (≤ 2 MB typical; exceptions documented)

⸻

Examples
	•	brand/ariel-logo-dark.svg — primary logomark (dark theme)
	•	icons/download-mono.svg — 16×16 UI glyph
	•	colors/magma-extended.json — sequential colormap for spectra heatmaps
	•	spectra/wasp-39b-water-band.png — sample annotated figure for docs
	•	papers/poster-ariel-challenge-2025.pdf — print‑ready A0 poster (with LICENSE or ATTRIBUTION)

⸻

Rationale

Keeping a clear, documented assets/ makes the repo:
	•	Maintainable: every file has a home and purpose
	•	Reproducible: figures and palettes used in docs/reports are tracked like code
	•	Portable: small, vector-first, license‑compliant artifacts travel well across environments

---

If you want the **next file**, tell me which you’d like me to generate, or I can proceed with one of these common follow‑ups (each as a standalone, copy‑pasteable file):

1) `assets/colors/magma-extended.json` (ready-to-use colormap)  
2) `assets/icons/download-mono.svg` (tiny UI icon example)  
3) Root `/.gitattributes` snippet to enable Git LFS for large PDFs in `assets/papers/`  

Just say **“next (1)”**, **“next (2)”**, or **“next (3)”** — or name any file/path you want.