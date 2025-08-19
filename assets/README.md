# Assets — SpectraMind V50

This directory holds **static resources** for documentation, dashboards, and GUIs.

## Structure
- `logos/` → Logos, icons, branding for SpectraMind and Ariel.
- `diagrams/` → Architecture diagrams (pipeline, CLI, symbolic layers).
- `plots/` → Example plots (spectrum, SHAP overlays, UMAP clusters).
- `gui/` → Stylesheets, color palettes, GUI mockups for diagnostics dashboard.
- `badges/` → Build, coverage, license badges.
- `fonts/` → Fonts for reports and dashboard rendering.

## Usage
- Referenced by `docs/` (MkDocs, notebooks).
- Embedded in diagnostics HTML (`generate_html_report.py`).
- Used in GitHub `README.md` for branding and badges.