
# assets/diagrams

**SpectraMind V50 — ArielSensorArray**  
*Repository diagrams, system schematics, and visualization sources*

---

## 📌 Purpose

This directory centralizes all **Mermaid diagrams, SVG schematics, PNG exports, and supporting visuals** used throughout the SpectraMind V50 project. It ensures that every architectural drawing, pipeline flow, and symbolic diagnostic graphic is:

- **Source-tracked** (Mermaid `.mmd` is the canonical source)
- **Exportable** (to `.svg`, `.png`, `.pdf` via GitHub Actions or `make diagrams`)
- **Reproducible** (diagram generation logged in CI, hashes stored in `v50_debug_log.md`)
- **Dashboard-ready** (HTML reports and notebooks load diagrams directly from here)

---

## 📂 Structure

assets/diagrams/
├── README.md                 # This file
├── architecture.mmd          # High-level pipeline architecture (Mermaid source)
├── architecture.svg          # Exported vector diagram
├── architecture.png          # Exported raster preview
├── cli_map.mmd               # CLI command → file/component map
├── cli_map.svg
├── cli_map.png
├── pipeline_flow.mmd         # DVC/Hydra/CLI pipeline flowchart
├── pipeline_flow.svg
├── pipeline_flow.png
├── symbolic_logic.mmd        # Symbolic logic engine, constraint graph
├── symbolic_logic.svg
├── symbolic_logic.png
└── test_diagrams.py          # Unit tests to validate .mmd → .svg/.png export integrity

---

## 🛠️ Workflow

1. **Edit Source**  
   Update the `.mmd` (Mermaid) file using any text editor or IDE with Mermaid preview.

2. **Export**  
   Run:
   ```bash
   make diagrams

or directly:

npx @mermaid-js/mermaid-cli -i assets/diagrams/architecture.mmd -o assets/diagrams/architecture.svg

	3.	Verify
Run the test:

pytest assets/diagrams/test_diagrams.py

This ensures:
	•	All .mmd have matching .svg/.png exports
	•	Files are referenced in ARCHITECTURE.md and report.html

⸻

📊 Integration
	•	Documentation
Diagrams are embedded in ARCHITECTURE.md, README.md, and docs in /docs/.
	•	Diagnostics Dashboard
HTML outputs (report.html, diagnostics_dashboard.html) include these diagrams inline.
	•	CI/CD
.github/workflows/mermaid-export.yml auto-renders .mmd sources to .svg/.png on PRs and pushes.

⸻

✅ Best Practices
	•	Always commit both the .mmd (source) and .svg/.png (exports).
	•	Keep diagrams minimal, legible, and color-blind safe.
	•	Use Mermaid class definitions for consistent styling.
	•	Cross-reference diagrams in ARCHITECTURE.md using relative paths.
	•	Update test_diagrams.py when adding a new diagram.

⸻

🔗 References
	•	Mermaid Documentation
	•	SpectraMind V50 ARCHITECTURE.md
	•	Diagnostics Dashboard
	•	.github/workflows/mermaid-export.yml

⸻
