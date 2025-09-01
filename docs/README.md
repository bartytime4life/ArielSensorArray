#!/usr/bin/env bash
set -euo pipefail

# Root of your repo (change if you’re running this from elsewhere)
REPO_ROOT="$(pwd)"

# ------------------------------------------------------------------------------
# 1) Create documentation hierarchy
# ------------------------------------------------------------------------------
mkdir -p "$REPO_ROOT/docs/references/physics"
mkdir -p "$REPO_ROOT/docs/references/ai"
mkdir -p "$REPO_ROOT/docs/references/cli"
mkdir -p "$REPO_ROOT/docs/references/gui"
mkdir -p "$REPO_ROOT/docs/references/kaggle"
mkdir -p "$REPO_ROOT/docs/references/meta"
mkdir -p "$REPO_ROOT/docs/references/philosophy"
mkdir -p "$REPO_ROOT/docs/project"

# Top-level docs README
cat > "$REPO_ROOT/docs/README.md" << 'EOF'
# 📚 SpectraMind V50 — Documentation

This `docs/` tree collects all technical references, physics background, AI/ML guides, CLI/GUI design materials, Kaggle challenge context, and project-specific plans/analyses that underpin SpectraMind V50’s NASA-grade, CLI-first, neuro-symbolic architecture.

**Subfolders**
- `references/physics/` – Physics, spectroscopy, optics, radiative transfer, astronomical systematics
- `references/ai/` – AI/ML engineering (Hydra, Hugging Face, decoding/encoding, Python/YAML)
- `references/cli/` – CLI design, UX, Master Coder Protocol
- `references/gui/` – GUI engineering references for the optional dashboard layer
- `references/kaggle/` – Kaggle platform mechanics and competitor baselines
- `references/meta/` – Docs tooling (e.g., Mermaid in GitHub)
- `references/philosophy/` – Vision and inspiration pieces
- `project/` – SpectraMind V50 plans, analyses, and strategy

All files here are **supporting documentation** (not source code). They anchor reproducibility, traceability, and scientific integrity across the repo.
EOF

# ------------------------------------------------------------------------------
# 2) Create per-folder READMEs with clear purpose and usage
# ------------------------------------------------------------------------------

# Physics
cat > "$REPO_ROOT/docs/references/physics/README.md" << 'EOF'
# 🔬 Physics & Astronomy References

Foundational references for the physics-informed aspects of SpectraMind V50:
- Radiative transfer, spectroscopy and “cosmic fingerprints”
- Observational limitations and mitigation (atmospheric seeing, diffraction, lensing)
- Radiation physics and computational physics methods

**Use in repo**
- Cite these docs in model cards, diagnostics reports, and symbolic rule notes.
- When adding a physics-based constraint or diagnostic, link the specific section you used here.
EOF

# AI/ML
cat > "$REPO_ROOT/docs/references/ai/README.md" << 'EOF'
# 🤖 AI / ML Engineering References

Core AI/ML implementation guides used across SpectraMind V50:
- Hydra configuration, Python/YAML patterns, decoding/encoding methods
- Hugging Face setup, training, model packaging, and deployment
- Advanced Python references for clean, type-hinted, testable code

**Use in repo**
- Reference when adding configs, training flows, or HF integrations.
- Keep snippets minimal here; code lives in `src/` with docstrings.
EOF

# CLI
cat > "$REPO_ROOT/docs/references/cli/README.md" << 'EOF'
# 🧰 CLI & UX References

CLI-first patterns and UX guidance aligned with the Master Coder Protocol:
- Discoverability (`--help` everywhere), consistency, progress feedback
- Scriptability, determinism, auditable logs

**Use in repo**
- When extending the Typer CLI, cross-check flags and help output with these guides.
EOF

# GUI
cat > "$REPO_ROOT/docs/references/gui/README.md" << 'EOF'
# 🖥️ GUI Engineering References (Optional Layer)

Cross-platform GUI design references (Qt/Electron/Flutter/React etc.).  
SpectraMind V50 is CLI-first; a GUI is optional and must mirror the CLI for reproducibility.

**Use in repo**
- If a dashboard or viewer is added, document the exact CLI artifacts it renders (HTML/PNG/JSON).
EOF

# Kaggle
cat > "$REPO_ROOT/docs/references/kaggle/README.md" << 'EOF'
# 🏁 Kaggle Platform & Challenge Context

Platform mechanics, notebooks/runtime constraints, leaderboard dynamics, and competitor baselines specific to the **NeurIPS 2025 Ariel Data Challenge**.

**Use in repo**
- Align submission packaging and runtime guardrails with these notes.
- Benchmark against competitor baselines documented here.
EOF

# Meta (docs tooling)
cat > "$REPO_ROOT/docs/references/meta/README.md" << 'EOF'
# 🧭 Meta — Documentation Tooling

Guides that improve repo documentation quality (e.g., Mermaid diagrams in GitHub Markdown).

**Use in repo**
- Place diagram `.mmd` blocks directly in Markdown with fenced ```mermaid code.
- Keep diagrams versioned alongside the docs they visualize.
EOF

# Philosophy
cat > "$REPO_ROOT/docs/references/philosophy/README.md" << 'EOF'
# 🌌 Vision & Philosophy

Inspiration and long-horizon framing for the project (e.g., quantum revolutions, cosmology/spectroscopy narratives). Useful for onboarding and design talks.
EOF

# Project
cat > "$REPO_ROOT/docs/project/README.md" << 'EOF'
# 🗺️ SpectraMind V50 — Project Docs

Project-specific materials:
- Technical plans and analysis
- Strategy for upgrades, gaps, and phased execution

**Use in repo**
- Treat these as the living blueprint; keep them synchronized with the implemented CLI/configs/tests.
EOF

# ------------------------------------------------------------------------------
# 3) (Optional) Move your uploaded files into their destinations
#    Adjust the LEFT-HAND PATHS below to match where the files currently live.
#    If they are already in the repo root, leave as-is.
# ------------------------------------------------------------------------------

move_if_exists () {
  local src="$1"
  local dst="$2"
  if [ -f "$src" ]; then
    mkdir -p "$(dirname "$dst")"
    git mv -f "$src" "$dst" 2>/dev/null || mv -f "$src" "$dst"
    echo "Moved: $src -> $dst"
  else
    echo "Skip (not found): $src"
  fi
}

# Physics
move_if_exists "$REPO_ROOT/Scientific Modeling and Simulation: A Comprehensive NASA-Grade Guide.pdf" \
               "$REPO_ROOT/docs/references/physics/Scientific Modeling and Simulation - NASA-Grade Guide.pdf"
move_if_exists "$REPO_ROOT/Computational Physics Modeling: Mechanics, Thermodynamics, Electromagnetism & Quantum.pdf" \
               "$REPO_ROOT/docs/references/physics/Computational Physics Modeling - MTEQ.pdf"
move_if_exists "$REPO_ROOT/Physics Modeling Using Computers: A Comprehensive Reference.pdf" \
               "$REPO_ROOT/docs/references/physics/Physics Modeling Using Computers - Reference.pdf"
move_if_exists "$REPO_ROOT/Gravitational Lensing and Astronomical Observation: Modeling and Mitigation.pdf" \
               "$REPO_ROOT/docs/references/physics/Gravitational Lensing - Modeling and Mitigation.pdf"
move_if_exists "$REPO_ROOT/Radiation: A Comprehensive Technical Reference.pdf" \
               "$REPO_ROOT/docs/references/physics/Radiation - Comprehensive Technical Reference.pdf"
move_if_exists "$REPO_ROOT/Cosmic Fingerprints .txt" \
               "$REPO_ROOT/docs/references/physics/Cosmic Fingerprints.txt"

# AI / ML
move_if_exists "$REPO_ROOT/AI Decoding and Processing Methods.pdf" \
               "$REPO_ROOT/docs/references/ai/AI Decoding and Processing Methods.pdf"
move_if_exists "$REPO_ROOT/AI Design and Modeling.pdf" \
               "$REPO_ROOT/docs/references/ai/AI Design and Modeling.pdf"
move_if_exists "$REPO_ROOT/Hydra for AI Projects: A Comprehensive Guide.pdf" \
               "$REPO_ROOT/docs/references/ai/Hydra for AI Projects - Comprehensive Guide.pdf"
move_if_exists "$REPO_ROOT/Advanced Python Mastery Reference Guide.pdf" \
               "$REPO_ROOT/docs/references/ai/Advanced Python Mastery Reference Guide.pdf"
move_if_exists "$REPO_ROOT/Advanced Python and YAML Configuration Guide.pdf" \
               "$REPO_ROOT/docs/references/ai/Advanced Python and YAML Configuration Guide.pdf"
move_if_exists "$REPO_ROOT/Hugging Face AI Mastery: Setup, Training, and Deployment Guide.pdf" \
               "$REPO_ROOT/docs/references/ai/Hugging Face AI Mastery - Setup Training Deployment.pdf"
move_if_exists "$REPO_ROOT/Using Hugging Face for the NeurIPS Ariel Data Challenge 2025.pdf" \
               "$REPO_ROOT/docs/references/ai/Using Hugging Face for the Ariel Data Challenge 2025.pdf"

# CLI
move_if_exists "$REPO_ROOT/Command Line Interfaces (CLI) Technical Reference (Master Coder Protocol).pdf" \
               "$REPO_ROOT/docs/references/cli/CLI Technical Reference - Master Coder Protocol.pdf"
move_if_exists "$REPO_ROOT/Mastering the Command Line: Comprehensive Guide to CLI Development and UX.pdf" \
               "$REPO_ROOT/docs/references/cli/Mastering the Command Line - CLI Dev and UX.pdf"
move_if_exists "$REPO_ROOT/Master Coder's Guide for Intermediate Programmers.pdf" \
               "$REPO_ROOT/docs/references/cli/Master Coder's Guide for Intermediate Programmers.pdf"

# GUI
move_if_exists "$REPO_ROOT/Engineering Guide to GUI Development Across Platforms.pdf" \
               "$REPO_ROOT/docs/references/gui/Engineering Guide to GUI Development Across Platforms.pdf"
move_if_exists "$REPO_ROOT/Comprehensive Guide to GUI Programming.pdf" \
               "$REPO_ROOT/docs/references/gui/Comprehensive Guide to GUI Programming.pdf"

# Kaggle
move_if_exists "$REPO_ROOT/Kaggle Platform: Comprehensive Technical Guide.pdf" \
               "$REPO_ROOT/docs/references/kaggle/Kaggle Platform - Comprehensive Technical Guide.pdf"
move_if_exists "$REPO_ROOT/Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf" \
               "$REPO_ROOT/docs/references/kaggle/Comparison of Kaggle Models - Ariel 2025.pdf"

# Meta
move_if_exists "$REPO_ROOT/Mermaid Diagrams in GitHub Markdown – Comprehensive Reference.pdf" \
               "$REPO_ROOT/docs/references/meta/Mermaid Diagrams in GitHub Markdown - Reference.pdf"

# Philosophy
move_if_exists "$REPO_ROOT/Light a 2nd Quantum Revolution .txt" \
               "$REPO_ROOT/docs/references/philosophy/Light a 2nd Quantum Revolution.txt"

# Project-specific
move_if_exists "$REPO_ROOT/SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf" \
               "$REPO_ROOT/docs/project/SpectraMind V50 - Technical Plan (Ariel 2025).pdf"
move_if_exists "$REPO_ROOT/SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf" \
               "$REPO_ROOT/docs/project/SpectraMind V50 - Project Analysis (Ariel 2025).pdf"
move_if_exists "$REPO_ROOT/Strategy for Updating and Extending SpectraMind V50 for NeurIPS 2025 Ariel Challenge.pdf" \
               "$REPO_ROOT/docs/project/SpectraMind V50 - Upgrade Strategy (Ariel 2025).pdf"

# ------------------------------------------------------------------------------
# 4) Root-level README pointers to docs (optional enhancement)
# ------------------------------------------------------------------------------
if [ -f "$REPO_ROOT/README.md" ]; then
  if ! grep -q "## Documentation" "$REPO_ROOT/README.md"; then
    cat >> "$REPO_ROOT/README.md" << 'EOF'

## Documentation

- See [`docs/`](docs/) for physics, AI/ML, CLI/GUI, Kaggle, and project-specific references that support
  SpectraMind V50’s reproducible, physics-informed pipeline.
EOF
  fi
fi

# ------------------------------------------------------------------------------
# 5) Show resulting tree
# ------------------------------------------------------------------------------
echo
echo "Final docs tree:"
( cd "$REPO_ROOT" && \
  find docs -maxdepth 3 -type d -print | sort )

echo
echo "Done. Review file moves above, then:"
echo "  git add -A && git commit -m 'docs: add structured references + READMEs for SpectraMind V50'"
