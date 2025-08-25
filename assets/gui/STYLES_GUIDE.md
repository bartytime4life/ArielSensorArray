# SpectraMind V50 — GUI Styles Guide (`/assets/GUI/STYLES_GUIDE.md`)

> **Scope**  
> This guide defines the visual language, design tokens, components, and interaction rules for every GUI surface in **SpectraMind V50** — including the HTML diagnostics dashboard, optional desktop shells, and any future thin UIs layered on top of the CLI-first pipeline. It exists to ensure **consistency, accessibility, and reproducibility** across all front-ends while keeping the **CLI as the source of truth** and the GUI as an **optional, lightweight layer**. [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:1‡Ubuntu CLI-Driven Architecture for Large-Scale Scientific Data Pipelines (NeurIPS 2025 Ariel Challen.pdf](file-service://file-Fdr46UbCyD9vDBpXSk9Yi1) [oai_citation:2‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL)

---

## 0) North Star Principles

1) **CLI-first, GUI-thin**  
   - All operations must be invokable and auditable via the Typer/Hydra CLI; GUIs surface the same capabilities without adding hidden state. [oai_citation:3‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:4‡Ubuntu CLI-Driven Architecture for Large-Scale Scientific Data Pipelines (NeurIPS 2025 Ariel Challen.pdf](file-service://file-Fdr46UbCyD9vDBpXSk9Yi1)  
   - Prefer headless generation of plots/HTML + static assets for portability; live views are read-only windows onto CLI artifacts. [oai_citation:5‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w)

2) **Reproducibility by design**  
   - Style configuration (themes, palettes, scales) is **config-as-code** with Hydra YAMLs; theme switches are deterministic and logged alongside run hashes. [oai_citation:6‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL) [oai_citation:7‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w)  
   - No style override that changes semantic meaning (e.g., alert colors) may occur outside committed tokens or Hydra configs.

3) **Event‑driven & declarative UI**  
   - UIs are event-driven; favor **retained/declarative** patterns (React/SwiftUI/QML) over ad‑hoc imperative redrawing for maintainability and testability. [oai_citation:8‡Engineering Guide to GUI Development Across Platforms.pdf](file-service://file-83fWWTfizJB76rJs6Hq4KD) [oai_citation:9‡Comprehensive Guide to GUI Programming.pdf](file-service://file-NiTQ7cdQw7zGnLUVCUpoRx)

4) **Accessible to everyone**  
   - Meet or exceed WCAG 2.1 AA: color contrast, keyboard nav, focus order, reduced motion, and screen-reader semantics.

5) **Scientific clarity beats flash**  
   - Visualizations prioritize legibility, consistent encodings, and domain semantics over spectacle. Keep legends, units, and uncertainty explicit. [oai_citation:10‡Scientific Modeling and Simulation: A Comprehensive NASA-Grade Guide.pdf](file-service://file-6bN8o3KiMBECrKT1k7HGhY)

---

## 1) Assets & Files

- `/assets/GUI/color_palette.json` — **Design tokens** for color (brand, semantic, data-viz ramps).  
- `/assets/GUI/widget_styles.css` — **Baseline CSS** implementing tokens as CSS variables + primitive utilities.  
- `/assets/GUI/assets-manifest.json` — **Declared inventory** (icons, fonts, sprites, ramp PNGs) with hashes for cache busting and provenance.  
- `/assets/GUI/STYLES_GUIDE.md` — **This document**.

> **Rule:** All UI code (web/desktop) reads style values from tokens (JSON/CSS vars) — never hardcode literal colors, spacing, or radii in components.

---

## 2) Design Tokens

Tokens are the single source of visual truth.

### 2.1 Naming & Structure

**JSON (authoritative)**
```json
{
  "color": {
    "brand": { "primary": "#3C7DFF", "secondary": "#00D1B2", "accent": "#F45D48" },
    "neutral": { "0": "#0B0D10", "100": "#12161C", "200": "#1A202A", "300": "#232B36", "400": "#2E3947", "500": "#3B495C", "600": "#50627B", "700": "#6E86A6", "800": "#9CB0CC", "900": "#D8E2EE", "1000": "#FFFFFF" },
    "semantic": {
      "info": "#3C7DFF",
      "success": "#22C55E",
      "warning": "#F59E0B",
      "danger": "#EF4444"
    },
    "viz": {
      "umap": ["#3C7DFF","#00D1B2","#F45D48","#F59E0B","#8B5CF6","#10B981","#EC4899","#14B8A6"],
      "fft":  ["#0EA5E9","#22D3EE","#A3E635","#F59E0B","#EF4444"],
      "molecule": {
        "H2O": "#3C7DFF",
        "CO2": "#EF4444",
        "CH4": "#22C55E",
        "CO":  "#F59E0B",
        "NH3": "#8B5CF6"
      }
    }
  },
  "typography": {
    "family": { "sans": "Inter, system-ui, Segoe UI, Roboto, Helvetica, Arial, sans-serif", "mono": "JetBrains Mono, SFMono-Regular, Menlo, Consolas, monospace" },
    "scale": { "xs": 12, "sm": 14, "md": 16, "lg": 18, "xl": 20, "2xl": 24, "3xl": 30, "4xl": 36 },
    "lineHeight": { "tight": 1.25, "normal": 1.5, "loose": 1.7 }
  },
  "space": { "1": 4, "2": 8, "3": 12, "4": 16, "5": 24, "6": 32, "7": 40, "8": 48 },
  "radius": { "sm": 6, "md": 12, "lg": 16, "xl": 24 },
  "shadow": {
    "sm": "0 1px 2px rgba(0,0,0,.10)",
    "md": "0 4px 12px rgba(0,0,0,.12)",
    "lg": "0 10px 24px rgba(0,0,0,.16)"
  }
}

CSS variables (runtime)

:root{
  /* brand */
  --sm-color-brand-primary:#3C7DFF;
  --sm-color-brand-secondary:#00D1B2;
  --sm-color-brand-accent:#F45D48;

  /* semantic */
  --sm-color-info:#3C7DFF;
  --sm-color-success:#22C55E;
  --sm-color-warning:#F59E0B;
  --sm-color-danger:#EF4444;

  /* neutrals */
  --sm-color-bg:#0B0D10;
  --sm-color-surface:#12161C;
  --sm-color-elev-1:#1A202A;
  --sm-color-elev-2:#232B36;
  --sm-color-border:#2E3947;
  --sm-color-text:#D8E2EE;
  --sm-color-text-inverse:#0B0D10;

  /* typography */
  --sm-font-sans:Inter,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
  --sm-font-mono:JetBrains Mono,SFMono-Regular,Menlo,Consolas,monospace;
  --sm-font-xs:12px;--sm-font-sm:14px;--sm-font-md:16px;--sm-font-lg:18px;--sm-font-xl:20px;--sm-font-2xl:24px;--sm-font-3xl:30px;--sm-font-4xl:36px;
  --sm-lh-tight:1.25;--sm-lh-normal:1.5;--sm-lh-loose:1.7;

  /* spacing (4px grid) */
  --sm-space-1:4px;--sm-space-2:8px;--sm-space-3:12px;--sm-space-4:16px;--sm-space-5:24px;--sm-space-6:32px;--sm-space-7:40px;--sm-space-8:48px;

  /* radius & elevation */
  --sm-radius-sm:6px;--sm-radius-md:12px;--sm-radius-lg:16px;--sm-radius-xl:24px;
  --sm-shadow-sm:0 1px 2px rgba(0,0,0,.10);
  --sm-shadow-md:0 4px 12px rgba(0,0,0,.12);
  --sm-shadow-lg:0 10px 24px rgba(0,0,0,.16);
}

Why tokens? They enable config-as-code theming with Hydra, keep the GUI thin & portable, and ensure consistent visual semantics across platforms. ￼ ￼

⸻

3) Theming & Modes
	•	Dark mode is default (diagnostics are often viewed on dark controls in labs).
	•	Light mode is supported; consumers switch by toggling a root class or prefers-color-scheme.
	•	High contrast variant increases contrasts by +25% and removes decorative shadows.

Switch example

/* light */
:root.theme-light{
  --sm-color-bg:#FFFFFF;
  --sm-color-surface:#F7FAFC;
  --sm-color-text:#0B0D10;
  --sm-color-border:#E2E8F0;
}

/* high contrast augment */
:root.hc{
  --sm-shadow-sm:none;--sm-shadow-md:none;--sm-shadow-lg:none;
  --sm-color-border:color-mix(in oklab, var(--sm-color-border) 70%, black 30%);
}

Hydra can select theme=dark|light|hc to generate dashboard variants during CLI export, and every export logs theme + token hash for provenance. ￼ ￼

⸻

4) Typography
	•	Families: Inter for UI & long-form; JetBrains Mono for code/data.
	•	Hierarchy:
	•	H1 --sm-font-4xl / --sm-lh-tight
	•	H2 --sm-font-3xl / --sm-lh-tight
	•	H3 --sm-font-2xl / --sm-lh-tight
	•	Body --sm-font-md / --sm-lh-normal
	•	Caption/Meta --sm-font-sm or --sm-font-xs
	•	Rules: Avoid text below 12px; body contrast ≥ 4.5:1; display contrast ≥ 3:1.

⸻

5) Layout & Spacing
	•	Grid: 4px base; multiples of 8px for primary rhythm.
	•	Breakpoints: sm 640, md 768, lg 1024, xl 1280 (px).
	•	Containers: Max width 1280px on wide screens; content gutters 24–32px.
	•	White space: Use --sm-space-* tokens; never arbitrary pixel literals.

⸻

6) Iconography
	•	Icons are monochrome SVGs tinted with currentColor.
	•	Minimum target area 32×32 px; stroke 1.5–2 px for legibility at small sizes.

⸻

7) Color Guidance (Semantics & Data Viz)

7.1 Semantic usage
	•	Info --sm-color-info — neutral status/metadata.
	•	Success --sm-color-success — pass/coverage OK.
	•	Warning --sm-color-warning — potential issue, user attention recommended.
	•	Danger --sm-color-danger — violation/failure.

Do not repurpose danger for non-error categories.

7.2 Data visualization ramps
	•	UMAP/t‑SNE: color.viz.umap qualitative palette (distinct, color‑blind aware).
	•	FFT/Power: sequential/cyclic ramp color.viz.fft (cool→warm) with fixed legend.
	•	Molecular fingerprints: deterministic mapping via color.viz.molecule (e.g., H₂O=blue, CO₂=red) to keep interpretability across views.

Use consistent encodings to help users recognize patterns and avoid accidental misinterpretation — a core requirement for scientific clarity. ￼

⸻

8) Components (Baseline)

All components use the same tokens and behavioral patterns across platforms.

8.1 Card

.sm-card{
  background:var(--sm-color-surface);
  color:var(--sm-color-text);
  border:1px solid var(--sm-color-border);
  border-radius:var(--sm-radius-lg);
  box-shadow:var(--sm-shadow-sm);
  padding:var(--sm-space-5);
}
.sm-card--elevated{ box-shadow:var(--sm-shadow-md); }

8.2 Button

.sm-btn{
  --bg:var(--sm-color-brand-primary);
  --fg:#fff;
  background:var(--bg); color:var(--fg);
  border:1px solid color-mix(in oklab, var(--bg) 80%, black 20%);
  border-radius:var(--sm-radius-md);
  padding:calc(var(--sm-space-2) + 2px) var(--sm-space-4);
  font:600 var(--sm-font-sm)/var(--sm-lh-tight) var(--sm-font-sans);
}
.sm-btn:is(:hover,:focus-visible){ filter:brightness(1.06); }
.sm-btn:disabled{ opacity:.45; cursor:not-allowed; }

.sm-btn--secondary{ --bg:var(--sm-color-brand-secondary); }
.sm-btn--danger{ --bg:var(--sm-color-danger); }
.sm-btn--ghost{
  --bg:transparent; --fg:var(--sm-color-text);
  background:transparent; color:var(--fg);
  border:1px solid var(--sm-color-border);
}

8.3 Tabs

.sm-tabs{ display:flex; gap:var(--sm-space-3); border-bottom:1px solid var(--sm-color-border); }
.sm-tab{
  padding:var(--sm-space-3) var(--sm-space-4);
  color:var(--sm-color-text); opacity:.8;
  border-bottom:2px solid transparent;
}
.sm-tab[aria-selected="true"]{
  opacity:1; border-bottom-color:var(--sm-color-brand-primary);
  font-weight:600;
}

8.4 Table (Dense)

.sm-table{ width:100%; border-collapse:separate; border-spacing:0; }
.sm-table th, .sm-table td{
  padding:var(--sm-space-3) var(--sm-space-4);
  border-bottom:1px solid var(--sm-color-border);
}
.sm-table thead th{ color:var(--sm-color-text); opacity:.85; font-weight:600; }
.sm-table tr:hover td{ background:color-mix(in oklab, var(--sm-color-surface) 85%, white 15%); }

8.5 Alert

.sm-alert{ border-radius:var(--sm-radius-md); padding:var(--sm-space-4); border:1px solid; }
.sm-alert--info{    border-color:color-mix(in oklab, var(--sm-color-info) 35%, black 15%);    background:color-mix(in oklab, var(--sm-color-info) 12%, var(--sm-color-surface)); }
.sm-alert--success{ border-color:color-mix(in oklab, var(--sm-color-success) 35%, black 15%); background:color-mix(in oklab, var(--sm-color-success) 12%, var(--sm-color-surface)); }
.sm-alert--warning{ border-color:color-mix(in oklab, var(--sm-color-warning) 35%, black 15%); background:color-mix(in oklab, var(--sm-color-warning) 12%, var(--sm-color-surface)); }
.sm-alert--danger{  border-color:color-mix(in oklab, var(--sm-color-danger) 35%, black 15%);  background:color-mix(in oklab, var(--sm-color-danger) 12%, var(--sm-color-surface)); }


⸻

9) Interaction Patterns
	•	Event Loop & Handlers: Use framework-native event loops; never block the UI thread. Offload long work to background tasks triggered by CLI runs or async calls. ￼
	•	Declarative Bindings: Favor MVVM/React-style bindings; make view state a pure function of model state to avoid drift. ￼
	•	Immediate feedback: Show progress bars/spinners during long operations; reflect CLI job status through log streaming and status chips (Idle/Running/Success/Fail). For CLI UX, follow helpful progress + structured logs best practices. ￼

⸻

10) Charts & Scientific Visuals
	•	Axes & Units: Always display units, scales, and uncertainty bars or shaded intervals when applicable.
	•	Legends: Keep stable order and color mapping across pages/runs (H₂O is always blue, etc.).
	•	Accessibility: Provide color-blind safe alternatives; ensure 3:1 contrast for line charts against surfaces.
	•	Export: Charts must export to PNG/SVG with embedded token hash & theme in the metadata for auditability.

Scientific diagnostics must be auditable and reproducible; embed config hashes, theme version, and data snapshot IDs in exports. ￼ ￼

⸻

11) Accessibility (WCAG 2.1 AA)
	•	Keyboard: All interactive elements are reachable in logical order; visible focus ring ≥ 2px with ≥ 3:1 contrast.
	•	ARIA: Use appropriate roles (tablist, tab, tabpanel, status, progressbar, etc.).
	•	Motion: Respect prefers-reduced-motion; animations are opt-in and under 200ms.
	•	Color: Never use color alone to convey meaning; add icons/labels.

⸻

12) Performance & Footprint
	•	Thin shell: Prefer static HTML + CSS + minimal JS; avoid heavy runtime GUIs unless needed.
	•	Budget: Initial dashboard load ≤ 200KB CSS, ≤ 300KB JS (gzipped) on typical screens.
	•	Lazy load: Defer charts & large modules until visible.
	•	Cache busting: All assets referenced via assets-manifest.json with content hashes.

This aligns with the CLI‑first mandate and keeps the GUI an efficient window into CLI outputs. ￼ ￼

⸻

13) Platform Notes

Web (HTML/React)
	•	Preferred for the diagnostics dashboard.
	•	Follow the retained/declarative model; bind component props to structured JSON outputs from CLI runs. ￼ ￼

Desktop (Qt/QML)
	•	Use QML for declarative bindings; load tokens as QML Theme singletons and bind properties (colors, radii, spacing). ￼

CLI overlays
	•	When presenting summaries inside terminal UIs, mirror styles semantically (e.g., info/success/warning/danger color mapping via Rich) and never add capabilities that are not available through CLI flags. ￼

⸻

14) Integration with CLI & Hydra
	•	Hydra integration:
	•	configs/ui/theme.yaml selects dark|light|hc and injects token file path(s).
	•	Exports include theme, token_hash, run_hash, and data_hash in an audit banner inside HTML. ￼ ￼
	•	Logging: The GUI must render summaries from structured logs (JSONL) produced by the pipeline; no GUI-only data sources are permitted. ￼

⸻

15) Coding Standards (Web)
	•	No hardcoded magic values — use CSS variables/tokens.
	•	BEM-ish utility classes for primitives (.sm-card, .sm-btn, .sm-table…).
	•	Testing:
	•	Visual regression snapshots per theme (dark/light/hc).
	•	a11y checks (axe-core) in CI.
	•	Storybook (or equivalent) for tokens and component states.

⸻

16) Examples

16.1 Card + Chart shell (HTML)

<section class="sm-card">
  <header style="display:flex;justify-content:space-between;align-items:center;margin-bottom:var(--sm-space-4)">
    <h2 style="font:600 var(--sm-font-xl)/var(--sm-lh-tight) var(--sm-font-sans);margin:0">FFT Power Spectrum</h2>
    <span class="sm-badge sm-badge--info" aria-label="Status: ready">Ready</span>
  </header>
  <figure id="fft-chart" role="img" aria-label="FFT power spectrum of μ with normalized frequency"></figure>
  <figcaption style="opacity:.8;margin-top:var(--sm-space-3)">Theme:<code>dark</code> • Tokens:<code>2a6e…</code> • Run:<code>c9f1…</code></figcaption>
</section>

16.2 React token hook (pseudo)

const useTokens = () => {
  const tokens = useMemo(()=>fetch('/assets/GUI/color_palette.json'),[]);
  return tokens;
};

16.3 QML theme binding (pseudo)

Rectangle {
  color: Theme.surface
  radius: Theme.radiusLg
  border.color: Theme.border
  Text { text: "Diagnostics"; color: Theme.text; font.pixelSize: Theme.font2xl }
}


⸻

17) QA Checklist (must‑pass)
	•	No hardcoded colors/spacing; only tokens used.
	•	Dark/light/hc render parity (no missing contrast).
	•	Legend encodings stable and documented.
	•	Keyboard traversal order & visible focus rings.
	•	Charts export with theme + hash metadata.
	•	Asset sizes within budget; manifests hashed.
	•	CI: a11y + visual diff + unit tests pass.
	•	Hydra config recorded in HTML banner & logs. ￼ ￼

⸻

18) Governance & Changes
	•	Propose token edits via PR with before/after screenshots (dark/light/hc), contrast metrics, and updated assets-manifest.json.
	•	Breaking changes (e.g., semantic color remap) require design review and a minor version bump of tokens.

⸻

19) References
	•	CLI-first & GUI-thin, provenance, logging: SpectraMind V50 plan. ￼
	•	Terminal-first & Typer/Hydra workflows: Ubuntu CLI-Driven blueprint. ￼
	•	Hydra config-as-code for themes: Hydra guide. ￼
	•	Event-driven & rendering models: Engineering Guide to GUI Development. ￼
	•	MVC/MVVM patterns & declarative UIs: Comprehensive Guide to GUI Programming. ￼
	•	Scientific rigor & auditability: NASA‑grade modeling & simulation guide. ￼
	•	CLI UX & progress/logging standards: Mastering the Command Line. ￼

⸻

20) Appendix — Utility Classes (extract from widget_styles.css)

/* text */
.sm-h1{ font:800 var(--sm-font-4xl)/var(--sm-lh-tight) var(--sm-font-sans); margin:0 0 var(--sm-space-4); }
.sm-h2{ font:700 var(--sm-font-3xl)/var(--sm-lh-tight) var(--sm-font-sans); margin:0 0 var(--sm-space-4); }
.sm-body{ font:400 var(--sm-font-md)/var(--sm-lh-normal) var(--sm-font-sans); }

/* layout */
.sm-row{ display:flex; gap:var(--sm-space-4); }
.sm-col{ display:flex; flex-direction:column; gap:var(--sm-space-4); }
.sm-gap-2{ gap:var(--sm-space-2); } .sm-gap-4{ gap:var(--sm-space-4); }

/* badges */
.sm-badge{ display:inline-flex; align-items:center; gap:8px; padding:2px 8px; border-radius:999px; font:600 var(--sm-font-xs)/1 var(--sm-font-sans); border:1px solid var(--sm-color-border); }
.sm-badge--info{ color:var(--sm-color-info); border-color:color-mix(in oklab, var(--sm-color-info) 40%, var(--sm-color-border)); }
.sm-badge--success{ color:var(--sm-color-success); border-color:color-mix(in oklab, var(--sm-color-success) 40%, var(--sm-color-border)); }
.sm-badge--warning{ color:var(--sm-color-warning); border-color:color-mix(in oklab, var(--sm-color-warning) 40%, var(--sm-color-border)); }
.sm-badge--danger{ color:var(--sm-color-danger); border-color:color-mix(in oklab, var(--sm-color-danger) 40%, var(--sm-color-border)); }


⸻

Closing

This guide encodes how SpectraMind V50 looks and behaves — predictably, accessibly, and reproducibly — while keeping the GUI deliberately thin over a CLI-first engine. If in doubt: align with tokens, document with Hydra, and favor declarative, event‑driven patterns supported by our references. ￼ ￼ ￼ ￼ ￼

