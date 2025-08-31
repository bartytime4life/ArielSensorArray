# ♿ UX & Accessibility — SpectraMind V50 (GUI-Optional Layer)

> Goal: ensure our **thin GUIs** (Streamlit/React/Qt) are **usable, accessible, and audit-safe** while preserving the **CLI-first, Hydra-safe** pipeline.  
> Scope: visual design, interaction, accessibility (A11y), i18n/l10n, error messaging, progress, and testing.

---

## 0) First Principles

1) **CLI-first, GUI-thin**  
   - All operations must be reproducible via `spectramind …`; the GUI **only** invokes CLI and renders artifacts (JSON/HTML/plots/logs).  
   - No hidden state; all parameters live in **Hydra configs** or explicit CLI overrides:contentReference[oaicite:0]{index=0}:contentReference[oaicite:1]{index=1}.

2) **Declarative UI & Event-Driven**  
   - Treat UI as a function of state (diagnostics JSON, run status).  
   - Event loop (clicks/timers/file updates) → invoke CLI → render outputs (state binding):contentReference[oaicite:2]{index=2}:contentReference[oaicite:3]{index=3}.

3) **Accessible by default**  
   - Follow industry patterns (MVC/MVVM), event-driven models, and accessibility guidelines to reduce coupling and improve testability:contentReference[oaicite:4]{index=4}:contentReference[oaicite:5]{index=5}.

4) **Human-friendly CLI UX**  
   - GUI surfaces **clear help**, **progress**, **actionable errors**, and **reaction for every action**, reflecting elite CLI UX standards:contentReference[oaicite:6]{index=6}.

---

## 1) A11y Checklist (WCAG-inspired, pragmatic)

### 1.1 Perceivable
- **Contrast** ≥ 4.5:1 for text/UI chrome; 3:1 for large text. Prefer neutral/high-contrast themes in charts and UIs.  
- **Color-independence**: never encode meaning with color alone; add shapes, patterns, or labels.  
- **Alt text** for images/plots; **captions** for embedded HTML reports.  
- Avoid tiny fonts; ensure responsive scaling (desktop ≥14–16 px base).

### 1.2 Operable
- **Keyboard first**: every action reachable via keyboard (see §4 map).  
- **Focus management** with visible focus rings; logical tab order; skip links for long pages.  
- **Avoid seizure/motion**: no flashing; prefer subtle transitions; respect reduced-motion settings.  
- Provide **pause/stop** or throttling for auto-refreshing logs/plots.

### 1.3 Understandable
- **Plain-language labels/help**; tooltips for advanced toggles.  
- Consistent layout (grid), consistent components (buttons/links), consistent icons.  
- Clear affordances (primary vs secondary actions).

### 1.4 Robust
- **ARIA roles / labels** on custom components; native elements where possible.  
- Accessible forms with programmatic labels; error states with role=alert; announce changes politely (live regions).

Reference architectures, rendering modes, and pattern guidance are captured in our GUI references:contentReference[oaicite:7]{index=7}:contentReference[oaicite:8]{index=8}.

---

## 2) SpectraMind-Specific UX: Reproducibility & Auditability

### 2.1 Provenance UI
- Surface the **exact CLI command** the GUI will run (copy-as-CLI).  
- Show **Hydra overrides** and the **composed config hash**.  
- Render **run id**, **git SHA**, **timestamps**, and link to `logs/v50_debug_log.md` for each action:contentReference[oaicite:9]{index=9}.

### 2.2 Error Design (Actionable)
- Map common failures to **plain-language** messages with remediation; include a “copy error context” button.  
- Preserve raw stderr in an expandable panel.  
- Follow CLI UX best practices: never fail silently; always explain **what happened** & **how to recover**:contentReference[oaicite:10]{index=10}.

### 2.3 Progress & Feedback
- Never leave users staring at a blank screen:  
  - Show **spinners/progress bars** within 100 ms;  
  - Periodic status updates for long operations (log tail);  
  - Success/failure toasts on completion:contentReference[oaicite:11]{index=11}.

---

## 3) Layout, Grids & Visualization

- Use **grid-based** layouts to avoid clutter and maintain visual rhythm across panels (dashboards, tables, detail views).  
- Encode scientific **patterns** (temporal series, spectral bins, symbolic overlays) with consistent axes, units, and legends.  
- Prefer declarative renderers (React/Plotly) and accessible chart practices (aria-describedby, alt text, keyboard navigation).  
- Keep a visual hierarchy: page > section > card > content; minimize cognitive load:contentReference[oaicite:12]{index=12}:contentReference[oaicite:13]{index=13}.

---

## 4) Keyboard & Shortcuts (default map)

| Action | Keys |
|---|---|
| Run current form | **Shift + R** |
| Open log panel | **g l** |
| Dashboard home | **g d** |
| Run launcher | **g r** |
| Focus CLI preview | **.** |
| Copy CLI command | **c** |
| Toggle auto-refresh | **t** |
| Next/Prev run | **]** / **[** |

> Provide a **“?” shortcuts** overlay accessible via keyboard. Ensure shortcuts don’t shadow native screen-reader keys.

---

## 5) Internationalization (i18n) & Localization (l10n)

- Externalize all user strings; load via locale files; default to **en** with fallback.  
- Support **LTR/RTL** text directions if applicable; date/time/number formatting via locale.  
- Avoid concatenated string building; prefer format placeholders.  
- Keep labels concise to prevent overflow in localized UIs:contentReference[oaicite:14]{index=14}.

---

## 6) Components & Patterns

### 6.1 Run Launcher (form)
- Labeled inputs bound to Hydra fields; validation with programmatic hints; **Run** button + CLI preview; optional dry-run.  
- Live **diff** view vs baseline config.

### 6.2 Artifact Viewers
- **HTML report**: embed in sandboxed frame with caption + “open in new tab.”  
- **JSON**: table for `metrics`, `per_planet`; raw JSON in a collapsible region.  
- **Plots**: responsive images/canvases; alt text; zoom/pan with keyboard equivalents.

### 6.3 Logs
- Tail `logs/v50_debug_log.md` with adjustable interval; pause/resume; search filter.  
- Copy block button; download log.

---

## 7) Performance & Reliability

- Progressive rendering: show shell containers immediately, then hydrate with data.  
- Debounce rapid toggles; throttle auto-refresh.  
- Avoid blocking the UI thread (Qt: QProcess; React: workers/async).  
- Offline-friendly: cache last diagnostics; handle missing artifacts gracefully (info banners).  
- Low-bandwidth mode: fetch only metadata first, lazy-load heavy assets.

---

## 8) QA & Accessibility Testing

- **Automated**:  
  - Lint ARIA roles; unit tests for ViewModels; Playwright for keyboard flows.  
  - Axe/Pa11y for web A11y scans; snapshot tests for focus order.  
- **Manual**:  
  - Screen-reader pass (NVDA/VoiceOver); keyboard-only navigation; high-contrast theme check; reduced-motion toggle.  
- **Regression**: log rendering, artifact embedding, and CLI preview must not regress.

---

## 9) Diagram: Thin Integration (for docs)

```mermaid
flowchart LR
  U[User] -->|Key/Click| G[GUI Shell]
  G -->|Invoke| C[spectramind …]
  C --> Y[Hydra Configs]
  C --> O[Artifacts (JSON/HTML/plots)]
  C --> L[Audit Log]
  O --> G
  L --> G
  G -->|Render| V[Accessible Views]
````

> Use Mermaid in docs and READMEs; GitHub renders it natively — helpful for onboarding and reviews.

---

## 10) Reference Patterns

* **GUI architecture & patterns** (MVC/MVP/MVVM, declarative UIs, event-driven loops)
* **CLI UX principles** (discoverability, progress, errors, scripting) inform GUI strings and flows
* **CLI-first, artifact-driven integration** for SpectraMind V50 (Hydra configs, logs, diagnostics)

---

```
```
