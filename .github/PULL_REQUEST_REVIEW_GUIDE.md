# 🔎 SpectraMind V50 — Pull Request Review Guide

This guide provides **maintainers and reviewers** with a structured checklist to evaluate every pull request (PR) in the **SpectraMind V50** repository.  
It mirrors the official PR template and enforces **NASA-grade reproducibility, physics-informed rigor, and CLI-first governance**  
[oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) · [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)

---

## 🎯 Review Principles

- **Reproducibility First** — every PR must provide CLI commands, Hydra configs, and run hashes  
  [oai_citation:2‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL).
- **Scientific Integrity** — outputs (μ, σ, GLL, calibration) must remain scientifically valid  
  [oai_citation:3‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) · [oai_citation:4‡Gravitational Lensing and Astronomical Observation: Modeling and Mitigation.pdf](file-service://file-HA6qQbdteZZaeRSStyD1Ha).
- **Runtime Discipline** — Kaggle 9h limit and memory budgets must be respected  
  [oai_citation:5‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) · [oai_citation:6‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).
- **Security & Compliance** — no secrets, pinned dependencies, safe configs  
  [oai_citation:7‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL).
- **Documentation Updated** — READMEs, configs, CLI help, and CHANGELOG kept current  
  [oai_citation:8‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

---

## ✅ Reviewer Checklist

### 1. Title & Issue Links
- [ ] PR title is imperative, concise.  
- [ ] Links to issues/milestones are valid.  
- [ ] Labels applied (area, type, CI, docs, security).

### 2. Summary & Motivation
- [ ] Clear rationale provided (scientific/engineering context).  
- [ ] Impact on μ/σ spectra, GLL, calibration, runtime, symbolic constraints addressed  
  [oai_citation:9‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw).

### 3. Design & Reproducibility
- [ ] CLI commands are **exact** (`spectramind …`), runnable locally and in CI  
  [oai_citation:10‡Command Line Interfaces (CLI) Technical Reference (MCP).pdf](file-service://file-HzYPacwmdGzogMDYWAL7cp).
- [ ] Hydra config diffs shown; no hidden code constants  
  [oai_citation:11‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL).
- [ ] DVC stages updated; `dvc repro` passes  
  [oai_citation:12‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).
- [ ] Run hash present in `v50_debug_log.md`; config hash appended to `run_hash_summary_v50.json`.

### 4. Scientific Integrity & Diagnostics
- [ ] Metrics table filled with baseline vs. new numbers.  
- [ ] Plots/reports attached (dashboard HTML, UMAP/t-SNE, FFT, GLL heatmap)  
  [oai_citation:13‡SpectraMind V50 Project Analysis.pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK) ·  
  [oai_citation:14‡Computational Physics Modeling.pdf](file-service://file-7kBHKQhMuzqB16Z34Pvmmz).
- [ ] Symbolic/physics rules respected (smoothness, nonnegativity, priors)  
  [oai_citation:15‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) · [oai_citation:16‡Gravitational Lensing.pdf](file-service://file-HA6qQbdteZZaeRSStyD1Ha).
- [ ] Calibration diagnostics included (σ vs residuals, coverage, COREL).

### 5. Compatibility & Risk
- [ ] Breaking changes (CLI flags, configs, schema) explicitly listed.  
- [ ] Migration path provided (fallbacks, scripts).  
- [ ] Risk assessment + mitigation documented.  
- [ ] Kaggle notebook parity checked (if applicable).

### 6. Tests & Validation
- [ ] Unit tests updated/added; pytest runs pass  
  [oai_citation:17‡Master Coder's Guide for Intermediate Programmers.pdf](file-service://file-GEcjUjB2vinWs8nyS7zAH6).
- [ ] `spectramind selftest --fast/--deep` passes.  
- [ ] CI smoke run completes with no regressions.  
- [ ] Determinism checked (same seed = same results).  

### 7. Performance & Runtime
- [ ] Runtime/memory budget respected (≤9h Kaggle, within VRAM limits)  
  [oai_citation:18‡SpectraMind V50 Technical Plan.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) · [oai_citation:19‡Kaggle Platform Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).
- [ ] Per-planet timing reported.  
- [ ] Variance across seeds measured (± tolerance acceptable).  

### 8. Security & Compliance
- [ ] No secrets or credentials in code/configs  
  [oai_citation:20‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL).
- [ ] New actions/dependencies pinned to versions/SHAs.  
- [ ] Licenses respected; PII absent.  
- [ ] If a vuln was fixed, advisory prepared per `SECURITY_ADVISORY_TEMPLATE.md`.

### 9. Documentation
- [ ] README updated if user-facing changes  
  [oai_citation:21‡SpectraMind V50 Project Analysis.pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).
- [ ] CLI `--help` accurate  
  [oai_citation:22‡CLI Technical Reference (MCP).pdf](file-service://file-HzYPacwmdGzogMDYWAL7cp).
- [ ] Configs documented (inline comments).  
- [ ] CHANGELOG updated (if feature/major fix).  

### 10. Post-Merge Tasks
- [ ] CI pipeline runs green on main.  
- [ ] Run hash tagged + artifacts published.  
- [ ] Dashboard backfilled and linked.  
- [ ] Stakeholders notified / issues closed.  

---

## 🧠 Review Heuristics (When to Block)

- CLI repro not possible (missing commands or configs).  
- Run/config hash missing or inconsistent.  
- Significant regression in GLL/calibration without justification.  
- Runtime > Kaggle 9h or memory over container limits.  
- Physics/symbolic rules broken without justification.  
- Docs/tests/security hygiene missing or failing.

## 🟢 Approval Criteria

- CLI, Hydra, and DVC **consistent** and **runnable**.  
- CI **green** (lint, tests, security, diagnostics, smoke).  
- Metrics **stable or improved** with credible diagnostics.  
- Reproducibility **demonstrated** (hashes, seeds, composed config, artifact logs).

---

## 📚 References

- SpectraMind V50 Technical Plan — config, logging, reproducibility  
  [oai_citation:25](file-service://file-6PdU5f5knreHjmSdSauj3w) · [oai_citation:26](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)  
- SpectraMind V50 Project Analysis — repo audit & governance  
  [oai_citation:27](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)  
- Hydra for AI Projects — config composition & override discipline  
  [oai_citation:28](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL)  
- Kaggle Platform Guide — leaderboard, runtime, and GPU/TPU constraints  
  [oai_citation:29](file-service://file-CrgG895i84phyLsyW9FQgf) · [oai_citation:30](file-service://file-CG661XRZ48CnBj69Lf5vTy)  
- Physics-informed checks  
  [oai_citation:31](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) · [oai_citation:32](file-service://file-HA6qQbdteZZaeRSStyD1Ha)

---

### 🔭 Mission Reminder
Every PR is a **flight check**. Treat the pipeline like a **spacecraft instrument**:  
only **reproducible, validated, Kaggle-compliant, and scientifically safe** changes fly.