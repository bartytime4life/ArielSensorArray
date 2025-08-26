# 🔎 SpectraMind V50 — Pull Request Review Guide

This guide provides **maintainers and reviewers** with a structured checklist to evaluate every pull request (PR) in the **SpectraMind V50** repository.  
It mirrors the official PR template and enforces **NASA-grade reproducibility, physics-informed rigor, and CLI-first governance** [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).

---

## 🎯 Review Principles

- **Reproducibility First** — every PR must provide CLI commands, Hydra configs, and run hashes [oai_citation:2‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL).  
- **Scientific Integrity** — outputs (μ, σ, GLL, calibration) must remain scientifically valid [oai_citation:3‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) [oai_citation:4‡Gravitational Lensing and Astronomical Observation: Modeling and Mitigation.pdf](file-service://file-HA6qQbdteZZaeRSStyD1Ha).  
- **Runtime Discipline** — Kaggle 9h limit and memory budgets must be respected [oai_citation:5‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:6‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- **Security & Compliance** — no secrets, pinned dependencies, safe configs [oai_citation:7‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL).  
- **Documentation Updated** — READMEs, configs, CLI help, and CHANGELOG kept current [oai_citation:8‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).  

---

## ✅ Reviewer Checklist

### 1. Title & Issue Links
- [ ] PR title is imperative, concise.  
- [ ] Links to issues/milestones are valid.  

### 2. Summary & Motivation
- [ ] Clear rationale provided (scientific/engineering context).  
- [ ] Impact on μ/σ spectra, GLL, calibration, runtime, symbolic constraints addressed [oai_citation:9‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw).  

### 3. Design & Reproducibility
- [ ] CLI commands are **exact** (`spectramind …`), runnable locally and in CI [oai_citation:10‡Command Line Interfaces (CLI) Technical Reference (Master Coder Protocol).pdf](file-service://file-HzYPacwmdGzogMDYWAL7cp).  
- [ ] Hydra config diffs shown; no hidden code constants [oai_citation:11‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL).  
- [ ] DVC stages updated; `dvc repro` passes [oai_citation:12‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).  
- [ ] Run hash present in `v50_debug_log.md`.  

### 4. Scientific Integrity & Diagnostics
- [ ] Metrics table filled with baseline vs. new numbers.  
- [ ] Plots/reports attached (dashboard HTML, UMAP/t-SNE, FFT, GLL heatmap) [oai_citation:13‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK) [oai_citation:14‡Computational Physics Modeling: Mechanics, Thermodynamics, Electromagnetism & Quantum.pdf](file-service://file-7kBHKQhMuzqB16Z34Pvmmz).  
- [ ] Symbolic/physics rules respected (smoothness, nonnegativity, priors) [oai_citation:15‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) [oai_citation:16‡Gravitational Lensing and Astronomical Observation: Modeling and Mitigation.pdf](file-service://file-HA6qQbdteZZaeRSStyD1Ha).  

### 5. Compatibility & Risk
- [ ] Breaking changes (CLI flags, configs, schema) explicitly listed.  
- [ ] Migration path provided (fallbacks, scripts).  
- [ ] Risk assessment + mitigation documented.  

### 6. Tests & Validation
- [ ] Unit tests updated/added; pytest runs pass [oai_citation:17‡Master Coder's Guide for Intermediate Programmers.pdf](file-service://file-GEcjUjB2vinWs8nyS7zAH6).  
- [ ] `spectramind selftest --fast/--deep` passes.  
- [ ] CI smoke run completes with no regressions.  
- [ ] Determinism checked (same seed = same results).  

### 7. Performance & Runtime
- [ ] Runtime/memory budget respected (≤9h Kaggle, within VRAM limits) [oai_citation:18‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:19‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf).  
- [ ] Per-planet timing reported.  
- [ ] Variance across seeds measured (± tolerance acceptable).  

### 8. Security & Compliance
- [ ] No secrets or credentials in code/configs [oai_citation:20‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL).  
- [ ] New actions/dependencies pinned to versions/SHAs.  
- [ ] Licenses respected; PII absent.  

### 9. Documentation
- [ ] README updated if user-facing changes [oai_citation:21‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK).  
- [ ] CLI `--help` accurate [oai_citation:22‡Command Line Interfaces (CLI) Technical Reference (Master Coder Protocol).pdf](file-service://file-HzYPacwmdGzogMDYWAL7cp).  
- [ ] Configs documented (comments inline).  
- [ ] CHANGELOG updated (if feature/major fix).  

### 10. Post-Merge Tasks
- [ ] CI pipeline runs green on main branch.  
- [ ] Run hash tagged + artifacts published.  
- [ ] Dashboard backfilled and linked.  
- [ ] Stakeholders notified / issues closed.  

---

## 🧠 Review Heuristics

- **Reject / Request Changes** if:  
  - CLI repro not possible (missing commands or configs).  
  - Run hash missing or inconsistent.  
  - GLL or calibration regresses beyond tolerance.  
  - Runtime exceeds Kaggle 9h constraint [oai_citation:23‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w).  
  - Physics/symbolic rules broken without justification [oai_citation:24‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw).  
  - Docs, tests, or security hygiene missing.  

- **Approve** only when:  
  - CLI, Hydra, and DVC are consistent.  
  - CI pipeline is green.  
  - Scientific metrics stable or improved.  
  - Logs/artifacts demonstrate reproducibility.  

---

## 📚 References

- SpectraMind V50 Technical Plan — config, logging, reproducibility [oai_citation:25‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) [oai_citation:26‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)  
- SpectraMind V50 Project Analysis — repo audit & governance [oai_citation:27‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)  
- Hydra for AI Projects — config composition & override discipline [oai_citation:28‡Hydra for AI Projects: A Comprehensive Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL)  
- Kaggle Platform Guide — leaderboard, runtime, and GPU/TPU constraints [oai_citation:29‡Kaggle Platform: Comprehensive Technical Guide.pdf](file-service://file-CrgG895i84phyLsyW9FQgf) [oai_citation:30‡Comparison of Kaggle Models from NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-CG661XRZ48CnBj69Lf5vTy)  
- Cosmic Fingerprints / Radiation References — physics-informed modeling checks [oai_citation:31‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) [oai_citation:32‡Radiation: A Comprehensive Technical Reference.pdf](file-service://file-Ta3DQ7U5AXfZBw4jAecJfL)  

---

### 🔭 Mission Reminder
Every PR is a **flight check**. Treat the pipeline like a **spacecraft instrument**:  
**only reproducible, validated, and scientifically safe changes fly.**