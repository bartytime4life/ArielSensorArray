# ✅ SpectraMind V50 — PR Review Checklist (Quick Triage)

This **one-pager** gives reviewers a **fast triage table** for pull requests.  
Use alongside the full [Pull Request Review Guide](PULL_REQUEST_REVIEW_GUIDE.md).  

Mark ✅ = pass, ❌ = fail, ⚠️ = needs clarification.

---

## 📝 Metadata
| Check | Status |
|-------|--------|
| PR title imperative & concise | ☐ |
| Linked to issue(s)/milestones | ☐ |
| Labels applied (area, type, CI, docs) | ☐ |
| Author followed PR template | ☐ |

---

## 🎯 Summary & Motivation
| Check | Status |
|-------|--------|
| Rationale clear & scientific context provided | ☐ |
| Impact on μ/σ, GLL, calibration, runtime, symbolic rules explained | ☐ |
| Alignment with Kaggle 9h runtime & rules stated | ☐ |

---

## ⚙️ Design & Reproducibility
| Check | Status |
|-------|--------|
| CLI commands exact & runnable (`spectramind …`) | ☐ |
| Hydra config diffs shown, no hidden constants | ☐ |
| DVC stages updated & `dvc repro` passes | ☐ |
| Run hash recorded in `v50_debug_log.md` | ☐ |
| Config hash updated in `run_hash_summary_v50.json` | ☐ |
| Selftest passes locally (`spectramind selftest`) | ☐ |

---

## 🔬 Scientific Integrity & Diagnostics
| Check | Status |
|-------|--------|
| Metrics table filled (baseline vs new) | ☐ |
| Dashboard/plots attached (HTML, UMAP/t-SNE, FFT, GLL heatmap) | ☐ |
| Symbolic/physics rules respected (smoothness, nonnegativity, priors) | ☐ |
| Calibration diagnostics included (σ vs residuals, COREL coverage) | ☐ |
| Reproducibility evidence provided (seed, config, commit) | ☐ |

---

## 🔄 Compatibility & Risk
| Check | Status |
|-------|--------|
| Breaking changes listed (CLI/config/schema) | ☐ |
| Migration/fallback path documented | ☐ |
| Risk assessment & mitigation provided | ☐ |
| Kaggle notebook parity checked (if applicable) | ☐ |

---

## 🧪 Tests & Validation
| Check | Status |
|-------|--------|
| Unit tests added/updated, pytest green | ☐ |
| `spectramind selftest --fast/--deep` passes | ☐ |
| CI smoke run passes | ☐ |
| Determinism checked (seeds consistent) | ☐ |
| Reproducibility verified with DVC artifacts | ☐ |

---

## ⚡ Performance & Runtime
| Check | Status |
|-------|--------|
| Runtime ≤ 9h Kaggle limit | ☐ |
| Memory/VRAM within container limits | ☐ |
| Variance across seeds acceptable | ☐ |
| Benchmark results compared vs baseline | ☐ |

---

## 🛡️ Security & Compliance
| Check | Status |
|-------|--------|
| No secrets/keys in code/configs | ☐ |
| New deps/actions pinned to versions/SHAs | ☐ |
| Licenses respected, no PII introduced | ☐ |
| Security scans (bandit, pip-audit, codeql, trivy) pass | ☐ |

---

## 📚 Docs & Changelog
| Check | Status |
|-------|--------|
| README/docs updated if user-facing | ☐ |
| CLI `--help` accurate | ☐ |
| Configs commented inline | ☐ |
| CHANGELOG updated | ☐ |
| Architecture diagrams/figures updated (if relevant) | ☐ |

---

## 🚀 Post-Merge Tasks
| Check | Status |
|-------|--------|
| CI pipeline green on main | ☐ |
| Run hash tagged & artifacts published | ☐ |
| Dashboard backfilled & linked | ☐ |
| Stakeholders notified / issues closed | ☐ |
| Kaggle notebook sync validated (if applicable) | ☐ |

---

### 🔭 Mission Reminder
**Every PR must be reproducible, validated, Kaggle-compliant, and scientifically safe before merge.**
