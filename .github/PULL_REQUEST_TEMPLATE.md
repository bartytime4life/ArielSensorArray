# 🛰️ SpectraMind V50 — Pull Request Template (ArielSensorArray)

> Fill out **every** section. If something doesn’t apply, write **“N/A”** and explain why.  
> This template enforces **NASA-grade reproducibility, physics-informed modeling, CLI-first design, and symbolic/diagnostics rigor** for the **NeurIPS 2025 Ariel Data Challenge**  
> [oai_citation:0‡SpectraMind V50 Technical Plan for the NeurIPS 2025 Ariel Data Challenge.pdf](file-service://file-6PdU5f5knreHjmSdSauj3w) · [oai_citation:1‡SpectraMind V50 Project Analysis (NeurIPS 2025 Ariel Data Challenge).pdf](file-service://file-QRDy8Xn69XgxEjZgtZZ8FK)

---

## 1) Title & Issue Links
**PR Title (imperative):**  
<!-- e.g., "Add COREL edge features to AIRS GNN" -->

Closes:
- #ISSUE_ID
- (Optional) External link(s):

Labels:
- `area:*` • `type:*` • `ci` • `docs` • `security` (as applicable)

---

## 2) Summary (What & Why)
**What changed (concise):**  
<!-- 3–5 bullets. List major files or modules touched. -->

**Why (motivation/scientific rationale):**  
<!-- cite design decisions; physics or competition constraints -->

Impact on:
- μ/σ spectra accuracy
- GLL / calibration
- Runtime (Kaggle 9h limit) [oai_citation:2‡SpectraMind V50 Technical Plan](file-service://file-6PdU5f5knreHjmSdSauj3w)
- Symbolic/physics integrity [oai_citation:3‡Cosmic Fingerprints .txt](file-service://file-HNCWW2WZZ9FkKvKZAfqMdw) · [oai_citation:4‡Gravitational Lensing & Observation.pdf](file-service://file-HA6qQbdteZZaeRSStyD1Ha)
- Reproducibility / DVC / Hydra [oai_citation:5‡Hydra Guide.pdf](file-service://file-MpHwv9Z1E3qqzGXaQ3agpL)

Scope touched:
- [ ] Code
- [ ] Configs (Hydra)
- [ ] Data pipeline / calibration
- [ ] Diagnostics / dashboard
- [ ] Docs
- [ ] CI / infra
- [ ] Security / compliance
- [ ] Other:

---

## 3) Design & Reproducibility

### 3.1 CLI Invocation(s)
Paste **exact** commands. Use unified Typer CLI (`spectramind …`)  
[oai_citation:6‡Command Line Interfaces (CLI) Technical Reference (MCP).pdf](file-service://file-HzYPacwmdGzogMDYWAL7cp)

```bash
spectramind selftest --deep
spectramind calibrate --config-name=config_v50
spectramind train --config-name=config_v50 +training.seed=1337
spectramind predict --out-csv outputs/submission.csv
spectramind diagnose dashboard --open

3.2 Hydra Config Diffs

Show minimal before → after diffs or new files.

# configs/model/v50.yaml
- decoder.head: "gaussian"
+ decoder.head: "gaussian_corel"
+ corel:
+   enable: true
+   edge_features: ["wavelength_dist", "molecule", "detector_region"]

3.3 Data & Artifacts (DVC)
	•	DVC stages updated in dvc.yaml: ☐ Yes
	•	dvc repro passes: ☐ Yes
	•	Large artifacts tracked in DVC (no binaries in Git): ☐ Yes

Stages touched: calibrate / train / predict / diagnose
New outputs: outputs/... (size, purpose)

3.4 Run Hash & Config Snapshot
	•	Run hash appended to run_hash_summary_v50.json: ☐ Yes
	•	Composed Hydra config saved (OmegaConf YAML) in artifacts: ☐ Yes

⸻

4) Scientific Integrity & Diagnostics

4.1 Metrics (attach numbers)

Metric	Baseline (commit/hash)	This PR	Δ
Public GLL (val)			
Private GLL			
MAE (μ)			
Calibration (z-score)			
Coverage@95% (σ)			
Runtime (planet avg)			

Seeds, dataset split, commands: paste exactly.

4.2 Plots / Reports
	•	Dashboard HTML: outputs/diagnostics/report_vX.html
	•	UMAP/t-SNE overlays
	•	GLL per-bin heatmap
	•	FFT / smoothness plots

Attach HTML or screenshots; link artifact paths.

4.3 Symbolic / Physics Checks
	•	Symbolic loss trend: ☐ stable ☐ improved
	•	Smoothness / priors respected
	•	Nonnegativity enforced
	•	No new symbolic rule violations

Explain degradations if any and justify scientifically.

⸻

5) Backward Compatibility & Risk

Breaking changes:
	•	CLI flags
	•	File/dir layout
	•	Config schema
	•	Model checkpoints

Migration notes (one-command recipe):

<!-- e.g., spectramind migrate --from v50.1 --to v50.2 -->


Risk:
	•	Impact: Low / Medium / High
	•	Mitigation: tests, feature flags, rollout plan

Kaggle notebook parity:
	•	Matching notebook updated / N/A
	•	Kernel runtime validated (≤9h) / N/A

⸻

6) Tests & Validation
	•	Unit tests (pytest) added/updated: list paths
	•	spectramind selftest --fast/--deep pass: ☐ Yes
	•	Toy pipeline sanity (make ci-smoke): ☐ Yes
	•	Repro run matches with same seed/config: ☐ Yes
	•	Determinism: ☐ Yes (note any non-det sources and controls)

Paste trimmed logs or link artifacts.

⸻

7) Performance & Runtime Budget

Target: ≤ 9h for ~1100 planets (Kaggle GPU limit)
	•	Per-planet wall time acceptable
	•	Memory within container limit
	•	Vectorization / batching respected
	•	--fast-dev-run documented

Numbers (machine, GPU, batch, precision, seq len):

⸻

8) Security & Compliance
	•	No secrets/keys in code/configs
	•	Licenses respected; attributions updated
	•	No PII; only synthetic/competition data
	•	New deps/actions pinned to versions/SHAs
	•	Security scans passed (bandit, pip-audit, CodeQL, Trivy)

(If a vuln is involved) Will publish advisory using SECURITY_ADVISORY_TEMPLATE.md: ☐ Yes / ☐ N/A

⸻

9) Docs
	•	README.md updated
	•	Docstrings & type hints added/updated
	•	CLI --help accurate
	•	Configs documented (inline comments)
	•	Architecture diagrams (Mermaid / PNG) updated if applicable

⸻

10) Checklist — Author
	•	Verified CLI runs produce outputs
	•	Captured Hydra configs + seeds
	•	Updated DVC; pushed/pulled artifacts
	•	Added/updated tests; selftest passes
	•	Produced dashboard HTML/plots
	•	Updated docs/help strings
	•	Passed pre-commit (ruff, black, isort, mypy)
	•	Updated CHANGELOG.md (if applicable)

Author: @your-handle
Date: YYYY-MM-DD

⸻

11) Checklist — Reviewer
	•	Clear problem + rationale
	•	Reproducible configs + seeds
	•	Tests sufficient; failure modes covered
	•	Metrics credible; GLL/calibration not regressed
	•	Runtime/memory budget respected
	•	Physics/symbolic rules respected
	•	No binary blobs in Git; DVC integrity OK
	•	CI green (lint, diagnostics, smoke, security)

Reviewer: @handle — Approve / Request changes / Comment

⸻

12) Post-Merge Tasks (Owner)
	•	Kick CI pipeline on main
	•	Tag run hash / update release notes
	•	Backfill dashboard / publish artifacts
	•	Notify stakeholders / update issue tracker
	•	Sync Kaggle notebook (if applicable)

⸻

Appendix A — Bench Run Log

Paste CLI call → Hydra config → hash → metrics.
Full logs → logs/v50_debug_log.md.

Appendix B — Config Snapshot

Paste exact composed Hydra config (OmegaConf.to_yaml) for the main run.

⸻

Contributor Notes
	•	Use unified Typer CLI; no ad-hoc scripts
	•	Treat configs as code; capture composed Hydra config
	•	Version large data/models with DVC
	•	Diagnostics are first-class: produce dashboard HTML every run
	•	Physics & symbolic constraints are mandatory — justify any deviation

