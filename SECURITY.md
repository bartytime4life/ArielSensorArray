Security Policy — SpectraMind V50 (ArielSensorArray)

Mission-grade security for a mission-grade, CLI-first research pipeline.
This policy defines supported versions, reporting channels, scope, SLAs, disclosure, supply-chain controls, and operator playbooks aligned with scientific reproducibility and competition timelines.
	•	Policy ID: SEC-SPC-V50-001
	•	Current Policy Version: 1.2.0
	•	Applies To: All first-party code, builds, containers, CI/CD, and published artifacts in the SpectraMind V50 repository unless superseded by a component-specific addendum.
	•	Last updated: 2025-08-31 (America/Chicago)

⸻

1) Supported Versions

We use semantic versioning and maintain one Active line (features + security), one LTS line (security-only), and retire older branches for archival reproducibility.

Branch / Version line	Status	Fixes Provided	Planned EOL*
0.50.x (current best: v0.50.0-best)	Active	Security + critical features	~6 months after next minor (0.51) ships
0.49.x	LTS	Security fixes only	0.51.x “stable” + 60 days
< 0.49	EOL	No fixes	Repro only (archival)

*EOL is aligned with competition gates; any changes are posted in CHANGELOG.md and Release Notes.

Rules
	•	Security fixes land on the highest maintained line, then backported to LTS if practical.
	•	Containers, notebooks, and CI workflows inherit the repo’s line status.

⸻

2) How to Report a Vulnerability (Private Only)

Preferred: GitHub Private Advisory
	1.	Repo → Security → Report a vulnerability.
	2.	Include a clear description, minimal PoC, affected versions/commits/tags, and impact.
	3.	Add logs, configs (scrubbed), and environment details.

Alternative: Email (encrypted preferred)
	•	To: security [at] 
	•	Subject: Security: <short summary>
	•	For encryption, request our PGP key in the advisory thread, or include your public key.

Acknowledgement SLA: ≤ 2 business days (see §4).

⸻

3) Scope

In-Scope
	•	Core pipeline: src/** (encoders/decoders, calibration, diagnostics, symbolic layers).
	•	CLI & entrypoints: Typer commands (spectramind …), selftest.py, submission/packaging tools.
	•	API & server: src/server/** (FastAPI, artifact serving, CLI bridge).
	•	GUI & docs: src/gui/**, docs site if it renders local/served artifacts.
	•	Build/runtime: Dockerfiles, docker-compose.yml, GitHub Actions, Make targets consumed by users.
	•	Configs & scripts: config/**, configs/**, bin/**, scripts/**, manifests and run hashes.

Out-of-Scope (unless you can prove an exploit path through SpectraMind)
	•	Pure third-party vulnerabilities without a SpectraMind attack path.
	•	DoS via unrealistic inputs or self-inflicted local misconfiguration.
	•	Social engineering, phishing, attacks on maintainers’ personal accounts.
	•	Self-XSS in locally edited, static HTML artifacts.
	•	Kaggle platform issues (report to Kaggle).

⸻

4) Coordinated Disclosure & SLAs

We follow responsible disclosure with a bias for rapid mitigation when research milestones are at risk.

Timeline targets (not guarantees):
	•	Triage & Acknowledgement: ≤ 2 business days
	•	Initial Technical Assessment: ≤ 7 days with provisional CVSS v3.1 base score
	•	Critical/High Fix or Mitigation: ≤ 30 days (faster if exploitation suspected/observed)
	•	Medium/Low Fix or Mitigation: ≤ 90 days
	•	Public Advisory: within 7 days of patch release or per agreed embargo lift

Embargo
If exploitation risk is high, we coordinate a short embargo to ship mitigations (e.g., config flags, CI rules, container base pin/patch) before full code changes.

⸻

5) Severity & Prioritization
	•	Primary rubric: CVSS v3.1 (base + temporal/environmental when relevant).
	•	Priority modifiers: exploit in the wild, data sensitivity (tokens, credentials), supply-chain blast radius, and competition timing.

⸻

6) What a Good Report Includes (Template)

Title: <concise name>
Version(s): <tag/commit range, container tag, workflow sha>
Component: <CLI | API | GUI | Docker | CI | Config | Diagnostics | Symbolic>
Environment: <OS, Python, GPU driver/container, cloud if any>

Summary:
<one paragraph describing the issue and why it matters>

Impact:
<data exfiltration? RCE? privilege escalation? integrity? availability?>

Reproduction:
1) <step>
2) <step>
Expected: <secure behavior>
Observed: <vulnerable behavior>

PoC:
<minimal code/commands; mask secrets; attach files safely>

Workarounds/Mitigations (if any):
<config flag, firewall rule, container arg, dependency pin>

Additional context/logs:
<stderr/stdout, screenshots, hashes>


⸻

7) Patch, Backport, and Disclosure
	1.	Private fix branch referencing the private advisory ID; restricted reviewers only.
	2.	Tests: regression + minimal PoC harness (safe to run in CI).
	3.	Release: bump highest maintained line (e.g., 0.50.1), then backport to LTS when feasible.
	4.	Release Notes: affected versions, impact summary, mitigation, and optional researcher credit.
	5.	Advisory: publish GitHub Security Advisory with CVSS, affected versions, mitigation.
	6.	Artifacts: images and SBOMs rebuilt; digests pinned where supported.

⸻

8) Supply Chain & Dependencies
	•	SBOM: make sbom → outputs/sbom/spdx_<RUN_ID>.json (CycloneDX/SPDX via syft; optional grype scan).
	•	Vuln Scans: make security runs pip-audit; optional Bandit static analysis for src/.
	•	Pinning: poetry.lock for Python deps; container bases pinned by tag or digest in CI; GitHub Actions pinned by SHA when feasible.
	•	Updates: Dependabot/Renovate for dependency bumps; human review required for security-relevant changes.
	•	Provenance: reproducible CLI/Make targets; run manifests (outputs/manifests/ci_run_manifest_*.json); config hashes (run_hash_summary_v50.json).
	•	Rebuild Triggers: On advisory publication, containers and notebooks rebuilt in CI (Kaggle image notes where applicable).

⸻

9) Secrets & Credentials
	•	Never commit secrets. Use environment variables and an optional .env (locally).
	•	Rotation: If a secret is exposed (e.g., Kaggle token), rotate immediately and note in the advisory thread.
	•	Validation: make validate-env enforces local env schema if scripts/validate_env.py is present.
	•	Least privilege: use short-lived tokens (OIDC) and minimal scopes in CI.

⸻

10) Hardening Guidance (Operators & Contributors)
	•	Containers: run with least privilege; mount only required volumes; prefer --read-only where compatible.
	•	Network: bind API/UI to localhost in dev; restrict ingress in shared environments.
	•	CLI usage: avoid untrusted configs/overrides; keep spectramind selftest green; verify hashes and manifests.
	•	Artifacts: serve dashboards read-only; disallow uploads in production contexts without sandboxing.
	•	CI: restrict workflow permissions; pin external actions; rotate tokens on policy or team changes.

⸻

11) Safe Harbor

We will not pursue action provided you:
	•	Act in good faith; avoid privacy violations and data destruction.
	•	Do not exploit beyond proof-of-existence.
	•	Use private reporting channels and respect coordinated disclosure timelines.
	•	Follow applicable laws.

When in doubt, contact us via the private advisory channel first.

⸻

12) Bug Bounty

No paid bounty at this time. We happily provide researcher credit in release notes/advisories (opt-in).

⸻

13) Contacts & Status
	•	Primary channel (preferred): GitHub → Security → Report a vulnerability
	•	Alternative: security [at] <your-team-alias> (PGP available upon request)
	•	Status updates: posted within the private advisory thread (ack → triage → fix → release).
	•	Public comms: Release Notes + GitHub Security Advisory upon patch.

⸻

14) Fast Mitigation Playbook (Operators)

If a critical advisory lands:
	1.	Containers: pull fixed tags/digests and redeploy.
	2.	Python envs: upgrade to patched 0.50.x/LTS; rebuild virtualenvs.
	3.	Configs: apply temporary hardening flags from the advisory (e.g., --no-open dashboards, disable live endpoints).
	4.	CI: rotate tokens; run make security sbom; validate ci_run_manifest_*.json.
	5.	Artifacts: regenerate dashboards with fixed code to avoid serving vulnerable bundles.
	6.	Notebook/Kaggle: re-run with patched environment; ensure competition submission tokens/secrets are rotated.

⸻

15) Policy Lifecycle & Governance
	•	Owner: Security Maintainer (delegated by Project Lead).
	•	Review cadence: Quarterly or after any high-severity advisory.
	•	Change control: Propose updates via PR; tag security-policy label; require approval from Owner + one core maintainer; bump Policy Version and update date.
	•	Records: Keep advisory IDs, CVSS, and patch links in SECURITY-ADVISORIES.md (redacted if under embargo).

⸻

16) Quick Reference (Cheat-Sheet)
	•	Report privately → GitHub Security Advisory
	•	Acknowledge: ≤2 business days
	•	Critical/High fix: ≤30 days
	•	Tools: make security (pip-audit/Bandit), make sbom (syft/grype), make ci (hardened pipeline)
	•	Proof safely; do not publish until coordinated release
	•	Containers & actions: prefer pinned tags/SHAs

⸻

Appendix A — Minimal Triage Matrix

Severity	Example Impact	Target Response
Critical	RCE, secret extraction, supply-chain compromise	Acknowledge ≤2d; fix ≤30d (faster)
High	Privilege escalation, auth bypass	Acknowledge ≤2d; fix ≤30d
Medium	SSRF, partial info leak, sandbox escape (limited)	Acknowledge ≤2d; fix ≤90d
Low	Best-practice violation, minor misconfig	Acknowledge ≤2d; fix ≤90d


⸻

Appendix B — Secure Development Guidelines (Snapshot)
	•	Inputs: validate file headers and shapes; enforce MIME/extension allow-lists for artifact loaders.
	•	Paths: resolve and confine to ARTIFACTS_DIR/LOGS_DIR; prevent .. traversal; never use untrusted --outdir without checks.
	•	Subprocess: avoid shell=True; pass explicit lists; sanitize env; capture return codes.
	•	Logging: avoid secrets; scrub tokens and PII; prefer structured JSON for security-relevant events.
	•	Crypto: use vetted libs; avoid custom primitives; pin versions for cryptography/TLS stacks in containers.
	•	Web/API: CORS least-privilege; disable auto-directory listings; rate-limit sensitive endpoints if exposed.
	•	Notebooks: treat as code; disable arbitrary file writes in shared environments; pin kernels and images.
	•	Reviews: security review required for changes to auth, network exposure, artifact serving, or container entrypoints.

⸻
