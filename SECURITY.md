# Security Policy — SpectraMind V50 (ArielSensorArray)

Mission-grade security for a mission-grade, CLI-first research pipeline. This document explains **what we support**, **how to report vulnerabilities**, **what’s in/out of scope**, and **how we disclose and patch**—with timelines that align to research competitions and reproducible science.

---

## Supported Versions

SpectraMind uses semantic versioning. We maintain one **Active** line (feature + security), one **LTS** line (security-only), and retire older branches.

| Branch / Version line                     | Status     | Fixes Provided               | Planned EOL\*                   |
| ----------------------------------------- | ---------- | ---------------------------- | ------------------------------- |
| **0.50.x** (current best: `v0.50.0-best`) | **Active** | Security + critical features | \~6 months after next minor     |
| **0.49.x**                                | **LTS**    | Security fixes only          | When 0.51.x is stable + 60 days |
| **< 0.49**                                | **EOL**    | No fixes                     | Repro only (archival)           |

\* We align EOL with competition milestones and will communicate any changes in the CHANGELOG and Release notes.

**Notes**

* All security fixes are released on the **highest maintained line** first, then backported to LTS if practical.
* Containers and CI workflows inherit the repo’s line status.

---

## How to Report a Vulnerability

**Please use a private channel**. Do **not** open public issues for suspected vulnerabilities.

### Preferred: GitHub Private Advisory

1. Go to the repository → **Security** tab → **Report a vulnerability**.
2. Provide a clear description, reproduction steps, and impact assessment (see template below).
3. Add logs, minimal PoCs, and affected commit(s)/tag(s) if possible.

### Alternative: Email (encrypted preferred)

* Send to: **security (at) project maintainers** *(replace with your team alias)*
* Subject: `Security: <short summary>`
* If you need encryption, attach a PGP key or request ours in the advisory thread.

We acknowledge receipt within **2 business days** (see SLA below).

---

## Coordinated Disclosure & SLAs

We follow responsible/coordinated disclosure principles.

**Timeline (targets, not guarantees):**

* **Triage & Acknowledgement:** ≤ **2 business days**
* **Initial Assessment:** ≤ **7 days** with provisional CVSS v3.1 score and scope
* **Fix or Mitigation for Critical/High:** ≤ **30 days** (expedited if actively exploited)
* **Fix or Mitigation for Medium/Low:** ≤ **90 days**
* **Advisory Publication:** Within **7 days** of a patch release or after a mutually agreed embargo

**Embargo:** If exploitation appears likely, we coordinate a short embargo to allow patching. We may ship mitigations (config flags, CI rules, container patches) before a full code fix.

---

## Scope

### In-Scope

* **Core pipeline code:** `src/**` (encoders/decoders, diagnostics, symbolic layers)
* **CLI & entrypoints:** `spectramind …` (Typer), `selftest.py`, packaging, submission tooling
* **Server & API:** `src/server/**` (FastAPI, artifact serving, CLI bridge)
* **GUI (read-only dashboards):** `src/gui/**` and docs site if it loads local/served artifacts
* **Build & runtime:** Dockerfiles, `docker-compose.yml`, GitHub Actions, Make targets that ship to users
* **Configs & scripts:** `configs/**`, `bin/**`, `scripts/**`, reproducibility manifests

### Out-of-Scope (unless leading to a real exploit in our environment)

* **Third-party vulnerabilities** with no demonstrated exploit path via SpectraMind
* **Denial of Service** via impractically large inputs or self-inflicted misconfiguration on local machines
* **Social engineering**, phishing, or attacks on maintainers’ personal accounts
* **Self-XSS** in local-only artifacts you hand-edit
* **Kaggle platform issues** (report to Kaggle)

---

## Severity & Scoring

We use **CVSS v3.1** (base score + temporal/environmental where appropriate). We may reprioritize based on:

* Exploitability (public PoC, active exploitation)
* Data sensitivity (e.g., credentials, submission tokens)
* Supply-chain blast radius (CI/CD, container bases)
* Competition timing (risk to participants & reproducibility)

---

## What a Good Report Includes (Template)

```
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
```

---

## Patch, Backport, and Disclosure Process

1. **Private fix branch** with restricted reviewers. Security-relevant commits reference the private advisory ID.
2. **Tests** added: regression + minimal PoC harness if safe.
3. **Fix release** on the **highest maintained line** (e.g., `0.50.x`) with a version bump (`0.50.1`), then **backport** to LTS where feasible.
4. **Release notes** include a short description, affected versions, and thanks (credit) if desired.
5. **Public advisory** (GitHub Security Advisory) with CVSS, affected versions, and mitigation steps.
6. **Artifacts** (containers, SBOMs) rebuilt and signed where supported.

---

## Supply Chain & Dependencies

* **SBOM:** Generate CycloneDX SBOM via `make sbom` (writes to `outputs/sbom.json`).
* **Vuln Scans:** `make audit` runs pip-audit (and optionally grype/trivy if installed).
* **Pinning:** Python deps pinned in `poetry.lock`; Docker bases pinned by digest or exact tag in CI where possible.
* **Updates:** We use automated checks (Dependabot/renovate) and human review for security-relevant bumps.
* **Build provenance:** Reproducible Make/CLI, run manifests (`outputs/manifests/run_manifest_*.json`), and config hashes (`run_hash_summary_v50.json`).

---

## Secrets & Credentials

* **Do not commit secrets.** Use `.env` files and environment variables.
* **Rotation:** If a secret is exposed (e.g., Kaggle token), rotate immediately and notify in the advisory thread.
* **Validation:** `make validate-env` can enforce local env schema (if `scripts/validate_env.py` present).

---

## Hardening Guidance (Recommended)

* **Containers:** Run with least privilege; map only needed volumes; prefer `--read-only` where possible.
* **Network:** Expose API/UI only to trusted hosts; prefer 127.0.0.1 during development.
* **CLI:** Avoid running untrusted configs; review overrides; keep `selftest` green.
* **Artifacts:** Serve dashboards read-only; never accept file uploads in production without additional sandboxing.
* **CI:** Use OIDC or short-lived tokens; restrict workflow permissions; pin actions by SHA.

---

## Safe Harbor

We will not pursue civil or criminal action, or ask law enforcement to investigate you, **provided that** you:

* Make a **good-faith** effort to avoid privacy violations and data destruction.
* Do not exploit a vulnerability beyond what is necessary to prove it exists.
* Use the **private reporting channels** above and **do not disclose** until we complete remediation or agree on a coordinated timeline.
* Follow applicable laws.

If in doubt, contact us first via the private advisory channel.

---

## No Bug Bounty (for now)

We **do not** run a paid bounty program. We’re happy to **credit** reporters in release notes/advisories (opt-in).

---

## Contact & Updates

* **Primary:** GitHub **Security → Report a vulnerability** (preferred)
* **Status updates:** We’ll post progress (ack → triage → fix → release) inside the advisory thread.
* **Public comms:** On patch release, we publish release notes and a GitHub Security Advisory.

---

## Appendix: Fast Mitigation Playbook (for Operators)

* **Containers:** Rebuild and redeploy the fixed tag; prefer digest-pinned images.
* **Pip installs:** Update to the patched `0.50.x` or LTS version; re-create venvs.
* **Configs:** Apply temporary hardening flags from the advisory (e.g., disable live API endpoints, switch to `--no-open` dashboard mode).
* **CI:** Rotate tokens; re-run `make audit sbom`.
* **Artifacts:** Regenerate dashboards with fixed code to avoid serving vulnerable bundles.

---

*Last updated:* 2025-08-31 (America/Chicago)
