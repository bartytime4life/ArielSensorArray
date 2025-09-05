# 🔐 Security Policy — SpectraMind V50

SpectraMind V50 is a **neuro-symbolic, physics-informed AI pipeline**  
for the [NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/neurips-2025-ariel).  
We treat **security as part of reproducibility** — vulnerabilities can compromise not only systems,  
but also **scientific validity** and **leaderboard integrity**.

---

## 📢 Supported Versions

We provide **security updates** for the actively developed branch only:

| Version          | Supported |
| ---------------- | ----------|
| `main` (default) | ✅        |
| older tags       | ❌        |

Older versions remain archived for **research reproducibility**,  
but security patches are only applied to the current release.

---

## 🛡️ Reporting a Vulnerability

If you discover a security vulnerability in SpectraMind V50:

1. **Do not open a public issue.**  
   Public disclosure without coordination can expose the community to risk.

2. **Report privately via email:**  
   `security@spectramind-v50.org` (PGP key available in `SECURITY.md`)

3. **Include in your report:**
   - Clear description of the vulnerability  
   - Steps to reproduce (CLI commands, configs, Hydra overrides)  
   - Potential impact on pipeline integrity, Kaggle runtime, or data security  
   - Suggested mitigations, if any  

4. **Response commitments:**  
   - Acknowledge within **72 hours**  
   - Provide a triage result and fix timeline  
   - Publish an advisory via [`SECURITY_ADVISORY_TEMPLATE.md`](./SECURITY_ADVISORY_TEMPLATE.md)  

---

## 🔒 Scope

We are most concerned with vulnerabilities that threaten:

- **Pipeline integrity**  
  (malicious Hydra overrides, DVC/lakeFS tampering, reproducibility hash bypass)

- **CLI / Config execution**  
  (unsafe Typer parameters, injection attacks, unvalidated inputs)

- **Docker & environment hardening**  
  (container escapes, CUDA driver exploits, unsafe base images)

- **Dependency vulnerabilities**  
  (Python, Poetry, CUDA, Docker, GitHub Actions)

- **Data security & privacy**  
  (sensitive log leaks, unsafe artifact packaging, metadata exposure)

---

## ✅ Security Best Practices for Contributors

- Run `pre-commit run --all-files` before committing  
  (includes `bandit`, `ruff`, `mypy`, YAML lint)

- Keep dependencies up-to-date (`poetry update`, Dependabot PRs)

- Never commit secrets, tokens, or datasets  
  (use `.env`, DVC, or GitHub Secrets)

- Verify reproducibility & integrity with:  
  ```bash
  spectramind selftest --deep


⸻

✦ Mission Reminder:
Security is part of science.
Every patch must preserve reproducibility, determinism, and Kaggle compliance.