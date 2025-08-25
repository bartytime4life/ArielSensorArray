# 🔐 Security Policy

SpectraMind V50 — Neuro-symbolic, physics-informed AI pipeline  
for the [NeurIPS 2025 Ariel Data Challenge](https://www.kaggle.com/competitions/neurips-2025-ariel).

---

## 📢 Supported Versions

We provide security updates and patches for the latest **active branch** only:

| Version          | Supported |
| ---------------- | ----------|
| `main` (default) | ✅        |
| older tags       | ❌        |

For research reproducibility, all versions remain archived, but only the **current release** is maintained for security.

---

## 🛡️ Reporting a Vulnerability

If you discover a security vulnerability in SpectraMind V50:

1. **Do not open a public issue.**
2. Email the maintainers at:

   ```
   security@spectramind.dev
   ```

3. Include:
   - A clear description of the vulnerability.
   - Steps to reproduce (if applicable).
   - Potential impact on the pipeline or data.

We will acknowledge receipt within **72 hours** and provide a timeline for a fix after triage.

---

## 🔍 Scope

This project is research-grade but **security-conscious**. We are especially interested in reports regarding:

- **Pipeline integrity**  
  (e.g., malicious config overrides, DVC/lakeFS data tampering, reproducibility hash bypass).

- **CLI / Config execution**  
  (e.g., unsafe parameter injection in Typer/Hydra commands).

- **Docker & environment hardening**  
  (e.g., container escapes, dependency exploits, unsafe GPU drivers).

- **Dependency vulnerabilities**  
  (Python, Poetry, Docker, GitHub Actions).

- **Data security & privacy**  
  (e.g., sensitive data leakage from logs or outputs).

---

## 🧰 Security Best Practices for Contributors

- Always run `pre-commit run --all-files` before committing (includes `bandit` checks).
- Keep dependencies updated (`poetry update` or rely on Dependabot PRs).
- Never commit secrets, tokens, or large datasets.  
  Use `.env`, DVC, or GitHub Actions secrets instead.
- Verify all configs with:
  ```bash
  spectramind selftest --deep
  ```

---

## 📦 Disclosure Policy

- We follow **Coordinated Disclosure**:  
  vulnerabilities are disclosed **privately first**, fixed, then announced once patches are released.

- Credit will be given to security researchers who responsibly disclose issues.

---

## 🔮 Future Work

- Integration with **CodeQL** scans (already enabled in CI).
- Automated dependency monitoring with **Dependabot**.
- Periodic **Docker image security scans**.
- Expanded **symbolic verification** for runtime security.

---

### 🙏 Thank You

Your diligence helps us keep the SpectraMind V50 pipeline **reproducible, secure, and trustworthy**.  
We deeply appreciate responsible disclosures from the community.