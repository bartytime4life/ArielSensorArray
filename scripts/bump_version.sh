#!/usr/bin/env bash
# SpectraMind V50 — version bump & tag helper (safe, flexible)

set -euo pipefail

# ----------------------------
# Usage & args
# ----------------------------
usage() {
  cat <<'USAGE'
Usage:
  scripts/bump_version.sh [--dry-run] [--no-push] [--sign] [--no-tag] [--pre <alpha|beta|rc>] <patch|minor|major>

Examples:
  scripts/bump_version.sh patch
  scripts/bump_version.sh --dry-run minor
  scripts/bump_version.sh --sign --pre rc major
USAGE
}

DRY_RUN=0
NO_PUSH=0
SIGN_TAG=0
NO_TAG=0
PREID=""

PART=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)   DRY_RUN=1; shift ;;
    --no-push)   NO_PUSH=1; shift ;;
    --sign)      SIGN_TAG=1; shift ;;
    --no-tag)    NO_TAG=1; shift ;;
    --pre)       PREID="${2:-}"; shift 2 ;;
    -h|--help)   usage; exit 0 ;;
    patch|minor|major) PART="$1"; shift ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ -z "${PART:-}" ]]; then
  echo "Error: missing version part (patch|minor|major)" >&2
  usage; exit 2
fi

if ! [[ "$PART" =~ ^(patch|minor|major)$ ]]; then
  echo "Error: version part must be patch|minor|major" >&2
  exit 2
fi

if [[ -n "${PREID}" && ! "${PREID}" =~ ^(alpha|beta|rc)$ ]]; then
  echo "Error: --pre must be one of: alpha|beta|rc" >&2
  exit 2
fi

# ----------------------------
# Helpers
# ----------------------------
say()  { printf "\033[36m[bump]\033[0m %s\n" "$*"; }
ok()   { printf "\033[32m[  ok]\033[0m %s\n" "$*"; }
warn() { printf "\033[33m[warn]\033[0m %s\n" "$*"; }
die()  { printf "\033[31m[fail]\033[0m %s\n" "$*" >&2; exit 1; }

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    printf "\033[2m[dry] %s\033[0m\n" "$*"
  else
    eval "$@"
  fi
}

require() {
  command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"
}

# ----------------------------
# Sanity checks
# ----------------------------
require git
if [[ -f pyproject.toml ]]; then
  require poetry
else
  die "pyproject.toml not found (run from repo root)."
fi

# Ensure we're inside a git repo
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || die "Not inside a git repository."

# Ensure clean working tree (no uncommitted changes)
if ! git diff --quiet || ! git diff --cached --quiet; then
  die "Working tree not clean. Commit or stash changes first."
fi

# Ensure not on a detached HEAD
if [[ "$(git rev-parse --abbrev-ref HEAD)" == "HEAD" ]]; then
  die "Detached HEAD. Checkout a branch before bumping."
fi

# Check remote presence (if we intend to push)
if [[ $NO_PUSH -eq 0 ]]; then
  git remote get-url origin >/dev/null 2>&1 || die "No 'origin' remote configured."
fi

# ----------------------------
# Bump with Poetry
# ----------------------------
say "Bumping version: ${PART}"
run "poetry version ${PART} >/dev/null"

NEW_VER="$(poetry version -s)"
# Basic prerelease support: convert 1.2.3 -> 1.2.3-rc.0 (PEP 440: 1.2.3rc0)
# We'll use PEP 440 compliant mapping: alpha->a0, beta->b0, rc->rc0
if [[ -n "${PREID}" ]]; then
  case "$PREID" in
    alpha) PRE_SUFFIX="a0" ;;
    beta)  PRE_SUFFIX="b0" ;;
    rc)    PRE_SUFFIX="rc0" ;;
  esac
  NEW_VER_PEP440="${NEW_VER}${PRE_SUFFIX}"
  say "Setting prerelease: ${NEW_VER_PEP440}"
  run "poetry version ${NEW_VER_PEP440} >/dev/null"
  NEW_VER="${NEW_VER_PEP440}"
fi

ok "New version: v${NEW_VER}"

# ----------------------------
# Sync src/asa/__init__.py
# ----------------------------
PKG_INIT="src/asa/__init__.py"
run "mkdir -p \"$(dirname \"$PKG_INIT\")\""
run "cat > \"$PKG_INIT\" <<'PY'
__all__ = [\"__version__\"]
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version(\"arielsensorarray\")
except Exception:
    __version__ = \"${NEW_VER}\"
PY"

# ----------------------------
# Commit changes
# ----------------------------
say "Committing version bump"
run "git add pyproject.toml \"$PKG_INIT\""
run "git commit -m \"release: bump version to v${NEW_VER}\""

# ----------------------------
# Tag (optional)
# ----------------------------
TAG="v${NEW_VER}"
if [[ $NO_TAG -eq 0 ]]; then
  if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
    die "Tag ${TAG} already exists."
  fi
  say "Creating tag ${TAG}"
  if [[ $SIGN_TAG -eq 1 ]]; then
    run "git tag -s \"${TAG}\" -m \"Release ${TAG}\""
  else
    run "git tag -a \"${TAG}\" -m \"Release ${TAG}\""
  fi
else
  warn "Skipping tag creation (--no-tag)"
fi

# ----------------------------
# Push (optional)
# ----------------------------
if [[ $NO_PUSH -eq 0 ]]; then
  BRANCH="$(git rev-parse --abbrev-ref HEAD)"
  say "Pushing branch ${BRANCH}"
  run "git push origin \"${BRANCH}\""
  if [[ $NO_TAG -eq 0 ]]; then
    say "Pushing tag ${TAG}"
    run "git push origin \"${TAG}\""
  fi
else
  warn "Skipping push (--no-push)"
fi

ok "Done. Current version: v${NEW_VER}"