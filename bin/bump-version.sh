#!/usr/bin/env bash
# bump-version.sh — SpectraMind V50
# Bump the project version across VERSION and pyproject.toml, commit, tag, and update CHANGELOG.
# Supports: major | minor | patch | set X.Y.Z | prerelease | build metadata | dry-run
# Safe defaults: fails on dirty git, validates semver, annotated tags, reproducible output.

set -Eeuo pipefail

### Config (edit if your repo is unusual) ######################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${REPO_ROOT}/VERSION"
PYPROJECT_FILE="${REPO_ROOT}/pyproject.toml"
CHANGELOG_FILE="${REPO_ROOT}/CHANGELOG.md"
TAG_PREFIX="v"                 # git tag will be vX.Y.Z
DEFAULT_PRE="rc"               # prerelease identifier if not provided
POETRY_BIN="${POETRY_BIN:-poetry}"  # override via env if needed
###############################################################################

# Colors
bold() { printf "\033[1m%s\033[0m" "$*"; }
warn() { printf "\033[33m%s\033[0m\n" "$*" >&2; }
err()  { printf "\033[31m%s\033[0m\n" "$*" >&2; }

usage() {
  cat <<EOF
$(basename "$0") — bump project version (SemVer)

USAGE:
  $(basename "$0") [command] [options]

COMMANDS:
  major                 Bump X.Y.Z -> (X+1).0.0
  minor                 Bump X.Y.Z -> X.(Y+1).0
  patch                 Bump X.Y.Z -> X.Y.(Z+1)
  prerelease            Add/advance prerelease (e.g., -rc.1 -> -rc.2; otherwise add -rc.1)
  set <X.Y.Z[-pre][+meta]>  Set an explicit version

OPTIONS:
  -p, --pre <id>        Prerelease id when adding one (default: ${DEFAULT_PRE})
  -m, --meta <data>     Build metadata to append (e.g., build.5 => +build.5)
  -n, --dry-run         Print planned changes but do not write
  --no-commit           Do not create a git commit
  --no-tag              Do not create a git tag
  -y, --yes             Non-interactive (assume yes)
  -h, --help            Show this help

EFFECT:
  * Updates VERSION
  * Updates tool.poetry.version in pyproject.toml (if present)
  * Updates CHANGELOG.md (creates an entry with git log since last tag)
  * Creates git commit and annotated tag (unless suppressed)

EXAMPLES:
  $(basename "$0") minor
  $(basename "$0") prerelease --pre beta
  $(basename "$0") set 1.2.3
  $(basename "$0") patch -m build.7
  $(basename "$0") minor --dry-run
EOF
}

# --- Helpers -----------------------------------------------------------------

require_clean_git() {
  if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    err "Not inside a git repository."
    exit 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    err "Working tree is dirty. Commit or stash changes first."
    exit 1
  fi
}

confirm() {
  local prompt="${1:-Proceed?} [y/N]: "
  if [[ "${ASSUME_YES:-0}" == "1" ]]; then
    return 0
  fi
  read -r -p "$prompt" ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

read_version_from_files() {
  local v=""
  if [[ -f "$VERSION_FILE" ]]; then
    v="$(sed -n '1s/[[:space:]]//gp' "$VERSION_FILE")"
  fi
  if [[ -z "$v" && -f "$PYPROJECT_FILE" ]]; then
    v="$(sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$PYPROJECT_FILE")"
  fi
  echo "$v"
}

is_semver() {
  # SemVer 2.0.0 (allowing prerelease & build)
  [[ "$1" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-[0-9A-Za-z]+(\.[0-9A-Za-z-]+)*)?(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?$ ]]
}

inc_numeric() { echo $(( $1 + 1 )); }

bump_part() {
  local ver="$1" part="$2"
  # strip prerelease/build for bumping
  local core="${ver%%-*}"
  core="${core%%+*}"
  local major minor patch
  IFS=. read -r major minor patch <<<"$core"
  case "$part" in
    major) major=$(inc_numeric "$major"); minor=0; patch=0 ;;
    minor) minor=$(inc_numeric "$minor"); patch=0 ;;
    patch) patch=$(inc_numeric "$patch") ;;
    *) err "Unknown part: $part"; exit 1 ;;
  esac
  echo "${major}.${minor}.${patch}"
}

advance_prerelease() {
  local ver="$1" id="$2"
  local core="${ver%%-*}"
  local rest="${ver#${core}}"
  local meta=""
  if [[ "$core" != "$ver" ]]; then
    # has prerelease or build; split build metadata if present
    if [[ "$rest" == *"+"* ]]; then
      meta="+${rest#*+}"
      rest="${rest%%+*}"
    fi
  else
    rest=""
  fi

  if [[ "$rest" =~ ^-([0-9A-Za-z.-]+)$ ]]; then
    local pre="${BASH_REMATCH[1]}"
    # if same id exists ending with .N -> increment N
    if [[ "$pre" =~ ^(${id})(\.([0-9]+))?$ ]]; then
      local num="${BASH_REMATCH[3]:-0}"
      num=$(inc_numeric "$num")
      echo "${core}-${id}.${num}${meta}"
      return
    fi
  fi
  # no prerelease or different id -> add id.1
  echo "${core}-${id}.1${meta}"
}

add_metadata() {
  local ver="$1" meta="$2"
  # Strip any existing +... then append new +meta
  local base="${ver%%+*}"
  echo "${base}+${meta}"
}

update_VERSION() {
  echo "$1" > "$VERSION_FILE"
}

update_pyproject() {
  if [[ -f "$PYPROJECT_FILE" ]]; then
    # update tool.poetry.version = "..."
    if grep -qE '^[[:space:]]*version[[:space:]]*=' "$PYPROJECT_FILE"; then
      sed -i.bak -E "s/^([[:space:]]*version[[:space:]]*=\s*)\"[^\"]+\"/\1\"$1\"/" "$PYPROJECT_FILE"
      rm -f "${PYPROJECT_FILE}.bak"
    fi
  fi
}

update_changelog() {
  local new="$1"
  local tag="${TAG_PREFIX}${new}"
  local today
  today="$(date +%Y-%m-%d)"
  local prev_tag=""
  if git tag --list "${TAG_PREFIX}*" >/dev/null; then
    prev_tag="$(git tag --list "${TAG_PREFIX}*" --sort=-creatordate | head -n1 || true)"
  fi

  {
    echo "## ${new} — ${today}"
    if [[ -n "$prev_tag" ]]; then
      git log --pretty=format:'- %s (%h)' "${prev_tag}..HEAD"
    else
      git log --pretty=format:'- %s (%h)'
    fi
    echo
  } | awk 'NR==1{print;next}1' | sed '1s/^/## /' >/dev/null # no-op to appease shellcheck

  # Prepend to CHANGELOG.md
  local tmp
  tmp="$(mktemp)"
  {
    echo "## ${new} — ${today}"
    if [[ -n "$prev_tag" ]]; then
      git log --pretty=format:'- %s (%h)' "${prev_tag}..HEAD"
    else
      git log --pretty=format:'- %s (%h)'
    fi
    echo
    if [[ -f "$CHANGELOG_FILE" ]]; then cat "$CHANGELOG_FILE"; fi
  } > "$tmp"
  mv "$tmp" "$CHANGELOG_FILE"
}

git_commit_and_tag() {
  local new="$1"
  git add "$VERSION_FILE"
  [[ -f "$PYPROJECT_FILE" ]] && git add "$PYPROJECT_FILE"
  [[ -f "$CHANGELOG_FILE" ]] && git add "$CHANGELOG_FILE"
  git commit -m "chore(release): bump version to ${new}"
  git tag -a "${TAG_PREFIX}${new}" -m "Release ${new}"
}

# --- Parse args --------------------------------------------------------------

CMD=""
EXPLICIT=""
PRE_ID="${DEFAULT_PRE}"
META=""
DRY_RUN=0
DO_COMMIT=1
DO_TAG=1
ASSUME_YES=0

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    major|minor|patch|prerelease|set) CMD="$1"; shift ;;
    -p|--pre) PRE_ID="${2:?}"; shift 2 ;;
    -m|--meta) META="${2:?}"; shift 2 ;;
    -n|--dry-run) DRY_RUN=1; shift ;;
    --no-commit) DO_COMMIT=0; shift ;;
    --no-tag) DO_TAG=0; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "$CMD" ]]; then
  err "No command provided."
  usage
  exit 2
fi

if [[ "$CMD" == "set" ]]; then
  EXPLICIT="${ARGS[0]:-}"
  if [[ -z "$EXPLICIT" ]]; then
    err "Usage: $(basename "$0") set X.Y.Z[-pre][+meta]"
    exit 2
  fi
fi

# --- Main --------------------------------------------------------------------

current="$(read_version_from_files || true)"
if [[ -z "$current" ]]; then
  warn "No version found; defaulting to 0.0.0"
  current="0.0.0"
fi

if ! is_semver "$current"; then
  err "Current version '${current}' is not valid SemVer."
  exit 1
fi

case "$CMD" in
  major|minor|patch)
    base="$(bump_part "$current" "$CMD")"
    new="$base"
    ;;
  prerelease)
    new="$(advance_prerelease "$current" "$PRE_ID")"
    ;;
  set)
    new="$EXPLICIT"
    ;;
esac

# Add build metadata if requested
if [[ -n "$META" ]]; then
  new="$(add_metadata "$new" "$META")"
fi

if ! is_semver "$new"; then
  err "Proposed version '${new}' is not valid SemVer."
  exit 1
fi

echo
echo "$(bold "Current:")  $current"
echo "$(bold "Proposed:") $new"
echo

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] Would update:"
  echo "  - $VERSION_FILE -> $new"
  echo "  - $PYPROJECT_FILE (tool.poetry.version) -> $new (if present)"
  echo "  - $CHANGELOG_FILE (prepend new section)"
  echo "  - Create git commit and tag ${TAG_PREFIX}${new}"
  exit 0
fi

require_clean_git

if ! confirm "Bump version to ${new}?"; then
  echo "Aborted."
  exit 1
fi

# Write files
update_VERSION "$new"
update_pyproject "$new"
update_changelog "$new"

# Git ops
if [[ "$DO_COMMIT" -eq 1 ]]; then
  git_commit_and_tag "$new"
  if [[ "$DO_TAG" -eq 0 ]]; then
    # delete the tag if created (rare path)
    git tag -d "${TAG_PREFIX}${new}" >/dev/null 2>&1 || true
  fi
else
  echo "Skipping git commit/tag as requested."
fi

echo
echo "$(bold "Done.") New version: $new"
[[ "$DO_COMMIT" -eq 1 ]] && echo "Created commit and tag ${TAG_PREFIX}${new}."
echo

# Optional: show Poetry version check
if command -v "$POETRY_BIN" >/dev/null 2>&1 && [[ -f "$PYPROJECT_FILE" ]]; then
  echo "Poetry reports version: $("$POETRY_BIN" version | awk '{print $2}')"
fi