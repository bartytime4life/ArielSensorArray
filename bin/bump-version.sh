#!/usr/bin/env bash
# ==============================================================================
# bin/bump-version.sh — SpectraMind V50 release bumper (upgraded)
# ------------------------------------------------------------------------------
# Bumps the project version across VERSION and pyproject.toml, updates CHANGELOG,
# commits, optionally signs/annotates a tag, and can push to origin. Designed to
# be CI- and developer-friendly with dry-run, unified diff preview, and JSON out.
#
# Preferred path:
#   • Uses bin/version_tools.py for robust PEP440-ish semantics & diffs
# Fallback path (if Python tool not present):
#   • Pure-bash semver bump with minimal guarantees
#
# Usage
#   bin/bump-version.sh <command> [options]
#
# Commands
#   major                  Bump X.Y.Z -> (X+1).0.0
#   minor                  Bump X.Y.Z -> X.(Y+1).0
#   patch                  Bump X.Y.Z -> X.Y.(Z+1)
#   prerelease             Add/advance pre (e.g., rc.1 -> rc.2; else add <id>.1)
#   set <X.Y.Z[-pre][+meta]>  Set explicit version
#
# Options
#   -p, --pre <id>         Pre id when adding one (alpha|beta|rc)  (default: rc)
#   -m, --meta <data>      Build metadata to append (e.g., build.7 => +build.7)
#   -n, --dry-run          Print planned changes; do not write
#       --no-commit        Do not create git commit
#       --no-tag           Do not create git tag
#       --gpg-sign         Sign tag (-s) instead of annotate (-a)
#       --push             git push (and --tags if a tag is created)
#       --json             Emit JSON result summary
#       --no-validate      Skip VERSION↔pyproject validation gate
#       --preview-diff     Show unified file diffs (VERSION/pyproject/CHANGELOG)
#   -y, --yes              Non-interactive (assume yes)
#   -h, --help             Show help
#
# Exit codes
#   0 OK, 1 failure (validation/git/parse), 2 usage
# ==============================================================================

set -Eeuo pipefail

# ---------- repo layout ----------
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VERSION_FILE="${REPO_ROOT}/VERSION"
PYPROJECT_FILE="${REPO_ROOT}/pyproject.toml"
CHANGELOG_FILE="${REPO_ROOT}/CHANGELOG.md"
TAG_PREFIX="v"
DEFAULT_PRE="rc"
POETRY_BIN="${POETRY_BIN:-poetry}"

# ---------- pretty ----------
is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RED="$(tput setaf 1)"
  GRN="$(tput setaf 2)"; YLW="$(tput setaf 3)"; CYN="$(tput setaf 6)"; RST="$(tput sgr0)"
else
  BOLD=""; DIM=""; RED=""; GRN=""; YLW=""; CYN=""; RST=""
fi
say()  { printf "%s[REL]%s %s\n" "$CYN" "$RST" "$*"; }
warn() { printf "%s[REL]%s %s\n" "$YLW" "$RST" "$*" >&2; }
err()  { printf "%s[REL]%s %s\n" "$RED" "$RST" "$*" >&2; }

usage() {
  sed -n '1,200p' "$0" | sed 's/^# \{0,1\}//'
  exit "${1:-0}"
}

# ---------- args ----------
CMD=""
EXPLICIT=""
PRE_ID="$DEFAULT_PRE"
META=""
DRY=0
DO_COMMIT=1
DO_TAG=1
GPG_SIGN=0
PUSH=0
JSON=0
VALIDATE=1
PREVIEW_DIFF=0
YES=0

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    major|minor|patch|prerelease|set) CMD="$1"; shift ;;
    -p|--pre)   PRE_ID="${2:?}"; shift 2 ;;
    -m|--meta)  META="${2:?}"; shift 2 ;;
    -n|--dry-run) DRY=1; shift ;;
    --no-commit) DO_COMMIT=0; shift ;;
    --no-tag)    DO_TAG=0; shift ;;
    --gpg-sign)  GPG_SIGN=1; shift ;;
    --push)      PUSH=1; shift ;;
    --json)      JSON=1; shift ;;
    --no-validate) VALIDATE=0; shift ;;
    --preview-diff) PREVIEW_DIFF=1; shift ;;
    -y|--yes)   YES=1; shift ;;
    -h|--help)  usage 0 ;;
    *)          ARGS+=("$1"); shift ;;
  esac
done

if [[ -z "${CMD}" ]]; then
  err "No command provided."; usage 2
fi
if [[ "$CMD" == "set" ]]; then
  EXPLICIT="${ARGS[0]:-}"
  [[ -n "$EXPLICIT" ]] || { err "Usage: $(basename "$0") set X.Y.Z[-pre][+meta]"; exit 2; }
fi

# ---------- git checks ----------
require_clean_git() {
  command -v git >/dev/null 2>&1 || { err "git not found"; exit 1; }
  git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { err "Not inside a git repository."; exit 1; }
  if [[ -n "$(git status --porcelain)" ]]; then
    err "Working tree is dirty. Commit or stash changes first."; exit 1
  fi
}
confirm() {
  local prompt="${1:-Proceed?} [y/N]: "
  [[ $YES -eq 1 ]] && return 0
  read -r -p "$prompt" ans
  [[ "$ans" =~ ^[Yy]$ ]]
}

# ---------- helpers ----------
read_version_from_files() {
  local v=""
  [[ -f "$VERSION_FILE" ]] && v="$(sed -n '1s/[[:space:]]//gp' "$VERSION_FILE")"
  if [[ -z "$v" && -f "$PYPROJECT_FILE" ]]; then
    v="$(sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$PYPROJECT_FILE" | head -n1)"
  fi
  echo "$v"
}
is_semver() {
  [[ "$1" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)(-([0-9A-Za-z]+(\.[0-9A-Za-z-]+)*))?(\+([0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*))?$ ]]
}
inc() { echo $(( $1 + 1 )); }
bump_part() {
  local ver="$1" part="$2" core="${ver%%-*}"; core="${core%%+*}"
  local major minor patch; IFS=. read -r major minor patch <<<"$core"
  case "$part" in
    major) major=$(inc "$major"); minor=0; patch=0 ;;
    minor) minor=$(inc "$minor"); patch=0 ;;
    patch) patch=$(inc "$patch") ;;
    *) err "Unknown part: $part"; exit 1 ;;
  esac
  echo "${major}.${minor}.${patch}"
}
advance_prerelease() {
  local ver="$1" id="$2"
  local base="${ver%%-*}"; local tail="${ver#${base}}"; local meta=""
  if [[ "$tail" == *"+"* ]]; then meta="+${tail#*+}"; tail="${tail%%+*}"; fi
  if [[ "$tail" =~ ^-([0-9A-Za-z.-]+)$ ]]; then
    local pre="${BASH_REMATCH[1]}"
    if [[ "$pre" =~ ^(${id})(\.([0-9]+))?$ ]]; then
      local n="${BASH_REMATCH[3]:-0}"; n=$(inc "$n"); echo "${base}-${id}.${n}${meta}"; return
    fi
  fi
  echo "${base}-${id}.1${meta}"
}
add_meta() {
  local ver="$1" meta="$2"
  echo "${ver%%+*}+${meta}"
}

write_version() { printf '%s\n' "$1" > "$VERSION_FILE"; }
write_pyproject() {
  [[ -f "$PYPROJECT_FILE" ]] || return 0
  if grep -qE '^[[:space:]]*version[[:space:]]*=' "$PYPROJECT_FILE"; then
    # GNU/BSD sed compatible
    perl -0777 -pe 's/^(\s*version\s*=\s*")[^"]*(".*)$/\1'"$1"'\2/m' -i "$PYPROJECT_FILE"
  fi
}
prepend_changelog() {
  local new="$1" tag="${TAG_PREFIX}${new}" today; today="$(date +%Y-%m-%d)"
  local prev_tag
  prev_tag="$(git tag --list "${TAG_PREFIX}*" --sort=-creatordate | tail -n1 || true)"
  local tmp; tmp="$(mktemp)"
  {
    echo "## ${new} — ${today}"
    if [[ -n "$prev_tag" ]]; then git log --pretty=format:'- %s (%h)' "${prev_tag}..HEAD"
    else git log --pretty=format:'- %s (%h)'; fi
    echo
    [[ -f "$CHANGELOG_FILE" ]] && cat "$CHANGELOG_FILE"
  } > "$tmp"
  mv "$tmp" "$CHANGELOG_FILE"
}

git_commit_and_tag() {
  local new="$1"; local tag="${TAG_PREFIX}${new}"
  git add "$VERSION_FILE" 2>/dev/null || true
  [[ -f "$PYPROJECT_FILE" ]] && git add "$PYPROJECT_FILE"
  [[ -f "$CHANGELOG_FILE" ]] && git add "$CHANGELOG_FILE"
  git commit -m "chore(release): v${new}"
  if [[ $DO_TAG -eq 1 ]]; then
    if [[ $GPG_SIGN -eq 1 ]]; then
      git tag -s "$tag" -m "$tag"
    else
      git tag -a "$tag" -m "$tag"
    fi
  fi
}

# ---------- primary path: Python tool ----------
PY_TOOL="$REPO_ROOT/bin/version_tools.py"
use_python_tool() {
  [[ -f "$PY_TOOL" ]] && command -v python >/dev/null 2>&1
}

# ---------- compute new version ----------
current="$(read_version_from_files || true)"
[[ -z "$current" ]] && { warn "No version found; defaulting to 0.0.0"; current="0.0.0"; }
if ! is_semver "$current"; then err "Current version '$current' is not valid SemVer."; exit 1; fi

new="$current"
case "$CMD" in
  major|minor|patch)
    if use_python_tool; then
      new=$(python "$PY_TOOL" --get | tr -d '\n' || echo "$current")
      new=$(python "$PY_TOOL" --bump "$CMD" --print-json | python -c 'import json,sys; print(json.load(sys.stdin)["new_version"])' 2>/dev/null || true)
      [[ -z "$new" ]] && new="$(bump_part "$current" "$CMD")"
    else
      new="$(bump_part "$current" "$CMD")"
    fi
    ;;
  prerelease)
    if use_python_tool; then
      new=$(python "$PY_TOOL" --get | tr -d '\n' || echo "$current")
      new=$(python "$PY_TOOL" --pre "$PRE_ID" --print-json | python -c 'import json,sys; print(json.load(sys.stdin)["new_version"])' 2>/dev/null || true)
      [[ -z "$new" ]] && new="$(advance_prerelease "$current" "$PRE_ID")"
    else
      new="$(advance_prerelease "$current" "$PRE_ID")"
    fi
    ;;
  set) new="$EXPLICIT" ;;
esac
[[ -n "$META" ]] && new="$(add_meta "$new" "$META")"
if ! is_semver "$new"; then err "Proposed version '$new' is not valid SemVer."; exit 1; fi

echo
say "Current : ${BOLD}${current}${RST}"
say "Proposed: ${BOLD}${new}${RST}"
echo

# ---------- dry-run preview ----------
if [[ $DRY -eq 1 ]]; then
  say "[dry-run] Would update files and create commit/tag:"
  echo "  - $VERSION_FILE -> $new"
  echo "  - $PYPROJECT_FILE (tool.poetry.version) -> $new (if present)"
  echo "  - $CHANGELOG_FILE (prepend new section)"
  echo "  - Commit: chore(release): v${new}"
  echo "  - Tag   : ${TAG_PREFIX}${new} $([[ $GPG_SIGN -eq 1 ]] && echo '(signed)')"
  exit 0
fi

# ---------- validation gate ----------
if [[ $VALIDATE -eq 1 ]]; then
  if [[ -f "$VERSION_FILE" && -f "$PYPROJECT_FILE" ]]; then
    v_file="$(sed -n '1s/[[:space:]]//gp' "$VERSION_FILE" || true)"
    v_py="$(sed -n 's/^[[:space:]]*version[[:space:]]*=[[:space:]]*"\(.*\)".*/\1/p' "$PYPROJECT_FILE" | head -n1 || true)"
    if [[ -n "$v_py" && -n "$v_file" && "$v_py" != "$v_file" ]]; then
      warn "VERSION ($v_file) differs from pyproject.toml ($v_py). (Will be fixed by bump.)"
    fi
  fi
fi

# ---------- git safety ----------
require_clean_git
[[ $YES -eq 1 ]] || confirm "Bump version to ${new}?" || { say "Aborted."; exit 1; }

# ---------- write files / diffs ----------
if use_python_tool; then
  # Use the tool to write and offer diff preview
  python "$PY_TOOL" --set "$new" --write-version-file --write-pyproject --write-changelog \
                    $([[ $PREVIEW_DIFF -eq 1 ]] && echo --preview-diff) \
                    --print-json >/dev/null || true
else
  # Manual writes
  printf '%s\n' "$new" > "$VERSION_FILE"
  write_pyproject "$new"
  prepend_changelog "$new"
fi

# Preview diffs (if asked)
if [[ $PREVIEW_DIFF -eq 1 ]]; then
  say "Unified diffs:"
  git --no-pager diff --no-index -- "$VERSION_FILE" 2>/dev/null || true
  [[ -f "$PYPROJECT_FILE" ]] && git --no-pager diff --no-index -- "$PYPROJECT_FILE" 2>/dev/null || true
  [[ -f "$CHANGELOG_FILE" ]] && git --no-pager diff --no-index -- "$CHANGELOG_FILE" 2>/dev/null || true
fi

# ---------- git commit/tag/push ----------
if [[ $DO_COMMIT -eq 1 ]]; then
  git_commit_and_tag "$new"
else
  say "Skipping git commit/tag by request."
fi

if [[ $PUSH -eq 1 ]]; then
  say "Pushing branch…"; git push
  if [[ $DO_TAG -eq 1 ]]; then say "Pushing tags…"; git push --tags; fi
fi

echo
say "Done. New version: ${BOLD}${new}${RST}"
if command -v "$POETRY_BIN" >/dev/null 2>&1 && [[ -f "$PYPROJECT_FILE" ]]; then
  pv="$("$POETRY_BIN" version 2>/dev/null | awk '{print $2}')"
  say "Poetry reports version: ${BOLD}${pv:-unknown}${RST}"
fi

# ---------- JSON summary ----------
if [[ $JSON -eq 1 ]]; then
  printf '{'
  printf '"ok": true, '
  printf '"old": "%s", "new": "%s", ' "$current" "$new"
  printf '"commit": %s, "tag": %s, "push": %s' \
    "$([[ $DO_COMMIT -eq 1 ]] && echo true || echo false)" \
    "$([[ $DO_TAG -eq 1 ]] && echo true || echo false)" \
    "$([[ $PUSH -eq 1 ]] && echo true || echo false)"
  printf '}\n'
fi
