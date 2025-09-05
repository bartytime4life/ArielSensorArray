#!/usr/bin/env bash

==============================================================================

bin/bump-version.sh — SpectraMind V50 release bumper (ultimate, upgraded)

——————————————————————————

What it does

• Bumps project version across VERSION and pyproject.toml (PEP621 or Poetry)

• Prepends a CHANGELOG section using git log since previous tag

• Git commit + tag (annotated by default, GPG-signed optional) + optional push

• Dry-run, unified diff preview, JSON summary, non-interactive (-y)

• Emits a structured line to logs/v50_debug_log.md for audit trails



Primary path (preferred):

• If bin/version_tools.py exists, uses it for robust PEP 440 semantics



Fallback path:

• Pure bash/awk/perl implementation with SemVer-ish handling + pre/meta



Usage:

bin/bump-version.sh  [options]



Commands:

major                       X.Y.Z → (X+1).0.0

minor                       X.Y.Z → X.(Y+1).0

patch                       X.Y.Z → X.Y.(Z+1)

prerelease                  add/advance pre (e.g., rc.1→rc.2, else add .1)

set <X.Y.Z[-pre][+meta]>    set explicit version



Options:

-p, –pre           Pre id when adding (alpha|beta|rc)     (default: rc)

-m, –meta        Build metadata to append (e.g. build.7 => +build.7)

-n, –dry-run           Print planned changes; do not write

–no-commit         Do not create git commit

–no-tag            Do not create git tag

–gpg-sign          Sign tag (-s) instead of annotate (-a)

–push              git push (and –tags if a tag is created)

–json              Emit JSON result summary to stdout

–json-path      Write JSON summary to a file

–no-validate       Skip VERSION↔pyproject validation gate

–preview-diff      Show unified diffs of changed files (post-write)

-y, –yes               Non-interactive (assume yes)

-h, –help              Show help



Exit codes:

0 OK, 1 failure (validation/git/parse), 2 usage

==============================================================================

set -Eeuo pipefail
: “${LC_ALL:=C}”; export LC_ALL
IFS=$’\n\t’

––––– repo layout –––––

REPO_ROOT=”$(cd “$(dirname “${BASH_SOURCE[0]}”)/..” && pwd)”
VERSION_FILE=”${REPO_ROOT}/VERSION”
PYPROJECT_FILE=”${REPO_ROOT}/pyproject.toml”
CHANGELOG_FILE=”${REPO_ROOT}/CHANGELOG.md”
TAG_PREFIX=”${TAG_PREFIX:-v}”
DEFAULT_PRE=“rc”
POETRY_BIN=”${POETRY_BIN:-poetry}”
PYTHON_BIN=”${PYTHON_BIN:-python3}”

––––– pretty –––––

is_tty() { [[ -t 1 ]]; }
if is_tty && command -v tput >/dev/null 2>&1; then
BOLD=”$(tput bold)”; DIM=”$(tput dim)”; RED=”$(tput setaf 1)”
GRN=”$(tput setaf 2)”; YLW=”$(tput setaf 3)”; CYN=”$(tput setaf 6)”; RST=”$(tput sgr0)”
else
BOLD=””; DIM=””; RED=””; GRN=””; YLW=””; CYN=””; RST=””
fi
say()  { printf “%s[REL]%s %s\n” “$CYN” “$RST” “$”; }
warn() { printf “%s[REL]%s %s\n” “$YLW” “$RST” “$” >&2; }
err()  { printf “%s[REL]%s %s\n” “$RED” “$RST” “$*” >&2; }

usage() {
sed -n ‘1,200p’ “$0” | sed ‘s/^# {0,1}//’
exit “${1:-0}”
}

––––– args –––––

CMD=””
EXPLICIT=””
PRE_ID=”$DEFAULT_PRE”
META=””
DRY=0
DO_COMMIT=1
DO_TAG=1
GPG_SIGN=0
PUSH=0
JSON=0
JSON_PATH=””
VALIDATE=1
PREVIEW_DIFF=0
YES=0

ARGS=()
while [[ $# -gt 0 ]]; do
case “$1” in
major|minor|patch|prerelease|set) CMD=”$1”; shift ;;
-p|–pre)           PRE_ID=”${2:?}”; shift 2 ;;
-m|–meta)          META=”${2:?}”; shift 2 ;;
-n|–dry-run)       DRY=1; shift ;;
–no-commit)        DO_COMMIT=0; shift ;;
–no-tag)           DO_TAG=0; shift ;;
–gpg-sign)         GPG_SIGN=1; shift ;;
–push)             PUSH=1; shift ;;
–json)             JSON=1; shift ;;
–json-path)        JSON_PATH=”${2:?}”; shift 2 ;;
–no-validate)      VALIDATE=0; shift ;;
–preview-diff)     PREVIEW_DIFF=1; shift ;;
-y|–yes)           YES=1; shift ;;
-h|–help)          usage 0 ;;
*)                  ARGS+=(”$1”); shift ;;
esac
done

if [[ -z “${CMD}” ]]; then
err “No command provided.”
usage 2
fi
if [[ “$CMD” == “set” ]]; then
EXPLICIT=”${ARGS[0]:-}”
[[ -n “$EXPLICIT” ]] || { err “Usage: $(basename “$0”) set X.Y.Z[-pre][+meta]”; exit 2; }
fi

––––– git checks –––––

require_clean_git() {
command -v git >/dev/null 2>&1 || { err “git not found”; exit 1; }
git rev-parse –is-inside-work-tree >/dev/null 2>&1 || { err “Not inside a git repository.”; exit 1; }
if [[ -n “$(git status –porcelain)” ]]; then
err “Working tree is dirty. Commit or stash changes first.”; exit 1
fi
}
confirm() {
local prompt=”${1:-Proceed?} [y/N]: “
[[ $YES -eq 1 ]] && return 0
read -r -p “$prompt” ans
[[ “$ans” =~ ^[Yy]$ ]]
}

git_sha_short() { git rev-parse –short HEAD 2>/dev/null || echo “nogit”; }

––––– file helpers –––––

have() { command -v “$1” >/dev/null 2>&1; }
mkdirp() { mkdir -p “$1” 2>/dev/null || true; }

read_version_from_files() {
local v=””
[[ -f “$VERSION_FILE” ]] && v=”$(sed -n ‘1s/[[:space:]]//gp’ “$VERSION_FILE”)”
if [[ -z “$v” && -f “$PYPROJECT_FILE” ]]; then
# Try Python tomllib first (safe for both [project] and [tool.poetry])
if have “$PYTHON_BIN”; then
v=”$(”$PYTHON_BIN” - <<‘PY’ 2>/dev/null || true
import sys, json, re, os
p=os.environ.get(“PYPROJECT_FILE”)
data=open(p,‘rb’).read()
try:
import tomllib
t=tomllib.loads(data.decode(‘utf-8’))
v=t.get(‘project’,{}).get(‘version’) or t.get(‘tool’,{}).get(‘poetry’,{}).get(‘version’) or ‘’
print(v or ‘’)
except Exception:

fallback regex

import re
m=re.search(r’^\sversion\s=\s*”([^”]+)”’, data.decode(‘utf-8’), re.M)
print(m.group(1) if m else ‘’)
PY
)”
fi
[[ -z “$v” ]] && v=”$(sed -n ‘s/^[[:space:]]version[[:space:]]=[[:space:]]”.”.*/\1/p’ “$PYPROJECT_FILE” | head -n1)”
fi
echo “$v”
}

write_version_file() { printf ‘%s\n’ “$1” > “$VERSION_FILE”; }

write_pyproject_version() {
local new=”$1”
[[ -f “$PYPROJECT_FILE” ]] || return 0
if have “$PYTHON_BIN”; then
# Accurate PEP621/Poetry update via tomllib + round-trip minimal editing
“$PYTHON_BIN” - “$new” <<‘PY’
import os, sys, re, io
new = sys.argv[1]
p = os.environ.get(“PYPROJECT_FILE”)
txt = open(p,‘r’,encoding=‘utf-8’).read()

Try to update [project].version if present; else [tool.poetry].version; else first version=

def replace_ver(text, section, key):
# naive but scoped replacement within section boundaries
pattern = r’(’+re.escape(section)+r’[\s\S]?\n)([\s\S]?)(?=\n[|\Z)’
m = re.search(pattern, text, re.M)
if not m: return None
body = m.group(2)
body2, n = re.subn(r’^(\s*’+re.escape(key)+r’\s*=\s*”)([^”])(”.)$’, r’\1’+new+r’\3’, body, flags=re.M)
if n:
return text[:m.start(2)] + body2 + text[m.end(2):]
return None

out = replace_ver(txt, ‘project’, ‘version’)
if out is None:
out = replace_ver(txt, ‘tool.poetry’, ‘version’)
if out is None:
# generic: first version=”…”
out, n = re.subn(r’^(\sversion\s=\s*”)([^”])(”.)$’, r’\1’+new+r’\3’, txt, flags=re.M)
if n==0:
out = txt  # no-op if missing (rare)
open(p,‘w’,encoding=‘utf-8’).write(out)
PY
else
# Perl in-place change of the first version=”…” occurrence
perl -0777 -pe ‘s/^(\sversion\s=\s*”)[^”](”.)$/\1’”$new”’\2/m’ -i “$PYPROJECT_FILE” || true
fi
}

prepend_changelog() {
local new=”$1” today; today=”$(date +%Y-%m-%d)”
local tmp; tmp=”$(mktemp)”
local prev_tag
prev_tag=”$(git tag –list “${TAG_PREFIX}*” –sort=-version:refname | head -n1 || true)”
{
echo “## ${new} — ${today}”
if [[ -n “$prev_tag” ]]; then
git log –no-merges –pretty=format:’- %s (%h)’ “${prev_tag}..HEAD”
else
git log –no-merges –pretty=format:’- %s (%h)’
fi
echo
[[ -f “$CHANGELOG_FILE” ]] && cat “$CHANGELOG_FILE”
} > “$tmp”
mv “$tmp” “$CHANGELOG_FILE”
}

––––– validators –––––

is_semver() {
[[ “$1” =~ ^([0-9]+).([0-9]+).([0-9]+)(-([0-9A-Za-z]+(.[0-9A-Za-z-]+)))?(+([0-9A-Za-z-]+(.[0-9A-Za-z-]+)))?$ ]]
}
inc() { echo $(( $1 + 1 )); }

bump_part() {
local ver=”$1” part=”$2” core=”${ver%%-}”; core=”${core%%+}”
local major minor patch; IFS=. read -r major minor patch <<<”$core”
case “$part” in
major) major=$(inc “$major”); minor=0; patch=0 ;;
minor) minor=$(inc “$minor”); patch=0 ;;
patch) patch=$(inc “$patch”) ;;
*) err “Unknown part: $part”; exit 1 ;;
esac
echo “${major}.${minor}.${patch}”
}

advance_prerelease() {
local ver=”$1” id=”$2”
local base=”${ver%%-}”; local tail=”${ver#${base}}”; local meta=””
if [[ “$tail” == ”+” ]]; then meta=”+${tail#+}”; tail=”${tail%%+*}”; fi
if [[ “$tail” =~ ^-([0-9A-Za-z.-]+)$ ]]; then
local pre=”${BASH_REMATCH[1]}”
if [[ “$pre” =~ ^(${id})(.([0-9]+))?$ ]]; then
local n=”${BASH_REMATCH[3]:-0}”; n=$(inc “$n”); echo “${base}-${id}.${n}${meta}”; return
fi
fi
echo “${base}-${id}.1${meta}”
}

add_meta() {
local ver=”$1” meta=”$2”

preserve existing meta if present

if [[ “$ver” == ”+” ]]; then
echo “${ver%+*}+${meta}”
else
echo “${ver}+${meta}”
fi
}

––––– version tool path –––––

PY_TOOL=”${REPO_ROOT}/bin/version_tools.py”
use_python_tool() { [[ -f “$PY_TOOL” ]] && have “$PYTHON_BIN”; }

––––– compute new version –––––

current=”$(read_version_from_files || true)”
[[ -z “$current” ]] && { warn “No version found; defaulting to 0.0.0”; current=“0.0.0”; }

Accept non-strict PEP440 via the python tool; otherwise require semver-ish

if ! use_python_tool; then
if ! is_semver “$current”; then err “Current version ‘$current’ is not valid SemVer.”; exit 1; fi
fi

new=”$current”
case “$CMD” in
major|minor|patch)
if use_python_tool; then
new=”$(”$PYTHON_BIN” “$PY_TOOL” –bump “$CMD” –print-json 2>/dev/null | “$PYTHON_BIN” -c ‘import sys,json;print(json.load(sys.stdin)[“new_version”])’ || true)”
[[ -z “$new” ]] && new=”$(bump_part “$current” “$CMD”)”
else
new=”$(bump_part “$current” “$CMD”)”
fi
;;
prerelease)
if use_python_tool; then
new=”$(”$PYTHON_BIN” “$PY_TOOL” –pre “$PRE_ID” –print-json 2>/dev/null | “$PYTHON_BIN” -c ‘import sys,json;print(json.load(sys.stdin)[“new_version”])’ || true)”
[[ -z “$new” ]] && new=”$(advance_prerelease “$current” “$PRE_ID”)”
else
new=”$(advance_prerelease “$current” “$PRE_ID”)”
fi
;;
set)
new=”$EXPLICIT”
;;
esac
[[ -n “$META” ]] && new=”$(add_meta “$new” “$META”)”

If no python tool: enforce semver-ish validity

if ! use_python_tool; then
if ! is_semver “$new”; then err “Proposed version ‘$new’ is not valid SemVer.”; exit 1; fi
fi

echo
say “Current : ${BOLD}${current}${RST}”
say “Proposed: ${BOLD}${new}${RST}”
echo

––––– dry-run preview –––––

if [[ $DRY -eq 1 ]]; then
say “[dry-run] Would update files and create commit/tag:”
echo “  - $VERSION_FILE -> $new”
echo “  - $PYPROJECT_FILE (version field) -> $new (if present)”
echo “  - $CHANGELOG_FILE (prepend new section with git log since last tag)”
echo “  - Commit: chore(release): v${new}”
echo “  - Tag   : ${TAG_PREFIX}${new} $([[ $GPG_SIGN -eq 1 ]] && echo ‘(signed)’)”

Optional JSON

if [[ $JSON -eq 1 || -n “${JSON_PATH:-}” ]]; then
payload=$(printf ‘{“ok”:true,“old”:”%s”,“new”:”%s”,“commit”:%s,“tag”:%s,“push”:%s}’ 
“$current” “$new” 
“$([[ $DO_COMMIT -eq 1 ]] && echo true || echo false)” 
“$([[ $DO_TAG -eq 1 ]] && echo true || echo false)” 
“$([[ $PUSH -eq 1 ]] && echo true || echo false)”)
[[ -n “${JSON_PATH:-}” ]] && { printf “%s\n” “$payload” > “$JSON_PATH”; say “Wrote JSON → $JSON_PATH”; }
[[ $JSON -eq 1 ]] && printf “%s\n” “$payload”
fi
exit 0
fi

––––– validation gate –––––

if [[ $VALIDATE -eq 1 && -f “$VERSION_FILE” && -f “$PYPROJECT_FILE” ]]; then
v_file=”$(sed -n ‘1s/[[:space:]]//gp’ “$VERSION_FILE” || true)”
v_py=”$(read_version_from_files || true)”
if [[ -n “$v_py” && -n “$v_file” && “$v_py” != “$v_file” ]]; then
warn “VERSION ($v_file) differs from pyproject.toml ($v_py). (Will be synchronized by bump.)”
fi
fi

––––– git safety –––––

require_clean_git
[[ $YES -eq 1 ]] || confirm “Bump version to ${new}?” || { say “Aborted.”; exit 1; }

––––– write files –––––

if use_python_tool; then

Delegate version writing + changelog to the tool (if it supports these flags)

if “$PYTHON_BIN” “$PY_TOOL” –set “$new” –write-version-file –write-pyproject –write-changelog –print-json >/dev/null 2>&1; then
:
else
# Fallback to manual writes if tool lacks some flags
printf ‘%s\n’ “$new” > “$VERSION_FILE”
write_pyproject_version “$new”
prepend_changelog “$new”
fi
else

Manual writes

write_version_file “$new”
write_pyproject_version “$new”
prepend_changelog “$new”
fi

––––– diffs preview –––––

if [[ $PREVIEW_DIFF -eq 1 ]]; then
say “Unified diffs:”

Use git diff against index (tree should be clean before write)

git –no-pager diff – “$VERSION_FILE” || true
[[ -f “$PYPROJECT_FILE” ]] && git –no-pager diff – “$PYPROJECT_FILE” || true
[[ -f “$CHANGELOG_FILE” ]] && git –no-pager diff – “$CHANGELOG_FILE” || true
fi

––––– commit / tag –––––

if [[ $DO_COMMIT -eq 1 ]]; then
git add “$VERSION_FILE” 2>/dev/null || true
[[ -f “$PYPROJECT_FILE” ]] && git add “$PYPROJECT_FILE”
[[ -f “$CHANGELOG_FILE” ]] && git add “$CHANGELOG_FILE”
git commit -m “chore(release): v${new}”
if [[ $DO_TAG -eq 1 ]]; then
tag=”${TAG_PREFIX}${new}”
if [[ $GPG_SIGN -eq 1 ]]; then
git tag -s “$tag” -m “$tag”
else
git tag -a “$tag” -m “$tag”
fi
fi
else
say “Skipping git commit/tag by request.”
fi

if [[ $PUSH -eq 1 ]]; then
say “Pushing branch…”; git push
if [[ $DO_TAG -eq 1 ]]; then say “Pushing tags…”; git push –tags; fi
fi

––––– poetry echo –––––

if command -v “$POETRY_BIN” >/dev/null 2>&1 && [[ -f “$PYPROJECT_FILE” ]]; then
pv=”$(”$POETRY_BIN” version 2>/dev/null | awk ‘{print $2}’)”
say “Poetry reports version: ${BOLD}${pv:-unknown}${RST}”
fi

––––– structured audit log –––––

mkdirp “${REPO_ROOT}/logs”
AUDIT_LOG=”${REPO_ROOT}/logs/v50_debug_log.md”
ts_iso=”$(date -u +%Y-%m-%dT%H:%M:%SZ)”
git_short=”$(git_sha_short)”
cfg_hash=”-”
if [[ -f “${REPO_ROOT}/run_hash_summary_v50.json” ]]; then
cfg_hash=”$(grep -oE ‘“config_hash”[[:space:]]:[[:space:]]”[^”]+”’ “${REPO_ROOT}/run_hash_summary_v50.json” 2>/dev/null | head -n1 | sed -E ‘s/.:”([^”]+)”./\1/’)”
[[ -z “$cfg_hash” ]] && cfg_hash=”-”
fi
printf ‘[%s] cmd=bump-version git=%s cfg_hash=%s tag=_ pred=_ bundle=_ notes=“old=%s;new=%s”\n’ 
“$ts_iso” “$git_short” “$cfg_hash” “$current” “$new” >> “$AUDIT_LOG”

––––– JSON summary –––––

if [[ $JSON -eq 1 || -n “${JSON_PATH:-}” ]]; then
payload=$(printf ‘{“ok”:true,“old”:”%s”,“new”:”%s”,“commit”:%s,“tag”:%s,“push”:%s}’ 
“$current” “$new” 
“$([[ $DO_COMMIT -eq 1 ]] && echo true || echo false)” 
“$([[ $DO_TAG -eq 1 ]] && echo true || echo false)” 
“$([[ $PUSH -eq 1 ]] && echo true || echo false)”)
[[ -n “${JSON_PATH:-}” ]] && { printf “%s\n” “$payload” > “$JSON_PATH”; say “Wrote JSON → $JSON_PATH”; }
[[ $JSON -eq 1 ]] && printf “%s\n” “$payload”
fi

echo
say “Done. New version: ${BOLD}${new}${RST}”
exit 0