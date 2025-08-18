#!/usr/bin/env bash
set -euo pipefail

# usage: scripts/bump_version.sh patch|minor|major
PART="${1:-patch}"
if ! [[ "$PART" =~ ^(patch|minor|major)$ ]]; then
  echo "Usage: $0 {patch|minor|major}" >&2
  exit 2
fi

# 1) bump pyproject version with Poetry
poetry version "$PART" >/dev/null
NEW_VER="$(poetry version -s)"

# 2) write src/asa/__init__.py to reflect version via importlib.metadata fallback
PKG_INIT="src/asa/__init__.py"
mkdir -p "$(dirname "$PKG_INIT")"
cat > "$PKG_INIT" <<PY
__all__ = ["__version__"]
try:
    from importlib.metadata import version as _pkg_version
    __version__ = _pkg_version("arielsensorarray")
except Exception:
    __version__ = "${NEW_VER}"
PY

# 3) commit + tag
git add pyproject.toml "$PKG_INIT"
git commit -m "release: bump version to v${NEW_VER}"
git tag -a "v${NEW_VER}" -m "Release v${NEW_VER}"

# 4) push code + tag
git push origin HEAD
git push origin "v${NEW_VER}"

echo "✅ Bumped to v${NEW_VER} and pushed with tag."
