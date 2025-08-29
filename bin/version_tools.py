#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bin/version_tools.py — SpectraMind V50 version management utility (upgraded)
============================================================================

Mission
-------
A safe, explicit, **CI/Kaggle-friendly** tool to **inspect, validate, bump, set, and synchronize**
the project version across:

  • `VERSION` file (single line; default canonical source)
  • `pyproject.toml` (`tool.poetry.version`)
  • Git (optional conventional commit + signed/annotated tag + push)
  • `CHANGELOG.md` (optional entry injection from recent commit subjects)
  • SpectraMind CLI banner (`spectramind --version`, optional smoke test)

This script assumes **no network** and **no non-standard dependencies**. It is deterministic and
shell-verbose by default, with explicit `--dry-run` and `--quiet` controls. Designed to be called
from **GitHub Actions**, local dev, and **Kaggle** environments.

Key Upgrades (over prior version)
---------------------------------
  • **PEP 440–style** semantic version handling w/ strict parser & helpful errors
  • Rich **bump logic**: major/minor/patch + pre-release (alpha/beta/rc) + finalize + build metadata
  • **Diff preview**: show unified diffs for changed files (VERSION/pyproject/changelog)
  • **JSON output** (`--print-json`) for CI consumption (old/new version, changed files, actions)
  • **Safety rails**:
      - `--ensure-git` fails if not inside a Git repo
      - `--ensure-clean` requires a clean working tree prior to changing files
      - refuse dangerous ops in `--dry-run` only if they *would* fail
  • **Conventional commit** message control + optional **signed** tag (`--gpg-sign`)
  • **Smarter pyproject.toml** editing that preserves formatting and inserts version if missing
  • **CHANGELOG** injection with ISO date and recent commit bullet list
  • **Tag source**: `--from-git` (latest tag `v?X.Y.Z[prerelease]`), fallback to `VERSION`
  • **Metadata controls**: `--meta set:VALUE` or `--meta inc` (increments trailing number)

Exit Codes
----------
0 OK
1 Failure (validation error, IO error, git failure, version parse error, etc.)

Usage (examples)
----------------
  # Print current version from VERSION
  bin/version_tools.py --get

  # Validate pyproject version equals VERSION (non-zero exit on mismatch)
  bin/version_tools.py --validate

  # Set an explicit version and sync pyproject (no git actions)
  bin/version_tools.py --set 0.2.0 --write-pyproject

  # Bump patch (0.2.0 -> 0.2.1), write pyproject, commit+tag, show diffs, JSON for CI
  bin/version_tools.py --bump patch --write-pyproject --commit --tag --preview-diff --print-json

  # Start a pre-release series (0.3.0 -> 0.3.0a1) and add CHANGELOG section
  bin/version_tools.py --bump minor --pre alpha --write-changelog

  # Increment existing pre (0.3.0a1 -> 0.3.0a2)
  bin/version_tools.py --pre alpha

  # Finalize (0.3.0rc3 -> 0.3.0)
  bin/version_tools.py --final

  # Derive VERSION from latest git tag (v0.3.1) and sync pyproject
  bin/version_tools.py --from-git --write-pyproject

  # Create signed tag and push branch+tags
  bin/version_tools.py --commit --tag --push --gpg-sign

Author / License
----------------
SpectraMind V50 — MIT License. © 2025.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import difflib
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple, List, Dict

# ------------------------------ Constants ------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # bin/ → repo root
DEFAULT_VERSION_FILE = REPO_ROOT / "VERSION"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# Strict PEP 440-ish semantic version with classic semver core + optional pre + optional +meta
SEMVER_RE = re.compile(
    r"""
    ^
    (?P<major>0|[1-9]\d*)
    \.
    (?P<minor>0|[1-9]\d*)
    \.
    (?P<patch>0|[1-9]\d*)
    (?:
        (?P<pre>(?:a|b|rc))
        (?P<pre_n>\d+)
    )?
    (?:
        \+(?P<meta>[0-9A-Za-z.-]+)
    )?
    $
    """,
    re.VERBOSE,
)

# Accept tags like v0.1.2, 0.1.2, v0.1.2a1, v0.1.2rc3+exp.1
TAG_RE = re.compile(r"^v?(?P<ver>\d+\.\d+\.\d+(?:a|b|rc)?\d*(?:\+[0-9A-Za-z.-]+)?)$")


# ------------------------------ Utilities ------------------------------


def echo(msg: str, quiet: bool = False) -> None:
    if not quiet:
        print(msg)


def run(cmd: list[str], dry: bool = False, quiet: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with clear echo; honor dry-run and quiet."""
    if dry:
        if not quiet:
            print(f"[dry-run] $ {' '.join(cmd)}")
        # Return a dummy success result
        return subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str, dry: bool = False, quiet: bool = False) -> None:
    """Write file content (creating parents). In dry-run, print a pretty preview block."""
    if dry:
        if not quiet:
            print(f"[dry-run] write -> {path}")
            print("---------8<---------")
            print(content.rstrip("\n"))
            print("---------8<---------")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def unified_diff(old: str, new: str, path: Path) -> str:
    """Return a unified diff string for preview purposes."""
    a = old.splitlines(keepends=True)
    b = new.splitlines(keepends=True)
    diff = difflib.unified_diff(a, b, fromfile=f"{path} (old)", tofile=f"{path} (new)")
    return "".join(diff)


def is_git_repo() -> bool:
    try:
        subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], check=True, capture_output=True)
        return True
    except Exception:
        return False


def git_root() -> Path:
    try:
        cp = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True)
        return Path(cp.stdout.decode().strip())
    except Exception:
        return REPO_ROOT


def git_dirty() -> bool:
    try:
        cp = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True)
        return bool(cp.stdout.strip())
    except Exception:
        return False


def now_utc_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# ------------------------------ Version Model ------------------------------


@dataclasses.dataclass(frozen=True)
class Version:
    major: int
    minor: int
    patch: int
    pre: Optional[str] = None  # 'a' | 'b' | 'rc'
    pre_n: Optional[int] = None
    meta: Optional[str] = None  # build metadata like '+exp.sha.abc'

    @classmethod
    def parse(cls, s: str) -> "Version":
        s = s.strip()
        m = SEMVER_RE.match(s)
        if not m:
            raise ValueError(
                f"Invalid version {s!r}. Expected 'X.Y.Z', optional pre 'aN|bN|rcN', optional '+meta'."
            )
        d = m.groupdict()
        return cls(
            major=int(d["major"]),
            minor=int(d["minor"]),
            patch=int(d["patch"]),
            pre=d["pre"],
            pre_n=int(d["pre_n"]) if d["pre_n"] else None,
            meta=d["meta"],
        )

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        pre = f"{self.pre}{self.pre_n}" if self.pre else ""
        meta = f"+{self.meta}" if self.meta else ""
        return f"{base}{pre}{meta}"

    # ---- bump ops ----
    def bumped(self, which: str, pre: Optional[str] = None) -> "Version":
        which = which.lower()
        if which not in {"major", "minor", "patch"}:
            raise ValueError("which must be one of: major|minor|patch")
        if which == "major":
            v = Version(self.major + 1, 0, 0)
        elif which == "minor":
            v = Version(self.major, self.minor + 1, 0)
        else:  # patch
            v = Version(self.major, self.minor, self.patch + 1)
        if pre:
            v = dataclasses.replace(v, pre=pre_map(pre), pre_n=1)
        return v

    def bumped_pre(self, pre: str) -> "Version":
        pre = pre_map(pre)
        if self.pre == pre:
            # a1 -> a2, rc3 -> rc4, etc.
            return dataclasses.replace(self, pre_n=(self.pre_n or 0) + 1)
        # switch or start pre-series
        return dataclasses.replace(self, pre=pre, pre_n=1)

    def without_pre(self) -> "Version":
        return dataclasses.replace(self, pre=None, pre_n=None)

    def with_meta(self, meta: Optional[str]) -> "Version":
        return dataclasses.replace(self, meta=meta)

    def inc_meta(self) -> "Version":
        """Increment trailing integer in build metadata if present, else start with +build.1."""
        if not self.meta:
            return dataclasses.replace(self, meta="build.1")
        parts = self.meta.split(".")
        if parts and parts[-1].isdigit():
            parts[-1] = str(int(parts[-1]) + 1)
        else:
            parts.append("1")
        return dataclasses.replace(self, meta=".".join(parts))


def pre_map(name: str) -> str:
    name = name.lower()
    if name in {"alpha", "a"}:
        return "a"
    if name in {"beta", "b"}:
        return "b"
    if name in {"rc", "release-candidate"}:
        return "rc"
    raise ValueError("pre must be one of: alpha|beta|rc")


# ------------------------------ Pyproject Sync ------------------------------


def read_pyproject_version(pyproject: Path) -> Optional[str]:
    if not pyproject.exists():
        return None
    text = read_text(pyproject)
    # Robust: match first version key within tool.poetry (or top-level)
    # Prefer an anchored version line to avoid collisions in other tables.
    m = re.search(r'(?m)^\s*version\s*=\s*"(.*?)"\s*$', text)
    return m.group(1) if m else None


def write_pyproject_version(pyproject: Path, new_version: str, dry: bool = False, quiet: bool = False) -> Tuple[str, str]:
    """
    Update the version = "X" entry in pyproject.toml.
    Returns (old_text, new_text) for diff preview.
    """
    if not pyproject.exists():
        raise FileNotFoundError(pyproject)
    old_text = read_text(pyproject)

    # First, try a straightforward replacement on the first "version = "..."" occurrence
    new_text, n = re.subn(r'(?m)^(\s*version\s*=\s*")(.+?)(".*)$', rf'\g<1>{new_version}\3', old_text, count=1)
    if n == 0:
        # Fallback: locate [tool.poetry] block and insert line if missing
        block_re = re.compile(r"(?ms)(^\s*\[tool\.poetry\]\s*)(.*?)(^\s*\[|\Z)")
        m = block_re.search(old_text + "\n")
        if m:
            head, block, tail_start = m.group(1), m.group(2), m.group(3)
            if re.search(r"(?m)^\s*version\s*=", block):
                block_new = re.sub(r'(?m)^\s*version\s*=.*$', f'version = "{new_version}"', block)
            else:
                # insert version near top of block
                block_lines = block.splitlines()
                block_lines.insert(0, f'version = "{new_version}"')
                block_new = "\n".join(block_lines) + ("\n" if block.endswith("\n") else "")
            new_text = old_text[:m.start(2)] + block_new + old_text[m.end(2):]
        else:
            raise ValueError("Could not find [tool.poetry] block to inject version.")

    if not quiet:
        echo(f"pyproject.toml: set version = {new_version}", quiet=quiet)
    if dry:
        return old_text, new_text
    write_text(pyproject, new_text, dry=dry, quiet=quiet)
    return old_text, new_text


# ------------------------------ CHANGELOG ------------------------------


def update_changelog(changelog: Path, new_version: str, dry: bool = False, quiet: bool = False) -> Tuple[str, str]:
    """
    Insert a new version header at the top, with a date and recent commits.
    Returns (old_text, new_text) for diff preview.

    Example entry:
      ## [0.3.0] - 2025-08-21
      - commit subject (abcd123)
      - commit subject (ef56789)
    """
    date_str = dt.date.today().isoformat()
    header = f"## [{new_version}] - {date_str}\n"
    bullets = git_log_bullets(max_items=12)
    body = "\n".join(f"- {s}" for s in bullets) or "- Internal changes."
    entry = f"{header}{body}\n\n"

    if changelog.exists():
        existing = read_text(changelog)
        # If file starts with H1, inject after H1 line; else prepend
        new_content: str
        if existing.lstrip().startswith("#"):
            # find first newline after first line
            first_nl = existing.find("\n")
            if first_nl == -1:
                new_content = existing + "\n\n" + entry
            else:
                new_content = existing[: first_nl + 1] + "\n" + entry + existing[first_nl + 1 :]
        else:
            new_content = entry + existing
        if not dry:
            write_text(changelog, new_content, dry=dry, quiet=quiet)
        echo(f"CHANGELOG: add entry for {new_version}", quiet=quiet)
        return (existing, new_content)
    else:
        new_content = f"# Changelog\n\n{entry}"
        if not dry:
            write_text(changelog, new_content, dry=dry, quiet=quiet)
        echo(f"CHANGELOG: create and add entry for {new_version}", quiet=quiet)
        return ("", new_content)


def git_log_bullets(max_items: int = 12) -> list[str]:
    if not is_git_repo():
        return []
    try:
        cp = subprocess.run(
            ["git", "log", f"-{max_items}", "--pretty=format:%s (%h)"],
            check=True,
            capture_output=True,
        )
        lines = cp.stdout.decode().splitlines()
        # Filter out noisy merge commits
        return [ln for ln in lines if not ln.lower().startswith("merge ")]
    except Exception:
        return []


# ------------------------------ Git Ops ------------------------------


def git_commit(paths: Iterable[Path], message: str, dry: bool = False, quiet: bool = False) -> None:
    rels = [str(p.relative_to(git_root())) for p in paths if p.exists()]
    if rels:
        if dry:
            echo(f"[dry-run] git add {' '.join(rels)}", quiet=quiet)
            echo(f"[dry-run] git commit -m {message!r}", quiet=quiet)
        else:
            run(["git", "add", *rels], dry=False, quiet=quiet)
            run(["git", "commit", "-m", message], dry=False, quiet=quiet)
    else:
        echo("git add (nothing) — no existing files staged", quiet=quiet)
        if not dry:
            run(["git", "commit", "-m", message], dry=False, quiet=quiet)


def git_tag(tag: str, message: str, sign: bool, dry: bool = False, quiet: bool = False) -> None:
    cmd = ["git", "tag"]
    if sign:
        cmd.append("-s")
    else:
        cmd.append("-a")
    cmd += [tag, "-m", message]
    run(cmd, dry=dry, quiet=quiet)


def git_push(with_tags: bool, dry: bool = False, quiet: bool = False) -> None:
    run(["git", "push"], dry=dry, quiet=quiet)
    if with_tags:
        run(["git", "push", "--tags"], dry=dry, quiet=quiet)


def latest_git_tag_version() -> Optional[str]:
    """Return version string from latest annotated tag if it matches our pattern."""
    if not is_git_repo():
        return None
    # Try `git describe --tags --abbrev=0` first
    candidates: List[str] = []
    try:
        cp = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], check=True, capture_output=True)
        candidates.append(cp.stdout.decode().strip())
    except Exception:
        pass
    # Fallback: list tags sorted by version
    try:
        cp = subprocess.run(["git", "tag", "--list"], check=True, capture_output=True)
        tags = [t.strip() for t in cp.stdout.decode().splitlines() if t.strip()]
        candidates += tags[::-1]  # last lines usually newer
    except Exception:
        pass

    for tag in candidates:
        m = TAG_RE.match(tag)
        if m:
            return m.group("ver")
    return None


# ------------------------------ Core R/W ------------------------------


def load_version_from_files(version_file: Path) -> Version:
    if not version_file.exists():
        raise FileNotFoundError(f"VERSION file not found at {version_file}")
    raw = read_text(version_file).strip()
    return Version.parse(raw)


def write_version_file(version_file: Path, v: Version, dry: bool = False, quiet: bool = False) -> Tuple[str, str]:
    """Write VERSION file and return (old_text, new_text) for diff preview."""
    old = read_text(version_file) if version_file.exists() else ""
    new = f"{v}\n"
    if not quiet:
        echo(f"VERSION: set {v}", quiet=quiet)
    write_text(version_file, new, dry=dry, quiet=quiet)
    return (old, new)


def validate(pyproject: Path, version_file: Path, quiet: bool = False) -> bool:
    ok = True
    v_file = read_text(version_file).strip() if version_file.exists() else None
    v_py = read_pyproject_version(pyproject)
    if v_file and v_py and v_file != v_py:
        print(f"[ERROR] Mismatch: VERSION={v_file}, pyproject.toml={v_py}", file=sys.stderr)
        ok = False
    elif v_file and v_py:
        echo(f"[OK] VERSION matches pyproject.toml: {v_file}", quiet=quiet)
    elif v_file and not v_py:
        echo(f"[WARN] pyproject.toml missing version; VERSION={v_file}", quiet=quiet)
    elif v_py and not v_file:
        echo(f"[WARN] VERSION file missing; pyproject.toml={v_py}", quiet=quiet)
    else:
        echo("[WARN] Neither VERSION nor pyproject.toml version found.", quiet=quiet)
    return ok


def ensure_cli_banner(quiet: bool = False) -> None:
    """Optional: invoke `spectramind --version` to display the banner (no parsing)."""
    for cmd in (["spectramind", "--version"], [sys.executable, "-m", "spectramind", "--version"]):
        try:
            run(cmd, dry=False, quiet=quiet)
            return
        except Exception:
            continue
    echo("[WARN] SpectraMind CLI not found to show --version banner.", quiet=quiet)


# ------------------------------ CLI ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpectraMind V50 version manager (VERSION + pyproject.toml + git + changelog).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_read = p.add_argument_group("Read / Validate")
    g_read.add_argument("--get", action="store_true", help="Print the current version and exit.")
    g_read.add_argument("--validate", action="store_true", help="Verify pyproject.toml version matches VERSION.")
    g_read.add_argument("--print-json", action="store_true", help="Print a JSON result summary (for CI).")

    g_src = p.add_argument_group("Source Selection")
    g_src.add_argument("--version-file", type=Path, default=DEFAULT_VERSION_FILE, help="Path to VERSION file.")
    g_src.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT, help="Path to pyproject.toml.")
    g_src.add_argument("--from-git", action="store_true", help="Use latest git tag (vX.Y.Z...) as version source.")

    g_set = p.add_argument_group("Set / Bump / Meta")
    g_set.add_argument("--set", dest="set_version", type=str, help="Set an explicit version (e.g., 0.3.0).")
    g_set.add_argument("--bump", choices=["major", "minor", "patch"], help="Bump version component.")
    g_set.add_argument("--pre", choices=["alpha", "beta", "rc"], help="Set or increment pre-release tag.")
    g_set.add_argument("--final", action="store_true", help="Strip pre-release (finalize).")
    g_set.add_argument("--meta", metavar="MODE", choices=["inc"], help="Build metadata management. Use 'inc' to increment trailing number.")
    g_set.add_argument("--meta-set", metavar="VALUE", help="Explicitly set build metadata string (e.g. 'exp.sha.abcdef').")

    g_sync = p.add_argument_group("Write Targets")
    g_sync.add_argument("--write-version-file", action="store_true", help="Write VERSION file.")
    g_sync.add_argument("--write-pyproject", action="store_true", help="Write pyproject.toml (tool.poetry.version).")
    g_sync.add_argument("--write-changelog", action="store_true", help="Insert a CHANGELOG entry for the new version.")
    g_sync.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG, help="Path to CHANGELOG.md.")
    g_sync.add_argument("--preview-diff", action="store_true", help="Show unified diffs for file changes.")

    g_git = p.add_argument_group("Git Actions")
    g_git.add_argument("--ensure-git", action="store_true", help="Fail if not inside a Git repository.")
    g_git.add_argument("--ensure-clean", action="store_true", help="Require a clean working tree before changes.")
    g_git.add_argument("--commit", action="store_true", help="Create a version bump commit.")
    g_git.add_argument("--tag", action="store_true", help="Create an annotated tag vX.Y.Z.")
    g_git.add_argument("--gpg-sign", action="store_true", help="Create a GPG-signed tag (-s) instead of annotated (-a).")
    g_git.add_argument("--push", action="store_true", help="Push branch and tags.")
    g_git.add_argument("--commit-message", type=str, default=None, help='Custom commit message (default: "chore(release): vX.Y.Z").')

    g_misc = p.add_argument_group("Execution Controls")
    g_misc.add_argument("--dry-run", action="store_true", help="Preview changes and commands without applying.")
    g_misc.add_argument("--quiet", action="store_true", help="Reduce logging.")
    g_misc.add_argument("--show-cli", action="store_true", help="Invoke `spectramind --version` after updates.")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)
    quiet = bool(args.quiet)
    dry = bool(args.dry_run)

    # Git environment checks (optional)
    if args.ensure-git and not is_git_repo():
        print("[ERROR] Not inside a Git repository (use --ensure-git to enforce).", file=sys.stderr)
        return 1
    if args.ensure_clean and is_git_repo() and git_dirty():
        print("[ERROR] Working tree not clean (use --ensure-clean to enforce).", file=sys.stderr)
        return 1

    # Determine current version (source)
    try:
        if args.from_git:
            ver_str = latest_git_tag_version()
            if not ver_str:
                print("[ERROR] Could not derive version from latest git tag.", file=sys.stderr)
                return 1
            current = Version.parse(ver_str)
        else:
            current = load_version_from_files(args.version_file)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    original = current  # keep original for JSON report

    # Early read/validate passthrough
    if args.get and not any([args.set_version, args.bump, args.pre, args.final, args.meta, args.meta_set]):
        print(str(current))
        if args.print_json:
            print(json.dumps({"ok": True, "version": str(current), "source": "git-tag" if args.from_git else "VERSION"}))
        return 0

    if args.validate and not any([args.set_version, args.bump, args.pre, args.final, args.meta, args.meta_set]):
        ok = validate(args.pyproject, args.version_file, quiet=quiet)
        if args.print_json:
            print(json.dumps({"ok": ok, "version_file": str(args.version_file), "pyproject": str(args.pyproject)}))
        return 0 if ok else 1

    # Compute new version
    new = current

    # Explicit set
    if args.set_version:
        try:
            new = Version.parse(args.set_version)
        except Exception as e:
            print(f"[ERROR] --set invalid: {e}", file=sys.stderr)
            return 1

    # Bump operations
    if args.bump:
        try:
            new = new.bumped(args.bump, pre=args.pre)
        except Exception as e:
            print(f"[ERROR] --bump failed: {e}", file=sys.stderr)
            return 1
    else:
        # Only pre handling if requested without bump
        if args.pre:
            try:
                new = new.bumped_pre(args.pre)
            except Exception as e:
                print(f"[ERROR] --pre failed: {e}", file=sys.stderr)
                return 1

    # Finalize (strip pre-release)
    if args.final:
        new = new.without_pre()

    # Metadata
    if args.meta_set:
        new = new.with_meta(args.meta_set)
    if args.meta == "inc":
        new = new.inc_meta()

    # If no change and no writes requested → no-op
    writes_requested = any([args.write_pyproject, args.write_version_file, args.write_changelog])
    if str(new) == str(current) and not writes_requested and not any([args.commit, args.tag, args.push, args.preview_diff]):
        echo(f"[INFO] Version unchanged: {new}", quiet=quiet)
        if args.show_cli:
            ensure_cli_banner(quiet=quiet)
        if args.print_json:
            print(json.dumps({"ok": True, "version": str(new), "changed": False}))
        return 0

    # Stage file edits (collect diffs if requested)
    changed_files: Dict[str, str] = {}
    diffs: Dict[str, str] = {}
    try:
        # VERSION
        if args.write_version_file or any([args.set_version, args.bump, args.pre, args.final, args.meta, args.meta_set]):
            old, new_text = write_version_file(args.version_file, new, dry=dry, quiet=quiet)
            changed_files[str(args.version_file)] = "updated"
            if args.preview_diff:
                diffs[str(args.version_file)] = unified_diff(old, new_text, args.version_file)

        # pyproject.toml
        if args.write_pyproject:
            try:
                old, new_text = write_pyproject_version(args.pyproject, str(new), dry=dry, quiet=quiet)
                changed_files[str(args.pyproject)] = "updated"
                if args.preview_diff:
                    diffs[str(args.pyproject)] = unified_diff(old, new_text, args.pyproject)
            except Exception as e:
                print(f"[ERROR] Failed to write pyproject.toml: {e}", file=sys.stderr)
                return 1

        # CHANGELOG.md
        if args.write_changelog:
            try:
                old, new_text = update_changelog(args.changelog, str(new), dry=dry, quiet=quiet)
                changed_files[str(args.changelog)] = "updated"
                if args.preview_diff:
                    diffs[str(args.changelog)] = unified_diff(old, new_text, args.changelog)
            except Exception as e:
                print(f"[ERROR] Failed to update CHANGELOG: {e}", file=sys.stderr)
                return 1
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1

    # Show diffs
    if args.preview_diff and diffs:
        print("\n# Preview diffs")
        for path, d in diffs.items():
            if d.strip():
                print(d.rstrip("\n"))
            else:
                print(f"--- {path} (no textual change detected)")

    # Git workflow
    if any([args.commit, args.tag, args.push]):
        if not is_git_repo():
            print("[ERROR] Not a git repository; cannot perform git actions.", file=sys.stderr)
            return 1
        if args.ensure_clean and git_dirty():
            print("[ERROR] Working tree not clean; aborting git actions.", file=sys.stderr)
            return 1

        # Commit
        if args.commit:
            msg = args.commit_message or f"chore(release): v{new}"
            paths_to_commit = [args.version_file, args.pyproject, args.changelog]
            if dry:
                echo(f"[dry-run] git add {' '.join(str(p) for p in paths_to_commit if p)}", quiet=quiet)
                echo(f"[dry-run] git commit -m {msg!r}", quiet=quiet)
            else:
                git_commit(paths_to_commit, msg, dry=False, quiet=quiet)

        # Tag
        if args.tag:
            tag = f"v{new}"
            if dry:
                echo(f"[dry-run] git tag {'-s' if args.gpg_sign else '-a'} {tag} -m {tag}", quiet=quiet)
            else:
                git_tag(tag, tag, sign=bool(args.gpg_sign), dry=False, quiet=quiet)

        # Push
        if args.push:
            if dry:
                echo("[dry-run] git push", quiet=quiet)
                if args.tag:
                    echo("[dry-run] git push --tags", quiet=quiet)
            else:
                git_push(with_tags=args.tag, dry=False, quiet=quiet)

    if not quiet:
        echo(f"[OK] New version: {new}", quiet=quiet)

    if args.show_cli:
        ensure_cli_banner(quiet=quiet)

    # JSON report for CI
    if args.print_json:
        print(json.dumps({
            "ok": True,
            "old_version": str(original),
            "new_version": str(new),
            "changed": str(original) != str(new) or bool(changed_files),
            "changed_files": changed_files,
            "from_git": bool(args.from_git),
            "dry_run": bool(args.dry_run),
            "timestamp": now_utc_iso(),
        }))

    return 0


if __name__ == "__main__":
    sys.exit(main())