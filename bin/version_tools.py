#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bin/version_tools.py — SpectraMind V50 version management utility
=================================================================

Purpose
-------
A safe, explicit, and automation‑friendly tool to **inspect, validate, bump, set,
and synchronize** the project version across:
  • `VERSION` file (single line, canonical source by default)
  • `pyproject.toml` (tool.poetry.version)
  • Git (commit/tag/push workflow, optional)
  • CHANGELOG.md (optional entry injection)
  • SpectraMind CLI banner (sanity check via `spectramind --version`, optional)

The script is designed to be CI‑safe and Kaggle‑safe (no outbound network calls; git
operations run only if requested and if a repository is present).

Highlights
----------
  • Read/print the current version.
  • Set a specific version (PEP 440-ish classic semantic versions are supported).
  • Bump major/minor/patch or pre-release (alpha/beta/rc) intelligently.
  • Synchronize `pyproject.toml` (Poetry) with the canonical `VERSION` file.
  • Create a conventional commit and optionally an annotated tag `vX.Y.Z`.
  • Generate a CHANGELOG entry (from recent git log) with a timestamp.
  • Dry-run mode shows **exact** file diffs and git commands without changing anything.
  • Verbose and explicit logging to stdout; optional quiet mode.

Usage
-----
  # Print current version from VERSION
  bin/version_tools.py --get

  # Set a specific version and sync pyproject
  bin/version_tools.py --set 0.2.0 --write-pyproject

  # Bump patch (e.g., 0.2.0 -> 0.2.1), write pyproject, commit and tag
  bin/version_tools.py --bump patch --write-pyproject --commit --tag

  # Bump minor pre-release (e.g., 0.2.0 -> 0.3.0a1), and update changelog
  bin/version_tools.py --bump minor --pre alpha --write-changelog

  # Enforce pyproject version = VERSION (no changes, non-zero exit if mismatch)
  bin/version_tools.py --validate

  # Derive VERSION from latest git tag (e.g., v0.3.1) and sync pyproject
  bin/version_tools.py --from-git --write-pyproject

  # Dry-run any operation (no changes)
  bin/version_tools.py ... --dry-run

Conventions
-----------
Default canonical source is `VERSION` at the repo root (single line, e.g., `0.2.0`).
If `--from-git` is provided, the latest annotated tag matching `v?X.Y.Z*` is parsed
and used as the source.

Pre-release flags:
  --pre alpha|beta|rc
  When combined with --bump (major/minor/patch), produces X.Y.Za1 / b1 / rc1.
  When used alone on a version already containing that pre-tag, increments aN->aN+1.

Exit Codes
----------
0 OK
1 Failure (validation error, IO error, git failure, version parse error, etc.)

Author / License
----------------
SpectraMind V50 — MIT License. © 2025.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

# ------------------------------ Constants ------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]  # bin/ → repo root
DEFAULT_VERSION_FILE = REPO_ROOT / "VERSION"
DEFAULT_PYPROJECT = REPO_ROOT / "pyproject.toml"
DEFAULT_CHANGELOG = REPO_ROOT / "CHANGELOG.md"

SEMVER_RE = re.compile(
    r"""
    ^
    (?P<major>0|[1-9]\d*)
    \.
    (?P<minor>0|[1-9]\d*)
    \.
    (?P<patch>0|[1-9]\d*)
    (?:
        (?P<pre>
            a|b|rc
        )
        (?P<pre_n>\d+)
    )?
    (?:
        \+(?P<meta>[0-9A-Za-z.-]+)
    )?
    $
    """,
    re.VERBOSE,
)

TAG_RE = re.compile(r"^v?(?P<ver>\d+\.\d+\.\d+(?:[ab]|rc)?\d*(?:\+[\w.-]+)?)$")


# ------------------------------ Utilities ------------------------------


def run(cmd: list[str], dry: bool = False, quiet: bool = False, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command with clear echo; honor dry-run and quiet."""
    if dry:
        if not quiet:
            print(f"[dry-run] $ {' '.join(cmd)}")
        # Return a dummy success result
        cp = subprocess.CompletedProcess(cmd, 0, stdout=b"", stderr=b"")
        return cp
    if not quiet:
        print(f"$ {' '.join(cmd)}")
    return subprocess.run(cmd, check=check)


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_text(path: Path, content: str, dry: bool = False, quiet: bool = False) -> None:
    if dry:
        if not quiet:
            print(f"[dry-run] write -> {path}")
            print("---------8<---------")
            print(content.rstrip("\n"))
            print("---------8<---------")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def is_git_repo() -> bool:
    try:
        run(["git", "rev-parse", "--is-inside-work-tree"], dry=False, quiet=True)
        return True
    except subprocess.CalledProcessError:
        return False


def git_root() -> Path:
    try:
        cp = subprocess.run(["git", "rev-parse", "--show-toplevel"], check=True, capture_output=True)
        return Path(cp.stdout.decode().strip())
    except Exception:
        return REPO_ROOT


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
            raise ValueError(f"Invalid version: {s!r}")
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
    # Simple, robust regex for tool.poetry.version = "X.Y.Z..."
    m = re.search(r'(?m)^\s*version\s*=\s*"(.*?)"\s*$', text)
    if m:
        return m.group(1)
    return None


def write_pyproject_version(pyproject: Path, new_version: str, dry: bool = False, quiet: bool = False) -> None:
    if not pyproject.exists():
        raise FileNotFoundError(pyproject)
    text = read_text(pyproject)
    new_text, n = re.subn(r'(?m)^(\s*version\s*=\s*")(.+?)(".*)$', rf'\g<1>{new_version}\3', text, count=1)
    if n == 0:
        # Fallback: attempt to locate [tool.poetry] block and insert/replace version
        new_text = text
        block_re = re.compile(r"(?ms)^\s*\[tool\.poetry\]\s*(.*?)\n\s*\[", re.DOTALL)
        bm = block_re.search(text + "\n[")  # sentinel
        if bm:
            block = bm.group(1)
            if re.search(r"(?m)^\s*version\s*=", block):
                new_block = re.sub(r'(?m)^\s*version\s*=.*$', f'version = "{new_version}"', block)
            else:
                # Insert version at top of block
                new_block = f'version = "{new_version}"\n' + block
            new_text = text.replace(block, new_block)
        else:
            raise ValueError("Could not find [tool.poetry] block to inject version.")
    if not quiet:
        print(f"pyproject.toml: set version = {new_version}")
    write_text(pyproject, new_text, dry=dry, quiet=quiet)


# ------------------------------ CHANGELOG ------------------------------


def update_changelog(changelog: Path, new_version: str, dry: bool = False, quiet: bool = False) -> None:
    """
    Insert a new version header at the top, with a timestamp and recent commits summary.

    Format:

    ## [0.3.0] - 2025-08-21
    - commit subject (abcd123)
    - commit subject (ef56789)
    """
    date_str = dt.date.today().isoformat()
    header = f"## [{new_version}] - {date_str}\n"
    bullets = git_log_bullets(max_items=10)
    body = "\n".join(f"- {s}" for s in bullets) or "- Internal changes."
    entry = f"{header}{body}\n\n"
    if changelog.exists():
        existing = read_text(changelog)
        # Place after first H1 or at beginning
        if existing.lstrip().startswith("#"):
            # after main title line
            first_nl = existing.find("\n")
            new_content = existing[: first_nl + 1] + "\n" + entry + existing[first_nl + 1 :]
        else:
            new_content = entry + existing
    else:
        new_content = f"# Changelog\n\n{entry}"
    if not quiet:
        print(f"CHANGELOG: add entry for {new_version}")
    write_text(changelog, new_content, dry=dry, quiet=quiet)


def git_log_bullets(max_items: int = 10) -> list[str]:
    if not is_git_repo():
        return []
    try:
        cp = subprocess.run(
            ["git", "log", f"-{max_items}", "--pretty=format:%s (%h)"],
            check=True,
            capture_output=True,
        )
        lines = cp.stdout.decode().splitlines()
        # Filter out merge commits noise
        return [ln for ln in lines if not ln.lower().startswith("merge ")]
    except Exception:
        return []


# ------------------------------ Git Ops ------------------------------


def git_dirty() -> bool:
    try:
        cp = subprocess.run(["git", "status", "--porcelain"], check=True, capture_output=True)
        return bool(cp.stdout.strip())
    except Exception:
        return False


def git_commit(msg: str, dry: bool = False, quiet: bool = False) -> None:
    # Stage VERSION, pyproject, CHANGELOG if present
    paths = []
    for p in (DEFAULT_VERSION_FILE, DEFAULT_PYPROJECT, DEFAULT_CHANGELOG):
        if p.exists():
            paths.append(str(p.relative_to(git_root())))
    if not quiet:
        print(f"git add {' '.join(paths) if paths else '(nothing)'}")
    if not dry and paths:
        subprocess.run(["git", "add", *paths], check=True)
    run(["git", "commit", "-m", msg], dry=dry, quiet=quiet)


def git_tag(tag: str, message: str, dry: bool = False, quiet: bool = False) -> None:
    run(["git", "tag", "-a", tag, "-m", message], dry=dry, quiet=quiet)


def git_push(with_tags: bool, dry: bool = False, quiet: bool = False) -> None:
    run(["git", "push"], dry=dry, quiet=quiet)
    if with_tags:
        run(["git", "push", "--tags"], dry=dry, quiet=quiet)


def latest_git_tag_version() -> Optional[str]:
    if not is_git_repo():
        return None
    try:
        cp = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], check=True, capture_output=True)
        tag = cp.stdout.decode().strip()
        m = TAG_RE.match(tag)
        if not m:
            return None
        return m.group("ver")
    except Exception:
        return None


# ------------------------------ Core Logic ------------------------------


def load_version_from_files(version_file: Path) -> Version:
    if not version_file.exists():
        raise FileNotFoundError(f"VERSION file not found at {version_file}")
    raw = read_text(version_file).strip()
    return Version.parse(raw)


def write_version_file(version_file: Path, v: Version, dry: bool = False, quiet: bool = False) -> None:
    if not quiet:
        print(f"VERSION: set {v}")
    write_text(version_file, f"{v}\n", dry=dry, quiet=quiet)


def validate(pyproject: Path, version_file: Path) -> bool:
    ok = True
    v_file = read_text(version_file).strip() if version_file.exists() else None
    v_py = read_pyproject_version(pyproject)
    if v_file and v_py and v_file != v_py:
        print(f"[ERROR] Mismatch: VERSION={v_file}, pyproject.toml={v_py}", file=sys.stderr)
        ok = False
    elif v_file and v_py:
        print(f"[OK] VERSION matches pyproject.toml: {v_file}")
    elif v_file and not v_py:
        print(f"[WARN] pyproject.toml missing version; VERSION={v_file}")
    elif v_py and not v_file:
        print(f"[WARN] VERSION file missing; pyproject.toml={v_py}")
    else:
        print("[WARN] Neither VERSION nor pyproject.toml version found.")
    return ok


def ensure_cli_banner() -> None:
    """Optional: Call spectramind --version to surface the banner (no parsing)."""
    for cmd in (["spectramind", "--version"], [sys.executable, "-m", "spectramind", "--version"]):
        try:
            run(cmd, dry=False, quiet=False)
            return
        except Exception:
            continue
    print("[WARN] SpectraMind CLI not found to show --version banner.")


# ------------------------------ CLI ------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="SpectraMind V50 version manager (VERSION + pyproject.toml + git + changelog).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g_read = p.add_argument_group("Read / Validate")
    g_read.add_argument("--get", action="store_true", help="Print the current version and exit.")
    g_read.add_argument("--validate", action="store_true", help="Verify pyproject.toml version matches VERSION.")

    g_src = p.add_argument_group("Source Selection")
    g_src.add_argument("--version-file", type=Path, default=DEFAULT_VERSION_FILE, help="Path to VERSION file.")
    g_src.add_argument("--pyproject", type=Path, default=DEFAULT_PYPROJECT, help="Path to pyproject.toml.")
    g_src.add_argument("--from-git", action="store_true", help="Use latest git tag (vX.Y.Z...) as version source.")

    g_set = p.add_argument_group("Set / Bump")
    g_set.add_argument("--set", dest="set_version", type=str, help="Set an explicit version (e.g., 0.3.0).")
    g_set.add_argument("--bump", choices=["major", "minor", "patch"], help="Bump version component.")
    g_set.add_argument("--pre", choices=["alpha", "beta", "rc"], help="Set or increment pre-release tag.")
    g_set.add_argument("--final", action="store_true", help="Strip pre-release (finalize).")

    g_sync = p.add_argument_group("Sync Targets")
    g_sync.add_argument("--write-version-file", action="store_true", help="Write VERSION file.")
    g_sync.add_argument("--write-pyproject", action="store_true", help="Write pyproject.toml (tool.poetry.version).")
    g_sync.add_argument("--write-changelog", action="store_true", help="Insert a CHANGELOG entry for the new version.")
    g_sync.add_argument("--changelog", type=Path, default=DEFAULT_CHANGELOG, help="Path to CHANGELOG.md.")

    g_git = p.add_argument_group("Git Actions")
    g_git.add_argument("--commit", action="store_true", help="Create a version bump commit.")
    g_git.add_argument("--tag", action="store_true", help="Create an annotated tag vX.Y.Z.")
    g_git.add_argument("--push", action="store_true", help="Push branch and tags.")
    g_git.add_argument("--commit-message", type=str, default=None, help='Custom commit message (default uses "chore(release): vX.Y.Z").')

    g_misc = p.add_argument_group("Execution Controls")
    g_misc.add_argument("--dry-run", action="store_true", help="Preview changes and commands without applying.")
    g_misc.add_argument("--quiet", action="store_true", help="Reduce logging.")
    g_misc.add_argument("--show-cli", action="store_true", help="Invoke `spectramind --version` after updates.")

    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    # Determine current version (source)
    if args.from_git:
        ver_str = latest_git_tag_version()
        if not ver_str:
            print("[ERROR] Could not derive version from latest git tag.", file=sys.stderr)
            return 1
        try:
            current = Version.parse(ver_str)
        except Exception as e:
            print(f"[ERROR] Invalid git tag version: {ver_str!r} ({e})", file=sys.stderr)
            return 1
    else:
        try:
            current = load_version_from_files(args.version_file)
        except Exception as e:
            print(f"[ERROR] {e}", file=sys.stderr)
            return 1

    # Early actions
    if args.get and not (args.set_version or args.bump or args.pre or args.final):
        print(str(current))
        return 0

    if args.validate and not (args.set_version or args.bump or args.pre or args.final):
        ok = validate(args.pyproject, args.version_file)
        return 0 if ok else 1

    new = current

    # --- Set explicit version ---
    if args.set_version:
        try:
            new = Version.parse(args.set_version)
        except Exception as e:
            print(f"[ERROR] --set invalid: {e}", file=sys.stderr)
            return 1

    # --- Bump operations ---
    if args.bump:
        try:
            new = new.bumped(args.bump, pre=args.pre)
        except Exception as e:
            print(f"[ERROR] --bump failed: {e}", file=sys.stderr)
            return 1
    else:
        # Only pre handling (increment or set), if requested without bump
        if args.pre:
            try:
                new = new.bumped_pre(args.pre)
            except Exception as e:
                print(f"[ERROR] --pre failed: {e}", file=sys.stderr)
                return 1

    # --- Finalize (strip pre-release) ---
    if args.final:
        new = new.without_pre()

    # If no change requested (accidental call), do nothing
    if str(new) == str(current) and not (args.write_pyproject or args.write_version_file or args.write_changelog):
        print(f"[INFO] Version unchanged: {new}")
        if args.show_cli:
            ensure_cli_banner()
        return 0

    # --- Apply changes to files ---
    if args.write_version_file or (args.set_version or args.bump or args.pre or args.final):
        write_version_file(args.version_file, new, dry=args.dry_run, quiet=args.quiet)

    if args.write_pyproject:
        try:
            write_pyproject_version(args.pyproject, str(new), dry=args.dry_run, quiet=args.quiet)
        except Exception as e:
            print(f"[ERROR] Failed to write pyproject.toml: {e}", file=sys.stderr)
            return 1

    if args.write_changelog:
        try:
            update_changelog(args.changelog, str(new), dry=args.dry_run, quiet=args.quiet)
        except Exception as e:
            print(f"[ERROR] Failed to update CHANGELOG: {e}", file=sys.stderr)
            return 1

    # --- Git workflow ---
    if any([args.commit, args.tag, args.push]):
        if not is_git_repo():
            print("[ERROR] Not a git repository; cannot perform git actions.", file=sys.stderr)
            return 1

        if args.commit:
            msg = args.commit_message or f"chore(release): v{new}"
            if args.dry_run:
                print(f"[dry-run] git commit -m {msg!r}")
            else:
                # commit even with a clean tree so the version bump is captured (we staged above)
                git_commit(msg, dry=args.dry_run, quiet=args.quiet)

        if args.tag:
            tag = f"v{new}"
            if args.dry_run:
                print(f"[dry-run] git tag -a {tag} -m {tag}")
            else:
                git_tag(tag, tag, dry=args.dry_run, quiet=args.quiet)

        if args.push:
            git_push(with_tags=args.tag, dry=args.dry_run, quiet=args.quiet)

    if not args.quiet:
        print(f"[OK] New version: {new}")

    if args.show_cli:
        ensure_cli_banner()

    return 0


if __name__ == "__main__":
    sys.exit(main())