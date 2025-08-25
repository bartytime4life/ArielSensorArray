#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tools/validate_assets_manifest.py

SpectraMind V50 — Assets Manifest Validator

Validates /assets/assets-manifest.json:
  • structure & required fields
  • file existence
  • sha256 integrity (matches manifest)
  • allowed asset types
  • optional autofix of placeholder hashes

Exit codes
  0 = all good
  1 = structural error (invalid manifest format)
  2 = missing files
  3 = hash mismatches
  4 = unknown/invalid asset types
  5 = untracked files (if --fail-on-untracked)
  9 = multiple error classes (bitwise OR-like aggregation)

Usage examples
  python tools/validate_assets_manifest.py
  python tools/validate_assets_manifest.py --root assets
  python tools/validate_assets_manifest.py --fix-placeholders
  python tools/validate_assets_manifest.py --update-all-hashes
  python tools/validate_assets_manifest.py --list-untracked
  python tools/validate_assets_manifest.py --fail-on-untracked
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Any, Tuple, List, Set


ALLOWED_TYPES: Set[str] = {
    "theme", "style", "image", "doc", "diagram", "html", "notebook"
}

DEFAULT_ROOT = Path("assets")
DEFAULT_MANIFEST = DEFAULT_ROOT / "assets-manifest.json"

PLACEHOLDER_PREFIX = "sha256-PLACEHOLDER_HASH"


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return f"sha256-{h.hexdigest()}"


def load_manifest(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise SystemExit(f"[ERROR] Manifest not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] Invalid JSON in manifest {path}: {e}")


def iter_assets(manifest: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    assets = manifest.get("assets", {})
    for group_name, group in assets.items():
        if not isinstance(group, dict):
            continue
        for logical_name, entry in group.items():
            yield group_name, {**entry, "_logical_name": logical_name}


def validate_structure(manifest: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    if "meta" not in manifest or "assets" not in manifest:
        errors.append("Manifest must contain 'meta' and 'assets' keys.")
        return errors

    # Minimal meta checks
    meta = manifest["meta"]
    for k in ("project", "component", "version"):
        if k not in meta:
            errors.append(f"meta.{k} missing")

    # Asset entries checks
    for group, entry in iter_assets(manifest):
        missing = [k for k in ("path", "type", "hash", "version") if k not in entry]
        if missing:
            errors.append(
                f"[{group}/{entry.get('_logical_name')}] missing fields: {', '.join(missing)}"
            )
        # type validation deferred to later

    return errors


def collect_manifest_paths(manifest: Dict[str, Any]) -> Dict[Path, Dict[str, Any]]:
    out: Dict[Path, Dict[str, Any]] = {}
    for _, entry in iter_assets(manifest):
        p = Path(entry["path"])
        out[p] = entry
    return out


def walk_asset_tree(root: Path) -> List[Path]:
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file():
            # ignore common non-source junk files
            if p.name.startswith(".DS_Store") or p.name.endswith(".map"):
                continue
            files.append(p.relative_to(root))
    return files


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate the assets-manifest.json integrity.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Assets root directory (default: ./assets)")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST, help="Manifest path (default: ./assets/assets-manifest.json)")
    ap.add_argument("--fix-placeholders", action="store_true", help="Replace placeholder hashes with computed sha256 values.")
    ap.add_argument("--update-all-hashes", action="store_true", help="Recompute and write ALL hashes (use with care in CI).")
    ap.add_argument("--list-untracked", action="store_true", help="List files under --root that are not referenced by manifest.")
    ap.add_argument("--fail-on-untracked", action="store_true", help="Return non-zero if untracked files are found.")
    args = ap.parse_args()

    root: Path = args.root
    manifest_path: Path = args.manifest

    manifest = load_manifest(manifest_path)
    structure_errors = validate_structure(manifest)
    if structure_errors:
        print("\n[STRUCTURE ERRORS]")
        for e in structure_errors:
            print(" -", e)
        # Don't bail out yet; continue to find more issues, but mark code
        exit_code = 1
    else:
        exit_code = 0

    manifest_entries = collect_manifest_paths(manifest)
    all_files = walk_asset_tree(root)

    # 1) Unknown/invalid types
    type_errors: List[str] = []
    for rel, entry in manifest_entries.items():
        t = entry.get("type")
        if t not in ALLOWED_TYPES:
            type_errors.append(f"{rel} has invalid type '{t}'. Allowed: {sorted(ALLOWED_TYPES)}")
    if type_errors:
        print("\n[TYPE ERRORS]")
        for e in type_errors:
            print(" -", e)
        exit_code = exit_code or 4

    # 2) Missing files
    missing: List[Path] = [rel for rel in manifest_entries.keys() if not (root / rel).exists()]
    if missing:
        print("\n[MISSING FILES]")
        for rel in missing:
            print(" -", rel)
        exit_code = exit_code or 2

    # 3) Hash mismatches & placeholder fixes
    mismatches: List[Tuple[Path, str, str]] = []  # rel, expected, actual
    updated = 0
    for rel, entry in manifest_entries.items():
        p = root / rel
        if not p.exists():
            continue
        actual = sha256_file(p)
        expected = entry.get("hash", "")
        is_placeholder = isinstance(expected, str) and expected.startswith(PLACEHOLDER_PREFIX)

        if args.update_all_hashes:
            entry["hash"] = actual
            updated += 1
            continue

        if is_placeholder and args.fix_placeholders:
            entry["hash"] = actual
            updated += 1
            continue

        if expected != actual:
            mismatches.append((rel, expected, actual))

    if mismatches:
        print("\n[HASH MISMATCHES]")
        for rel, exp, act in mismatches:
            print(f" - {rel}: expected {exp}, actual {act}")
        exit_code = exit_code or 3

    # 4) Untracked files
    manifest_paths_set = set(manifest_entries.keys())
    untracked: List[Path] = [f for f in all_files if f not in manifest_paths_set and f.name != manifest_path.name]
    if untracked and (args.list_untracked or args.fail_on_untracked):
        print("\n[UNTRACKED FILES]")
        for rel in untracked:
            print(" -", rel)
        if args.fail_on_untracked:
            exit_code = exit_code or 5

    # Write back manifest if modified
    if updated > 0:
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        print(f"\n[UPDATED] Wrote {updated} hash value(s) to {manifest_path}")

    # Summary
    print("\n=== Summary ===")
    print(f"Root:            {root.resolve()}")
    print(f"Manifest:        {manifest_path.resolve()}")
    print(f"Files tracked:   {len(manifest_entries)}")
    print(f"Missing:         {len(missing)}")
    print(f"Mismatches:      {len(mismatches)}")
    if args.list_untracked or args.fail_on_untracked:
        print(f"Untracked:       {len(untracked)}")
    print(f"Exit code:       {exit_code}")

    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()