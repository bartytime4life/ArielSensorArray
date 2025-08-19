#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
validate_env.py — SpectraMind V50
Mission‑grade validator for `.env` against `.env.schema.json`

Overview
--------
This CLI:
  1) Loads a `.env` file (default: ./.env)
  2) Loads a JSON Schema (default: ./ .env.schema.json)
  3) Parses and *type‑coerces* environment values according to schema
  4) Applies defaults (unless disabled) and optional strict unknown‑key checks
  5) Validates with Draft 2020‑12 JSON Schema
  6) Emits a helpful report and appropriate exit code

Exit codes
----------
  0 = success (valid)
  1 = validation errors (schema violations, missing/invalid values)
  2 = runtime errors (I/O, schema parse error, missing deps, etc.)

Usage
-----
  python validate_env.py
  python validate_env.py --env-file .env --schema-file .env.schema.json
  python validate_env.py --apply-defaults --strict-unknown-keys --output outputs/env_resolved.json

Notes
-----
- No network access required.
- Only dependency needed is `jsonschema` (install via: `poetry add jsonschema` or `pip install jsonschema`).
- This validator is *schema-driven* and robust to comment/blank lines in `.env`.
- It provides rich, human-friendly error output for CI and local dev.

Author
------
SpectraMind V50 Engineering — NeurIPS 2025 Ariel Data Challenge
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Optional import guard for jsonschema
try:
    import jsonschema
    from jsonschema import Draft202012Validator  # Explicit to lock draft used
except Exception as e:  # pragma: no cover
    print(
        "[FATAL] Missing dependency: jsonschema\n"
        "Install with one of:\n"
        "  • poetry add jsonschema\n"
        "  • pip install jsonschema\n",
        file=sys.stderr,
    )
    sys.exit(2)


# --------------------------------------------------------------------------------------
# Lightweight .env parser (no external deps). Supports:
#   - Comment lines starting with '#'
#   - KEY=VAL (leading/trailing whitespace trimmed)
#   - Quoted values ("value" or 'value') -> quotes stripped
#   - Preserves blank keys with empty values (for nullable handling)
# --------------------------------------------------------------------------------------
_ENV_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_\-]*)\s*=\s*(.*)\s*$")


def parse_dotenv(env_text: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    for ln in env_text.splitlines():
        line = ln.strip()
        if not line or line.startswith("#"):
            continue
        m = _ENV_LINE.match(line)
        if not m:
            # tolerate lines like `export KEY=VAL` (strip 'export ')
            if line.startswith("export "):
                maybe = line[len("export ") :]
                m2 = _ENV_LINE.match(maybe)
                if not m2:
                    # ignore malformed lines but warn
                    print(f"[WARN] Ignoring malformed env line: {ln}", file=sys.stderr)
                    continue
                key, value = m2.group(1), m2.group(2)
            else:
                print(f"[WARN] Ignoring malformed env line: {ln}", file=sys.stderr)
                continue
        else:
            key, value = m.group(1), m.group(2)

        # Strip surrounding quotes if present
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]

        env[key] = value
    return env


# --------------------------------------------------------------------------------------
# Types & helpers
# --------------------------------------------------------------------------------------
Json = Dict[str, Any]


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]
    resolved: Json


def _load_json(path: Path) -> Json:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"[FATAL] Schema file not found: {path}", file=sys.stderr)
        sys.exit(2)
    except json.JSONDecodeError as e:
        print(f"[FATAL] Failed to parse JSON schema: {path}\n{e}", file=sys.stderr)
        sys.exit(2)


def _load_env_file(path: Path) -> Dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"[FATAL] .env file not found: {path}", file=sys.stderr)
        sys.exit(2)
    return parse_dotenv(text)


def _coerce_value(key: str, raw: str, prop_schema: Json) -> Any:
    """
    Coerces a raw .env string to the type expected by the schema for top-level properties only.
    Rules:
      - If schema type == "integer" -> int
      - If schema type == "number" -> float
      - If schema type == "string"  -> keep as string (strip)
      - Nullable strings: if raw is empty => None
      - For booleans represented as strings in schema (enum ["true","false"]), keep as string (lowercased)
      - URIs remain strings; validation handled by jsonschema format check
    """
    # Handle nullable strings (common in our schema)
    if prop_schema.get("type") == "string":
        if prop_schema.get("nullable") and raw == "":
            return None
        # Preserve as string; for typical bool-like string fields, normalize canonical form
        enum_vals = prop_schema.get("enum")
        if isinstance(enum_vals, list) and all(isinstance(v, str) for v in enum_vals):
            lc = raw.strip().lower()
            # if enum contains 'true'/'false', normalize case
            if {"true", "false"}.issubset(set(map(str.lower, enum_vals))):
                if lc in {"true", "false"}:
                    return lc
        return raw

    # Numeric coercion
    if prop_schema.get("type") == "integer":
        if raw == "":
            # Let the schema handle empty vs required
            return raw
        try:
            return int(raw, 10)
        except ValueError:
            return raw  # Let schema emit error with better message

    if prop_schema.get("type") == "number":
        if raw == "":
            return raw
        try:
            return float(raw)
        except ValueError:
            return raw

    # Fallback: leave as-is
    return raw


def _apply_defaults(resolved: Json, schema: Json) -> None:
    """
    Apply top‑level defaults from schema.properties when property is missing.
    JSON Schema does not automatically apply defaults; this is a pragmatic top‑level merge.
    """
    props = schema.get("properties", {})
    for key, prop_schema in props.items():
        if key not in resolved and "default" in prop_schema:
            resolved[key] = prop_schema["default"]


def _strict_unknown_keys(env_dict: Dict[str, Any], schema: Json) -> List[str]:
    props = schema.get("properties", {})
    unknown = [k for k in env_dict.keys() if k not in props]
    return unknown


def _build_resolved_config(
    env_map: Dict[str, str],
    schema: Json,
    apply_defaults: bool,
    coerce: bool,
) -> Json:
    """
    Construct the resolved config:
      - Optionally apply defaults from schema
      - Optionally coerce values to expected JSON types
    """
    resolved: Json = {}
    if apply_defaults:
        _apply_defaults(resolved, schema)

    props = schema.get("properties", {})
    for key, prop_schema in props.items():
        if key in env_map:
            raw = env_map[key]
            value = _coerce_value(key, raw, prop_schema) if coerce else raw
            resolved[key] = value
        # else: keep default if present; otherwise missing -> validator will handle
    return resolved


def _format_jsonschema_error(err: jsonschema.ValidationError) -> str:
    """
    Produce a detailed, human-friendly error message from jsonschema.ValidationError
    """
    loc = ".".join(map(str, err.path)) if err.path else "(root)"
    # Underlying cause may carry useful info
    cause = f" | cause: {err.cause}" if getattr(err, "cause", None) else ""
    ctx = ""
    if err.context:
        ctx = " | anyOf/oneOf context:\n    - " + "\n    - ".join([c.message for c in err.context])
    return f"[SCHEMA] at {loc}: {err.message}{cause}{ctx}"


def validate_env(
    env_file: Path,
    schema_file: Path,
    apply_defaults: bool = True,
    strict_unknown_keys: bool = False,
    coerce_types: bool = True,
    output_path: Optional[Path] = None,
) -> ValidationResult:
    """
    Main validation pipeline:
      - Load schema + env
      - Check unknown keys (optional strict)
      - Build resolved config (defaults + type coercion)
      - Validate with Draft 2020‑12
      - Write resolved (optional)
    """
    schema = _load_json(schema_file)
    raw_env = _load_env_file(env_file)

    errors: List[str] = []
    warnings: List[str] = []

    # Unknown keys check (optional)
    unknown = _strict_unknown_keys(raw_env, schema)
    if unknown:
        msg = f"[UNKNOWN] Keys in .env but not in schema: {', '.join(sorted(unknown))}"
        if strict_unknown_keys:
            errors.append(msg)
        else:
            warnings.append(msg)

    # Build resolved config
    resolved = _build_resolved_config(raw_env, schema, apply_defaults=apply_defaults, coerce=coerce_types)

    # Validate against schema
    validator = Draft202012Validator(schema)
    all_errors = sorted(validator.iter_errors(resolved), key=lambda e: e.path)
    if all_errors:
        for e in all_errors:
            errors.append(_format_jsonschema_error(e))

    # Optionally write the resolved JSON (useful for downstream tools/CI artifacts)
    if output_path and not errors:
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(resolved, indent=2, sort_keys=True), encoding="utf-8")
        except Exception as e:
            errors.append(f"[I/O] Failed to write resolved JSON to {output_path}: {e}")

    ok = len(errors) == 0
    return ValidationResult(ok=ok, errors=errors, warnings=warnings, resolved=resolved)


def _print_report(res: ValidationResult, env_file: Path, schema_file: Path) -> None:
    # Header
    print("\n=== SpectraMind V50 — .env Validation Report ===")
    print(f" • .env file    : {env_file}")
    print(f" • schema       : {schema_file}\n")

    # Warnings
    for w in res.warnings:
        print(f"WARNING: {w}")

    # Errors
    if res.errors:
        print("\nValidation errors:")
        for i, err in enumerate(res.errors, 1):
            print(f"  {i:02d}) {err}")

    # Key summary (a few high-signal keys)
    print("\nKey settings (resolved snapshot):")
    keys_to_show = [
        "PROJECT_NAME",
        "PROJECT_STAGE",
        "CONFIG_FILE",
        "LOGS_DIR",
        "OUT_DIR",
        "KAGGLE_COMPETITION",
        "KAGGLE_SUBMISSION_FILE",
        "DOCKER_IMAGE_NAME",
        "DOCKER_CONTAINER_NAME",
        "SPECTRAMIND_CLI",
        "DIAGNOSTICS_REPORT",
        "ENABLE_RICH_LOGGING",
    ]
    for k in keys_to_show:
        v = res.resolved.get(k, "(missing)")
        print(f"  - {k:24s}: {v}")

    print("\nResult:", "✅ VALID" if res.ok else "❌ INVALID")
    print("==============================================\n")


def _cli(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="validate_env.py",
        description="Validate a SpectraMind V50 .env file against the JSON schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--env-file", type=Path, default=Path(".env"), help="Path to .env file")
    p.add_argument("--schema-file", type=Path, default=Path(".env.schema.json"), help="Path to JSON Schema file")
    p.add_argument(
        "--apply-defaults",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply schema defaults for missing keys before validation",
    )
    p.add_argument(
        "--strict-unknown-keys",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Fail if .env contains keys not present in schema.properties",
    )
    p.add_argument(
        "--no-coerce",
        action="store_true",
        help="Disable type coercion of values from `.env` (treat all as strings)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Write resolved config (with defaults/coercions) to this JSON file when valid",
    )

    args = p.parse_args(argv)

    result = validate_env(
        env_file=args.env_file,
        schema_file=args.schema_file,
        apply_defaults=args.apply_defaults,
        strict_unknown_keys=args.strict_unknown_keys,
        coerce_types=not args.no_coerce,
        output_path=args.output,
    )
    _print_report(result, env_file=args.env_file, schema_file=args.schema_file)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(_cli())