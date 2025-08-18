#!/usr/bin/env python3
"""
SpectraMind V50 — submission validator

Usage:
  scripts/validate_submission.py outputs/submission.csv
  cat outputs/submission.csv | scripts/validate_submission.py -
  scripts/validate_submission.py --ids data/expected_ids.txt outputs/submission.csv
  scripts/validate_submission.py --bins 283 outputs/submission.csv
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import math
import sys
from pathlib import Path
from typing import Iterable, Set, TextIO


def fail(msg: str, code: int = 2) -> "NoReturn":
    print(f"[submission:invalid] {msg}", file=sys.stderr)
    raise SystemExit(code)


def ok(msg: str) -> None:
    print(f"[submission:ok] {msg}")


def open_maybe_gzip(path: str) -> TextIO:
    if path == "-":
        # read text from stdin with universal newlines
        return io.TextIOWrapper(sys.stdin.buffer, encoding="utf-8", newline="")
    p = Path(path)
    if not p.exists():
        fail(f"file not found: {p}")
    if p.suffix == ".gz":
        return io.TextIOWrapper(gzip.open(p, "rb"), encoding="utf-8", newline="")
    return p.open("r", encoding="utf-8", newline="")


def load_expected_ids(path: Path) -> Set[str]:
    if not path.exists():
        fail(f"--ids file not found: {path}")
    ids: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                ids.add(s)
    if not ids:
        fail(f"--ids file is empty: {path}")
    return ids


def validate_header(header: list[str], bins: int) -> list[str]:
    if not header:
        fail("empty CSV (no header row found)")
    if header[0] != "planet_id":
        fail('first column must be "planet_id"')
    expected = ["planet_id"] + [f"bin{i}" for i in range(bins)]
    if header != expected:
        fail(
            f"header mismatch; expected {len(expected)} columns "
            f"(planet_id + bin0..bin{bins-1}), got {len(header)}"
        )
    return expected


def validate_value(val: str, row_idx: int, col_idx: int) -> None:
    # disallow leading/trailing whitespace (common source of parse surprises)
    if val != val.strip():
        fail(f"row {row_idx} col {col_idx} has surrounding whitespace: {val!r}")
    if val == "":
        fail(f"row {row_idx} col {col_idx} is empty")
    try:
        x = float(val)
    except ValueError:
        fail(f"row {row_idx} col {col_idx} not numeric: {val!r}")
    if not math.isfinite(x):
        fail(f"row {row_idx} col {col_idx} is not finite (NaN/Inf): {val!r}")


def validate_stream(
    fh: TextIO,
    bins: int,
    expected_ids: Set[str] | None = None,
    quiet: bool = False,
) -> None:
    reader = csv.reader(fh)
    try:
        header = next(reader)
    except StopIteration:
        fail("empty CSV")

    expected_header = validate_header(header, bins)

    seen_ids: Set[str] = set()
    n_rows = 0

    for row_idx, row in enumerate(reader, start=1):
        # skip pure empty lines (csv module usually doesn’t yield them, but be safe)
        if not row:
            fail(f"row {row_idx} is empty")
        if len(row) != len(expected_header):
            fail(f"row {row_idx} has {len(row)} columns (expected {len(expected_header)})")

        pid = row[0]
        if pid == "" or pid != pid.strip():
            fail(f"row {row_idx} planet_id is empty or has whitespace: {pid!r}")
        if pid in seen_ids:
            fail(f"duplicate planet_id at row {row_idx}: {pid!r}")
        seen_ids.add(pid)

        # validate all numeric cells (1..N)
        for col_idx, val in enumerate(row[1:], start=1):
            validate_value(val, row_idx, col_idx)

        n_rows += 1

    if expected_ids is not None:
        missing = expected_ids - seen_ids
        extra = seen_ids - expected_ids
        if missing:
            fail(f"missing {len(missing)} planet_id(s) vs --ids (e.g., {sorted(list(missing))[:5]})")
        if extra:
            fail(f"unexpected {len(extra)} planet_id(s) not in --ids (e.g., {sorted(list(extra))[:5]})")

    if not quiet:
        ok(f"{n_rows} predictions, {len(expected_header)-1} bins each")


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Validate Ariel challenge submission CSV (planet_id, bin0..binN)."
    )
    ap.add_argument("csv", help='submission CSV path (use "-" for stdin; ".gz" supported)')
    ap.add_argument("--bins", type=int, default=283, help="number of spectral bins (default: 283)")
    ap.add_argument(
        "--ids",
        type=Path,
        help="path to text file of expected planet_id (one per line) to enforce exact ID set",
    )
    ap.add_argument("--quiet", action="store_true", help="suppress OK detail output")
    return ap.parse_args(list(argv))


def main(argv: Iterable[str]) -> None:
    args = parse_args(argv)
    expected_ids = load_expected_ids(args.ids) if args.ids else None
    with open_maybe_gzip(args.csv) as fh:
        validate_stream(fh, bins=args.bins, expected_ids=expected_ids, quiet=args.quiet)


if __name__ == "__main__":
    try:
        main(sys.argv[1:])
    except SystemExit:
        raise
    except Exception as e:
        fail(f"unhandled error: {e!r}")