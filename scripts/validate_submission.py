#!/usr/bin/env python3
import csv
import pathlib
import sys


def fail(msg: str, code: int = 2):
    print(f"[submission:invalid] {msg}")
    raise SystemExit(code)


def main(path: str):
    p = pathlib.Path(path)
    if not p.exists():
        fail(f"file not found: {p}")

    with p.open(newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        fail("empty CSV")

    header = rows[0]
    if header[0] != "planet_id":
        fail('first column must be "planet_id"')

    expected_bins = 283
    expected_header = ["planet_id"] + [f"bin{i}" for i in range(expected_bins)]
    if header != expected_header:
        fail(
            f"header mismatch; expected {len(expected_header)} columns "
            f"(planet_id + bin0..bin{expected_bins-1}), got {len(header)}"
        )

    for i, row in enumerate(rows[1:], start=1):
        if len(row) != len(expected_header):
            fail(f"row {i} has {len(row)} columns (expected {len(expected_header)})")
        for j, val in enumerate(row[1:], start=1):
            try:
                float(val)
            except ValueError:
                fail(f"row {i} col {j} not numeric: {val!r}")

    print(f"[submission:ok] {p} looks good with {len(rows)-1} predictions.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: validate_submission.py outputs/submission.csv")
        raise SystemExit(2)
    main(sys.argv[1])
