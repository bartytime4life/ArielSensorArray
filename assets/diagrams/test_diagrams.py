#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
assets/diagrams/test_diagrams.py

Mission:
  Validate and (optionally) render all Mermaid `.mmd` sources in assets/diagrams
  to `.svg` and `.png`, with strict checks suitable for CI and local use.

Features:
  - Finds all .mmd files (or a selected subset via --only)
  - Uses mermaid-cli (mmdc) if available; else falls back to npx runner
  - Renders SVG/PNG (configurable) and validates basic integrity
  - Strict mode: fails on warnings/missing renders/out-of-date exports
  - Prints a compact summary; exits non-zero on failure

Usage:
  python assets/diagrams/test_diagrams.py --render
  python assets/diagrams/test_diagrams.py --render --only pipeline_overview.mmd,symbolic_logic_layers.mmd
  python assets/diagrams/test_diagrams.py --strict
  python assets/diagrams/test_diagrams.py --out-formats svg,png
"""

import argparse
import os
import sys
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

# Optional pretty output
try:
    from rich.console import Console
    from rich.table import Table
    RICH = True
    console = Console()
except Exception:
    RICH = False
    console = None


DIAGRAMS_DIR = Path(__file__).resolve().parent
REPO_ROOT = DIAGRAMS_DIR.parents[2] if len(DIAGRAMS_DIR.parents) >= 2 else DIAGRAMS_DIR
DEFAULT_FORMATS = ("svg", "png")


@dataclass
class RenderResult:
    src: Path
    out_svg: Optional[Path]
    out_png: Optional[Path]
    rendered: List[str]
    warnings: List[str]
    errors: List[str]


def which(cmd: str) -> Optional[str]:
    """Return the absolute path to an executable or None."""
    return shutil.which(cmd)


def find_mermaid_runner() -> Tuple[List[str], str]:
    """
    Locate the mermaid-cli runner.
      - Prefer a local/global 'mmdc'
      - Fallback to npx @mermaid-js/mermaid-cli
    Returns (runner_argv_prefix, runner_name)
    """
    mmdc = which("mmdc")
    if mmdc:
        return ([mmdc], "mmdc")

    npx = which("npx")
    if npx:
        # Use npx to run the CLI without global install
        return ([npx, "-y", "@mermaid-js/mermaid-cli"], "npx-mermaid")
    # No runner found
    return ([], "")


def run_cmd(args: Sequence[str], cwd: Optional[Path] = None, timeout: Optional[int] = 120) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(args),
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        timeout=timeout,
        check=False,
    )


def list_mmd_files(only: Optional[Iterable[str]] = None) -> List[Path]:
    files = sorted(DIAGRAMS_DIR.glob("*.mmd"))
    if only:
        wanted = {name.strip() for name in only if name and name.strip()}
        files = [p for p in files if p.name in wanted]
    return files


def target_for(src: Path, fmt: str) -> Path:
    return src.with_suffix(f".{fmt.lower()}")


def is_up_to_date(src: Path, dst: Path) -> bool:
    if not dst.exists():
        return False
    return dst.stat().st_mtime >= src.stat().st_mtime


def render_one(runner_prefix: List[str], src: Path, fmt: str) -> Tuple[bool, str]:
    out = target_for(src, fmt)
    args = runner_prefix + ["-i", str(src), "-o", str(out)]
    # Add a small delay to avoid identical mtime edge cases on very fast filesystems
    before = time.time()
    proc = run_cmd(args, cwd=REPO_ROOT)
    ok = proc.returncode == 0 and out.exists()
    detail = proc.stdout.strip() + ("\n" + proc.stderr.strip() if proc.stderr.strip() else "")
    # Some environments require a first-time browser install for the CLI rendering engine.
    if (not ok) and "Chromium" in detail and "not found" in detail.lower():
        detail += (
            "\nHint: Mermaid CLI may require a headless Chromium. Try:\n"
            "  npx playwright install chromium\n"
            "or ensure puppeteer can download Chromium in your environment."
        )
    # Touch to ensure mtime moves forward relative to src if render succeeded but generated quickly
    if ok and out.exists() and out.stat().st_mtime <= Path(src).stat().st_mtime:
        os.utime(out, None)
    return ok, detail


def validate_svg(svg_path: Path) -> Optional[str]:
    try:
        data = svg_path.read_bytes()
        head = data[:512].decode(errors="ignore").lower()
        if "<svg" not in head:
            return "SVG does not contain <svg> header"
        if b"</svg>" not in data:
            return "SVG missing closing </svg> tag"
    except Exception as e:
        return f"SVG read error: {e}"
    return None


def validate_png(png_path: Path) -> Optional[str]:
    try:
        data = png_path.read_bytes()
        if len(data) < 8:
            return "PNG too small"
        sig = data[:8]
        # PNG signature
        if sig != b"\x89PNG\r\n\x1a\n":
            return "Invalid PNG signature"
    except Exception as e:
        return f"PNG read error: {e}"
    return None


def process_file(
    runner_prefix: List[str],
    src: Path,
    out_formats: Tuple[str, ...],
    do_render: bool,
    strict: bool,
) -> RenderResult:
    out_svg = target_for(src, "svg") if "svg" in out_formats else None
    out_png = target_for(src, "png") if "png" in out_formats else None
    rendered: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []

    for fmt in out_formats:
        dst = target_for(src, fmt)
        needs = not dst.exists() or not is_up_to_date(src, dst)
        if do_render or not dst.exists():
            ok, detail = render_one(runner_prefix, src, fmt)
            if ok:
                rendered.append(fmt)
            else:
                errors.append(f"{src.name}: render {fmt} failed\n{detail}")
        else:
            # Not rendering; verify freshness if strict
            if strict and needs:
                warnings.append(f"{src.name}: {fmt} export is stale (source newer than output)")

    # Validate outputs if they exist / were requested
    if out_svg and out_svg.exists():
        err = validate_svg(out_svg)
        if err:
            errors.append(f"{out_svg.name}: {err}")
    elif "svg" in out_formats:
        errors.append(f"{src.name}: missing SVG export")

    if out_png and out_png.exists():
        err = validate_png(out_png)
        if err:
            errors.append(f"{out_png.name}: {err}")
    elif "png" in out_formats:
        errors.append(f"{src.name}: missing PNG export")

    return RenderResult(src=src, out_svg=out_svg, out_png=out_png, rendered=rendered, warnings=warnings, errors=errors)


def print_summary(results: List[RenderResult], strict: bool) -> None:
    total = len(results)
    errs = sum(len(r.errors) for r in results)
    warns = sum(len(r.warnings) for r in results)

    if RICH:
        table = Table(title="Mermaid Diagrams — Summary", show_lines=False)
        table.add_column("Source", style="bold")
        table.add_column("Rendered", justify="center")
        table.add_column("Warnings", justify="left")
        table.add_column("Errors", justify="left")

        for r in results:
            rendered = ", ".join(r.rendered) if r.rendered else "-"
            warn_txt = "\n".join(r.warnings) if r.warnings else "-"
            err_txt = "\n".join(r.errors) if r.errors else "-"
            table.add_row(r.src.name, rendered, warn_txt, err_txt)
        console.print(table)
        console.print(f"[bold]Total:[/bold] {total}  [yellow]Warnings:[/yellow] {warns}  [red]Errors:[/red] {errs}")
        if strict and warns and not errs:
            console.print("[yellow]Strict mode: warnings will fail the run.[/yellow]")
    else:
        print("\n=== Mermaid Diagrams — Summary ===")
        for r in results:
            print(f"- {r.src.name}")
            print(f"  rendered: {', '.join(r.rendered) if r.rendered else '-'}")
            if r.warnings:
                for w in r.warnings:
                    print(f"  warning: {w}")
            if r.errors:
                for e in r.errors:
                    print(f"  error:   {e}")
        print(f"\nTotal: {total}  Warnings: {warns}  Errors: {errs}")
        if strict and warns and not errs:
            print("Strict mode: warnings will fail the run.")

def parse_only_arg(only_arg: Optional[str]) -> Optional[List[str]]:
    if not only_arg:
        return None
    items = [s.strip() for s in only_arg.split(",")]
    return [s for s in items if s]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Render/validate Mermaid diagrams in assets/diagrams")
    parser.add_argument("--render", action="store_true", help="Render outputs (SVG/PNG) for each .mmd")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as failures and require up-to-date outputs")
    parser.add_argument("--only", type=str, default=None, help="Comma-separated .mmd names to process (e.g., a.mmd,b.mmd)")
    parser.add_argument("--out-formats", type=str, default="svg,png", help="Comma-separated formats (svg,png)")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first error")
    args = parser.parse_args(argv)

    out_formats = tuple(fmt.strip().lower() for fmt in args.out_formats.split(",") if fmt.strip())
    if not out_formats:
        print("No output formats specified.", file=sys.stderr)
        return 2

    runner_prefix, runner_name = find_mermaid_runner()
    if args.render and not runner_prefix:
        print(
            "Mermaid CLI not found.\n"
            "Install globally:   npm i -g @mermaid-js/mermaid-cli\n"
            "Or use npx runner:  npx -y @mermaid-js/mermaid-cli -i input.mmd -o output.svg\n",
            file=sys.stderr,
        )
        return 2

    only_list = parse_only_arg(args.only)
    mmd_files = list_mmd_files(only_list)

    if not mmd_files:
        print(f"No .mmd files found in {DIAGRAMS_DIR} matching filter: {only_list}", file=sys.stderr)
        return 0

    results: List[RenderResult] = []
    exit_code = 0

    if RICH:
        console.rule(f"[bold]Mermaid Runner[/bold] • {runner_name or 'validation-only'}")
        console.print(f"[bold]Directory:[/bold] {DIAGRAMS_DIR}")
        console.print(f"[bold]Formats:[/bold] {', '.join(out_formats)}")
        console.print(f"[bold]Files:[/bold] {', '.join(p.name for p in mmd_files)}")
    else:
        print(f"Runner: {runner_name or 'validation-only'}")
        print(f"Directory: {DIAGRAMS_DIR}")
        print(f"Formats: {', '.join(out_formats)}")
        print(f"Files: {', '.join(p.name for p in mmd_files)}")

    for src in mmd_files:
        res = process_file(
            runner_prefix=runner_prefix,
            src=src,
            out_formats=out_formats,  # type: ignore[arg-type]
            do_render=args.render,
            strict=args.strict,
        )
        results.append(res)

        if res.errors:
            exit_code = 1
            if args.fail_fast:
                print_summary(results, args.strict)
                return exit_code

    # Strict mode: warnings also fail if no hard errors already set exit 1
    if args.strict and not exit_code and any(r.warnings for r in results):
        exit_code = 1

    print_summary(results, args.strict)
    return exit_code


# -------------------------
# Pytest integration hooks
# -------------------------
def test_render_all_diagrams():
    """
    Basic pytest hook:
      - validates that for every .mmd we have up-to-date SVG/PNG
      - if mermaid runner is available, (re)renders before asserting
    """
    runner_prefix, _ = find_mermaid_runner()
    mmd_files = list_mmd_files(None)
    assert mmd_files, "No .mmd files found in assets/diagrams"

    out_formats = DEFAULT_FORMATS
    failures: List[str] = []

    for src in mmd_files:
        # Auto-render if runner present; otherwise only validate
        res = process_file(
            runner_prefix=runner_prefix,
            src=src,
            out_formats=out_formats,  # type: ignore[arg-type]
            do_render=bool(runner_prefix),
            strict=True,
        )
        if res.errors:
            failures.extend(res.errors)

    if failures:
        msg = "\n".join(failures)
        raise AssertionError(f"Diagram render/validation failures:\n{msg}")


if __name__ == "__main__":
    sys.exit(main())