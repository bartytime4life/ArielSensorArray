#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_mermaid.py — Extracts ```mermaid blocks from Markdown files and renders SVG/PNG via mermaid-cli (mmdc).
Outputs go to docs/diagrams/<source-basename>/<slug>-<index>.svg|png

Usage:
  python scripts/export_mermaid.py ARCHITECTURE.md README.md docs/extra.md
Environment:
  MERMAID_CLI=path to mmdc (optional; default 'npx --yes @mermaid-js/mermaid-cli -p puppeteer-config.json -o .')
  EXPORT_PNG=1 to also export PNG alongside SVG
  THEME=dark|default|forest|neutral (overrides .mermaidrc.json theme just for CLI run)
"""
import os
import re
import sys
import json
import shlex
import subprocess
from pathlib import Path
from hashlib import md5
from typing import List, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "docs" / "diagrams"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
PUPPETEER_CFG = ROOT / "puppeteer-config.json"

MERMAID_CLI = os.environ.get(
    "MERMAID_CLI",
    "npx --yes @mermaid-js/mermaid-cli"
)
EXPORT_PNG = os.environ.get("EXPORT_PNG", "0") == "1"
THEME = os.environ.get("THEME", "").strip()

MERMAID_BLOCK_RE = re.compile(
    r"```mermaid\s*\n(.*?)\n```",
    re.DOTALL | re.IGNORECASE
)

SLUG_RE = re.compile(r"[^a-z0-9]+", re.IGNORECASE)


def slugify(text: str, max_len: int = 60) -> str:
    text = text.strip().lower()
    text = SLUG_RE.sub("-", text).strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "diagram"


def extract_mermaid_blocks(md_text: str) -> List[str]:
    return [m.group(1).strip() for m in MERMAID_BLOCK_RE.finditer(md_text)]


def guess_title_for_block(block: str) -> str:
    # Try to derive a stable slug from first non-empty line, else a hash
    first_line = next((ln for ln in block.splitlines() if ln.strip()), "")
    if first_line:
        # E.g., "flowchart LR" → use the keyword
        return slugify(first_line.split()[0])
    return md5(block.encode("utf-8")).hexdigest()[:8]


def write_temp_mmd(block: str, tmp_dir: Path, name: str) -> Path:
    tmp_dir.mkdir(parents=True, exist_ok=True)
    f = tmp_dir / f"{name}.mmd"
    f.write_text(block, encoding="utf-8")
    return f


def run_cmd(cmd: str) -> None:
    print(f"[CMD] {cmd}")
    proc = subprocess.run(cmd, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {cmd}")


def render_with_mmdc(input_mmd: Path, out_svg: Path, out_png: Path = None, theme: str = "") -> None:
    base_cmd = f"{MERMAID_CLI} -i {shlex.quote(str(input_mmd))} -o {shlex.quote(str(out_svg))}"
    if PUPPETEER_CFG.exists():
        base_cmd += f" -p {shlex.quote(str(PUPPETEER_CFG))}"
    if (ROOT / ".mermaidrc.json").exists():
        base_cmd += f" -c {shlex.quote(str(ROOT / '.mermaidrc.json'))}"
    if theme:
        base_cmd += f" -t {shlex.quote(theme)}"
    run_cmd(base_cmd)
    if out_png is not None:
        png_cmd = base_cmd.replace(str(out_svg), str(out_png))
        # Ensure PNG output extension
        if not png_cmd.endswith(".png"):
            png_cmd += " -e png"
        else:
            png_cmd += ""
        run_cmd(png_cmd)


def process_markdown(md_path: Path) -> List[Tuple[Path, Path]]:
    rel = md_path.relative_to(ROOT)
    out_dir = OUT_ROOT / rel.parent / rel.stem
    tmp_dir = ROOT / ".mermaid_tmp" / rel.parent / rel.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    md_text = md_path.read_text(encoding="utf-8")
    blocks = extract_mermaid_blocks(md_text)
    results = []

    if not blocks:
        print(f"[INFO] No mermaid blocks in {rel}")
        return results

    for idx, block in enumerate(blocks, 1):
        slug = guess_title_for_block(block)
        # Make filename stable across edits: include a short digest of the block to avoid collisions
        digest = md5(block.encode("utf-8")).hexdigest()[:6]
        base_name = f"{slug}-{idx:02d}-{digest}"
        mmd_file = write_temp_mmd(block, tmp_dir, base_name)
        out_svg = out_dir / f"{base_name}.svg"
        out_png = (out_dir / f"{base_name}.png") if EXPORT_PNG else None
        print(f"[RENDER] {rel} → {out_svg.name}")
        render_with_mmdc(mmd_file, out_svg, out_png, theme=THEME)
        results.append((out_svg, out_png or Path()))
    return results


def ensure_puppeteer_cfg():
    # In CI we run headless; allow sandbox-less if needed
    if not PUPPETEER_CFG.exists():
        PUPPETEER_CFG.write_text(
            json.dumps({
                "args": ["--no-sandbox", "--disable-setuid-sandbox"]
            }, indent=2),
            encoding="utf-8"
        )


def main(args: List[str]) -> None:
    ensure_puppeteer_cfg()
    if not args:
        print("Usage: export_mermaid.py <markdown files...>")
        sys.exit(2)

    any_out = False
    for a in args:
        p = (ROOT / a).resolve()
        if not p.exists():
            print(f"[WARN] File not found: {a}")
            continue
        if p.suffix.lower() not in {".md", ".markdown"}:
            print(f"[SKIP] Not a Markdown file: {a}")
            continue
        res = process_markdown(p)
        if res:
            any_out = True

    if any_out:
        print(f"[DONE] Diagrams exported under: {OUT_ROOT}")
    else:
        print("[DONE] No diagrams produced.")


if __name__ == "__main__":
    main(sys.argv[1:])