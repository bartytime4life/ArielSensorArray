\#!/usr/bin/env python3

# -*- coding: utf-8 -*-

#

# export\_mermaid.py — Extract Mermaid diagrams from Markdown fences and .mmd files, render via Mermaid CLI (mmdc),

# and write results to a predictable docs/diagrams tree suitable for CI artifacts.

#

# Key Upgrades

# • Supports BOTH Markdown \`\`\`mermaid fences AND standalone .mmd files

# • Concurrent rendering with stable, content-hashed filenames

# • Robust command construction: uses MERMAID\_CLI if provided (string or path) or discovers 'mmdc'

# • Optional fail-on-warn, width/height, theme/background, multi-format output (svg/png/pdf)

# • Directory discovery (recursive), per-source subdirectories: docs/diagrams/<rel>/<source-stem>/

# • JSON manifest output for CI traceability

# • Graceful logs + non-zero exit on failures (CI-friendly)

#

# Usage Examples

# poetry run python scripts/export\_mermaid.py --inputs README.md ARCHITECTURE.md --outdir docs/diagrams --formats svg png

# poetry run python scripts/export\_mermaid.py --inputs docs --outdir docs/diagrams --theme dark --background transparent

# poetry run python scripts/export\_mermaid.py --inputs README.md docs --list-only

#

# Environment Variables (optional)

# MERMAID\_CLI  : Custom Mermaid CLI command or path (e.g., "mmdc" or "npx --yes @mermaid-js/mermaid-cli")

# EXPORT\_PNG   : "1" to emit png even if --formats omitted (legacy compatibility)

# THEME        : Override theme (default|neutral|dark|forest) if not passed via CLI

#

# Exit Codes

# 0 success, 1 usage error/unsupported option, 2 CLI (mmdc) missing, 3 render failure

#

from **future** import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Dict

# ---------- Defaults / Constants ----------

ROOT = Path(**file**).resolve().parents\[1]
DEFAULT\_OUTDIR = ROOT / "docs" / "diagrams"
DEFAULT\_PUPPETEER\_CFG = ROOT / "puppeteer-config.json"
DEFAULT\_MERMAIDRC = ROOT / ".mermaidrc.json"

MERMAID\_FENCE\_RE = re.compile(
r"(^|\n)`mermaid\s*\n(?P<code>.*?)(?:\n)`",
re.IGNORECASE | re.DOTALL,
)

SLUG\_RE = re.compile(r"\[^a-z0-9]+", re.IGNORECASE)

SUPPORTED\_FORMATS = {"svg", "png", "pdf"}

ENV\_MERMAID\_CLI = os.environ.get("MERMAID\_CLI", "").strip()
ENV\_EXPORT\_PNG = os.environ.get("EXPORT\_PNG", "0").strip() == "1"
ENV\_THEME = os.environ.get("THEME", "").strip()

# ---------- Dataclasses ----------

@dataclass(frozen=True)
class DiagramTask:
"""Unit of work for rendering one Mermaid diagram (fence or file)."""
kind: str  # 'fence' or 'file'
source\_path: Path
code: str
out\_dir: Path        # directory to place outputs for this task
base\_name: str       # base file name (without extension)
fence\_index: int     # only for fences; 0 for files
fence\_line: int      # approximate starting line number; 1 for files

@dataclass(frozen=True)
class RenderResult:
"""Result for a single diagram task format render."""
task: DiagramTask
fmt: str
out\_path: Path

# ---------- Utilities ----------

def log(msg: str) -> None:
print(f"\[export-mermaid] {msg}", file=sys.stderr)

def die(msg: str, code: int = 1) -> None:
log(f"ERROR: {msg}")
sys.exit(code)

def sha256\_hex(s: str, n: int = 10) -> str:
return hashlib.sha256(s.encode("utf-8")).hexdigest()\[:n]

def slugify(text: str, max\_len: int = 60) -> str:
text = text.strip().lower()
text = SLUG\_RE.sub("-", text).strip("-")
if len(text) > max\_len:
text = text\[:max\_len].rstrip("-")
return text or "diagram"

def read\_text(p: Path) -> str:
try:
return p.read\_text(encoding="utf-8")
except UnicodeDecodeError:
return p.read\_text(encoding="utf-8", errors="replace")

def discover\_inputs(raws: Iterable\[str]) -> List\[Path]:
out: List\[Path] = \[]
for raw in raws:
p = Path(raw)
if not p.exists():
log(f"WARN: Input not found: {p}")
continue
if p.is\_dir():
for ext in ("*.md", "*.markdown", "\*.mmd"):
out.extend(sorted(p.rglob(ext)))
else:
if p.suffix.lower() in {".md", ".markdown", ".mmd"}:
out.append(p)
else:
log(f"WARN: Skipping unsupported file (not .md/.markdown/.mmd): {p}")
\# de-dup while preserving order
seen = set()
uniq: List\[Path] = \[]
for p in out:
if p.resolve() in seen:
continue
uniq.append(p)
seen.add(p.resolve())
return uniq

def extract\_fences(md\_path: Path) -> List\[Tuple\[str, int]]:
content = read\_text(md\_path)
fences: List\[Tuple\[str, int]] = \[]
for m in MERMAID\_FENCE\_RE.finditer(content):
code = m.group("code").strip()
start\_pos = m.start()
line\_no = content.count("\n", 0, start\_pos) + 1
fences.append((code, line\_no))
return fences

def fence\_stem(block: str, md\_path: Path, idx: int) -> str:
first\_line = next((ln for ln in block.splitlines() if ln.strip()), "")
token = first\_line.split()\[0] if first\_line else "diagram"
slug = slugify(token)
digest = sha256\_hex(block, n=8)
return f"{md\_path.stem}-fence-{idx:02d}-{slug}-{digest}"

def file\_stem(code: str, file\_path: Path) -> str:
digest = sha256\_hex(code, n=8)
return f"{file\_path.stem}-{digest}"

def ensure\_puppeteer\_config(path: Path) -> None:
if not path.exists():
path.write\_text(
json.dumps({"args": \["--no-sandbox", "--disable-setuid-sandbox"]}, indent=2),
encoding="utf-8",
)
log(f"Created Puppeteer config: {path}")

def discover\_mmdc() -> Optional\[str]:
"""
Resolve the Mermaid CLI runner command.

```
Priority:
  1) MERMAID_CLI env (use as-is; may contain spaces e.g. 'npx --yes @mermaid-js/mermaid-cli')
  2) 'mmdc' discoverable on PATH
  3) fallback: 'npx --yes @mermaid-js/mermaid-cli'
"""
if ENV_MERMAID_CLI:
    return ENV_MERMAID_CLI
mmdc_path = shutil.which("mmdc")
if mmdc_path:
    return mmdc_path
return "npx --yes @mermaid-js/mermaid-cli"
```

def run\_cmd\_capture(cmd: List\[str] | str, use\_shell: bool = False) -> subprocess.CompletedProcess:
if use\_shell:
return subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def build\_mmdc\_command(
mmdc: str,
in\_file: Path,
out\_file: Path,
theme: Optional\[str],
background: Optional\[str],
width: Optional\[int],
height: Optional\[int],
mermaidrc: Optional\[Path],
puppeteer\_cfg: Optional\[Path],
) -> Tuple\[List\[str] | str, bool]:
"""
Build the command to invoke mmdc. If mmdc is a multi-word string (e.g., 'npx --yes ...'),
we return a shell string and use\_shell=True. Otherwise, return a list for execv.
"""
args = \[]
if " " in mmdc or "\t" in mmdc:
\# treat as full shell command
base = f"{mmdc} -i {shlex.quote(str(in\_file))} -o {shlex.quote(str(out\_file))}"
if puppeteer\_cfg:
base += f" -p {shlex.quote(str(puppeteer\_cfg))}"
if mermaidrc and mermaidrc.exists():
base += f" -c {shlex.quote(str(mermaidrc))}"
if theme:
base += f" -t {shlex.quote(theme)}"
if background:
base += f" -b {shlex.quote(background)}"
if width:
base += f" -w {int(width)}"
if height:
base += f" -H {int(height)}"
return base, True
else:
args = \[mmdc, "-i", str(in\_file), "-o", str(out\_file)]
if puppeteer\_cfg:
args += \["-p", str(puppeteer\_cfg)]
if mermaidrc and mermaidrc.exists():
args += \["-c", str(mermaidrc)]
if theme:
args += \["-t", theme]
if background:
args += \["-b", background]
if width:
args += \["-w", str(int(width))]
if height:
args += \["-H", str(int(height))]
return args, False

# ---------- Task Collection ----------

def collect\_tasks(
inputs: List\[Path],
outdir\_root: Path,
theme: Optional\[str],
group\_by\_source: bool = True,
) -> List\[DiagramTask]:
tasks: List\[DiagramTask] = \[]
for src in inputs:
rel = src.relative\_to(ROOT) if src.is\_absolute() else src
\# Compute per-source directory: docs/diagrams/<rel-parent>/<source-stem>/
subdir = outdir\_root / (rel.parent if group\_by\_source else Path()) / src.stem
subdir.mkdir(parents=True, exist\_ok=True)

```
    if src.suffix.lower() in {".md", ".markdown"}:
        fences = extract_fences(src)
        if not fences:
            log(f"INFO: No mermaid fences in {rel}")
            continue
        for i, (code, line_no) in enumerate(fences, start=1):
            base_name = fence_stem(code, src, i)
            tasks.append(
                DiagramTask(
                    kind="fence",
                    source_path=src,
                    code=code,
                    out_dir=subdir,
                    base_name=base_name,
                    fence_index=i,
                    fence_line=line_no,
                )
            )
    elif src.suffix.lower() == ".mmd":
        code = read_text(src)
        base_name = file_stem(code, src)
        tasks.append(
            DiagramTask(
                kind="file",
                source_path=src,
                code=code,
                out_dir=subdir,
                base_name=base_name,
                fence_index=0,
                fence_line=1,
            )
        )
return tasks
```

# ---------- Rendering ----------

def render\_task(
task: DiagramTask,
mmdc\_cmd: str,
formats: Tuple\[str, ...],
theme: Optional\[str],
background: Optional\[str],
width: Optional\[int],
height: Optional\[int],
mermaidrc: Optional\[Path],
puppeteer\_cfg: Optional\[Path],
fail\_on\_warn: bool,
verbose: bool,
) -> List\[RenderResult]:
written: List\[RenderResult] = \[]

```
task.out_dir.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory(prefix="mmd_") as tdir:
    tmp_in = Path(tdir) / f"{task.base_name}.mmd"
    tmp_in.write_text(task.code, encoding="utf-8")

    for fmt in formats:
        out_path = task.out_dir / f"{task.base_name}.{fmt}"
        cmd, use_shell = build_mmdc_command(
            mmdc=mmdc_cmd,
            in_file=tmp_in,
            out_file=out_path,
            theme=theme,
            background=background,
            width=width,
            height=height,
            mermaidrc=mermaidrc if mermaidrc and mermaidrc.exists() else None,
            puppeteer_cfg=puppeteer_cfg if puppeteer_cfg and puppeteer_cfg.exists() else None,
        )
        if verbose:
            if use_shell:
                log(f"CMD(shell): {cmd}")
            else:
                log(f"CMD(exec) : {' '.join(shlex.quote(c) for c in cmd if isinstance(c, str))}")

        proc = run_cmd_capture(cmd, use_shell=use_shell)
        if proc.returncode != 0:
            log("----- mmdc stderr -----")
            log(proc.stderr.strip())
            log("----- mmdc stdout -----")
            log(proc.stdout.strip())
            die(
                f"Render failed for {task.kind} at {task.source_path} "
                f"(line {task.fence_line}) [{fmt}]",
                code=3,
            )

        # Some warnings appear in stderr even on RC=0
        if fail_on_warn and proc.stderr.strip():
            log("----- mmdc stderr (warn) -----")
            log(proc.stderr.strip())
            die(
                f"Render emitted warnings for {task.kind} at {task.source_path} "
                f"(line {task.fence_line}) with --fail-on-warn.",
                code=3,
            )

        written.append(RenderResult(task=task, fmt=fmt, out_path=out_path))

return written
```

# ---------- CLI ----------

def parse\_args(argv: Optional\[List\[str]] = None) -> argparse.Namespace:
p = argparse.ArgumentParser(
prog="export\_mermaid.py",
description="Render Mermaid diagrams from Markdown code fences and .mmd files using Mermaid CLI (mmdc).",
)
p.add\_argument(
"--inputs",
nargs="+",
required=True,
help="Files and/or directories to scan (.md/.markdown/.mmd). Directories are scanned recursively.",
)
p.add\_argument(
"--outdir",
default=str(DEFAULT\_OUTDIR),
help=f"Output root directory (default: {DEFAULT\_OUTDIR})",
)
p.add\_argument(
"--formats",
nargs="+",
default=None,
help="Output formats: svg png pdf. Default: svg (and png if EXPORT\_PNG=1).",
)
p.add\_argument(
"--theme",
default=ENV\_THEME or None,
help="Mermaid theme: default | neutral | dark | forest (env THEME respected if not provided).",
)
p.add\_argument(
"--background",
default="transparent",
help="Background color, e.g., 'transparent' or '#ffffff' (default: transparent).",
)
p.add\_argument("--width", type=int, default=None, help="Width in pixels (optional).")
p.add\_argument("--height", type=int, default=None, help="Height in pixels (optional).")
p.add\_argument("--max-workers", type=int, default=max(1, os.cpu\_count() or 1), help="Concurrency (default: CPU count).")
p.add\_argument("--fail-on-warn", action="store\_true", help="Treat Mermaid CLI warnings as errors.")
p.add\_argument("--list-only", action="store\_true", help="List discovered tasks and exit.")
p.add\_argument("--verbose", action="store\_true", help="Verbose command logging.")
p.add\_argument("--no-source-subdirs", action="store\_true", help="Do NOT group outputs under per-source subdirectories.")
p.add\_argument("--manifest", default=None, help="Write a JSON manifest mapping sources to outputs at this path.")
return p.parse\_args(argv)

def main(argv: Optional\[List\[str]] = None) -> None:
args = parse\_args(argv)

```
# Validate formats
if args.formats:
    formats = tuple(f.lower() for f in args.formats)
    for f in formats:
        if f not in SUPPORTED_FORMATS:
            die(f"Unsupported format: {f}. Supported: {', '.join(sorted(SUPPORTED_FORMATS))}", 1)
else:
    # Default: svg (+ png if legacy env set)
    formats = ("svg", "png") if ENV_EXPORT_PNG else ("svg",)

# Discover Mermaid CLI
mmdc_cmd = discover_mmdc()
if not mmdc_cmd:
    die("Mermaid CLI not found; set MERMAID_CLI or install 'mmdc' / npx @mermaid-js/mermaid-cli", 2)

# Ensure Puppeteer config for sandboxed CI environments
ensure_puppeteer_config(DEFAULT_PUPPETEER_CFG)

# Collect inputs
inputs = discover_inputs(args.inputs)
if not inputs:
    die("No valid inputs discovered (.md/.markdown/.mmd).", 1)

# Collect tasks
outdir_root = Path(args.outdir)
outdir_root.mkdir(parents=True, exist_ok=True)
tasks = collect_tasks(
    inputs=inputs,
    outdir_root=outdir_root,
    theme=args.theme,
    group_by_source=not args.no_source_subdirs,
)
if not tasks:
    log("No Mermaid fences or .mmd files found. Nothing to do.")
    sys.exit(0)

if args.list_only:
    for t in tasks:
        print(json.dumps({
            "kind": t.kind,
            "source": str(t.source_path),
            "line": t.fence_line,
            "out_dir": str(t.out_dir),
            "base_name": t.base_name,
        }))
    sys.exit(0)

# Render concurrently
log(f"Mermaid CLI: {mmdc_cmd}")
log(f"Inputs     : {len(inputs)} files/dirs")
log(f"Tasks      : {len(tasks)}")
log(f"Outdir     : {outdir_root.resolve()}")
log(f"Formats    : {', '.join(formats)}")
if args.theme:
    log(f"Theme      : {args.theme}")
log(f"Background : {args.background}")
if args.width or args.height:
    log(f"Size       : {args.width or 'auto'} x {args.height or 'auto'}")
if args.fail_on_warn:
    log("Fail-on-warn: ENABLED")

rendered: List[RenderResult] = []
manifest: Dict[str, Dict[str, List[str]]] = {}  # {source: {fmt: [paths...]}}
with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
    futures = []
    for task in tasks:
        futures.append(
            ex.submit(
                render_task,
                task,
                mmdc_cmd,
                formats,
                args.theme,
                args.background,
                args.width,
                args.height,
                DEFAULT_MERMAIDRC if DEFAULT_MERMAIDRC.exists() else None,
                DEFAULT_PUPPETEER_CFG if DEFAULT_PUPPETEER_CFG.exists() else None,
                args.fail_on_warn,
                args.verbose,
            )
        )
    # gather
    for fut in concurrent.futures.as_completed(futures):
        try:
            results = fut.result()
            rendered.extend(results)
        except SystemExit:
            raise
        except Exception as e:
            die(f"Unhandled exception during render: {e}", 3)

# Log written paths + build manifest
for r in rendered:
    print(r.out_path)
    src = str(r.task.source_path)
    manifest.setdefault(src, {})
    manifest[src].setdefault(r.fmt, [])
    manifest[src][r.fmt].append(str(r.out_path))

if args.manifest:
    Path(args.manifest).parent.mkdir(parents=True, exist_ok=True)
    Path(args.manifest).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    log(f"Wrote manifest: {args.manifest}")

log(f"Wrote {len(rendered)} files total.")
sys.exit(0)
```

if **name** == "**main**":
main(sys.argv\[1:])
