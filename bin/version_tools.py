#!/usr/bin/env python3
import re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"
PYPROJECT = ROOT / "pyproject.toml"
INIT = ROOT / "src/asa/__init__.py"

def read_version():
    return VERSION_FILE.read_text().strip()

def write_version(v):
    VERSION_FILE.write_text(v.strip() + "\n")

def bump(part):
    major, minor, patch = map(int, read_version().split("."))
    if part == "major":
        major += 1; minor = 0; patch = 0
    elif part == "minor":
        minor += 1; patch = 0
    elif part == "patch":
        patch += 1
    else:
        raise SystemExit(f"Unknown bump part: {part}")
    v = f"{major}.{minor}.{patch}"
    write_version(v)
    return v

def set_version(v):
    if not re.fullmatch(r"\d+\.\d+\.\d+", v):
        raise SystemExit("Version must be MAJOR.MINOR.PATCH")
    write_version(v)
    return v

def _replace(path, pattern, repl):
    text = path.read_text()
    text2, n = re.subn(pattern, repl, text, flags=re.M)
    if n == 0:
        raise SystemExit(f"Failed to update {path} with pattern {pattern}")
    path.write_text(text2)

def sync_files(v):
    _replace(PYPROJECT,
             r'(?m)^(version\s*=\s*")\d+\.\d+\.\d+(")', rf'\g<1>{v}\2')
    _replace(INIT,
             r'(?m)^__version__\s*=\s*"\d+\.\d+\.\d+"',
             f'__version__ = "{v}"')

def main():
    if len(sys.argv) < 2:
        print("Usage: version_tools.py [bump-major|bump-minor|bump-patch|set X.Y.Z]")
        sys.exit(2)
    cmd = sys.argv[1]
    if cmd.startswith("bump-"):
        v = bump(cmd.split("-",1)[1])
    elif cmd == "set":
        v = set_version(sys.argv[2])
    else:
        raise SystemExit("Unknown command")
    sync_files(v)
    print(v)

if __name__ == "__main__":
    main()
