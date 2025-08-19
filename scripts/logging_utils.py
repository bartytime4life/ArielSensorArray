from pathlib import Path
from datetime import datetime
from typing import Optional
from rich.console import Console
from rich.table import Table

_console = Console()

def console() -> Console:
    return _console

def log_event_jsonl(event_path: Path, payload: dict) -> None:
    """
    Append a single JSON line record to an events log for auditability.
    """
    import json
    event_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"ts": datetime.utcnow().isoformat(timespec="seconds") + "Z", **payload}
    with event_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")

def summary_table(title: str, mapping: dict) -> None:
    table = Table(title=title, show_header=False)
    table.add_column("Key", style="bold cyan", no_wrap=True)
    table.add_column("Value", style="white")
    for k, v in mapping.items():
        table.add_row(str(k), str(v))
    _console.print(table)

def notify_done(message: str, path: Optional[Path] = None) -> None:
    if path:
        _console.print(f"✅ {message} → [bold green]{path}[/]")
    else:
        _console.print(f"✅ {message}")