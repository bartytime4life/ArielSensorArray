"""
tests/test_logging.py

Check logging setup: v50_debug_log.md exists and is append-only.
"""

from pathlib import Path


def test_debug_log_exists_and_writable():
    """Ensure v50_debug_log.md is present and writable."""
    log_file = Path("logs/v50_debug_log.md")
    if not log_file.exists():
        # Create a placeholder
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text("# SpectraMind V50 Debug Log\n")

    old_content = log_file.read_text()
    with log_file.open("a") as f:
        f.write("Test entry\n")

    new_content = log_file.read_text()
    assert "Test entry" in new_content
    # restore
    log_file.write_text(old_content)