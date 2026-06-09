"""Executable entry point for source runs and PyInstaller."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_book_batch_writer.app import run
from ai_book_batch_writer.logging_config import configure_logging


def main() -> None:
    """Configure diagnostics and launch the GUI."""
    configure_logging()
    run()


if __name__ == "__main__":
    main()

