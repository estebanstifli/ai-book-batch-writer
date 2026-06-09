"""Application paths and environment-backed configuration."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

APP_NAME = "AI Book Batch Writer"
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parents[1]


def resource_path(relative_path: str | Path) -> Path:
    """Resolve a source or PyInstaller-bundled resource path."""
    relative = Path(relative_path)
    candidates: list[Path] = []

    if hasattr(sys, "_MEIPASS"):
        candidates.append(Path(getattr(sys, "_MEIPASS")) / relative)

    configured_home = os.getenv("AI_BOOK_WRITER_HOME")
    if configured_home:
        candidates.append(Path(configured_home).expanduser() / relative)

    candidates.extend(
        [
            PROJECT_ROOT / relative,
            PACKAGE_DIR / relative,
            Path(sys.prefix) / relative,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else PROJECT_ROOT / relative


def user_data_dir() -> Path:
    """Return a writable per-user data directory."""
    if sys.platform == "win32":
        base = Path(os.getenv("APPDATA", Path.home()))
    else:
        base = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    path = base / "AI Book Batch Writer"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_environment() -> None:
    """Load environment variables from a local .env file when present."""
    load_dotenv(PROJECT_ROOT / ".env", override=False)
