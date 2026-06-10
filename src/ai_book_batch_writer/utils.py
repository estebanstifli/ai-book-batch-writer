"""General utilities shared across the application."""

from __future__ import annotations

import threading
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any


class GenerationCancelled(RuntimeError):
    """Raised when a user requests cancellation."""


class CancelToken:
    """Thread-safe cancellation flag checked between model calls."""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self.is_cancelled:
            raise GenerationCancelled("Generation was cancelled.")


def ensure_parent_dir(path: str | Path) -> Path:
    """Create the parent directory for a target file and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def create_project_directory(
    base_directory: str | Path,
    title: str,
    created_at: datetime,
) -> Path:
    """Create and return a unique timestamped directory for one project."""
    base = Path(base_directory)
    base.mkdir(parents=True, exist_ok=True)
    slug = safe_filename(title) or "untitled-book"
    stem = f"{created_at.astimezone().strftime('%Y%m%d-%H%M%S')}-{slug}"
    candidate = base / stem
    suffix = 2
    while candidate.exists():
        candidate = base / f"{stem}-{suffix}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def safe_filename(value: str) -> str:
    """Convert user-facing text to a conservative ASCII filename."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    characters = [
        character.lower() if character.isalnum() else "-"
        for character in ascii_value
    ]
    return "-".join(filter(None, "".join(characters).split("-")))[:80]


def message_content_to_text(message: Any) -> str:
    """Normalize LangChain message content to plain text."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()
