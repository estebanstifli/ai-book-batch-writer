"""General utilities shared across the application."""

from __future__ import annotations

import threading
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

