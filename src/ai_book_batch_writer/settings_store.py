"""Persistence for non-secret desktop preferences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_book_batch_writer.config import user_data_dir


class SettingsStore:
    """Store appearance and provider preferences without credentials."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else user_data_dir() / "settings.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (OSError, json.JSONDecodeError):
            return {}

    def save(self, settings: dict[str, Any]) -> None:
        safe = {
            key: value
            for key, value in settings.items()
            if "key" not in key.lower() and "secret" not in key.lower()
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(safe, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

