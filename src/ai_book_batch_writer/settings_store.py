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
        safe = self._remove_secrets(settings)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(safe, handle, ensure_ascii=False, indent=2)
            handle.write("\n")

    def save_merged(self, updates: dict[str, Any]) -> dict[str, Any]:
        """Merge and persist preference updates, returning the stored data."""
        settings = self.load()
        settings.update(updates)
        self.save(settings)
        return self._remove_secrets(settings)

    @classmethod
    def _remove_secrets(cls, value: Any) -> Any:
        """Recursively remove fields that look like credentials."""
        if isinstance(value, dict):
            return {
                key: cls._remove_secrets(item)
                for key, item in value.items()
                if not cls._is_secret_key(key)
            }
        if isinstance(value, list):
            return [cls._remove_secrets(item) for item in value]
        return value

    @staticmethod
    def _is_secret_key(key: str) -> bool:
        normalized = key.lower().replace("-", "_")
        return (
            "api_key" in normalized
            or "secret" in normalized
            or normalized in {"token", "access_token", "auth_token", "password"}
        )
