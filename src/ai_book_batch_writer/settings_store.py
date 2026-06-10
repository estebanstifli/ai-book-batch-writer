"""Persistence for non-secret desktop preferences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_book_batch_writer.config import user_data_dir

SETTINGS_SCHEMA_VERSION = 2


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
            if not isinstance(data, dict):
                return {}
            migrated = self._migrate(data)
            if migrated != data:
                self.save(migrated)
            return migrated
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

    @staticmethod
    def _migrate(settings: dict[str, Any]) -> dict[str, Any]:
        """Upgrade safe preferences written by earlier application versions."""
        migrated = dict(settings)
        version = migrated.get("settings_schema_version", 1)
        try:
            version_number = int(version)
        except (TypeError, ValueError):
            version_number = 1

        if version_number < 2:
            global_llm = migrated.get("global_llm")
            if isinstance(global_llm, dict):
                global_llm = dict(global_llm)
                if global_llm.get("max_tokens") in {None, 3000, "3000"}:
                    global_llm["max_tokens"] = 8096
                migrated["global_llm"] = global_llm

        migrated["settings_schema_version"] = SETTINGS_SCHEMA_VERSION
        return migrated
