"""Local cache for provider model catalogs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ai_book_batch_writer.config import user_data_dir


class ModelCatalogStore:
    """Persist model identifiers without storing provider credentials."""

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else user_data_dir() / "model_cache.json"

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"schema_version": 1, "providers": {}}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"schema_version": 1, "providers": {}}
        if not isinstance(data, dict) or not isinstance(
            data.get("providers"), dict
        ):
            return {"schema_version": 1, "providers": {}}
        return data

    def models_for(self, provider: str) -> list[str]:
        """Return cached model identifiers for one provider."""
        entry = self.load()["providers"].get(provider, {})
        models = entry.get("models", []) if isinstance(entry, dict) else []
        if not isinstance(models, list):
            return []
        return sorted(
            {
                model.strip()
                for model in models
                if isinstance(model, str) and model.strip()
            },
            key=str.casefold,
        )

    def update(self, provider: str, models: list[str]) -> list[str]:
        """Store a normalized provider catalog and return it."""
        normalized = sorted(
            {model.strip() for model in models if model.strip()},
            key=str.casefold,
        )
        data = self.load()
        data["providers"][provider] = {
            "models": normalized,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)
        return normalized
