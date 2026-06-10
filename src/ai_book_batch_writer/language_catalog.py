"""Load the bundled Unicode CLDR language catalog."""

from __future__ import annotations

import json
from pathlib import Path

from ai_book_batch_writer.config import resource_path


def load_output_languages(path: str | Path | None = None) -> list[str]:
    """Return sorted unique language display names from the bundled JSON."""
    catalog_path = Path(path) if path else resource_path("data/languages_en.json")
    try:
        payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ["English", "Spanish"]
    items = payload.get("languages", []) if isinstance(payload, dict) else []
    names = {
        str(item.get("name")).strip()
        for item in items
        if isinstance(item, dict) and item.get("name")
    }
    names.update({"English", "Spanish"})
    return sorted(names, key=str.casefold)
