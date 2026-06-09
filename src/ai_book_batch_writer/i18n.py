"""Small JSON-backed translation service with nested key support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ai_book_batch_writer.config import resource_path


class Translator:
    """Load and query locale JSON files."""

    def __init__(
        self,
        language: str = "en",
        locales_dir: str | Path | None = None,
        fallback_language: str = "en",
    ) -> None:
        self.locales_dir = Path(locales_dir) if locales_dir else resource_path("locales")
        self.fallback_language = fallback_language
        self.language = language
        self._fallback = self._load(fallback_language)
        self._translations = (
            self._fallback if language == fallback_language else self._load(language)
        )

    def _load(self, language: str) -> dict[str, Any]:
        path = self.locales_dir / f"{language}.json"
        if not path.exists():
            if language == self.fallback_language:
                raise FileNotFoundError(f"Missing fallback locale: {path}")
            return {}
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Locale root must be a JSON object: {path}")
        return data

    @staticmethod
    def _resolve(data: dict[str, Any], key: str) -> Any:
        value: Any = data
        for part in key.split("."):
            if not isinstance(value, dict) or part not in value:
                return None
            value = value[part]
        return value

    def set_language(self, language: str) -> None:
        """Switch the active locale, retaining English fallback behavior."""
        self.language = language
        self._translations = (
            self._fallback if language == self.fallback_language else self._load(language)
        )

    def available_languages(self) -> list[str]:
        """Return locale codes found in the locale directory."""
        return sorted(path.stem for path in self.locales_dir.glob("*.json"))

    def t(self, key: str, **kwargs: Any) -> str:
        """Translate a nested key and apply optional format arguments."""
        value = self._resolve(self._translations, key)
        if value is None:
            value = self._resolve(self._fallback, key)
        if not isinstance(value, str):
            return key
        try:
            return value.format(**kwargs)
        except (KeyError, ValueError):
            return value


_default_translator: Translator | None = None


def get_translator() -> Translator:
    """Return the process-wide translator used by the desktop UI."""
    global _default_translator
    if _default_translator is None:
        _default_translator = Translator()
    return _default_translator


def t(key: str, **kwargs: Any) -> str:
    """Translate a key with the default translator."""
    return get_translator().t(key, **kwargs)

