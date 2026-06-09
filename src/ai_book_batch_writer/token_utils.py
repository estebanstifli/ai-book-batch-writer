"""Lightweight word and token usage helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def count_words(text: str) -> int:
    """Count whitespace-delimited words."""
    return len(text.split())


def estimate_tokens(text: str) -> int:
    """Estimate tokens when a provider does not return usage metadata."""
    return max(1, round(len(text) / 4)) if text else 0


def total_tokens_from_message(message: Any) -> int:
    """Read LangChain usage metadata with a conservative fallback."""
    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, Mapping):
        total = usage.get("total_tokens")
        if isinstance(total, int):
            return total

    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, Mapping):
        token_usage = metadata.get("token_usage")
        if isinstance(token_usage, Mapping):
            total = token_usage.get("total_tokens")
            if isinstance(total, int):
                return total
    return 0

