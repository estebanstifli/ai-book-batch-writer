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
    """Read total input and output tokens from common LangChain metadata."""
    usage = getattr(message, "usage_metadata", None)
    total = _tokens_from_mapping(usage)
    if total:
        return total

    metadata = getattr(message, "response_metadata", None)
    if isinstance(metadata, Mapping):
        for key in ("token_usage", "usage", "usage_metadata"):
            total = _tokens_from_mapping(metadata.get(key))
            if total:
                return total
        total = _tokens_from_mapping(metadata)
        if total:
            return total
    return 0


def _tokens_from_mapping(value: Any) -> int:
    if not isinstance(value, Mapping):
        return 0

    for key in ("total_tokens", "total_token_count"):
        total = _non_negative_int(value.get(key))
        if total is not None:
            return total

    input_tokens = _first_int(
        value,
        (
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
            "prompt_eval_count",
            "input_token_count",
        ),
    )
    output_tokens = _first_int(
        value,
        (
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
            "eval_count",
            "output_token_count",
        ),
    )
    if input_tokens is not None or output_tokens is not None:
        return (input_tokens or 0) + (output_tokens or 0)
    return 0


def _first_int(
    value: Mapping[str, Any],
    keys: tuple[str, ...],
) -> int | None:
    for key in keys:
        parsed = _non_negative_int(value.get(key))
        if parsed is not None:
            return parsed
    return None


def _non_negative_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value >= 0:
        return value
    return None
