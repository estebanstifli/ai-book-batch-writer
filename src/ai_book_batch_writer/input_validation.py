"""Friendly validation for values entered in desktop forms."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass(frozen=True)
class InputIssue:
    """One user-correctable form issue."""

    field_key: str
    message_key: str
    params: dict[str, object]


class InputValidationError(ValueError):
    """Collection of form issues suitable for localized display."""

    def __init__(self, issues: list[InputIssue]) -> None:
        self.issues = issues
        super().__init__("Invalid form input")


@dataclass(frozen=True)
class ValidatedLLMInput:
    model: str
    api_base: str | None
    temperature: float
    max_tokens: int


@dataclass(frozen=True)
class ValidatedBookInput:
    idea: str
    output_language: str
    tone: str
    target_audience: str
    chapter_count: int
    words_per_chapter: int


def validate_llm_input(
    *,
    model: str,
    api_base: str | None,
    temperature: str,
    max_tokens: str,
    api_key_required: bool,
    api_key_available: bool,
    api_base_required: bool,
) -> ValidatedLLMInput:
    """Validate model configuration and return parsed values."""
    issues: list[InputIssue] = []
    clean_model = _required(model, "label.model", issues)
    clean_api_base = (api_base or "").strip() or None
    if api_key_required and not api_key_available:
        issues.append(InputIssue("label.api_key", "validation.required", {}))
    if api_base_required:
        clean_api_base = _required(
            clean_api_base or "",
            "label.api_base",
            issues,
        )
        if clean_api_base and not _is_http_url(clean_api_base):
            issues.append(
                InputIssue("label.api_base", "validation.url", {})
            )
    parsed_temperature = _float_in_range(
        temperature,
        "label.temperature",
        0.0,
        2.0,
        issues,
    )
    parsed_max_tokens = _int_in_range(
        max_tokens,
        "label.max_tokens",
        128,
        128_000,
        issues,
    )
    if issues:
        raise InputValidationError(issues)
    return ValidatedLLMInput(
        model=clean_model,
        api_base=clean_api_base,
        temperature=parsed_temperature,
        max_tokens=parsed_max_tokens,
    )


def validate_book_input(
    *,
    idea: str,
    output_language: str,
    tone: str,
    target_audience: str,
    chapter_count: str,
    words_per_chapter: str,
) -> ValidatedBookInput:
    """Validate book fields and return parsed values."""
    issues: list[InputIssue] = []
    clean_idea = _required(idea, "label.idea", issues)
    clean_language = _required(
        output_language,
        "label.output_language",
        issues,
    )
    clean_tone = _required(tone, "label.tone", issues)
    clean_audience = _required(
        target_audience,
        "label.target_audience",
        issues,
    )
    parsed_chapters = _int_in_range(
        chapter_count,
        "label.chapter_count",
        1,
        100,
        issues,
    )
    parsed_words = _int_in_range(
        words_per_chapter,
        "label.words_per_chapter",
        100,
        20_000,
        issues,
    )
    if issues:
        raise InputValidationError(issues)
    return ValidatedBookInput(
        idea=clean_idea,
        output_language=clean_language,
        tone=clean_tone,
        target_audience=clean_audience,
        chapter_count=parsed_chapters,
        words_per_chapter=parsed_words,
    )


def _required(
    value: str,
    field_key: str,
    issues: list[InputIssue],
) -> str:
    cleaned = value.strip()
    if not cleaned:
        issues.append(InputIssue(field_key, "validation.required", {}))
    return cleaned


def _int_in_range(
    value: str,
    field_key: str,
    minimum: int,
    maximum: int,
    issues: list[InputIssue],
) -> int:
    try:
        parsed = int(value.strip())
    except (AttributeError, ValueError):
        issues.append(InputIssue(field_key, "validation.integer", {}))
        return minimum
    if not minimum <= parsed <= maximum:
        issues.append(
            InputIssue(
                field_key,
                "validation.range",
                {"minimum": minimum, "maximum": maximum},
            )
        )
    return parsed


def _float_in_range(
    value: str,
    field_key: str,
    minimum: float,
    maximum: float,
    issues: list[InputIssue],
) -> float:
    try:
        parsed = float(value.strip())
    except (AttributeError, ValueError):
        issues.append(InputIssue(field_key, "validation.number", {}))
        return minimum
    if not minimum <= parsed <= maximum:
        issues.append(
            InputIssue(
                field_key,
                "validation.range",
                {"minimum": minimum, "maximum": maximum},
            )
        )
    return parsed


def _is_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
