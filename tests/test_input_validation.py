from __future__ import annotations

import pytest

from ai_book_batch_writer.input_validation import (
    InputValidationError,
    validate_book_input,
    validate_llm_input,
)


def test_llm_validation_collects_multiple_friendly_issues() -> None:
    with pytest.raises(InputValidationError) as captured:
        validate_llm_input(
            model="",
            api_base="not-a-url",
            temperature="hot",
            max_tokens="12",
            api_key_required=True,
            api_key_available=False,
            api_base_required=True,
        )

    fields = [issue.field_key for issue in captured.value.issues]
    assert fields == [
        "label.model",
        "label.api_key",
        "label.api_base",
        "label.temperature",
        "label.max_tokens",
    ]


def test_llm_validation_accepts_8096_tokens() -> None:
    result = validate_llm_input(
        model="gpt-4o-mini",
        api_base=None,
        temperature="0.7",
        max_tokens="8096",
        api_key_required=True,
        api_key_available=True,
        api_base_required=False,
    )

    assert result.max_tokens == 8096


def test_book_validation_requires_text_and_numeric_ranges() -> None:
    with pytest.raises(InputValidationError) as captured:
        validate_book_input(
            idea="",
            output_language="",
            tone="",
            target_audience="",
            chapter_count="0",
            words_per_chapter="many",
        )

    assert len(captured.value.issues) == 6
