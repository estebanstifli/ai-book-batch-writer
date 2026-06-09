from __future__ import annotations

import pytest

from ai_book_batch_writer.json_utils import (
    extract_json_from_text,
    parse_outline_document,
)

VALID_OUTLINE = """
{
  "title": "A Test Book",
  "subtitle": null,
  "chapters": [
    {
      "number": 1,
      "title": "First Chapter",
      "summary": "The opening chapter.",
      "sections": [
        {"title": "First Section", "summary": "A useful section."}
      ]
    }
  ]
}
""".strip()


def test_extracts_json_from_markdown_fence() -> None:
    wrapped = f"Here is the outline:\\n```json\\n{VALID_OUTLINE}\\n```"
    assert extract_json_from_text(wrapped) == VALID_OUTLINE


def test_balanced_extraction_handles_braces_inside_strings() -> None:
    text = 'Prefix {"title": "Use {carefully}", "chapters": []} suffix'
    assert extract_json_from_text(text) == (
        '{"title": "Use {carefully}", "chapters": []}'
    )


def test_parse_outline_validates_pydantic_schema() -> None:
    outline = parse_outline_document(VALID_OUTLINE)
    assert outline.title == "A Test Book"
    assert outline.chapters[0].sections[0].title == "First Section"


def test_invalid_outline_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="validation"):
        parse_outline_document('{"title": "Missing chapters"}')

