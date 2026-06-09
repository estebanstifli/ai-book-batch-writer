from __future__ import annotations

from ai_book_batch_writer.prompts import (
    build_chapter_prompt,
    build_json_repair_prompt,
    build_outline_prompt,
    build_section_prompt,
)


def test_outline_prompt_declares_expected_inputs() -> None:
    prompt = build_outline_prompt()
    assert {
        "idea",
        "chapter_count",
        "output_language",
        "words_per_chapter",
    }.issubset(prompt.input_variables)

    messages = prompt.format_messages(
        title="Test",
        idea="A useful book",
        output_language="English",
        tone="Clear",
        target_audience="Developers",
        chapter_count=3,
        words_per_chapter=1000,
        additional_instructions="None",
    )
    assert '"chapters"' in messages[-1].content
    assert "exactly 3 chapters" in messages[-1].content


def test_generation_and_repair_prompts_are_langchain_templates() -> None:
    assert "chapter_title" in build_chapter_prompt().input_variables
    assert "section_title" in build_section_prompt().input_variables
    assert build_json_repair_prompt().input_variables == ["malformed_json"]

