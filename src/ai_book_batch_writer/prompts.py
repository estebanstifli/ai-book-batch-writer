"""LangChain prompt templates for outline and long-form generation."""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_outline_prompt() -> ChatPromptTemplate:
    """Return the strict JSON outline prompt."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are a senior book architect. Design coherent, useful, "
                    "non-repetitive long-form structures. Return only valid JSON."
                ),
            ),
            (
                "human",
                """
Create a detailed outline for a long-form book or guide.

Working title: {title}
Main idea: {idea}
Output language: {output_language}
Tone: {tone}
Target audience: {target_audience}
Number of chapters: {chapter_count}
Approximate words per chapter: {words_per_chapter}
Additional instructions: {additional_instructions}

Requirements:
- Return exactly {chapter_count} chapters.
- Use the requested output language for all reader-facing text.
- Give each chapter a clear purpose and logical progression.
- Include 2 to 5 useful sections per chapter.
- Return only one JSON object with no Markdown fences or commentary.
- Follow this shape exactly:

{{
  "title": "Book title",
  "subtitle": "Optional subtitle or null",
  "chapters": [
    {{
      "number": 1,
      "title": "Chapter title",
      "summary": "What this chapter covers",
      "sections": [
        {{
          "title": "Section title",
          "summary": "What this section covers"
        }}
      ]
    }}
  ]
}}
""".strip(),
            ),
        ]
    )


def build_chapter_prompt() -> ChatPromptTemplate:
    """Return the prompt for a chapter without explicit sections."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert long-form writer. Produce accurate, coherent "
                    "draft prose that follows the supplied outline and avoids padding."
                ),
            ),
            (
                "human",
                """
Write the complete draft for this chapter.

Book title: {book_title}
Book idea: {idea}
Output language: {output_language}
Tone: {tone}
Target audience: {target_audience}
Target length: about {target_words} words
Additional instructions: {additional_instructions}

Full outline:
{outline_context}

Current chapter:
Number: {chapter_number}
Title: {chapter_title}
Summary: {chapter_summary}

Recent prior context:
{previous_context}

Write only the chapter prose. Use helpful Markdown subheadings where appropriate.
Do not repeat the chapter title as a top-level heading.
""".strip(),
            ),
        ]
    )


def build_section_prompt() -> ChatPromptTemplate:
    """Return the prompt for one section within a chapter."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You are an expert long-form writer working one section at a time. "
                    "Maintain continuity, factual caution, and a consistent voice."
                ),
            ),
            (
                "human",
                """
Write one section of a chapter.

Book title: {book_title}
Book idea: {idea}
Output language: {output_language}
Tone: {tone}
Target audience: {target_audience}
Target length for this section: about {target_words} words
Additional instructions: {additional_instructions}

Full outline:
{outline_context}

Chapter {chapter_number}: {chapter_title}
Chapter purpose: {chapter_summary}
Section title: {section_title}
Section purpose: {section_summary}

Recent prior context:
{previous_context}

Write only the section prose. Do not repeat the section title as a heading.
Connect naturally to the broader chapter and avoid summarizing material that has
not yet been written.
""".strip(),
            ),
        ]
    )


def build_json_repair_prompt() -> ChatPromptTemplate:
    """Return a prompt that repairs malformed outline JSON."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                (
                    "You repair JSON. Preserve the intended data, remove commentary, "
                    "and return only one valid JSON object."
                ),
            ),
            (
                "human",
                """
Repair the following model output so it is valid JSON with this shape:

{{
  "title": "Book title",
  "subtitle": "Optional subtitle or null",
  "chapters": [
    {{
      "number": 1,
      "title": "Chapter title",
      "summary": "Chapter summary",
      "sections": [
        {{"title": "Section title", "summary": "Section summary"}}
      ]
    }}
  ]
}}

Malformed output:
{malformed_json}
""".strip(),
            ),
        ]
    )


def build_continuation_prompt() -> ChatPromptTemplate:
    """Return a future-ready continuation prompt for truncated responses."""
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Continue long-form prose without repeating existing text.",
            ),
            (
                "human",
                """
Continue the draft in {output_language} until the requested section is complete.
Keep the same tone and formatting. Return only the continuation.

Existing ending:
{existing_ending}
""".strip(),
            ),
        ]
    )

