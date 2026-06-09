from __future__ import annotations

import pytest

from ai_book_batch_writer.models import (
    BookChapter,
    BookProject,
    BookSection,
    BookSettings,
    LLMSettings,
)


@pytest.fixture
def sample_project() -> BookProject:
    return BookProject(
        settings=BookSettings(
            title="Reliable AI Systems",
            idea="A practical guide to reliable AI-assisted software.",
            chapter_count=1,
            words_per_chapter=800,
        ),
        llm_settings=LLMSettings(
            provider="openai",
            model="test-model",
            api_key="must-not-be-saved",
        ),
        subtitle="A Practical Introduction",
        chapters=[
            BookChapter(
                number=1,
                title="Start With the Problem",
                summary="Define goals and constraints.",
                sections=[
                    BookSection(
                        title="Map the Workflow",
                        summary="Understand the current process.",
                        content="Document the current process before automating it.",
                        status="completed",
                    )
                ],
                status="completed",
            )
        ],
    )

