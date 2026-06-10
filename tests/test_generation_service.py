from __future__ import annotations

from unittest.mock import Mock

from langchain_core.language_models.fake_chat_models import (
    FakeListChatModel,
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from tenacity import wait_none

from ai_book_batch_writer.generation_service import (
    MAX_MODEL_ATTEMPTS,
    BookGenerationService,
)
from ai_book_batch_writer.models import (
    BookChapter,
    BookProject,
    BookSection,
    BookSettings,
    LLMSettings,
)


def test_fake_llm_generates_outline_and_sections() -> None:
    outline_json = """
    {
      "title": "Test Book",
      "subtitle": null,
      "chapters": [
        {
          "number": 1,
          "title": "A Chapter",
          "summary": "A chapter summary.",
          "sections": [
            {"title": "One", "summary": "First section."},
            {"title": "Two", "summary": "Second section."}
          ]
        }
      ]
    }
    """
    fake = FakeListChatModel(
        responses=[
            outline_json,
            "Content for the first section.",
            "Content for the second section.",
        ]
    )
    llm_settings = LLMSettings(provider="ollama", model="fake")
    book_settings = BookSettings(
        idea="Test a complete generation flow.",
        chapter_count=1,
        words_per_chapter=500,
    )
    service = BookGenerationService(llm_settings, llm=fake)

    outline = service.generate_outline_document(book_settings)
    project = BookProject(
        settings=book_settings.model_copy(update={"title": outline.title}),
        llm_settings=llm_settings,
        chapters=outline.chapters,
        token_usage=service.usage.model_copy(deep=True),
    )
    outline_tokens = project.token_usage.outline_tokens
    events = []
    result = service.generate_book(project, progress_callback=events.append)

    assert result.chapters[0].status == "completed"
    assert result.chapters[0].sections[0].content.startswith("Content")
    assert result.chapters[0].sections[1].content.startswith("Content")
    assert any(event.message_key == "log.book_completed" for event in events)
    assert result.token_usage.outline_tokens == outline_tokens
    assert result.token_usage.book_tokens > 0
    assert result.token_usage.total_tokens == (
        result.token_usage.outline_tokens + result.token_usage.book_tokens
    )
    assert result.token_usage.estimated is True


def test_invalid_outline_uses_llm_repair() -> None:
    repaired = """
    {
      "title": "Repaired Book",
      "subtitle": null,
      "chapters": [
        {
          "number": 1,
          "title": "Recovered",
          "summary": "Recovered summary.",
          "sections": []
        }
      ]
    }
    """
    fake = FakeListChatModel(
        responses=[
            '{"title": "Broken", "chapters": [',
            repaired,
        ]
    )
    service = BookGenerationService(
        LLMSettings(provider="ollama", model="fake"),
        llm=fake,
    )

    outline = service.generate_outline_document(
        BookSettings(idea="Repair malformed JSON.", chapter_count=1)
    )
    assert outline.title == "Repaired Book"
    assert service.usage.outline_tokens > 0


def test_generation_uses_reported_provider_tokens() -> None:
    fake = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="Generated chapter.",
                usage_metadata={
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                },
            )
        ]
    )
    settings = LLMSettings(provider="ollama", model="fake")
    project = BookProject(
        settings=BookSettings(title="Tokens", idea="Count real tokens"),
        llm_settings=settings,
        chapters=[
            BookChapter(number=1, title="One", summary="Summary")
        ],
    )

    result = BookGenerationService(settings, llm=fake).generate_book(project)

    assert result.token_usage.book_tokens == 150
    assert result.token_usage.estimated is False


def test_model_request_uses_three_attempts() -> None:
    fake = Mock()
    fake.invoke.side_effect = [
        RuntimeError("temporary one"),
        RuntimeError("temporary two"),
        AIMessage(content="Recovered"),
    ]
    service = BookGenerationService(
        LLMSettings(provider="ollama", model="fake"),
        llm=fake,
    )
    invoke_without_wait = service._invoke_prompt.retry_with(wait=wait_none())

    result = invoke_without_wait(
        service,
        ChatPromptTemplate.from_messages([("human", "{request}")]),
        {"request": "Write"},
    )

    assert result.text == "Recovered"
    assert fake.invoke.call_count == MAX_MODEL_ATTEMPTS


def test_resume_skips_completed_sections_and_checkpoints() -> None:
    fake = FakeListChatModel(responses=["Recovered second section."])
    settings = LLMSettings(provider="ollama", model="fake")
    first_content = "Already generated."
    project = BookProject(
        settings=BookSettings(
            title="Resume",
            idea="Resume an interrupted project",
            words_per_chapter=500,
        ),
        llm_settings=settings,
        chapters=[
            BookChapter(
                number=1,
                title="One",
                summary="Summary",
                status="failed",
                sections=[
                    BookSection(
                        title="First",
                        summary="Done",
                        content=first_content,
                        status="completed",
                    ),
                    BookSection(
                        title="Second",
                        summary="Retry",
                        status="failed",
                        error="Temporary failure",
                    ),
                ],
            )
        ],
    )
    checkpoints: list[BookProject] = []

    result = BookGenerationService(settings, llm=fake).generate_book(
        project,
        checkpoint_callback=lambda value: checkpoints.append(
            value.model_copy(deep=True)
        ),
    )

    assert MAX_MODEL_ATTEMPTS == 3
    assert result.chapters[0].status == "completed"
    assert result.chapters[0].sections[0].content == first_content
    assert result.chapters[0].sections[1].content == "Recovered second section."
    assert checkpoints
    assert any(
        checkpoint.chapters[0].sections[1].status == "completed"
        for checkpoint in checkpoints
    )
