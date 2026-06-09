from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from ai_book_batch_writer.generation_service import BookGenerationService
from ai_book_batch_writer.models import BookProject, BookSettings, LLMSettings


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
