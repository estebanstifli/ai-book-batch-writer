"""Provider-independent outline and long-form generation orchestration."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from ai_book_batch_writer.json_utils import (
    parse_outline_document,
)
from ai_book_batch_writer.llm_providers import create_chat_model
from ai_book_batch_writer.models import (
    BookChapter,
    BookOutline,
    BookProject,
    BookSection,
    BookSettings,
    GenerationEvent,
    LLMSettings,
    TokenUsage,
)
from ai_book_batch_writer.prompts import (
    build_chapter_prompt,
    build_json_repair_prompt,
    build_outline_prompt,
    build_section_prompt,
)
from ai_book_batch_writer.token_utils import (
    count_words,
    estimate_tokens,
    total_tokens_from_message,
)
from ai_book_batch_writer.utils import (
    CancelToken,
    GenerationCancelled,
    message_content_to_text,
)

logger = logging.getLogger(__name__)
ProgressCallback = Callable[[GenerationEvent], None]


@dataclass(frozen=True)
class ModelResult:
    """Normalized result from one LangChain model invocation."""

    text: str
    words: int
    tokens: int
    estimated_tokens: bool


def _is_retryable(exception: BaseException) -> bool:
    return not isinstance(
        exception,
        (GenerationCancelled, ValidationError, ValueError),
    )


class BookGenerationService:
    """Generate validated outlines and book content with LangChain."""

    def __init__(
        self,
        llm_settings: LLMSettings,
        llm: BaseChatModel | None = None,
    ) -> None:
        self.llm_settings = llm_settings
        self.llm = llm or create_chat_model(llm_settings)
        self.usage = TokenUsage()

    def _reset_usage(self) -> None:
        self.usage = TokenUsage()

    @retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=8),
        reraise=True,
    )
    def _invoke_prompt(
        self,
        prompt: ChatPromptTemplate,
        values: dict[str, Any],
        usage_bucket: str = "book",
    ) -> ModelResult:
        chain = prompt | self.llm
        message = chain.invoke(values)
        text = message_content_to_text(message)
        if not text:
            raise RuntimeError("The model returned an empty response.")
        reported_tokens = total_tokens_from_message(message)
        estimated = reported_tokens <= 0
        result = ModelResult(
            text=text,
            words=count_words(text),
            tokens=reported_tokens or estimate_tokens(text),
            estimated_tokens=estimated,
        )
        if usage_bucket == "outline":
            self.usage.add_outline(result.tokens, estimated)
        else:
            self.usage.add_book(result.tokens, estimated)
        return result

    def generate_outline_document(self, book_settings: BookSettings) -> BookOutline:
        """Generate, extract, repair when needed, and validate an outline."""
        self._reset_usage()
        result = self._invoke_prompt(
            build_outline_prompt(),
            {
                "title": book_settings.title or "Let the model propose a title",
                "idea": book_settings.idea,
                "output_language": book_settings.output_language,
                "tone": book_settings.tone,
                "target_audience": book_settings.target_audience,
                "chapter_count": book_settings.chapter_count,
                "words_per_chapter": book_settings.words_per_chapter,
                "additional_instructions": (
                    book_settings.additional_instructions or "None"
                ),
            },
            usage_bucket="outline",
        )

        try:
            outline = parse_outline_document(result.text)
        except ValueError as first_error:
            logger.warning("Outline validation failed; attempting repair: %s", first_error)
            repaired = self._invoke_prompt(
                build_json_repair_prompt(),
                {"malformed_json": result.text},
                usage_bucket="outline",
            ).text
            try:
                outline = parse_outline_document(repaired)
            except ValueError as second_error:
                raise ValueError(
                    "The generated outline remained invalid after JSON repair."
                ) from second_error

        for number, chapter in enumerate(outline.chapters, start=1):
            chapter.number = number
            chapter.status = "pending"
            chapter.error = None
            chapter.content = None
            for section in chapter.sections:
                section.status = "pending"
                section.error = None
                section.content = None
        return outline

    def generate_outline(self, book_settings: BookSettings) -> list[BookChapter]:
        """Generate and return validated outline chapters."""
        return self.generate_outline_document(book_settings).chapters

    @staticmethod
    def _outline_context(project: BookProject) -> str:
        outline = [
            {
                "number": chapter.number,
                "title": chapter.title,
                "summary": chapter.summary,
                "sections": [
                    {"title": section.title, "summary": section.summary}
                    for section in chapter.sections
                ],
            }
            for chapter in project.chapters
        ]
        return json.dumps(outline, ensure_ascii=False, indent=2)

    @staticmethod
    def _recent_context(project: BookProject, chapter_number: int) -> str:
        previous_parts: list[str] = []
        for chapter in project.chapters:
            if chapter.number >= chapter_number:
                break
            if chapter.content:
                previous_parts.append(chapter.content)
            else:
                previous_parts.extend(
                    section.content or ""
                    for section in chapter.sections
                    if section.content
                )
        context = "\n\n".join(previous_parts).strip()
        return context[-6000:] if context else "No previous generated content."

    def _common_prompt_values(
        self,
        project: BookProject,
        chapter: BookChapter,
    ) -> dict[str, Any]:
        settings = project.settings
        return {
            "book_title": settings.title or "Untitled Book",
            "idea": settings.idea,
            "output_language": settings.output_language,
            "tone": settings.tone,
            "target_audience": settings.target_audience,
            "additional_instructions": settings.additional_instructions or "None",
            "outline_context": self._outline_context(project),
            "chapter_number": chapter.number,
            "chapter_title": chapter.title,
            "chapter_summary": chapter.summary,
            "previous_context": self._recent_context(project, chapter.number),
        }

    def _generate_section(
        self,
        project: BookProject,
        chapter: BookChapter,
        section: BookSection,
        target_words: int,
        cancel_token: CancelToken,
    ) -> ModelResult:
        cancel_token.raise_if_cancelled()
        values = self._common_prompt_values(project, chapter)
        prior_section_text = "\n\n".join(
            item.content or ""
            for item in chapter.sections
            if item is not section and item.content
        )
        if prior_section_text:
            values["previous_context"] = (
                f"{values['previous_context']}\n\n"
                f"Earlier sections in this chapter:\n{prior_section_text[-4000:]}"
            )
        values.update(
            {
                "section_title": section.title,
                "section_summary": section.summary,
                "target_words": target_words,
            }
        )
        return self._invoke_prompt(build_section_prompt(), values)

    def generate_chapter(
        self,
        project: BookProject,
        chapter: BookChapter,
        cancel_token: CancelToken | None = None,
    ) -> BookChapter:
        """Generate one chapter, using sections when present."""
        self._reset_usage()
        token = cancel_token or CancelToken()
        chapter.status = "generating"
        chapter.error = None

        try:
            if chapter.sections:
                target_words = max(
                    200,
                    project.settings.words_per_chapter // len(chapter.sections),
                )
                for section in chapter.sections:
                    token.raise_if_cancelled()
                    section.status = "generating"
                    section.error = None
                    result = self._generate_section(
                        project,
                        chapter,
                        section,
                        target_words,
                        token,
                    )
                    section.content = result.text
                    section.status = "completed"
                chapter.content = None
            else:
                values = self._common_prompt_values(project, chapter)
                values["target_words"] = project.settings.words_per_chapter
                result = self._invoke_prompt(build_chapter_prompt(), values)
                chapter.content = result.text
            chapter.status = "completed"
        except GenerationCancelled:
            chapter.status = "pending"
            raise
        except Exception as exc:
            chapter.status = "failed"
            chapter.error = str(exc)
            raise
        finally:
            project.token_usage.merge(self.usage)
        return chapter

    def generate_book(
        self,
        project: BookProject,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
    ) -> BookProject:
        """Generate all pending chapters sequentially with progress events."""
        self._reset_usage()
        try:
            return self._generate_book_impl(
                project,
                progress_callback=progress_callback,
                cancel_token=cancel_token,
            )
        finally:
            project.token_usage.merge(self.usage)
            project.touch()

    def _generate_book_impl(
        self,
        project: BookProject,
        progress_callback: ProgressCallback | None = None,
        cancel_token: CancelToken | None = None,
    ) -> BookProject:
        token = cancel_token or CancelToken()
        callback = progress_callback or (lambda event: None)
        total_units = sum(max(1, len(chapter.sections)) for chapter in project.chapters)
        completed_units = 0

        callback(
            GenerationEvent(
                kind="status",
                message_key="log.book_started",
                total=total_units,
            )
        )

        for chapter in project.chapters:
            token.raise_if_cancelled()
            chapter_start_units = completed_units
            chapter.status = "generating"
            chapter.error = None
            callback(
                GenerationEvent(
                    kind="chapter",
                    message_key="log.chapter_started",
                    params={"number": chapter.number, "title": chapter.title},
                    current=completed_units,
                    total=total_units,
                )
            )

            try:
                if chapter.sections:
                    section_words = max(
                        200,
                        project.settings.words_per_chapter // len(chapter.sections),
                    )
                    for section in chapter.sections:
                        token.raise_if_cancelled()
                        section.status = "generating"
                        section.error = None
                        callback(
                            GenerationEvent(
                                kind="section",
                                message_key="log.section_started",
                                params={
                                    "chapter": chapter.number,
                                    "section": section.title,
                                },
                                current=completed_units,
                                total=total_units,
                            )
                        )
                        result = self._generate_section(
                            project,
                            chapter,
                            section,
                            section_words,
                            token,
                        )
                        section.content = result.text
                        section.status = "completed"
                        completed_units += 1
                        callback(
                            GenerationEvent(
                                kind="progress",
                                message_key="log.section_completed",
                                params={
                                    "section": section.title,
                                    "words": result.words,
                                    "tokens": result.tokens,
                                    "total_tokens": (
                                        project.token_usage.total_tokens
                                        + self.usage.total_tokens
                                    ),
                                },
                                current=completed_units,
                                total=total_units,
                            )
                        )
                    chapter.content = None
                else:
                    values = self._common_prompt_values(project, chapter)
                    values["target_words"] = project.settings.words_per_chapter
                    result = self._invoke_prompt(build_chapter_prompt(), values)
                    chapter.content = result.text
                    completed_units += 1
                    callback(
                        GenerationEvent(
                            kind="progress",
                            message_key="log.chapter_completed",
                            params={
                                "number": chapter.number,
                                "words": result.words,
                                "tokens": result.tokens,
                                "total_tokens": (
                                    project.token_usage.total_tokens
                                    + self.usage.total_tokens
                                ),
                            },
                            current=completed_units,
                            total=total_units,
                        )
                    )
                chapter.status = "completed"
            except GenerationCancelled:
                chapter.status = "pending"
                raise
            except Exception as exc:
                chapter.status = "failed"
                chapter.error = str(exc)
                for section in chapter.sections:
                    if section.status == "generating":
                        section.status = "failed"
                        section.error = str(exc)
                chapter_units = max(1, len(chapter.sections))
                completed_in_chapter = completed_units - chapter_start_units
                completed_units += max(0, chapter_units - completed_in_chapter)
                callback(
                    GenerationEvent(
                        kind="error",
                        message_key="log.chapter_failed",
                        params={"number": chapter.number, "error": str(exc)},
                        current=min(completed_units, total_units),
                        total=total_units,
                    )
                )

        failed_chapters = sum(
            chapter.status == "failed" for chapter in project.chapters
        )
        callback(
            GenerationEvent(
                kind="status",
                message_key=(
                    "log.book_completed_with_errors"
                    if failed_chapters
                    else "log.book_completed"
                ),
                params={
                    "failed": failed_chapters,
                    "total_tokens": (
                        project.token_usage.total_tokens
                        + self.usage.total_tokens
                    ),
                },
                current=total_units,
                total=total_units,
            )
        )
        return project
