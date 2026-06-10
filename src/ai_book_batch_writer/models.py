"""Validated domain models used by the UI and generation services."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

ProviderName = Literal[
    "openai",
    "openrouter",
    "anthropic",
    "gemini",
    "ollama",
]
GenerationStatus = Literal["pending", "generating", "completed", "failed"]


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class LLMSettings(BaseModel):
    """Configuration for a supported chat model provider."""

    provider: ProviderName
    model: str = Field(min_length=1)
    api_key: str | None = Field(default=None, repr=False, exclude=True)
    api_base: str | None = None
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=8096, ge=128, le=128_000)

    @field_validator("model")
    @classmethod
    def strip_model(cls, value: str) -> str:
        return value.strip()

    @field_validator("api_key", "api_base")
    @classmethod
    def blank_to_none(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class BookSettings(BaseModel):
    """User-defined writing goals for a book project."""

    title: str | None = None
    idea: str = Field(min_length=1)
    output_language: str = Field(default="English", min_length=1)
    tone: str = Field(default="Clear and professional", min_length=1)
    target_audience: str = Field(default="General readers", min_length=1)
    chapter_count: int = Field(default=8, ge=1, le=100)
    words_per_chapter: int = Field(default=1200, ge=100, le=20_000)
    additional_instructions: str | None = None

    @field_validator(
        "title",
        "idea",
        "output_language",
        "tone",
        "target_audience",
        "additional_instructions",
    )
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        return stripped or None


class BookSection(BaseModel):
    """A section within a generated chapter."""

    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    content: str | None = None
    status: GenerationStatus = "pending"
    error: str | None = None


class BookChapter(BaseModel):
    """A chapter and its optional section breakdown."""

    number: int = Field(ge=1)
    title: str = Field(min_length=1)
    summary: str = Field(min_length=1)
    sections: list[BookSection] = Field(default_factory=list)
    content: str | None = None
    status: GenerationStatus = "pending"
    error: str | None = None


class BookOutline(BaseModel):
    """Structured model output for the editable outline."""

    title: str = Field(min_length=1)
    subtitle: str | None = None
    chapters: list[BookChapter] = Field(min_length=1)


class TokenUsage(BaseModel):
    """Persisted token usage for outline and book generation."""

    outline_tokens: int = Field(default=0, ge=0)
    book_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    estimated: bool = False

    @model_validator(mode="after")
    def synchronize_total(self) -> "TokenUsage":
        self.total_tokens = self.outline_tokens + self.book_tokens
        return self

    def add_outline(self, tokens: int, estimated: bool = False) -> None:
        self.outline_tokens += max(0, tokens)
        self.total_tokens += max(0, tokens)
        self.estimated = self.estimated or estimated

    def add_book(self, tokens: int, estimated: bool = False) -> None:
        self.book_tokens += max(0, tokens)
        self.total_tokens += max(0, tokens)
        self.estimated = self.estimated or estimated

    def merge(self, other: "TokenUsage") -> None:
        self.outline_tokens += other.outline_tokens
        self.book_tokens += other.book_tokens
        self.total_tokens += other.total_tokens
        self.estimated = self.estimated or other.estimated


class BookProject(BaseModel):
    """Complete serializable state of a writing project."""

    model_config = ConfigDict(validate_assignment=True)

    schema_version: int = 2
    settings: BookSettings
    llm_settings: LLMSettings | None = None
    subtitle: str | None = None
    chapters: list[BookChapter] = Field(default_factory=list)
    token_usage: TokenUsage = Field(default_factory=TokenUsage)
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)

    def touch(self) -> None:
        """Update the modification timestamp."""
        self.updated_at = utc_now()


class GenerationEvent(BaseModel):
    """Provider-agnostic progress event emitted by generation services."""

    kind: Literal["status", "progress", "chapter", "section", "error"]
    message_key: str
    params: dict[str, Any] = Field(default_factory=dict)
    current: int = 0
    total: int = 0

    @property
    def fraction(self) -> float:
        if self.total <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current / self.total))
