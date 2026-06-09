from __future__ import annotations

import pytest

from ai_book_batch_writer.llm_providers import (
    PROVIDER_SPECS,
    create_chat_model,
    provider_requires_api_key,
    provider_supports_api_base,
)
from ai_book_batch_writer.models import LLMSettings


@pytest.mark.parametrize(
    ("provider", "model", "class_name"),
    [
        ("openai", "gpt-4o-mini", "ChatOpenAI"),
        ("openrouter", "openai/gpt-4o-mini", "ChatOpenRouter"),
        ("anthropic", "claude-sonnet-4-6", "ChatAnthropic"),
        ("gemini", "gemini-2.5-flash", "ChatGoogleGenerativeAI"),
        ("ollama", "llama3.1", "ChatOllama"),
    ],
)
def test_all_provider_adapters_construct_without_network_calls(
    provider: str,
    model: str,
    class_name: str,
) -> None:
    settings = LLMSettings(
        provider=provider,
        model=model,
        api_key=None if provider == "ollama" else "test-key",
    )
    assert type(create_chat_model(settings)).__name__ == class_name


def test_only_ollama_exposes_api_base() -> None:
    assert set(PROVIDER_SPECS) == {
        "openai",
        "openrouter",
        "anthropic",
        "gemini",
        "ollama",
    }
    assert provider_supports_api_base("ollama") is True
    assert provider_requires_api_key("ollama") is False

    for provider in {"openai", "openrouter", "anthropic", "gemini"}:
        assert provider_supports_api_base(provider) is False
        assert provider_requires_api_key(provider) is True
