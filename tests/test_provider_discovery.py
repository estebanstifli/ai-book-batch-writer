from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from ai_book_batch_writer.models import LLMSettings
from ai_book_batch_writer.provider_discovery import (
    ProviderDiscoveryError,
    list_provider_models,
    test_provider_connection as check_provider_connection,
)


class FakeResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


@pytest.mark.parametrize(
    ("settings", "payload", "expected"),
    [
        (
            LLMSettings(
                provider="openai",
                model="gpt-4o-mini",
                api_key="test",
            ),
            {
                "data": [
                    {"id": "gpt-4o-mini"},
                    {"id": "text-embedding-3-small"},
                    {"id": "gpt-realtime"},
                ]
            },
            ["gpt-4o-mini"],
        ),
        (
            LLMSettings(
                provider="anthropic",
                model="claude-sonnet-4-6",
                api_key="test",
            ),
            {"data": [{"id": "claude-sonnet-4-6"}]},
            ["claude-sonnet-4-6"],
        ),
        (
            LLMSettings(
                provider="gemini",
                model="gemini-2.5-flash",
                api_key="test",
            ),
            {
                "models": [
                    {
                        "name": "models/gemini-2.5-flash",
                        "baseModelId": "gemini-2.5-flash",
                        "supportedGenerationMethods": ["generateContent"],
                    },
                    {
                        "name": "models/text-embedding-004",
                        "supportedGenerationMethods": ["embedContent"],
                    },
                ]
            },
            ["gemini-2.5-flash"],
        ),
        (
            LLMSettings(
                provider="ollama",
                model="llama3.1",
                api_base="http://localhost:11434",
            ),
            {"models": [{"model": "llama3.1:latest"}]},
            ["llama3.1:latest"],
        ),
    ],
)
def test_list_provider_models_parses_catalogs(
    settings: LLMSettings,
    payload: dict,
    expected: list[str],
) -> None:
    with patch(
        "ai_book_batch_writer.provider_discovery.urlopen",
        return_value=FakeResponse(payload),
    ):
        assert list_provider_models(settings) == expected


def test_openrouter_connection_uses_key_endpoint() -> None:
    settings = LLMSettings(
        provider="openrouter",
        model="openai/gpt-4o-mini",
        api_key="test",
    )
    with patch(
        "ai_book_batch_writer.provider_discovery.urlopen",
        return_value=FakeResponse({"data": {"label": "test"}}),
    ) as mocked:
        check_provider_connection(settings)

    assert mocked.call_args.args[0].full_url.endswith("/api/v1/key")


def test_discovery_requires_api_key() -> None:
    settings = LLMSettings(provider="openai", model="gpt-4o-mini")

    with patch(
        "ai_book_batch_writer.provider_discovery.provider_api_key",
        return_value=None,
    ):
        with pytest.raises(ProviderDiscoveryError, match="API key"):
            list_provider_models(settings)
