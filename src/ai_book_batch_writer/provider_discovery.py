"""Provider connection checks and model discovery using official REST APIs."""

from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from ai_book_batch_writer.config import load_environment
from ai_book_batch_writer.llm_providers import (
    get_provider_spec,
    provider_api_key,
)
from ai_book_batch_writer.models import LLMSettings

REQUEST_TIMEOUT_SECONDS = 30


class ProviderDiscoveryError(RuntimeError):
    """Raised when a provider cannot be reached or rejects a request."""


def test_provider_connection(settings: LLMSettings) -> None:
    """Validate credentials or local connectivity without generating text."""
    load_environment()
    if settings.provider == "openrouter":
        _request_json(
            "https://openrouter.ai/api/v1/key",
            headers=_bearer_headers(_required_api_key(settings)),
        )
        return
    list_provider_models(settings)


def list_provider_models(settings: LLMSettings) -> list[str]:
    """Fetch model identifiers available from the selected provider."""
    load_environment()
    provider = settings.provider

    if provider == "openai":
        payload = _request_json(
            "https://api.openai.com/v1/models",
            headers=_bearer_headers(_required_api_key(settings)),
        )
        models = _ids_from_items(payload.get("data"), "id")
        return [model for model in models if _is_openai_text_model(model)]

    if provider == "openrouter":
        payload = _request_json(
            "https://openrouter.ai/api/v1/models",
            headers={
                **_bearer_headers(_required_api_key(settings)),
                "HTTP-Referer": "https://github.com/estebanstifli/ai-book-batch-writer",
                "X-Title": "AI Book Batch Writer",
            },
        )
        return _ids_from_items(payload.get("data"), "id")

    if provider == "anthropic":
        payload = _request_json(
            "https://api.anthropic.com/v1/models?limit=1000",
            headers={
                "x-api-key": _required_api_key(settings),
                "anthropic-version": "2023-06-01",
            },
        )
        return _ids_from_items(payload.get("data"), "id")

    if provider == "gemini":
        payload = _request_json(
            "https://generativelanguage.googleapis.com/v1beta/models?"
            + urlencode({"pageSize": 1000}),
            headers={"x-goog-api-key": _required_api_key(settings)},
        )
        items = payload.get("models")
        if not isinstance(items, list):
            return []
        models: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            methods = item.get("supportedGenerationMethods", [])
            if "generateContent" not in methods:
                continue
            name = item.get("baseModelId") or item.get("name")
            if isinstance(name, str):
                models.append(name.removeprefix("models/"))
        return _normalize_models(models)

    if provider == "ollama":
        base_url = (
            settings.api_base
            or get_provider_spec("ollama").default_api_base
            or "http://localhost:11434"
        ).rstrip("/")
        payload = _request_json(f"{base_url}/api/tags")
        items = payload.get("models")
        if not isinstance(items, list):
            return []
        models = [
            str(item.get("model") or item.get("name"))
            for item in items
            if isinstance(item, dict) and (item.get("model") or item.get("name"))
        ]
        return _normalize_models(models)

    raise ProviderDiscoveryError(f"Unsupported provider: {provider}")


def _required_api_key(settings: LLMSettings) -> str:
    api_key = provider_api_key(settings)
    if not api_key:
        raise ProviderDiscoveryError(
            f"API key is required for {settings.provider}."
        )
    return api_key


def _bearer_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _request_json(
    url: str,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "AI-Book-Batch-Writer/0.4",
            **(headers or {}),
        },
        method="GET",
    )
    try:
        with urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = _http_error_detail(exc)
        raise ProviderDiscoveryError(
            f"Provider returned HTTP {exc.code}: {detail}"
        ) from exc
    except URLError as exc:
        raise ProviderDiscoveryError(
            f"Could not connect to the provider: {exc.reason}"
        ) from exc
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ProviderDiscoveryError(
            "The provider returned an invalid JSON response."
        ) from exc
    if not isinstance(payload, dict):
        raise ProviderDiscoveryError(
            "The provider returned an unexpected response."
        )
    return payload


def _http_error_detail(error: HTTPError) -> str:
    try:
        payload = json.loads(error.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        return error.reason or "request failed"
    if not isinstance(payload, dict):
        return error.reason or "request failed"
    detail = payload.get("error")
    if isinstance(detail, dict):
        return str(detail.get("message") or detail.get("status") or error.reason)
    return str(detail or payload.get("message") or error.reason)


def _ids_from_items(items: Any, key: str) -> list[str]:
    if not isinstance(items, list):
        return []
    return _normalize_models(
        [
            str(item[key])
            for item in items
            if isinstance(item, dict) and item.get(key)
        ]
    )


def _normalize_models(models: list[str]) -> list[str]:
    return sorted(
        {model.strip() for model in models if model.strip()},
        key=str.casefold,
    )


def _is_openai_text_model(model: str) -> bool:
    lowered = model.lower()
    allowed_prefixes = ("gpt-", "o1", "o3", "o4", "chatgpt-")
    excluded_fragments = (
        "audio",
        "image",
        "realtime",
        "search",
        "transcribe",
        "tts",
    )
    return lowered.startswith(allowed_prefixes) and not any(
        fragment in lowered for fragment in excluded_fragments
    )
