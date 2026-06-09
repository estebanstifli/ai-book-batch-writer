"""Factory for supported LangChain chat model integrations."""

from __future__ import annotations

import os
from dataclasses import dataclass

from langchain_core.language_models.chat_models import BaseChatModel

from ai_book_batch_writer.config import load_environment
from ai_book_batch_writer.models import LLMSettings


@dataclass(frozen=True)
class ProviderSpec:
    """UI and credential metadata for one chat model provider."""

    code: str
    default_model: str
    api_key_env_vars: tuple[str, ...] = ()
    supports_api_base: bool = False
    default_api_base: str | None = None

    @property
    def requires_api_key(self) -> bool:
        return bool(self.api_key_env_vars)


PROVIDER_SPECS: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        code="openai",
        default_model="gpt-4o-mini",
        api_key_env_vars=("OPENAI_API_KEY",),
    ),
    "openrouter": ProviderSpec(
        code="openrouter",
        default_model="openai/gpt-4o-mini",
        api_key_env_vars=("OPENROUTER_API_KEY",),
    ),
    "anthropic": ProviderSpec(
        code="anthropic",
        default_model="claude-sonnet-4-6",
        api_key_env_vars=("ANTHROPIC_API_KEY",),
    ),
    "gemini": ProviderSpec(
        code="gemini",
        default_model="gemini-2.5-flash",
        api_key_env_vars=("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    ),
    "ollama": ProviderSpec(
        code="ollama",
        default_model="llama3.1",
        supports_api_base=True,
        default_api_base="http://localhost:11434",
    ),
}


def get_provider_spec(provider: str) -> ProviderSpec:
    """Return metadata for a supported provider."""
    try:
        return PROVIDER_SPECS[provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported provider: {provider}") from exc


def provider_api_key(settings: LLMSettings) -> str | None:
    """Resolve a session key or provider-specific environment variable."""
    if settings.api_key:
        return settings.api_key
    spec = get_provider_spec(settings.provider)
    return next(
        (os.getenv(name) for name in spec.api_key_env_vars if os.getenv(name)),
        None,
    )


def create_chat_model(settings: LLMSettings) -> BaseChatModel:
    """Create a LangChain chat model for the selected provider."""
    load_environment()

    api_key = provider_api_key(settings)

    if settings.provider == "openai":
        from langchain_openai import ChatOpenAI

        if not api_key:
            raise ValueError("OpenAI API key is required.")

        return ChatOpenAI(
            model=settings.model,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_retries=0,
            timeout=180,
        )

    if settings.provider == "openrouter":
        from langchain_openrouter import ChatOpenRouter

        if not api_key:
            raise ValueError("OpenRouter API key is required.")
        return ChatOpenRouter(
            model=settings.model,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_retries=0,
            timeout=180,
            app_title="AI Book Batch Writer",
        )

    if settings.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not api_key:
            raise ValueError("Anthropic API key is required.")
        return ChatAnthropic(
            model=settings.model,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            max_retries=0,
            timeout=180,
        )

    if settings.provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        if not api_key:
            raise ValueError("Google Gemini API key is required.")
        return ChatGoogleGenerativeAI(
            model=settings.model,
            api_key=api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            retries=0,
            request_timeout=180,
        )

    if settings.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=settings.model,
            base_url=(
                settings.api_base
                or os.getenv("OLLAMA_BASE_URL")
                or get_provider_spec("ollama").default_api_base
            ),
            temperature=settings.temperature,
            num_predict=settings.max_tokens,
        )

    raise ValueError(f"Unsupported provider: {settings.provider}")


def provider_requires_api_key(provider: str) -> bool:
    """Return whether a provider requires a user credential."""
    return get_provider_spec(provider).requires_api_key


def provider_supports_api_base(provider: str) -> bool:
    """Return whether the provider exposes a configurable API base URL."""
    return get_provider_spec(provider).supports_api_base


def provider_key_env_display(provider: str) -> str:
    """Return the preferred environment variable shown to users."""
    spec = get_provider_spec(provider)
    return " / ".join(spec.api_key_env_vars)
