"""
Chat-model providers built on the official integrations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Any

from langchain_anthropic import ChatAnthropic
from langchain_core.language_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import LLMProvider, Settings, get_settings
from app.providers._common import require_api_key

ChatModelBuilder = Callable[[Settings], BaseChatModel]


def _tuning(settings: Settings) -> dict[str, Any]:
    """Provider-agnostic options shared by all chat models."""
    return {
        "temperature": settings.llm_temperature,
        "max_retries": settings.llm_max_retries,
        "timeout": settings.llm_timeout_seconds,
    }


def _build_groq(settings: Settings) -> BaseChatModel:
    return ChatGroq(
        model=settings.groq_model,
        api_key=require_api_key(settings.groq_api_key, "GROQ_API_KEY"),
        **_tuning(settings),
    )


def _build_openai(settings: Settings) -> BaseChatModel:
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=require_api_key(settings.openai_api_key, "OPENAI_API_KEY"),
        **_tuning(settings),
    )


def _build_anthropic(settings: Settings) -> BaseChatModel:
    return ChatAnthropic(
        model=settings.anthropic_model,
        api_key=require_api_key(settings.anthropic_api_key, "ANTHROPIC_API_KEY"),
        **_tuning(settings),
    )


_BUILDERS: dict[LLMProvider, ChatModelBuilder] = {
    LLMProvider.GROQ: _build_groq,
    LLMProvider.OPENAI: _build_openai,
    LLMProvider.ANTHROPIC: _build_anthropic,
}


def build_chat_model(settings: Settings) -> BaseChatModel:
    """Build a chat model for ``settings.llm_provider``."""
    try:
        builder = _BUILDERS[settings.llm_provider]
    except KeyError as exc:
        raise ValueError(f"Unsupported LLM provider: {settings.llm_provider!r}") from exc
    return builder(settings)


@lru_cache(maxsize=1)
def get_chat_model() -> BaseChatModel:
    """Return the shared chat model configured from the environment."""
    return build_chat_model(get_settings())
