"""Offline tests for the provider factories (no network calls)."""

from __future__ import annotations

import pytest
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

from app.config import EmbeddingProvider, LLMProvider, Settings
from app.providers import build_chat_model, build_embeddings


@pytest.mark.parametrize(
    ("provider", "expected_type"),
    [
        (LLMProvider.GROQ, ChatGroq),
        (LLMProvider.OPENAI, ChatOpenAI),
        (LLMProvider.ANTHROPIC, ChatAnthropic),
    ],
)
def test_build_chat_model_dispatches_to_provider(
    provider: LLMProvider,
    expected_type: type,
) -> None:
    settings = Settings(
        llm_provider=provider,
        llm_temperature=0,
        **{f"{provider.value}_api_key": "test-key"},
    )

    assert isinstance(build_chat_model(settings), expected_type)


def test_build_chat_model_requires_api_key() -> None:
    settings = Settings(llm_provider=LLMProvider.GROQ, groq_api_key=None)

    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        build_chat_model(settings)


def test_build_chat_model_rejects_unknown_provider() -> None:
    settings = Settings.model_construct(llm_provider="unknown")

    with pytest.raises(ValueError, match="Unsupported LLM provider"):
        build_chat_model(settings)


def test_build_embeddings_fake_is_offline() -> None:
    settings = Settings(embedding_provider=EmbeddingProvider.FAKE, fake_embedding_size=8)
    embeddings = build_embeddings(settings)

    vectors = embeddings.embed_documents(["first document", "second document"])

    assert len(vectors) == 2
    assert all(len(vector) == 8 for vector in vectors)
    assert len(embeddings.embed_query("query")) == 8


def test_build_embeddings_rejects_unknown_provider() -> None:
    settings = Settings.model_construct(embedding_provider="unknown")

    with pytest.raises(ValueError, match="Unsupported embedding provider"):
        build_embeddings(settings)
