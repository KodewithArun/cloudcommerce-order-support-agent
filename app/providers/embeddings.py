"""
Embedding providers built on the official LangChain integrations.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_core.embeddings import Embeddings, FakeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from app.config import EmbeddingProvider, Settings, get_settings
from app.providers._common import require_api_key

EmbeddingBuilder = Callable[[Settings], Embeddings]


def _build_huggingface(settings: Settings) -> Embeddings:
    return HuggingFaceEmbeddings(
        model_name=settings.hf_embedding_model,
        model_kwargs={"device": settings.hf_device},
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_openai(settings: Settings) -> Embeddings:
    return OpenAIEmbeddings(
        model=settings.openai_embedding_model,
        api_key=require_api_key(settings.openai_api_key, "OPENAI_API_KEY"),
    )


def _build_fake(settings: Settings) -> Embeddings:
    return FakeEmbeddings(size=settings.fake_embedding_size)


_BUILDERS: dict[EmbeddingProvider, EmbeddingBuilder] = {
    EmbeddingProvider.HUGGINGFACE: _build_huggingface,
    EmbeddingProvider.OPENAI: _build_openai,
    EmbeddingProvider.FAKE: _build_fake,
}


def build_embeddings(settings: Settings) -> Embeddings:
    """Build an embeddings client for ``settings.embedding_provider``."""
    try:
        builder = _BUILDERS[settings.embedding_provider]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported embedding provider: {settings.embedding_provider!r}"
        ) from exc
    return builder(settings)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """Return the shared embeddings client configured from the environment."""
    return build_embeddings(get_settings())
