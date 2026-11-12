"""Persistent Chroma policy index and retriever."""

from __future__ import annotations

from functools import lru_cache

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever

from app.config import Settings, get_settings
from app.providers import get_embeddings
from app.rag.loader import load_policy_documents

_COLLECTION_CONFIGURATION = {"hnsw": {"space": "cosine"}}


def build_policy_store(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
) -> Chroma:
    """Open the persistent policy store, indexing documents when empty."""
    store = Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings or get_embeddings(),
        persist_directory=settings.chroma_persist_path,
        collection_configuration=_COLLECTION_CONFIGURATION,
    )
    if not store.get(limit=1)["ids"]:
        store.add_documents(
            load_policy_documents(
                settings.policy_docs_path,
                chunk_size=settings.rag_chunk_size,
                chunk_overlap=settings.rag_chunk_overlap,
            )
        )
    return store


def build_policy_retriever(
    settings: Settings,
    *,
    embeddings: Embeddings | None = None,
) -> VectorStoreRetriever:
    """Build a score-threshold retriever over the policy store."""
    return build_policy_store(settings, embeddings=embeddings).as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "k": settings.rag_top_k,
            "score_threshold": settings.rag_min_score,
        },
    )


@lru_cache(maxsize=1)
def get_policy_retriever() -> VectorStoreRetriever:
    """Return the shared policy retriever configured from the environment."""
    return build_policy_retriever(get_settings())
