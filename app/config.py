"""
Centralised application settings loaded from environment / .env file.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# ── Provider enums ────────────────────────────────────────────────────


class LLMProvider(StrEnum):
    """Supported LLM back-ends."""

    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class EmbeddingProvider(StrEnum):
    """Supported embedding back-ends."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    FAKE = "fake"  # random offline vectors for CI / tests


# ── Settings ──────────────────────────────────────────────────────────


class Settings(BaseSettings):
    """Application-wide configuration.

    Values are read (in priority order) from:
    1. Explicit kwargs at construction time
    2. Environment variables
    3. The ``.env`` file in the project root
    4. Field defaults below
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ───────────────────────────────────────────────────────
    app_env: str = "development"
    log_level: str = "INFO"

    # ── LLM ───────────────────────────────────────────────────────────
    llm_provider: LLMProvider = LLMProvider.GROQ

    # Groq
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"

    # OpenAI
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o"

    # Anthropic
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # Shared LLM params
    llm_temperature: float = 0
    llm_max_retries: int = 2
    llm_timeout_seconds: int = 30

    # ── Embeddings ────────────────────────────────────────────────────
    embedding_provider: EmbeddingProvider = EmbeddingProvider.HUGGINGFACE

    # HuggingFace (sentence-transformers — runs locally, no API key)
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hf_device: str = "cpu"

    # OpenAI embeddings
    openai_embedding_model: str = "text-embedding-3-small"

    # Fake / offline embeddings (for tests)
    fake_embedding_size: int = 384

    # ── RAG ────────────────────────────────────────────────────────────
    rag_chunk_size: int = 700
    rag_chunk_overlap: int = 120
    rag_top_k: int = 4
    rag_min_score: float = 0.35
    chroma_persist_path: str = "data/chroma"
    chroma_collection_name: str = "policies"

    # ── Business policy ───────────────────────────────────────────────
    return_window_days: int = 14
    refund_approval_threshold: float = 100.0
    reference_date: str | None = None  # ISO date; empty = system date

    # ── Storage ───────────────────────────────────────────────────────
    database_path: str = "data/agent.sqlite3"
    tool_log_path: str = "logs/tool_calls.jsonl"
    policy_docs_path: str = "data/policies"
    orders_path: str = "data/orders.json"

    # ── Orders service resilience ─────────────────────────────────────
    orders_simulate_failure: str | None = None  # "timeout" | "error" | None
    orders_timeout_seconds: float = 2.0
    orders_max_retries: int = 2


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance (reads .env once)."""
    return Settings()
