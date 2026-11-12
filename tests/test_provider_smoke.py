"""Simple smoke test for the configured LLM and embedding providers.

Run with ``pytest tests/test_provider_smoke.py -s`` to see both outputs.
"""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.providers import get_chat_model, get_embeddings

settings = get_settings()
has_api_key = bool(getattr(settings, f"{settings.llm_provider.value}_api_key"))


@pytest.mark.skipif(
    not has_api_key,
    reason=f"No API key configured for {settings.llm_provider!r}",
)
def test_llm_provider() -> None:
    answer = (
        get_chat_model()
        .invoke("In one short sentence, what does a customer support agent do?")
        .content
    )

    print(f"\n[LLM] {answer}")
    assert isinstance(answer, str) and answer.strip()


def test_embedding_provider() -> None:
    vector = get_embeddings().embed_query("CloudCommerce handles customer order support.")

    print(f"\n[EMBEDDING] dimensions={len(vector)} first5={[round(x, 4) for x in vector[:5]]}")
    assert len(vector) > 0
