"""Offline tests for policy loading, retrieval and citations."""

from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from app.config import Settings
from app.rag import (
    UNKNOWN_ANSWER,
    answer_policy_question,
    build_policy_chain,
    build_policy_retriever,
    format_citations,
    load_policy_documents,
)

POLICY_DIR = Path(__file__).resolve().parents[1] / "data" / "policies"


class KeywordEmbeddings(Embeddings):
    """Deterministic bag-of-keywords embeddings for offline tests."""

    _KEYWORDS = ("return", "refund", "cancel", "ship", "warrant", "deliver")

    def _embed(self, text: str) -> list[float]:
        lowered = text.lower()
        return [float(lowered.count(keyword)) for keyword in self._KEYWORDS]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OrthogonalEmbeddings(Embeddings):
    """Documents and queries point in different directions (score 0)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        return [0.0, 1.0]


def _settings(persist_path: Path, *, min_score: float = 0.15) -> Settings:
    return Settings(
        policy_docs_path=str(POLICY_DIR),
        rag_chunk_size=700,
        rag_chunk_overlap=120,
        rag_top_k=6,
        rag_min_score=min_score,
        chroma_persist_path=str(persist_path),
        chroma_collection_name="policy-test",
    )


@pytest.fixture(scope="module")
def retriever(tmp_path_factory: pytest.TempPathFactory):
    persist_path = tmp_path_factory.mktemp("chroma")
    return build_policy_retriever(
        _settings(persist_path),
        embeddings=KeywordEmbeddings(),
    )


def test_load_policy_documents_adds_citation_metadata() -> None:
    documents = load_policy_documents(POLICY_DIR, chunk_size=700, chunk_overlap=120)

    assert documents
    assert {document.metadata["source"] for document in documents} == {
        "cancellations.md",
        "returns.md",
        "shipping.md",
        "warranty.md",
    }
    assert all(document.metadata["title"] for document in documents)
    assert all(document.metadata["section"] for document in documents)
    assert all(document.page_content.strip() for document in documents)


def test_load_policy_documents_rejects_missing_directory() -> None:
    with pytest.raises(FileNotFoundError):
        load_policy_documents(Path("missing-policies"), chunk_size=700, chunk_overlap=120)


def test_returns_question_retrieves_returns_policy(retriever) -> None:
    documents = retriever.invoke("Can I return an item within 14 days of delivery?")

    assert documents
    assert any(document.metadata["source"] == "returns.md" for document in documents)


def test_warranty_question_retrieves_warranty_policy(retriever) -> None:
    documents = retriever.invoke("Is a manufacturing defect covered by the warranty?")

    assert documents
    assert any(document.metadata["source"] == "warranty.md" for document in documents)


def test_refund_question_formats_citations(retriever) -> None:
    documents = retriever.invoke("How long do refunds take?")

    rendered = format_citations(documents)

    assert documents
    assert rendered.splitlines()
    assert all(line.startswith("- ") for line in rendered.splitlines())


def test_uncovered_question_returns_no_documents(tmp_path: Path) -> None:
    retriever = build_policy_retriever(
        _settings(tmp_path / "chroma"),
        embeddings=OrthogonalEmbeddings(),
    )

    assert retriever.invoke("Do you offer student discounts?") == []


def test_build_policy_retriever_uses_settings_and_custom_embeddings(tmp_path: Path) -> None:
    retriever = build_policy_retriever(
        _settings(tmp_path / "chroma"),
        embeddings=KeywordEmbeddings(),
    )

    documents = retriever.invoke("Can I cancel while processing?")

    assert documents
    assert any(document.metadata["source"] == "cancellations.md" for document in documents)


def test_policy_chain_returns_answer_with_context(
    retriever,
    tmp_path: Path,
) -> None:
    chain = build_policy_chain(
        _settings(tmp_path / "chain-chroma"),
        llm=FakeListChatModel(responses=["Refunds take 5-7 business days."]),
        retriever=retriever,
    )

    response = chain.invoke({"input": "How long do refunds take?"})

    assert response["answer"] == "Refunds take 5-7 business days."
    assert response["context"]
    assert any(document.metadata["source"] == "returns.md" for document in response["context"])


class EmptyRetrievalChain:
    def invoke(self, inputs: dict) -> dict:
        return {"input": inputs["input"], "context": [], "answer": "model answer"}


def test_answer_policy_question_falls_back_without_context() -> None:
    response = answer_policy_question(
        "Do you offer student discounts?",
        chain=EmptyRetrievalChain(),
    )

    assert response["answer"] == UNKNOWN_ANSWER
    assert response["context"] == []
