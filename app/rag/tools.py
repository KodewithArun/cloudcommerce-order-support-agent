"""LangChain tool for searching policy documents."""

from __future__ import annotations

from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import tool

from app.rag.chain import UNKNOWN_ANSWER
from app.rag.citations import citation_for
from app.rag.store import get_policy_retriever


def search_policy_text(query: str, retriever: BaseRetriever) -> str:
    """Return cited policy excerpts, or an explicit no-match instruction."""
    documents = retriever.invoke(query)
    if not documents:
        return f"No matching policy found. Reply with exactly: {UNKNOWN_ANSWER}"
    return "\n\n".join(
        f"[{citation_for(document)}]\n{document.page_content.strip()}" for document in documents
    )


@tool
def search_policies(query: str) -> str:
    """Search CloudCommerce policy documents for returns, refunds, cancellations,
    shipping or warranty information. Cite the returned source and section.
    """
    return search_policy_text(query, get_policy_retriever())
