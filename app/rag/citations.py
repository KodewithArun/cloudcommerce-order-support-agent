"""Render policy citations for answers."""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document


def citation_for(document: Document) -> str:
    """Return the source and section for a policy chunk."""
    source = document.metadata.get("source", "unknown")
    section = document.metadata.get("section") or document.metadata.get("title", "General")
    return f"{source} › {section}"


def format_citations(documents: Sequence[Document]) -> str:
    """Render one Markdown citation line per retrieved document."""
    return "\n".join(f"- {citation_for(document)}" for document in documents)
