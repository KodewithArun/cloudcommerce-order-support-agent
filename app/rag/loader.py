"""Load and chunk Markdown policy documents."""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

_HEADERS = [("#", "title"), ("##", "section")]


def load_policy_documents(
    directory: str | Path,
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Document]:
    """Load Markdown policies and split them into cited chunks."""
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Policy directory not found: {directory}")

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=_HEADERS,
        strip_headers=False,
    )
    chunker = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    documents: list[Document] = []
    for path in sorted(directory.glob("*.md")):
        raw_documents = TextLoader(str(path), encoding="utf-8").load()
        for raw_document in raw_documents:
            sections = header_splitter.split_text(raw_document.page_content)
            fallback_title = _title_from_filename(path)
            for section in sections:
                title = section.metadata.get("title") or fallback_title
                section.metadata.update(
                    {
                        "source": path.name,
                        "title": title,
                        "section": section.metadata.get("section") or title,
                    }
                )
            documents.extend(chunker.split_documents(sections))

    return documents


def _title_from_filename(path: Path) -> str:
    return path.stem.replace("-", " ").replace("_", " ").title()
