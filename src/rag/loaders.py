"""Document loading and chunking utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT

SUPPORTED_SUFFIXES = {".md", ".txt"}


@dataclass
class SimpleDocument:
    """Small fallback document type used when LangChain is not installed."""

    page_content: str
    metadata: dict[str, Any]


def _document_type() -> type:
    try:
        from langchain_core.documents import Document

        return Document
    except ImportError:
        return SimpleDocument


def _relative_source(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_documents(data_dir: Path) -> list[Any]:
    """Load Markdown and text files from the knowledge-base directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {data_dir}")

    document_type = _document_type()
    documents: list[Any] = []
    for file_path in sorted(data_dir.rglob("*")):
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        text = file_path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        documents.append(
            document_type(
                page_content=text,
                metadata={
                    "source": _relative_source(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower().lstrip("."),
                },
            )
        )

    if not documents:
        raise FileNotFoundError(f"No supported documents found in: {data_dir}")
    return documents


def split_documents(
    documents: Iterable[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Any]:
    """Split documents while preserving source metadata."""

    document_list = list(documents)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "；", ";", " ", ""],
        )
        chunks = splitter.split_documents(document_list)
    except ImportError:
        chunks = _fallback_split_documents(document_list, chunk_size, chunk_overlap)

    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata)
        chunk.metadata.setdefault("chunk_id", index)
    return chunks


def _fallback_split_documents(
    documents: list[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[SimpleDocument]:
    """Dependency-light splitter for local tests before LangChain is installed."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    overlap = max(0, min(chunk_overlap, chunk_size - 1))
    step = chunk_size - overlap

    chunks: list[SimpleDocument] = []
    for document in documents:
        content = document.page_content
        metadata = dict(document.metadata)
        for start in range(0, len(content), step):
            text = content[start : start + chunk_size].strip()
            if not text:
                continue
            chunks.append(SimpleDocument(page_content=text, metadata=metadata.copy()))
            if start + chunk_size >= len(content):
                break
    return chunks

