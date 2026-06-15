"""Vector index creation and loading."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import RAGSettings, get_settings
from .loaders import load_documents, split_documents


def create_embeddings(settings: RAGSettings) -> Any:
    """Create the OpenAI embedding client used by Chroma."""

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def is_index_ready(persist_dir: Path) -> bool:
    """Return whether the local Chroma index directory appears initialized."""

    return persist_dir.exists() and any(persist_dir.iterdir())


def build_vector_store(settings: RAGSettings | None = None, force_rebuild: bool = False) -> tuple[Any, int, int]:
    """Build or rebuild the Chroma vector store from sample documents."""

    settings = settings or get_settings()
    if force_rebuild and settings.persist_dir.exists():
        shutil.rmtree(settings.persist_dir)

    documents = load_documents(settings.data_dir)
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    from langchain_chroma import Chroma

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=create_embeddings(settings),
        persist_directory=str(settings.persist_dir),
        collection_name=settings.collection_name,
    )
    if hasattr(vector_store, "persist"):
        vector_store.persist()
    return vector_store, len(documents), len(chunks)


def load_vector_store(settings: RAGSettings | None = None) -> Any:
    """Load an existing Chroma vector store."""

    settings = settings or get_settings()
    if not is_index_ready(settings.persist_dir):
        raise FileNotFoundError("Vector index is not ready. Build it first.")

    from langchain_chroma import Chroma

    return Chroma(
        persist_directory=str(settings.persist_dir),
        embedding_function=create_embeddings(settings),
        collection_name=settings.collection_name,
    )


def get_or_create_vector_store(settings: RAGSettings | None = None) -> tuple[Any, bool]:
    """Load the index if present, otherwise create it."""

    settings = settings or get_settings()
    if is_index_ready(settings.persist_dir):
        return load_vector_store(settings), False
    vector_store, _, _ = build_vector_store(settings)
    return vector_store, True

