"""Vector index creation and loading.

学习提示：
这个文件对应 RAG 中的“建库”阶段：
原始文档/上传文档 -> 文本切分 -> Embedding 向量化 -> 写入 Chroma 向量库。
问答时不应该每次都重新建库，所以索引会持久化到 vectorstore/。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .config import RAGSettings, get_settings
from .loaders import load_documents_from_dirs, split_documents


def create_embeddings(settings: RAGSettings) -> Any:
    """Create the OpenAI embedding client used by Chroma.

    Embedding 模型负责把文本片段转成向量。检索时，用户问题也会被转成
    向量，再和库里的片段向量比较相似度。
    """

    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )


def is_index_ready(persist_dir: Path) -> bool:
    """Return whether the local Chroma index directory appears initialized."""

    return persist_dir.exists() and any(persist_dir.iterdir())


def build_vector_store(settings: RAGSettings | None = None, force_rebuild: bool = False) -> tuple[Any, int, int]:
    """Build or rebuild the Chroma vector store from sample and uploaded documents."""

    settings = settings or get_settings()
    if force_rebuild and settings.persist_dir.exists():
        # 重建索引时删除旧向量库，确保本地知识库变更能被重新写入。
        shutil.rmtree(settings.persist_dir)

    # 1. 读取知识库原文：内置示例 + 用户上传目录。
    documents = load_documents_from_dirs(settings.knowledge_dirs)
    # 2. 切成适合检索的小片段。
    chunks = split_documents(
        documents,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    from langchain_chroma import Chroma

    # 3. 对每个 chunk 做 Embedding，并写入本地 Chroma。
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
    """Load an existing Chroma vector store.

    注意：加载已有索引仍然需要 embedding_function，因为检索时要把用户问题
    转成同一向量空间中的向量。
    """

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
    """Load the index if present, otherwise create it.

    返回值里的 bool 表示这次是否新建了索引，页面可以据此提示用户。
    """

    settings = settings or get_settings()
    if is_index_ready(settings.persist_dir):
        return load_vector_store(settings), False
    vector_store, _, _ = build_vector_store(settings)
    return vector_store, True
