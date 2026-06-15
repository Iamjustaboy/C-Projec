"""Configuration helpers for the RAG demo.

学习提示：
RAG 项目通常有很多可调参数，例如模型名称、Embedding 模型、切分大小、
检索 Top-K、向量库目录、上传目录。把这些参数集中放在配置层，可以避免业务代码里
到处出现硬编码。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is installed in the app env.
    def load_dotenv(*_: object, **__: object) -> bool:
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RAGSettings:
    """Runtime settings loaded from environment variables.

    data_dir: 原始知识库文档目录。
    upload_dir: 用户通过页面上传的知识库文档目录。
    persist_dir: Chroma 向量库持久化目录。
    top_k: 每次提问时召回的知识片段数量。
    chunk_size/chunk_overlap: 文本切分大小和重叠长度。
    """

    data_dir: Path
    upload_dir: Path
    persist_dir: Path
    collection_name: str
    openai_api_key: str
    openai_model: str
    embedding_model: str
    top_k: int
    chunk_size: int
    chunk_overlap: int
    temperature: float

    @property
    def has_api_key(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def knowledge_dirs(self) -> list[Path]:
        """Knowledge-base directories read when rebuilding the vector index."""

        return [self.data_dir, self.upload_dir]


def _int_env(name: str, default: int) -> int:
    """Read an integer env var, falling back to a safe default."""

    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        return int(raw_value)
    except ValueError:
        return default


def get_settings() -> RAGSettings:
    """Load settings from `.env` and process environment variables."""

    # python-dotenv 会读取项目根目录下的 .env，方便本地开发时配置 API Key。
    load_dotenv(PROJECT_ROOT / ".env")
    return RAGSettings(
        data_dir=Path(os.getenv("RAG_DATA_DIR", PROJECT_ROOT / "data" / "sample_docs")),
        upload_dir=Path(os.getenv("RAG_UPLOAD_DIR", PROJECT_ROOT / "data" / "uploaded_docs")),
        persist_dir=Path(os.getenv("RAG_VECTORSTORE_DIR", PROJECT_ROOT / "vectorstore")),
        collection_name=os.getenv("RAG_COLLECTION_NAME", "enterprise_knowledge_base"),
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        top_k=_int_env("RAG_TOP_K", 4),
        chunk_size=_int_env("RAG_CHUNK_SIZE", 900),
        chunk_overlap=_int_env("RAG_CHUNK_OVERLAP", 160),
        temperature=float(os.getenv("RAG_TEMPERATURE", "0.1")),
    )
