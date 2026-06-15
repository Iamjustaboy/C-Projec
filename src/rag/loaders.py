"""Document loading and chunking utilities.

学习提示：
RAG 的第一步是把“外部知识”变成统一的 Document 列表。每个 Document
至少包含两部分：
1. page_content: 文档正文。
2. metadata: 来源、文件名、chunk_id 等追踪信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import PROJECT_ROOT

SUPPORTED_SUFFIXES = {".md", ".txt", ".pdf"}


@dataclass
class SimpleDocument:
    """Small fallback document type used when LangChain is not installed.

    项目测试不应该强依赖 OpenAI 或完整 LangChain 环境，所以这里提供一个
    最小 Document 替身，让加载和切分逻辑可以独立测试。
    """

    page_content: str
    metadata: dict[str, Any]


def _document_type() -> type:
    """Use LangChain's Document in the real app and SimpleDocument in tests."""

    try:
        from langchain_core.documents import Document

        return Document
    except ImportError:
        return SimpleDocument


def _relative_source(path: Path) -> str:
    """Store readable relative paths in metadata for source citations."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def load_documents(data_dir: Path) -> list[Any]:
    """Load Markdown, text, and PDF files from one knowledge-base directory."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory does not exist: {data_dir}")

    document_type = _document_type()
    documents: list[Any] = []
    for file_path in sorted(data_dir.rglob("*")):
        # 当前支持 Markdown/TXT/PDF；以后可以在这里扩展 DOCX、网页加载器。
        if not file_path.is_file() or file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
            continue

        text = _read_supported_file(file_path).strip()
        if not text:
            continue

        documents.append(
            document_type(
                page_content=text,
                metadata={
                    # source 会在回答后展示给用户，是“可溯源回答”的关键。
                    "source": _relative_source(file_path),
                    "filename": file_path.name,
                    "file_type": file_path.suffix.lower().lstrip("."),
                },
            )
        )

    if not documents:
        raise FileNotFoundError(f"No supported documents found in: {data_dir}")
    return documents


def load_documents_from_dirs(data_dirs: Iterable[Path]) -> list[Any]:
    """Load documents from multiple knowledge-base directories.

    示例资料和用户上传资料分开放置，但构建向量库时需要合并成一个文档列表。
    不存在的上传目录会被跳过，避免首次运行时因为没有上传文件而报错。
    """

    documents: list[Any] = []
    missing_required_dirs: list[Path] = []
    for index, data_dir in enumerate(data_dirs):
        if not data_dir.exists():
            if index == 0:
                missing_required_dirs.append(data_dir)
            continue
        try:
            documents.extend(load_documents(data_dir))
        except FileNotFoundError:
            continue

    if missing_required_dirs:
        missing = ", ".join(str(path) for path in missing_required_dirs)
        raise FileNotFoundError(f"Required knowledge-base directory does not exist: {missing}")
    if not documents:
        raise FileNotFoundError("No supported documents found in configured knowledge-base directories.")
    return documents


def _read_supported_file(file_path: Path) -> str:
    """Read a supported knowledge-base file as plain text."""

    suffix = file_path.suffix.lower()
    if suffix in {".md", ".txt"}:
        return file_path.read_text(encoding="utf-8")
    if suffix == ".pdf":
        return _read_pdf(file_path)
    raise ValueError(f"Unsupported file type: {file_path.suffix}")


def _read_pdf(file_path: Path) -> str:
    """Extract text from a PDF file with pypdf."""

    try:
        from pypdf import PdfReader
    except ImportError as error:
        raise ImportError("PDF loading requires `pypdf`. Run `pip install -r requirements.txt`.") from error

    reader = PdfReader(str(file_path))
    page_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(page_text)


def split_documents(
    documents: Iterable[Any],
    chunk_size: int,
    chunk_overlap: int,
) -> list[Any]:
    """Split documents while preserving source metadata.

    为什么要切分：
    大模型上下文长度有限，向量检索也需要更细粒度的片段。chunk_overlap
    能让相邻片段保留少量重复内容，降低一句话被切断导致语义丢失的风险。
    """

    document_list = list(documents)
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            # 中文文档优先按段落、换行和中文标点切分，语义会更自然。
            separators=["\n\n", "\n", "。", "；", ";", " ", ""],
        )
        chunks = splitter.split_documents(document_list)
    except ImportError:
        chunks = _fallback_split_documents(document_list, chunk_size, chunk_overlap)

    for index, chunk in enumerate(chunks):
        chunk.metadata = dict(chunk.metadata)
        # chunk_id 让同一个源文件中的不同片段可以被区分和定位。
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
    # step 是每次窗口向前移动的长度；overlap 越大，相邻片段重复越多。
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
