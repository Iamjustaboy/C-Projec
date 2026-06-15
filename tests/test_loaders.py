from src.rag.config import get_settings
from src.rag.loaders import load_documents, load_documents_from_dirs, split_documents


def test_sample_documents_load_with_source_metadata():
    # 学习重点：RAG 的回答必须可溯源，所以加载阶段就要写入 source。
    settings = get_settings()

    documents = load_documents(settings.data_dir)

    assert len(documents) >= 3
    assert all(document.page_content for document in documents)
    assert all(document.metadata.get("source") for document in documents)


def test_split_documents_preserves_source_metadata():
    # 学习重点：切分不能丢掉 metadata，否则检索结果无法展示引用来源。
    settings = get_settings()
    documents = load_documents(settings.data_dir)

    chunks = split_documents(documents, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > len(documents)
    assert all(chunk.metadata.get("source") for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)


def test_load_documents_from_multiple_dirs(tmp_path):
    # 学习重点：示例文档和上传文档分目录保存，但建库时要合并加载。
    sample_dir = tmp_path / "sample_docs"
    upload_dir = tmp_path / "uploaded_docs"
    sample_dir.mkdir()
    upload_dir.mkdir()
    (sample_dir / "policy.md").write_text("# 制度\n年假提前 3 天申请。", encoding="utf-8")
    (upload_dir / "faq.txt").write_text("知识库支持 PDF 上传。", encoding="utf-8")

    documents = load_documents_from_dirs([sample_dir, upload_dir])

    assert len(documents) == 2
    assert {document.metadata["filename"] for document in documents} == {"policy.md", "faq.txt"}
