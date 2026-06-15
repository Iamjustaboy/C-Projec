from src.rag.config import get_settings
from src.rag.loaders import load_documents, split_documents


def test_sample_documents_load_with_source_metadata():
    settings = get_settings()

    documents = load_documents(settings.data_dir)

    assert len(documents) >= 3
    assert all(document.page_content for document in documents)
    assert all(document.metadata.get("source") for document in documents)


def test_split_documents_preserves_source_metadata():
    settings = get_settings()
    documents = load_documents(settings.data_dir)

    chunks = split_documents(documents, chunk_size=120, chunk_overlap=20)

    assert len(chunks) > len(documents)
    assert all(chunk.metadata.get("source") for chunk in chunks)
    assert all("chunk_id" in chunk.metadata for chunk in chunks)

