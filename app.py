"""Streamlit entrypoint for the enterprise RAG assistant."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.rag.chain import answer_question
from src.rag.config import PROJECT_ROOT, get_settings
from src.rag.index import build_vector_store, get_or_create_vector_store, is_index_ready


EXAMPLE_QUESTIONS = [
    "新员工入职前需要准备什么？",
    "InsightFlow Enterprise 支持哪些知识库能力？",
    "年假申请需要提前多久提交？",
]


def _render_source(document: object, index: int) -> None:
    metadata = getattr(document, "metadata", {})
    content = getattr(document, "page_content", "")
    source = metadata.get("source", "unknown")
    chunk_id = metadata.get("chunk_id", "n/a")
    with st.expander(f"来源 {index}: {source} / chunk {chunk_id}"):
        st.write(content[:1200])


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def main() -> None:
    st.set_page_config(page_title="企业知识库助手", layout="wide")
    settings = get_settings()

    st.title("企业知识库助手")
    st.caption("基于内置企业资料的 RAG 问答 Demo，回答会附带检索来源。")

    with st.sidebar:
        st.header("运行配置")
        st.write(f"模型：`{settings.openai_model}`")
        st.write(f"Embedding：`{settings.embedding_model}`")
        st.write(f"知识库目录：`{_display_path(settings.data_dir)}`")

        selected_k = st.slider(
            "检索 Top-K",
            min_value=2,
            max_value=8,
            value=min(max(settings.top_k, 2), 8),
            step=1,
        )

        ready = is_index_ready(settings.persist_dir)
        st.write("索引状态：" + ("已构建" if ready else "未构建"))

        if not settings.has_api_key:
            st.warning("请先在 `.env` 中配置 `OPENAI_API_KEY`。")

        if st.button("重建知识库索引", type="primary", use_container_width=True):
            if not settings.has_api_key:
                st.error("缺少 OPENAI_API_KEY，无法调用 Embedding 模型。")
            else:
                with st.spinner("正在读取文档并重建 Chroma 索引..."):
                    _, doc_count, chunk_count = build_vector_store(settings, force_rebuild=True)
                st.success(f"索引已更新：{doc_count} 篇文档，{chunk_count} 个片段。")

    if "active_question" not in st.session_state:
        st.session_state.active_question = ""

    st.subheader("示例问题")
    columns = st.columns(len(EXAMPLE_QUESTIONS))
    for column, question in zip(columns, EXAMPLE_QUESTIONS, strict=True):
        if column.button(question, use_container_width=True):
            st.session_state.active_question = question

    question = st.chat_input("输入一个企业知识库问题")
    if question:
        st.session_state.active_question = question

    active_question = st.session_state.active_question
    if active_question:
        st.chat_message("user").write(active_question)

        if not settings.has_api_key:
            st.chat_message("assistant").warning(
                "缺少 `OPENAI_API_KEY`。配置 `.env` 后即可构建索引并生成回答。"
            )
            return

        try:
            with st.spinner("正在检索知识库并生成回答..."):
                vector_store, created = get_or_create_vector_store(settings)
                result = answer_question(
                    active_question,
                    vector_store,
                    settings=settings,
                    top_k=selected_k,
                )
        except ImportError as error:
            st.error("依赖未安装完整，请先运行 `pip install -r requirements.txt`。")
            st.exception(error)
            return
        except Exception as error:  # pragma: no cover - displayed in UI.
            st.error("问答流程执行失败，请检查 API Key、网络连接或本地索引。")
            st.exception(error)
            return

        if created:
            st.info("首次提问时已自动构建本地知识库索引。")

        st.chat_message("assistant").write(result["answer"])
        st.subheader("引用来源")
        for index, document in enumerate(result["sources"], start=1):
            _render_source(document, index)


if __name__ == "__main__":
    main()
