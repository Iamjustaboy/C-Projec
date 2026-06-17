"""Question answering chain built on top of the vector index.

学习提示：
这个文件对应 RAG 中的“问答”阶段：
用户问题 -> 检索相关 chunks -> 拼成上下文 -> 调用 LLM -> 返回回答和来源。
"""

from __future__ import annotations

from typing import Any

from .config import RAGSettings, get_settings
from .prompts import build_prompt, format_documents


def retrieve_documents(vector_store: Any, question: str, top_k: int) -> list[Any]:
    """Retrieve the most relevant chunks for a user question.

    top_k 越大，模型能看到的资料越多，但也更容易引入噪声并增加 token 成本。
    """

    # as_retriever 把 Chroma 向量库包装成 LangChain 通用检索器接口。
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(question))
    return list(retriever.get_relevant_documents(question))


def answer_question(
    question: str,
    vector_store: Any,
    settings: RAGSettings | None = None,
    top_k: int | None = None,
) -> dict[str, Any]:
    """Retrieve context and ask the chat model for a grounded answer."""

    settings = settings or get_settings()
    selected_k = top_k or settings.top_k
    # 1. 先检索，不直接把问题丢给大模型。
    documents = retrieve_documents(vector_store, question, selected_k)
    # 2. 把检索结果格式化为带 source/chunk_id 的上下文。
    context = format_documents(documents)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.temperature,
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
    )
    # 3. LCEL 管道：PromptTemplate -> ChatModel -> 字符串解析器。
    chain = build_prompt() | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    # 4. sources 原样返回给 UI，用于展示引用片段。
    return {"answer": answer, "sources": documents}
