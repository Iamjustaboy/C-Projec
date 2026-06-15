"""Prompt templates and context formatting.

学习提示：
Prompt 是 RAG 的“回答规则”。检索只能决定模型看到什么资料，
Prompt 决定模型应该如何使用这些资料、资料不足时如何拒答。
"""

from __future__ import annotations

from typing import Any, Iterable

SYSTEM_PROMPT = """你是企业知识库助手，只能根据检索到的企业资料回答问题。
回答要求：
1. 优先给出直接、可执行的答案。
2. 如果资料中没有答案，必须明确说“根据当前知识库资料，我不知道答案。”不要编造。
3. 涉及流程、数字、时间、责任人时，必须保持与资料一致。
4. 回答末尾用简短列表总结引用来源。"""


def build_prompt() -> Any:
    """Create a LangChain chat prompt template.

    system 消息定义助手身份和约束，human 消息注入用户问题和检索上下文。
    """

    from langchain_core.prompts import ChatPromptTemplate

    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "问题：{question}\n\n"
                "检索到的企业资料：\n{context}\n\n"
                "请基于以上资料回答。",
            ),
        ]
    )


def format_documents(documents: Iterable[Any]) -> str:
    """Format retrieved chunks as grounded context for the model."""

    formatted_chunks: list[str] = []
    for document in documents:
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", "n/a")
        # 把来源信息直接放进上下文，模型回答时更容易引用对应资料。
        formatted_chunks.append(
            f"[source: {source} | chunk: {chunk_id}]\n{document.page_content}"
        )
    return "\n\n---\n\n".join(formatted_chunks)
