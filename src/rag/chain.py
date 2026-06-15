"""Question answering chain built on top of the vector index."""

from __future__ import annotations

from typing import Any

from .config import RAGSettings, get_settings
from .prompts import build_prompt, format_documents


def retrieve_documents(vector_store: Any, question: str, top_k: int) -> list[Any]:
    """Retrieve the most relevant chunks for a user question."""

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
    documents = retrieve_documents(vector_store, question, selected_k)
    context = format_documents(documents)

    from langchain_core.output_parsers import StrOutputParser
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        model=settings.openai_model,
        temperature=settings.temperature,
        api_key=settings.openai_api_key,
    )
    chain = build_prompt() | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
    return {"answer": answer, "sources": documents}

