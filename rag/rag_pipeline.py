"""End-to-end retrieval-augmented answer generation."""

from __future__ import annotations

from typing import Any

from config import TOP_K
from rag.llm_client import call_llm
from rag.prompts import build_rag_prompt
from rag.retriever import retrieve_context


NO_CONTEXT_ANSWER = "I do not know based on the available documents."


def run_rag(
    query: str,
    top_k: int = TOP_K,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> str:
    """Retrieve private context and return only the LLM-generated answer."""
    context = retrieve_context(
        query,
        top_k=top_k,
        metadata_filter=metadata_filter,
        score_threshold=score_threshold,
    )
    if not context:
        return NO_CONTEXT_ANSWER
    return call_llm(build_rag_prompt(query, context))
