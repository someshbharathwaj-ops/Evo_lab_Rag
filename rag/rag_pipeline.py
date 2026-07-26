"""End-to-end retrieval-augmented answer generation wrapper for backward compatibility."""

from __future__ import annotations

from typing import Any

from config import TOP_K
from rag.pipeline import NO_CONTEXT_ANSWER, run_pipeline


def run_rag(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> str:
    """Retrieve context and return LLM-generated answer using the full RAG pipeline."""
    result = run_pipeline(
        query=query,
        top_k=top_k,
        metadata_filter=metadata_filter,
        score_threshold=score_threshold,
    )
    return result.answer

