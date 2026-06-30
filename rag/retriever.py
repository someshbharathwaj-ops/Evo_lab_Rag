"""Semantic retrieval service over the pgvector store."""

from __future__ import annotations

from typing import Any

from config import SCORE_THRESHOLD, TOP_K
from ingestion.embeddings.embedding import embed_text
from ingestion.vectorstore.pgvector_store import similarity_search
from rag.prompts import build_context


def retrieve(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Return internal ranked matches for a query (not an end-user response)."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")
    limit = TOP_K if top_k is None else top_k
    threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold
    query_embedding = embed_text(query)
    return similarity_search(
        query_embedding,
        top_k=limit,
        metadata_filter=metadata_filter,
        score_threshold=threshold,
    )


def retrieve_context(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> str:
    """Retrieve, deduplicate, and format context for generation."""
    return build_context(
        retrieve(query, top_k, metadata_filter, score_threshold)
    )
