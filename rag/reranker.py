"""
Cross-encoder reranking stage for the RAG pipeline.

Model selection is entirely driven by the RERANKER_MODEL environment variable.
This module does NOT hardcode any model name.

To enable reranking, set in .env:
    ENABLE_RERANKING=true
    RERANKER_MODEL=<any HuggingFace cross-encoder model>

Examples of compatible models (user's choice):
    z-ai/glm-5.2                                     # GLM reranker
    cross-encoder/ms-marco-MiniLM-L-6-v2        # fast, lightweight
    cross-encoder/ms-marco-MiniLM-L-12-v2       # balanced
    jinaai/jina-reranker-v2-base-multilingual    # multilingual

The model is lazy-loaded on first call and cached for subsequent requests.
If sentence-transformers is not installed, reranking is gracefully skipped
with a warning rather than crashing the server.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_cross_encoder(model_name: str):
    """
    Lazy-load and cache a CrossEncoder model.

    Raises:
        ImportError: if sentence-transformers is not installed.
        RuntimeError: if the model fails to load.
    """
    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for reranking. "
            "Install it with: pip install sentence-transformers"
        ) from exc

    logger.info("[Reranker] Loading cross-encoder model: %s", model_name)
    try:
        model = CrossEncoder(model_name)
        logger.info("[Reranker] Model loaded successfully: %s", model_name)
        return model
    except Exception as exc:
        raise RuntimeError(
            f"[Reranker] Failed to load cross-encoder model '{model_name}': {exc}\n"
            "Check that RERANKER_MODEL is a valid HuggingFace model identifier."
        ) from exc


def rerank(
    query: str,
    chunks: list[dict[str, Any]],
    top_n: int,
    model_name: str,
) -> list[dict[str, Any]]:
    """
    Rerank chunks using a cross-encoder model and return the top_n most relevant.

    Args:
        query      : The user's query string.
        chunks     : List of chunk dicts from the retriever (must have "text" key).
        top_n      : Number of chunks to return after reranking.
        model_name : HuggingFace cross-encoder model identifier.

    Returns:
        Top-N chunks sorted by cross-encoder relevance score (highest first).
        Each chunk dict gets an added "rerank_score" field.

    Raises:
        ImportError: if sentence-transformers is not installed.
        RuntimeError: if the model cannot be loaded.
    """
    if not chunks:
        return []
    if not model_name:
        raise ValueError(
            "RERANKER_MODEL is not set. "
            "Set it in your .env to a HuggingFace cross-encoder model name, "
            "e.g. RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2"
        )

    model = _get_cross_encoder(model_name)

    # Build (query, passage) pairs for the cross-encoder
    pairs = [(query, chunk.get("text", "")) for chunk in chunks]

    logger.info(
        "[Reranker] Scoring %d candidates with '%s'...", len(pairs), model_name
    )
    scores: list[float] = model.predict(pairs).tolist()

    # Attach score and sort descending
    scored = sorted(
        [
            {**chunk, "rerank_score": float(score)}
            for chunk, score in zip(chunks, scores)
        ],
        key=lambda c: c["rerank_score"],
        reverse=True,
    )

    top = scored[:top_n]
    logger.info(
        "[Reranker] Reranked %d → top %d  |  top score=%.4f  bottom score=%.4f",
        len(chunks),
        len(top),
        top[0]["rerank_score"] if top else 0.0,
        top[-1]["rerank_score"] if top else 0.0,
    )
    return top
