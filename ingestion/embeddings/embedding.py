"""Shared SentenceTransformer embedding adapter."""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    try:
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load embedding model '{EMBEDDING_MODEL}': {exc}"
        ) from exc


def get_embedding_dimension() -> int:
    """Read the actual output dimension from the configured model."""
    dimension = _get_model().get_sentence_embedding_dimension()
    if not dimension:
        dimension = len(embed_text("dimension probe"))
    return int(dimension)


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")
    try:
        embedding = _get_model().encode(text, normalize_embeddings=True)
    except Exception as exc:
        raise RuntimeError(f"Failed to generate embedding: {exc}") from exc
    return embedding.tolist()


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text or not text.strip() for text in texts):
        raise ValueError("Cannot embed empty text")
    try:
        embeddings = _get_model().encode(
            list(texts),
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to generate batch embeddings: {exc}") from exc
    return embeddings.tolist()
