"""Backward-compatible wrappers around the active pgvector implementation."""

from __future__ import annotations

from ingestion.embeddings.embedding import embed_texts
from ingestion.vectorstore.pgvector_store import insert_chunks
from rag.retriever import retrieve


def retrieve_chunks(query: str, top_k: int = 5) -> list[dict]:
    return retrieve(query, top_k)


def add_documents_to_store(chunks: list[dict]) -> bool:
    missing = [chunk for chunk in chunks if "embedding" not in chunk]
    if missing:
        embeddings = embed_texts([chunk["text"] for chunk in missing])
        for chunk, embedding in zip(missing, embeddings, strict=True):
            chunk["embedding"] = embedding
    insert_chunks(chunks)
    return True


def close_store() -> None:
    # Connections are scoped per operation; retained for old callers.
    return None
