"""Semantic retrieval service over the pgvector store."""

from __future__ import annotations

from typing import Any
import numpy as np
from psycopg import sql
from psycopg.types.json import Jsonb

from config import SCORE_THRESHOLD, TOP_K
from ingestion.embeddings.embedding import embed_text
from ingestion.vectorstore.pgvector_store import _connection, _qualified_table
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
    q_norm = float(np.linalg.norm(query_embedding))
    if q_norm == 0:
        return []

    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "q_norm": q_norm,
        "limit": limit,
    }

    # Similarity expression using unnest to calculate dot product and magnitude database-side
    similarity_expr = sql.SQL(
        "(SELECT SUM(q * e) FROM unnest(%(query_embedding)s::real[], embedding) AS x(q, e)) / "
        "NULLIF(sqrt((SELECT SUM(e * e) FROM unnest(embedding) AS x(e))) * %(q_norm)s, 0)"
    )

    filters = []
    if metadata_filter:
        filters.append(sql.SQL("metadata @> %(metadata_filter)s"))
        params["metadata_filter"] = Jsonb(metadata_filter)
    if threshold is not None:
        filters.append(sql.SQL("similarity >= %(threshold)s"))
        params["threshold"] = threshold

    where_clause = sql.SQL(" WHERE ") + sql.SQL(" AND ").join(filters) if filters else sql.SQL("")

    statement = sql.SQL(
        """
        WITH similarity_calc AS (
            SELECT chunk_id, text, source, page, metadata,
                   {} AS similarity
            FROM {}
        )
        SELECT chunk_id, text, source, page, metadata, similarity
        FROM similarity_calc
        {}
        ORDER BY similarity DESC
        LIMIT %(limit)s
        """
    ).format(similarity_expr, _qualified_table(), where_clause)

    results = []
    with _connection() as conn, conn.cursor() as cur:
        cur.execute(statement, params)
        for row in cur.fetchall():
            chunk_id, text, source, page, metadata, similarity = row
            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "page": page,
                "metadata": metadata or {},
                "similarity": float(similarity) if similarity is not None else 0.0,
            })

    return results


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

