"""Semantic retrieval service over the pgvector store."""

from __future__ import annotations

import logging
import math
from typing import Any
import numpy as np
from psycopg import sql
from psycopg.types.json import Jsonb

from config import SCORE_THRESHOLD, TOP_K
from ingestion.embeddings.embedding import embed_text
from ingestion.vectorstore.pgvector_store import _connection, _qualified_table, table_exists
from rag.prompts import build_context

logger = logging.getLogger(__name__)


def retrieve(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Return internal ranked matches for a query (not an end-user response)."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if not table_exists():
        logger.info("[Retriever] Vector table does not exist. Returning empty context.")
        return []

    limit = TOP_K if top_k is None else top_k
    threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold

    try:
        query_embedding = embed_text(query)
    except Exception as exc:
        logger.warning("[Retriever] Failed to generate query embedding (%s). Returning empty context.", exc)
        return []

    if not query_embedding or any(math.isnan(x) or math.isinf(x) for x in query_embedding):
        logger.warning("[Retriever] Query embedding contains NaN or Inf. Returning empty context.")
        return []

    q_norm = float(np.linalg.norm(query_embedding))
    if q_norm == 0 or math.isnan(q_norm):
        return []

    params: dict[str, Any] = {
        "query_embedding": query_embedding,
        "q_norm": q_norm,
        "limit": limit,
    }

    # Native pgvector cosine similarity: 1 - (A <=> B)
    similarity_expr = sql.SQL(
        "1 - (embedding::vector <=> %(query_embedding)s::vector)"
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
    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(statement, params)
            for row in cur.fetchall():
                chunk_id, text, source, page, metadata, similarity = row
                # Filter out any individual returned row with invalid similarity
                sim_val = float(similarity) if similarity is not None and not math.isnan(similarity) else 0.0
                results.append({
                    "chunk_id": chunk_id,
                    "text": text,
                    "source": source,
                    "page": page,
                    "metadata": metadata or {},
                    "similarity": sim_val,
                })
    except Exception as exc:
        logger.warning(
            "[Retriever] Database vector query error (%s). Returning empty context.", exc
        )
        return []

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

