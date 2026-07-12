"""PostgreSQL/pgvector persistence and cosine-similarity search."""

from __future__ import annotations

import hashlib
import json
import re
from contextlib import contextmanager
from typing import Iterator

import psycopg
from psycopg import sql
from psycopg.types.json import Jsonb

from config import (
    DATABASE_URL,
    DB_CONNECT_TIMEOUT,
    DB_HOST,
    DB_NAME,
    DB_PASSWORD,
    DB_PORT,
    DB_SCHEMA,
    DB_SSLMODE,
    DB_TABLE,
    DB_USER,
    EMBEDDING_MODEL,
    SCORE_THRESHOLD,
)


class VectorStoreError(RuntimeError):
    """Raised when the pgvector store cannot complete an operation."""


def _validate_identifier(value: str, setting: str) -> str:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise ValueError(f"{setting} is not a valid PostgreSQL identifier: {value!r}")
    return value


SCHEMA = _validate_identifier(DB_SCHEMA, "DB_SCHEMA")
TABLE = _validate_identifier(DB_TABLE, "DB_TABLE")


def _qualified_table() -> sql.Composed:
    return sql.SQL("{}.{}").format(sql.Identifier(SCHEMA), sql.Identifier(TABLE))


def _connect() -> psycopg.Connection:
    try:
        if DATABASE_URL:
            return psycopg.connect(
                DATABASE_URL,
                autocommit=True,
                connect_timeout=DB_CONNECT_TIMEOUT,
                prepare_threshold=None,
            )
        return psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            sslmode=DB_SSLMODE,
            autocommit=True,
            connect_timeout=DB_CONNECT_TIMEOUT,
            prepare_threshold=None,
        )
    except Exception as exc:
        target = "DATABASE_URL" if DATABASE_URL else f"{DB_HOST}:{DB_PORT}/{DB_NAME}"
        raise VectorStoreError(f"Could not connect to PostgreSQL via {target}: {exc}") from exc


@contextmanager
def _connection() -> Iterator[psycopg.Connection]:
    conn = _connect()
    try:
        yield conn
    finally:
        conn.close()


def _get_connection() -> psycopg.Connection:
    """Compatibility helper; callers own and must close the returned connection."""
    return _connect()


def _stored_dimension(cur: psycopg.Cursor) -> int | None:
    try:
        cur.execute(
            sql.SQL("SELECT array_length(embedding, 1) FROM {} LIMIT 1").format(_qualified_table())
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None
    except Exception:
        return None


def table_exists() -> bool:
    """Check if the pgvector chunk table exists in the database."""
    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s 
                    AND table_name = %s
                )
                """,
                (SCHEMA, TABLE),
            )
            return bool(cur.fetchone()[0])
    except Exception:
        return False


def create_table(embedding_dimension: int | None = None) -> None:
    """Create/migrate the chunk table and validate its vector dimension."""
    if embedding_dimension is None:
        from ingestion.embeddings.embedding import get_embedding_dimension

        embedding_dimension = get_embedding_dimension()
    if embedding_dimension <= 0:
        raise ValueError("embedding_dimension must be greater than zero")

    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("CREATE SCHEMA IF NOT EXISTS {}").format(sql.Identifier(SCHEMA)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        chunk_id TEXT PRIMARY KEY,
                        text TEXT NOT NULL,
                        source TEXT,
                        page INTEGER,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        content_hash TEXT,
                        embedding_model TEXT NOT NULL DEFAULT '',
                        embedding real[] NOT NULL,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                    )
                    """
                ).format(_qualified_table())
            )

            # Upgrade the prototype's earlier table without discarding its data.
            for statement in (
                "ADD COLUMN IF NOT EXISTS metadata JSONB NOT NULL DEFAULT '{}'::jsonb",
                "ADD COLUMN IF NOT EXISTS content_hash TEXT",
                "ADD COLUMN IF NOT EXISTS embedding_model TEXT NOT NULL DEFAULT ''",
                "ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now()",
                "ADD COLUMN IF NOT EXISTS updated_at TIMESTAMPTZ NOT NULL DEFAULT now()",
            ):
                cur.execute(
                    sql.SQL("ALTER TABLE {} ").format(_qualified_table()) + sql.SQL(statement)
                )

            stored_dimension = _stored_dimension(cur)
            if stored_dimension is not None and stored_dimension != embedding_dimension:
                raise VectorStoreError(
                    f"Vector dimension mismatch for {SCHEMA}.{TABLE}: table uses "
                    f"{stored_dimension}, but '{EMBEDDING_MODEL}' produces "
                    f"{embedding_dimension}. Recreate or migrate the table before ingesting."
                )

            cur.execute(
                sql.SQL("CREATE UNIQUE INDEX IF NOT EXISTS {} ON {} (content_hash)").format(
                    sql.Identifier(f"{TABLE}_content_hash_idx"), _qualified_table()
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING gin (metadata)").format(
                    sql.Identifier(f"{TABLE}_metadata_idx"), _qualified_table()
                )
            )
    except VectorStoreError:
        raise
    except Exception as exc:
        raise VectorStoreError(
            f"Could not initialize pgvector table {SCHEMA}.{TABLE}: {exc}. "
            "On Supabase, run db/schema.sql in the SQL editor if this database "
            "role cannot create extensions."
        ) from exc


def _content_hash(chunk: dict) -> str:
    existing = chunk.get("content_hash")
    if existing:
        return str(existing)
    payload = json.dumps(
        {"text": chunk["text"], "metadata": chunk.get("metadata", {})},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def insert_chunks(chunks: list[dict]) -> int:
    """Idempotently insert or update embedded chunks; return processed count."""
    if not chunks:
        return 0
    dimension = len(chunks[0].get("embedding", []))
    if not dimension:
        raise ValueError("Every chunk must include a non-empty embedding")
    for chunk in chunks:
        if len(chunk.get("embedding", [])) != dimension:
            raise ValueError("All chunk embeddings must have the same dimension")

    create_table(dimension)
    statement = sql.SQL(
        """
        INSERT INTO {} (
            chunk_id, text, source, page, metadata, content_hash,
            embedding_model, embedding, updated_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::real[], now())
        ON CONFLICT (content_hash) DO UPDATE SET
            text = EXCLUDED.text,
            source = EXCLUDED.source,
            page = EXCLUDED.page,
            metadata = EXCLUDED.metadata,
            embedding_model = EXCLUDED.embedding_model,
            embedding = EXCLUDED.embedding,
            updated_at = now()
        """
    ).format(_qualified_table())
    values = []
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        values.append(
            (
                chunk["chunk_id"],
                chunk["text"],
                metadata.get("source", ""),
                metadata.get("page"),
                Jsonb(metadata),
                _content_hash(chunk),
                EMBEDDING_MODEL,
                chunk["embedding"],
            )
        )

    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.executemany(statement, values)
        return len(values)
    except Exception as exc:
        raise VectorStoreError(
            f"Failed to upload {len(chunks)} chunks to {SCHEMA}.{TABLE}: {exc}"
        ) from exc


def similarity_search(
    query_embedding: list[float],
    top_k: int = 5,
    metadata_filter: dict | None = None,
    score_threshold: float | None = None,
) -> list[dict]:
    """Return the highest cosine-similarity matches from pgvector."""
    if not query_embedding:
        raise ValueError("query_embedding cannot be empty")
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    threshold = SCORE_THRESHOLD if score_threshold is None else score_threshold

    filters = []
    params: list[object] = []
    if metadata_filter:
        filters.append(sql.SQL("metadata @> %s"))
        params.append(Jsonb(metadata_filter))

    where_clause = sql.SQL(" WHERE ").join(filters) if filters else sql.SQL("")
    statement = sql.SQL(
        """
        SELECT chunk_id, text, source, page, metadata, embedding
        FROM {}
        {}
        """
    ).format(_qualified_table(), where_clause)

    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(statement, params)
            rows = cur.fetchall()
    except psycopg.errors.UndefinedTable as exc:
        raise VectorStoreError(
            f"Vector table {SCHEMA}.{TABLE} does not exist. Run ingestion or db/schema.sql first."
        ) from exc
    except Exception as exc:
        raise VectorStoreError(f"Database query failed: {exc}") from exc

    import numpy as np

    results = []
    q_vec = np.array(query_embedding, dtype=np.float32)
    q_norm = np.linalg.norm(q_vec)

    if q_norm == 0:
        return []

    for row in rows:
        chunk_id, text, source, page, metadata, emb_list = row
        if not emb_list:
            continue
        c_vec = np.array(emb_list, dtype=np.float32)
        c_norm = np.linalg.norm(c_vec)
        if c_norm == 0:
            similarity = 0.0
        else:
            similarity = float(np.dot(q_vec, c_vec) / (q_norm * c_norm))

        if similarity >= threshold:
            results.append({
                "chunk_id": chunk_id,
                "text": text,
                "source": source,
                "page": page,
                "metadata": metadata or {},
                "similarity": similarity,
            })

    results.sort(key=lambda x: x["similarity"], reverse=True)
    return results[:top_k]


def count_chunks() -> int:
    """Return the number of stored chunks for health checks and verification."""
    try:
        with _connection() as conn, conn.cursor() as cur:
            cur.execute(sql.SQL("SELECT count(*) FROM {}").format(_qualified_table()))
            return int(cur.fetchone()[0])
    except psycopg.errors.UndefinedTable:
        return 0
    except Exception as exc:
        raise VectorStoreError(f"Could not count stored chunks: {exc}") from exc
