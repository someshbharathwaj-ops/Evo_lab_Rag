"""Script to find and delete NaN, NULL, or empty rows in the pgvector database."""

from __future__ import annotations

import sys
from psycopg import sql
from ingestion.vectorstore.pgvector_store import _connection, _qualified_table, table_exists


def clean_nan_rows() -> int:
    if not table_exists():
        print("[Clean DB] Vector table does not exist. Nothing to clean.")
        return 0

    print("[Clean DB] Checking PostgreSQL database for NaN / NULL / empty records...")

    deleted_total = 0

    with _connection() as conn, conn.cursor() as cur:
        # 1. Count total rows
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(_qualified_table()))
        total_before = cur.fetchone()[0]
        print(f"[Clean DB] Total chunks before cleanup: {total_before:,}")

        if total_before == 0:
            print("[Clean DB] Database table is empty.")
            return 0

        # 2. Delete rows with NULL, empty, or 'nan' text / chunk_id
        delete_text_stmt = sql.SQL(
            """
            DELETE FROM {}
            WHERE text IS NULL
               OR trim(text) = ''
               OR lower(trim(text)) = 'nan'
               OR lower(trim(text)) = 'null'
               OR lower(trim(text)) = 'none'
               OR chunk_id IS NULL
               OR lower(trim(chunk_id)) = 'nan'
            """
        ).format(_qualified_table())
        cur.execute(delete_text_stmt)
        deleted_text = cur.rowcount
        deleted_total += deleted_text
        print(f"[Clean DB] Deleted {deleted_text:,} rows with invalid/NaN text or chunk_id.")

        # 3. Delete rows with NULL or empty embeddings
        delete_emb_stmt = sql.SQL(
            """
            DELETE FROM {}
            WHERE embedding IS NULL
               OR array_length(embedding, 1) IS NULL
               OR array_length(embedding, 1) = 0
            """
        ).format(_qualified_table())
        cur.execute(delete_emb_stmt)
        deleted_emb = cur.rowcount
        deleted_total += deleted_emb
        print(f"[Clean DB] Deleted {deleted_emb:,} rows with missing or zero-length embeddings.")

        # 4. Count total rows after cleanup
        cur.execute(sql.SQL("SELECT count(*) FROM {}").format(_qualified_table()))
        total_after = cur.fetchone()[0]
        print(f"[Clean DB] Total chunks after cleanup: {total_after:,}")
        print(f"[Clean DB] Total NaN/invalid rows removed: {deleted_total:,}")

    return deleted_total


if __name__ == "__main__":
    clean_nan_rows()
