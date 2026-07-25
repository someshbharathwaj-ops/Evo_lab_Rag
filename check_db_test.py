"""Quick DB connectivity test for audit."""
from ingestion.vectorstore.pgvector_store import _connection, count_chunks

try:
    with _connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT version()")
        ver = cur.fetchone()[0]
        print("DB Connected:", ver[:70])
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
        print("pgvector:", row[0] if row else "NOT INSTALLED")
    cnt = count_chunks()
    print("Chunk count:", cnt)
except Exception as e:
    print("DB Error:", e)
