import psycopg
from config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD, EMBEDDING_DIMENSION

_conn = None


def _get_connection():
    global _conn
    if _conn is None or _conn.closed:
        _conn = psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            autocommit=True
        )
    return _conn


def create_table():
    conn = _get_connection()
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                source TEXT,
                page INTEGER,
                embedding vector({EMBEDDING_DIMENSION})
            )
        """)


def insert_chunks(chunks: list[dict]):
    conn = _get_connection()
    with conn.cursor() as cur:
        for chunk in chunks:
            cur.execute(
                """
                INSERT INTO chunks (chunk_id, text, source, page, embedding)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (chunk_id) DO NOTHING
                """,
                (
                    chunk["chunk_id"],
                    chunk["text"],
                    chunk["metadata"].get("source", ""),
                    chunk["metadata"].get("page", 0),
                    str(chunk["embedding"])
                )
            )


def similarity_search(query_embedding: list[float], top_k: int = 5) -> list[dict]:
    conn = _get_connection()
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT chunk_id, text, source, page,
                   1 - (embedding <=> %s::vector) AS similarity
            FROM chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (str(query_embedding), str(query_embedding), top_k)
        )
        rows = cur.fetchall()

    results = []
    for row in rows:
        results.append({
            "chunk_id": row[0],
            "text": row[1],
            "source": row[2],
            "page": row[3],
            "similarity": row[4]
        })
    return results
