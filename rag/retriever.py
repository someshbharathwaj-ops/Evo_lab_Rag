from ingestion.embeddings.embedding import embed_text
from ingestion.vectorstore.pgvector_store import similarity_search
from config import TOP_K


def retrieve(query: str, top_k: int = None) -> list[dict]:
    if top_k is None:
        top_k = TOP_K
    query_embedding = embed_text(query)
    results = similarity_search(query_embedding, top_k)
    return results
