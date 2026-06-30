

_store_instance = None


def get_store():
    global _store_instance
    if _store_instance is None:
        _store_instance = WeaviateStore()
    return _store_instance


def retrieve_chunks(query: str, top_k: int = 5):
    """
    Retrieve relevant chunks from Weaviate.
    Falls back to mock data if store is unavailable.
    """
    try:
        store = get_store()
        results = store.search(query, limit=top_k)

        formatted_results = []
        for result in results:
            formatted_results.append({
                "text": result["text"],
                "metadata": result.get("metadata", {})
            })

        return formatted_results

    except Exception:
        # Safe fallback for development/testing
        return [
            {
                "text": "The paper proposes a new method for efficient information retrieval.",
                "metadata": {"source": "paper1.pdf"}
            },
            {
                "text": "Experimental results demonstrate improved performance over baseline methods.",
                "metadata": {"source": "paper1.pdf"}
            }
        ]


def add_documents_to_store(chunks):
    try:
        store = get_store()
        store.add_documents(chunks)
        return True
    except Exception:
        return False


def close_store():
    global _store_instance
    if _store_instance:
        _store_instance.close()
        _store_instance = None
