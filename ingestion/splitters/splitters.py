import hashlib
import json
from typing import List, Dict

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None


def token_based_splitter(
    documents: List[Dict],
    model_name: str = "gpt-4o-mini",
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Dict]:
    """
    Split a list of documents into token-based chunks.
    
    Args:
        documents: List of dicts with 'text' and 'metadata' keys
        model_name: Tokenizer model to use
        chunk_size: Max tokens per chunk
        chunk_overlap: Token overlap between chunks
    
    Returns:
        List of chunk dicts with chunk_id, text, and metadata
    """
    if tiktoken is None:
        raise RuntimeError("tiktoken is required for token-based chunking")

    if chunk_size is None or chunk_overlap is None:
        from config import CHUNK_OVERLAP, CHUNK_SIZE

        chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
        chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    encoding = tiktoken.encoding_for_model(model_name)
    all_chunks = []
    
    for doc in documents:
        text = doc["text"]
        metadata = doc["metadata"]
        
        tokens = encoding.encode(text)
        start = 0
        total_tokens = len(tokens)
        
        while start < total_tokens:
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)
            
            chunk_metadata = {
                **metadata,
                "token_start": start,
                "token_end": min(end, total_tokens),
            }
            identity = json.dumps(
                {"text": chunk_text, "metadata": chunk_metadata},
                sort_keys=True,
                ensure_ascii=False,
            ).encode("utf-8")
            chunk_id = hashlib.sha256(identity).hexdigest()

            all_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": chunk_metadata,
            })
            
            if end >= total_tokens:
                break
            
            start = end - chunk_overlap
    
    return all_chunks
