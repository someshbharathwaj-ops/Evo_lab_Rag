import uuid
from typing import List, Dict

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None


def token_based_splitter(
    documents: List[Dict],
    model_name: str = "gpt-4o-mini",
    chunk_size: int = 512,
    chunk_overlap: int = 100
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
            
            all_chunks.append({
                "chunk_id": str(uuid.uuid4()),
                "text": chunk_text,
                "metadata": {
                    **metadata,
                    "token_start": start,
                    "token_end": min(end, total_tokens)
                }
            })
            
            if end >= total_tokens:
                break
            
            start = end - chunk_overlap
    
    return all_chunks

