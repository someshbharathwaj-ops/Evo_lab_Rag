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
    Split a list of documents into overlapping token-based chunks.

    Works with any document type (PDF, PPTX, DOCX, image) as long as
    the input follows the standard loader format:
        [{"text": str, "metadata": {"source": str, "page": int | None}}]

    Args:
        documents    : List of dicts from any loader (load_pdf, load_pptx, etc.)
        model_name   : Tiktoken encoding to use (default: gpt-4o-mini / cl100k_base)
        chunk_size   : Max tokens per chunk (default: config.CHUNK_SIZE = 800)
        chunk_overlap: Token overlap between consecutive chunks
                       (default: config.CHUNK_OVERLAP = 200)

    Returns:
        List of chunk dicts with:
            - chunk_id  : deterministic SHA-256 hex ID
            - text      : chunk text
            - metadata  : original metadata + token_start / token_end offsets
    """
    if tiktoken is None:
        raise RuntimeError(
            "tiktoken is required for token-based chunking. "
            "Install it with: pip install tiktoken>=0.7"
        )

    if chunk_size is None or chunk_overlap is None:
        from config import CHUNK_OVERLAP, CHUNK_SIZE
        chunk_size = CHUNK_SIZE if chunk_size is None else chunk_size
        chunk_overlap = CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")

    try:
        encoding = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Fallback to cl100k_base for unknown model names
        encoding = tiktoken.get_encoding("cl100k_base")

    all_chunks: List[Dict] = []

    for doc in documents:
        text = doc.get("text", "")
        metadata = doc.get("metadata", {})

        if not text or not text.strip():
            continue

        tokens = encoding.encode(text)
        total_tokens = len(tokens)
        start = 0

        while start < total_tokens:
            end = min(start + chunk_size, total_tokens)
            chunk_tokens = tokens[start:end]
            chunk_text = encoding.decode(chunk_tokens)

            if chunk_text.strip():
                chunk_metadata = {
                    **metadata,
                    "token_start": start,
                    "token_end": end,
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
