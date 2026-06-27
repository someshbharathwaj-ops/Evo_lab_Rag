import json
import os
from typing import Dict, List

from ingestion.loaders.loaders import load_pdf
from ingestion.splitters.splitters import token_based_splitter


CHUNKS_PATH = os.path.join("data", "chunks", "chunks.json")


def run_ingestion(pdf_path: str) -> List[Dict]:
    """
    Full ingestion pipeline:
    PDF -> text -> chunks -> local JSON storage.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"[Ingestion] Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)

    if not documents:
        raise ValueError("No documents loaded from PDF")

    print(f"[Ingestion] Loaded {len(documents)} pages")
    print("[Ingestion] Splitting text into chunks...")
    chunks = token_based_splitter(documents)

    if not chunks:
        raise ValueError("No chunks created during ingestion")

    print(f"[Ingestion] Created {len(chunks)} chunks")

    os.makedirs(os.path.dirname(CHUNKS_PATH), exist_ok=True)
    with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
        json.dump(chunks, file, indent=2, ensure_ascii=False)

    print(f"[Ingestion] Chunks saved to {CHUNKS_PATH}")
    print("[Ingestion] Vector store upload skipped until pgvector migration is added")
    print("[Ingestion] Ingestion complete")

    return chunks
