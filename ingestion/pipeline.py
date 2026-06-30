"""PDF ingestion orchestration: load, chunk, embed, persist, and audit."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import argparse

from config import CHUNKS_PATH
from ingestion.embeddings.embedding import embed_texts
from ingestion.loaders.loaders import load_pdf
from ingestion.splitters.splitters import token_based_splitter
from ingestion.vectorstore.pgvector_store import insert_chunks


def _write_debug_chunks(chunks: list[dict[str, Any]]) -> None:
    path = Path(CHUNKS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep embeddings out of the debug artifact: they make it huge and are stored in pgvector.
    debug_chunks = [
        {key: value for key, value in chunk.items() if key != "embedding"}
        for chunk in chunks
    ]
    path.write_text(
        json.dumps(debug_chunks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def run_ingestion(pdf_path: str) -> list[dict[str, Any]]:
    """Ingest one PDF into pgvector and return chunks without embedding payloads."""
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    print(f"[Ingestion] Loading PDF: {pdf_path}")
    documents = load_pdf(pdf_path)
    if not documents:
        raise ValueError(f"No readable text was found in PDF: {pdf_path}")

    print(f"[Ingestion] Loaded {len(documents)} pages; splitting into chunks...")
    chunks = token_based_splitter(documents)
    if not chunks:
        raise ValueError("No chunks were created during ingestion")

    print(f"[Ingestion] Generating embeddings for {len(chunks)} chunks...")
    embeddings = embed_texts([chunk["text"] for chunk in chunks])
    for chunk, embedding in zip(chunks, embeddings, strict=True):
        chunk["embedding"] = embedding

    print("[Ingestion] Uploading embeddings to PostgreSQL/pgvector...")
    uploaded = insert_chunks(chunks)
    _write_debug_chunks(chunks)
    print(f"[Ingestion] Stored {uploaded} chunks; debug copy written to {CHUNKS_PATH}")

    return [
        {key: value for key, value in chunk.items() if key != "embedding"}
        for chunk in chunks
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest PDFs into Supabase pgvector")
    parser.add_argument("pdfs", nargs="+", help="one or more PDF paths")
    args = parser.parse_args(argv)
    all_chunks: list[dict[str, Any]] = []
    for pdf in args.pdfs:
        all_chunks.extend(run_ingestion(pdf))
    _write_debug_chunks(all_chunks)
    print(
        f"[Ingestion] Complete: processed {len(all_chunks)} chunks "
        f"from {len(args.pdfs)} PDF(s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
