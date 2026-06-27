"""
Test script to run the ingestion pipeline on local PDFs.
"""
import os
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.pipeline import run_ingestion

CHUNKS_PATH = PROJECT_ROOT / "data" / "chunks" / "chunks.json"


def main():
    pdf_files = [
        "data/raw/1-s2.0-S1877050924021860-main.pdf",
        "data/raw/2504.07615v2.pdf",
    ]

    all_chunks = []

    for pdf_path in pdf_files:
        if os.path.exists(pdf_path):
            print(f"\n{'=' * 60}")
            print(f"Processing: {pdf_path}")
            print(f"{'=' * 60}")

            try:
                chunks = run_ingestion(pdf_path)
                all_chunks.extend(chunks)
                print(f"OK: Successfully ingested {len(chunks)} chunks from {pdf_path}")
            except Exception as exc:
                print(f"ERROR: Error processing {pdf_path}: {exc}")
        else:
            print(f"WARNING: File not found: {pdf_path}")

    if all_chunks:
        CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(CHUNKS_PATH, "w", encoding="utf-8") as file:
            json.dump(all_chunks, file, indent=2, ensure_ascii=False)
        print(f"\nOK: Wrote {len(all_chunks)} total chunks to {CHUNKS_PATH}")


if __name__ == "__main__":
    main()
