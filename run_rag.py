"""Interactive terminal entry point for the local RAG pipeline."""

from __future__ import annotations

from config import LLM_MODEL, OLLAMA_BASE_URL
from ingestion.vectorstore.pgvector_store import count_chunks
from rag.rag_pipeline import run_rag


def print_header() -> None:
    print("===================================")
    print("Local RAG Assistant")
    print("===================================")
    print(f"Ollama: {OLLAMA_BASE_URL} | model: {LLM_MODEL}")
    print("Type 'exit' to quit.\n")


def check_vector_store() -> None:
    count = count_chunks()
    if count == 0:
        raise RuntimeError("The vector table is empty. Run ingestion before asking questions.")
    print(f"Vector store ready ({count} chunks).\n")


def prompt_loop() -> None:
    while True:
        question = input("Question:\n").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not question:
            print("Please enter a question or type 'exit' to quit.\n")
            continue
        try:
            answer = run_rag(question)
        except Exception as exc:
            print(f"Request failed: {exc}\n")
            continue
        print(f"\nAnswer\n------\n{answer}\n")


def main() -> int:
    print_header()
    try:
        check_vector_store()
    except Exception as exc:
        print(f"Startup failed: {exc}")
        return 1
    prompt_loop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
