"""Interactive terminal entry point for the existing RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RagComponents:
    """Runtime references to the existing project components."""

    top_k: int
    retrieve: Any
    rag_prompt: str
    call_llm: Any


def print_header() -> None:
    """Print the terminal application header."""
    print("===================================")
    print("RAG Assistant")
    print("===================================")
    print()
    print("Type 'exit' to quit.")
    print()


def load_components() -> RagComponents:
    """Import and validate the existing RAG modules without rebuilding data."""
    try:
        print("Loading configuration...")
        from config import OPENROUTER_API_KEY, TOP_K

        if not OPENROUTER_API_KEY:
            print(
                "Warning: OPENROUTER_API_KEY is not set. "
                "LLM calls may fail until it is configured."
            )

        print("Loading embedding model...")
        from ingestion.embeddings.embedding import embed_text

        if not callable(embed_text):
            raise RuntimeError("Embedding function embed_text is not callable.")

        print("Connecting to vector database...")
        from ingestion.vectorstore import pgvector_store

        try:
            pgvector_store._get_connection()
        except Exception as exc :
           
            print(f"Actual database error: {exc}")
            raise
                

        print("Retriever initialized...")
        from rag.retriever import retrieve
        from rag.prompts import RAG_PROMPT
        from rag.llm_client import call_llm

        print("Ready.")
        print()
        return RagComponents(
            top_k=TOP_K,
            retrieve=retrieve,
            rag_prompt=RAG_PROMPT,
            call_llm=call_llm,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import a required project module or dependency. "
            f"Original error: {exc}"
        ) from exc


def get_chunk_text(chunk: dict[str, Any]) -> str:
    """Return chunk text from the existing result shape."""
    return str(chunk.get("text") or "")


def get_chunk_source(chunk: dict[str, Any]) -> str:
    """Return a readable source value from a retrieved chunk."""
    metadata = chunk.get("metadata")
    if isinstance(metadata, dict) and metadata.get("source"):
        return str(metadata["source"])
    return str(chunk.get("source") or "Unknown")


def get_chunk_score(chunk: dict[str, Any]) -> str:
    """Return a readable similarity score from a retrieved chunk."""
    score = chunk.get("similarity", chunk.get("score"))
    if score is None:
        return "N/A"
    if isinstance(score, float):
        return f"{score:.4f}"
    return str(score)


def print_retrieved_chunks(chunks: list[dict[str, Any]]) -> None:
    """Display retrieved chunks for debugging before answer generation."""
    print("Retrieved Chunks")
    print()
    print("----------------------------")
    print()

    for index, chunk in enumerate(chunks, start=1):
        print(f"Chunk {index}")
        print(f"Source: {get_chunk_source(chunk)}")
        print(f"Score: {get_chunk_score(chunk)}")
        print()
        print(get_chunk_text(chunk))
        print()
        print("----------------------------")
        print()


def build_prompt(question: str, chunks: list[dict[str, Any]], template: str) -> str:
    """Build the RAG prompt using the existing prompt template."""
    context = "\n\n".join(get_chunk_text(chunk) for chunk in chunks)
    return template.format(context=context, question=question)


def print_answer(answer: str) -> None:
    """Print the final answer in the requested format."""
    print("Answer")
    print()
    print("--------------------------------")
    print()
    print(answer)
    print()
    print("--------------------------------")
    print()


def answer_question(question: str, components: RagComponents) -> None:
    """Run retrieval, prompt construction, LLM generation, and display."""
    try:
        chunks = components.retrieve(question, components.top_k)
    except Exception as exc:
        print(
            "Retrieval failed. Check that embeddings can load and that the "
            "pgvector chunks table exists with indexed data."
        )
        print(f"Details: {exc}")
        print()
        return

    if not chunks:
        print("No relevant information found.")
        print()
        return

    print_retrieved_chunks(chunks)

    try:
        prompt = build_prompt(question, chunks, components.rag_prompt)
    except Exception as exc:
        print("Prompt construction failed.")
        print(f"Details: {exc}")
        print()
        return

    try:
        answer = components.call_llm(prompt)
    except Exception as exc:
        print("LLM call failed. Check your API key, model, and network access.")
        print(f"Details: {exc}")
        print()
        return

    print_answer(answer)


def prompt_loop(components: RagComponents) -> None:
    """Continuously accept user questions until exit or quit."""
    while True:
        question = input("Question:\n").strip()
        if question.lower() in {"exit", "quit"}:
            print("Goodbye.")
            return
        if not question:
            print("Please enter a question or type 'exit' to quit.")
            print()
            continue
        answer_question(question, components)


def main() -> int:
    """Program entry point."""
    print_header()
    try:
        components = load_components()
    except RuntimeError as exc:
        print("Startup failed.")
        print(str(exc))
        return 1

    prompt_loop(components)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
