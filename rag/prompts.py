"""Prompt and context construction for grounded answer generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from config import MAX_CONTEXT_CHARS


RAG_PROMPT = """You are an expert AI assistant providing authoritative, comprehensive, and perfectly structured answers based strictly on the provided retrieved context documents.

## Formatting & Response Rules (MUST follow strictly):

### 1. Factual Grounding & Missing Information Handling:
- Base your answer **ONLY** on the facts explicitly stated in the **Retrieved Context**.
- Do **NOT** assume, extrapolate, or introduce outside knowledge.
- If the **Retrieved Context** contains the required information, provide a thorough, accurate, and perfectly structured response.
- **Handling Missing Details**: If the retrieved context does **NOT** contain enough details to fully answer the question (or specific parts of the question), you MUST explicitly state which requested details or facts are not present in the provided documents, while still answering any parts that are supported by the context.

### 2. Markdown Structure & Style:
- **Sections & Headings**: Structure every answer with clear headings (`## Main Section`, `### Subsection`).
- **Lists**: Use numbered lists (`1.`, `2.`, `3.`) for ordered steps, algorithms, or processes. Use bullet points (`-`) for features, lists, or unordered properties.
- **Emphasis**: Use **bold text** for key terms on first mention.
- **Tables**: Use markdown tables (`| Header 1 | Header 2 |` with separator `|---|---|`) for comparisons, parameters, or structured data.

### 3. Mathematical Notation (Critical):
- ALWAYS render inline math using single dollar signs: $formula$ (e.g., $f(x_i)$).
- ALWAYS render display / block equations using double dollar signs on dedicated lines:
  $$formula$$
- Use standard LaTeX notation inside dollar signs.
- NEVER use HTML tags like <sub>, <sup>, <i>, or <b> for mathematical expressions or formulas.

---

Context:
{context}

---

Question:
{question}

Answer:"""


def build_no_context_answer(question: str) -> str:
    """Return a structured, informative response when no context matches the query."""
    q_clean = question.strip()
    return (
        f"## Information Not Present in Document\n\n"
        f"I searched the ingested document database for your query: **\"{q_clean}\"**, but could not find relevant context chunks matching this topic.\n\n"
        f"### Details:\n"
        f"- **Requested Topic:** {q_clean}\n"
        f"- **Status:** Details not present in the available document context.\n\n"
        f"### Next Steps:\n"
        f"1. Verify that the relevant document or chapter has been ingested.\n"
        f"2. Try rephrasing your question or using broader keywords."
    )


def build_context(
    chunks: Iterable[dict[str, Any]],
    max_chars: int = MAX_CONTEXT_CHARS,
) -> str:
    """Deduplicate matches and format a bounded, source-labelled context."""
    sections: list[str] = []
    seen_text: set[str] = set()
    current_length = 0

    for chunk in chunks:
        text = str(chunk.get("text", "")).strip()
        normalized = " ".join(text.split()).casefold()
        if not text or normalized in seen_text:
            continue
        seen_text.add(normalized)

        metadata = chunk.get("metadata") or {}
        source = metadata.get("source") or chunk.get("source") or "unknown"
        page = metadata.get("page") or chunk.get("page")
        label = Path(str(source)).name
        if page is not None:
            label += f", page {page}"
        section = f"[Source: {label}]\n{text}"

        remaining = max_chars - current_length
        if remaining <= 0:
            break
        if len(section) > remaining:
            section = section[:remaining].rstrip()
        sections.append(section)
        current_length += len(section) + 2

    return "\n\n".join(sections)


def build_rag_prompt(question: str, context: str) -> str:
    if not question.strip():
        raise ValueError("Question cannot be empty")
    if not context.strip():
        raise ValueError("Context cannot be empty")
    return RAG_PROMPT.format(context=context, question=question.strip())
