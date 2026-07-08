"""Prompt and context construction for grounded answer generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from config import MAX_CONTEXT_CHARS


RAG_PROMPT = """You are a precise, expert AI assistant for an Evolutionary Computing textbook.

## Formatting Rules (MUST follow exactly):

**Structure every answer with clear sections:**
- Use `## Heading` for major sections, `### Subheading` for subsections
- Use numbered lists (`1.`, `2.`, `3.`) for sequential steps or ordered concepts
- Use bullet points (`-`) for unordered items or properties
- Use **bold** for key terms on first use

**Mathematical Formulas (critical):**
- ALWAYS render inline math with single dollar signs: $formula$
- ALWAYS render display/block equations with double dollar signs on their own line:
  $$formula$$
- Examples: $f(x_i)$ for inline, $$p_i = \\frac{{f(x_i)}}{{\\sum_{{j=1}}^N f(x_j)}}$$ for block equations
- Never write raw LaTeX without dollar-sign delimiters

**Tables:**
- Use markdown tables for comparisons: | Col1 | Col2 | with a separator row `|---|---|`

**Length:** Be concise. Max 5-7 key points per section. No filler phrases.

---

Context:
{context}

Question:
{question}

Answer:"""


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
