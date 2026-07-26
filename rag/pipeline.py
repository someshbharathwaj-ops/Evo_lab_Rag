"""
Production RAG pipeline orchestrator.

Full pipeline:

  User Query
      ↓
  Embedding Search  (retriever.py)
      ↓
  [Optional] Cross-Encoder Reranking  (reranker.py)
      ↓
  Context Builder   (prompts.py)
      ↓
  Generator LLM     (llm_client.py)
      ↓
  [Optional] LLM-as-a-Judge  (judge.py)
      ↓
  Final Answer

Feature flags (all default to off for backward compatibility):
    ENABLE_RERANKING=true/false
    RERANKER_MODEL=<any HuggingFace cross-encoder>
    ENABLE_JUDGE=true/false
    JUDGE_MODEL=<model name, defaults to LLM_MODEL>
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from config import (
    ENABLE_JUDGE,
    ENABLE_RERANKING,
    FINAL_TOP_K,
    JUDGE_MAX_TOKENS,
    JUDGE_MODEL,
    MAX_REGENERATE_ATTEMPTS,
    RERANK_CANDIDATE_K,
    RERANKER_MODEL,
    TOP_K,
)
from rag.llm_client import call_llm
from rag.prompts import build_context, build_rag_prompt
from rag.retriever import retrieve

logger = logging.getLogger(__name__)

NO_CONTEXT_ANSWER = "I do not know based on the available documents."


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """Full result including answer, sources, and diagnostic metadata."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)

    # Diagnostics (populated regardless of feature flags)
    retrieved_count: int = 0
    reranked_count: int = 0
    reranking_used: bool = False
    judge_used: bool = False
    judge_passed: bool | None = None
    judge_score: float | None = None
    judge_feedback: str = ""
    attempts: int = 1


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    query: str,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    score_threshold: float | None = None,
) -> PipelineResult:
    """
    Execute the full RAG pipeline and return a PipelineResult.

    Args:
        query           : User's natural language question.
        top_k           : Override TOP_K from config (pre-reranking if reranking enabled).
        metadata_filter : Optional pgvector metadata filter dict.
        score_threshold : Minimum cosine similarity score (0–1).

    Returns:
        PipelineResult with the final answer and diagnostic metadata.
    """
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    result = PipelineResult(answer=NO_CONTEXT_ANSWER)

    # ------------------------------------------------------------------
    # Step 1: Retrieve
    # ------------------------------------------------------------------
    # When reranking is active, over-fetch candidates then rerank down.
    if ENABLE_RERANKING and RERANKER_MODEL:
        candidate_k = RERANK_CANDIDATE_K
    else:
        candidate_k = top_k if top_k is not None else TOP_K

    logger.info("[Pipeline] Retrieving top-%d candidates for query: %r", candidate_k, query[:80])

    chunks = retrieve(
        query,
        top_k=candidate_k,
        metadata_filter=metadata_filter,
        score_threshold=score_threshold,
    )
    result.retrieved_count = len(chunks)
    logger.info("[Pipeline] Retrieved %d chunks", len(chunks))

    if not chunks:
        logger.info("[Pipeline] No chunks retrieved — returning no-context answer")
        return result

    # ------------------------------------------------------------------
    # Step 2: Rerank (optional)
    # ------------------------------------------------------------------
    final_chunks = chunks
    if ENABLE_RERANKING:
        if not RERANKER_MODEL:
            logger.warning(
                "[Pipeline] ENABLE_RERANKING=true but RERANKER_MODEL is not set. "
                "Skipping reranking. Set RERANKER_MODEL in .env to a cross-encoder model."
            )
        else:
            try:
                from rag.reranker import rerank
                final_top_k = top_k if top_k is not None else FINAL_TOP_K
                final_chunks = rerank(
                    query=query,
                    chunks=chunks,
                    top_n=final_top_k,
                    model_name=RERANKER_MODEL,
                )
                result.reranking_used = True
                result.reranked_count = len(final_chunks)
                logger.info(
                    "[Pipeline] Reranked %d → %d chunks", len(chunks), len(final_chunks)
                )
            except ImportError as exc:
                logger.warning(
                    "[Pipeline] Reranking skipped — sentence-transformers not installed: %s", exc
                )
            except Exception as exc:
                logger.warning(
                    "[Pipeline] Reranking failed (%s). Falling back to embedding-ranked chunks.", exc
                )
                # Fall back to top-FINAL_TOP_K by embedding similarity
                final_top_k = top_k if top_k is not None else FINAL_TOP_K
                final_chunks = chunks[:final_top_k]
    else:
        # No reranking — just trim to requested top_k
        effective_k = top_k if top_k is not None else TOP_K
        final_chunks = chunks[:effective_k]

    # Populate sources from final chunks
    result.sources = [
        {
            "source": c.get("source") or (c.get("metadata") or {}).get("source", ""),
            "page": c.get("page") or (c.get("metadata") or {}).get("page"),
        }
        for c in final_chunks
    ]

    # ------------------------------------------------------------------
    # Step 3: Build context
    # ------------------------------------------------------------------
    context = build_context(final_chunks)
    if not context.strip():
        logger.info("[Pipeline] Context is empty after dedup — returning no-context answer")
        return result

    # ------------------------------------------------------------------
    # Step 4: Generate answer (with optional retry loop)
    # ------------------------------------------------------------------
    prompt = build_rag_prompt(query, context)
    answer = ""
    attempt = 0

    while attempt < max(1, MAX_REGENERATE_ATTEMPTS if ENABLE_JUDGE else 1):
        attempt += 1
        logger.info("[Pipeline] Generating answer (attempt %d)...", attempt)

        try:
            answer = call_llm(prompt)
        except Exception as exc:
            logger.error("[Pipeline] LLM generation failed on attempt %d: %s", attempt, exc)
            result.answer = NO_CONTEXT_ANSWER
            result.attempts = attempt
            return result

        logger.info("[Pipeline] Answer generated (%d chars)", len(answer))

        # --------------------------------------------------------------
        # Step 5: Judge (optional)
        # --------------------------------------------------------------
        if not ENABLE_JUDGE:
            break

        result.judge_used = True
        try:
            from rag.judge import judge_answer
            verdict = judge_answer(
                question=query,
                context=context,
                answer=answer,
                judge_model=JUDGE_MODEL,
                max_tokens=JUDGE_MAX_TOKENS,
            )
        except Exception as exc:
            logger.warning("[Pipeline] Judge raised an exception (%s). Returning answer as-is.", exc)
            break

        result.judge_passed = verdict.passed
        result.judge_score = verdict.score
        result.judge_feedback = verdict.feedback

        if verdict.passed:
            logger.info(
                "[Pipeline] ✓ Answer APPROVED by judge  score=%.2f", verdict.score
            )
            # Use judge's revised answer if it provided one and it's substantial
            if verdict.revised_answer and len(verdict.revised_answer) > 50:
                logger.info("[Pipeline] Using judge's revised answer")
                answer = verdict.revised_answer
            break
        else:
            logger.info(
                "[Pipeline] ✗ Answer REJECTED by judge  score=%.2f  feedback=%s",
                verdict.score,
                verdict.feedback[:120],
            )
            if attempt < MAX_REGENERATE_ATTEMPTS:
                # Regenerate with stronger grounding instructions
                prompt = _build_retry_prompt(query, context, answer, verdict.feedback)
                logger.info("[Pipeline] Regenerating with corrective prompt (attempt %d)...", attempt + 1)
            else:
                logger.info(
                    "[Pipeline] Max regeneration attempts reached. "
                    "Returning best available answer."
                )
                # If judge provided a revised answer, use it
                if verdict.revised_answer and len(verdict.revised_answer) > 50:
                    answer = verdict.revised_answer

    result.answer = answer
    result.attempts = attempt
    return result


def _build_retry_prompt(
    query: str, context: str, previous_answer: str, judge_feedback: str
) -> str:
    """Build a corrective prompt for answer regeneration after judge rejection."""
    return (
        f"You previously answered a question but your answer was flagged for quality issues.\n\n"
        f"Judge feedback: {judge_feedback}\n\n"
        f"Previous answer (do NOT copy this verbatim):\n{previous_answer}\n\n"
        f"You MUST correct the issues above. Only use information from the context below.\n"
        f"Do NOT add any information that is not explicitly present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}\n\n"
        f"Corrected Answer:"
    )
