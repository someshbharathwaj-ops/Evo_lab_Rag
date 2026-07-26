"""
LLM-as-a-Judge answer verification for the RAG pipeline.

After the generator LLM produces a candidate answer, this module asks a
judge model to evaluate it for factual correctness, grounding, hallucination,
and completeness — then returns a structured verdict.

The judge uses the same LLM_API_KEY / OLLAMA_BASE_URL infrastructure but
can be pointed at a different model via JUDGE_MODEL in .env.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge result dataclass
# ---------------------------------------------------------------------------

@dataclass
class JudgeResult:
    """Structured verdict from the LLM judge."""

    passed: bool                          # True = answer is acceptable
    score: float                          # 0.0 – 1.0 overall quality
    hallucination: bool                   # True if unsupported claims detected
    missing_information: list[str] = field(default_factory=list)
    unsupported_claims: list[str] = field(default_factory=list)
    feedback: str = ""                    # Human-readable explanation
    revised_answer: str = ""             # Judge's improved answer (may be empty)
    raw_response: str = ""               # Full raw LLM output for debugging


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are a strict factual accuracy evaluator for a Retrieval-Augmented Generation system.

## Your Task

Evaluate whether the provided **Generated Answer** is:
1. Factually correct and fully grounded in the **Retrieved Context**
2. Free of hallucinations (claims not supported by the context)
3. Complete — does it answer what the question actually asks?
4. Consistent — no internal contradictions?

## Inputs

**Question:**
{question}

**Retrieved Context:**
{context}

**Generated Answer:**
{answer}

## Evaluation Criteria

Score each dimension 0.0–1.0:

| Criterion | Description |
|---|---|
| Relevance | Does the answer address the question? |
| Groundedness | Are all claims supported by the retrieved context? |
| Completeness | Does it cover the key points from context relevant to the question? |
| Correctness | Are the facts accurate relative to the context? |
| No-Hallucination | Absence of invented facts not in the context? |

## Output Format

Respond with ONLY a valid JSON object. No markdown, no explanation, just the JSON:

{{
  "passed": <true if score >= 0.75 and hallucination is false, else false>,
  "score": <weighted average of all criteria, 0.0 to 1.0>,
  "hallucination": <true if any claim is not supported by context>,
  "missing_information": ["<point from context that should have been included>", ...],
  "unsupported_claims": ["<exact claim from answer that is not in context>", ...],
  "feedback": "<1–3 sentence explanation of your verdict>",
  "revised_answer": "<improved answer if you can do better, otherwise empty string>"
}}

Be strict. If the answer introduces ANY fact not present in the retrieved context, set hallucination=true and passed=false."""


# ---------------------------------------------------------------------------
# Judge function
# ---------------------------------------------------------------------------

def judge_answer(
    question: str,
    context: str,
    answer: str,
    judge_model: str,
    max_tokens: int = 1200,
) -> JudgeResult:
    """
    Ask the judge LLM to evaluate a generated answer.

    Args:
        question    : The original user query.
        context     : The retrieved context passed to the generator.
        answer      : The candidate answer produced by the generator LLM.
        judge_model : HuggingFace / API model identifier for the judge.
        max_tokens  : Token budget for the judge's JSON response.

    Returns:
        JudgeResult with structured verdict.
    """
    from rag.llm_client import _get_client  # reuse the existing OpenAI-compat client

    prompt = JUDGE_PROMPT.format(
        question=question.strip(),
        context=context.strip(),
        answer=answer.strip(),
    )

    logger.info("[Judge] Evaluating answer with model '%s'...", judge_model)

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,          # deterministic judgement
            max_tokens=max_tokens,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:
        logger.warning(
            "[Judge] LLM call failed (%s). Treating as PASS to avoid blocking.", exc
        )
        return JudgeResult(
            passed=True,
            score=0.5,
            hallucination=False,
            feedback=f"Judge unavailable ({exc}). Answer returned unverified.",
            raw_response="",
        )

    logger.debug("[Judge] Raw response: %s", raw[:500])
    return _parse_judge_response(raw)


def _parse_judge_response(raw: str) -> JudgeResult:
    """Parse judge JSON response defensively. Falls back to PASS on parse failure."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    # Try to extract the first {...} block if there's extra text around it
    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning(
            "[Judge] Could not parse JSON response (%s). Defaulting to PASS.\nRaw: %s",
            exc,
            raw[:300],
        )
        return JudgeResult(
            passed=True,
            score=0.5,
            hallucination=False,
            feedback="Judge response was not valid JSON. Answer returned unverified.",
            raw_response=raw,
        )

    passed = bool(data.get("passed", True))
    score = float(data.get("score", 0.5))
    hallucination = bool(data.get("hallucination", False))
    missing = [str(m) for m in data.get("missing_information", [])]
    unsupported = [str(u) for u in data.get("unsupported_claims", [])]
    feedback = str(data.get("feedback", ""))
    revised = str(data.get("revised_answer", ""))

    result = JudgeResult(
        passed=passed,
        score=score,
        hallucination=hallucination,
        missing_information=missing,
        unsupported_claims=unsupported,
        feedback=feedback,
        revised_answer=revised,
        raw_response=raw,
    )

    logger.info(
        "[Judge] Verdict: %s  score=%.2f  hallucination=%s  feedback=%s",
        "PASS" if passed else "FAIL",
        score,
        hallucination,
        feedback[:120],
    )
    return result
