"""Local Ollama client using its OpenAI-compatible endpoint."""

from __future__ import annotations

from functools import lru_cache

from openai import OpenAI

from config import (
    LLM_MODEL,
    LLM_TIMEOUT,
    MAX_TOKENS,
    OLLAMA_BASE_URL,
    OLLAMA_REASONING_EFFORT,
    TEMPERATURE,
)


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    return OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",
        timeout=LLM_TIMEOUT,
        max_retries=1,
    )


def call_llm(prompt: str) -> str:
    """Generate an answer with local Qwen3, raising actionable failures."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    try:
        response = _get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=MAX_TOKENS,
            temperature=TEMPERATURE,
            reasoning_effort=OLLAMA_REASONING_EFFORT,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed at {OLLAMA_BASE_URL} using model "
            f"'{LLM_MODEL}': {exc}. Confirm Ollama is running and execute "
            f"'ollama pull {LLM_MODEL}'."
        ) from exc

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(f"Ollama model '{LLM_MODEL}' returned an empty response")
    return content.strip()
