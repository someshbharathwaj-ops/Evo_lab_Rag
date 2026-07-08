"""Ollama client using the openai SDK."""

from __future__ import annotations

from functools import lru_cache
import openai

from config import (
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    OLLAMA_BASE_URL,
)


@lru_cache(maxsize=1)
def _get_client() -> openai.OpenAI:
    headers = {}
    if "ngrok" in OLLAMA_BASE_URL:
        headers["ngrok-skip-browser-warning"] = "true"
    return openai.OpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key="ollama",  # Placeholder required by the OpenAI interface
        default_headers=headers,
        timeout=120.0,
        max_retries=2,
    )


def call_llm(prompt: str) -> str:
    """Generate an answer with local Ollama, raising actionable failures."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed using model '{LLM_MODEL}': {exc}"
        ) from exc

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(f"Ollama model '{LLM_MODEL}' returned an empty response")
    return content.strip()

