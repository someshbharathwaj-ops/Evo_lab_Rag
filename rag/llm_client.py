"""Google Gemini client using the official google-genai SDK."""

from __future__ import annotations

from functools import lru_cache
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY,
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
)


@lru_cache(maxsize=1)
def _get_client() -> genai.Client:
    return genai.Client(api_key=GEMINI_API_KEY or None)


def call_llm(prompt: str) -> str:
    """Generate an answer with Google Gemini, raising actionable failures."""
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")
    try:
        client = _get_client()
        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                max_output_tokens=MAX_TOKENS,
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            f"Gemini request failed using model '{LLM_MODEL}': {exc}"
        ) from exc

    content = response.text
    if not content or not content.strip():
        raise RuntimeError(f"Gemini model '{LLM_MODEL}' returned an empty response")
    return content.strip()
