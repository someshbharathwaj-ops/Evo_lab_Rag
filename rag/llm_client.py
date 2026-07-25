"""Ollama client using the openai SDK."""

from __future__ import annotations

from functools import lru_cache
import openai

import re
from config import (
    LLM_MODEL,
    MAX_TOKENS,
    TEMPERATURE,
    OLLAMA_BASE_URL,
    OLLAMA_REASONING_EFFORT,
)


def clean_html_math(text: str) -> str:
    if not text:
        return text
    # Convert f(x<sub>i</sub>) -> $f(x_i)$
    text = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)\(([^<]*?)<sub>([^<]+?)</sub>\)',
        r'$\1(\2_{\3})$',
        text
    )
    # Convert x<sub>i</sub> -> $x_i$
    text = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)<sub>([^<]+?)</sub>',
        r'$\1_{\2}$',
        text
    )
    # Convert x<sup>2</sup> -> $x^{2}$
    text = re.sub(
        r'([a-zA-Z_][a-zA-Z0-9_]*)<sup>([^<]+?)</sup>',
        r'$\1^{\2}$',
        text
    )
    # Convert any remaining <sub>content</sub> -> _{content}
    text = re.sub(r'<sub>([^<]+?)</sub>', r'_{\1}', text)
    # Convert any remaining <sup>content</sup> -> ^{\1}
    text = re.sub(r'<sup>([^<]+?)</sup>', r'^{\1}', text)
    # Clean up double wrapped dollars like $$x_i$$ or similar into $x_i$
    text = re.sub(r'\$\$([^$]+?)\$\$', r'$\1$', text)
    return text


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
        extra_body = {}
        if OLLAMA_REASONING_EFFORT.lower() == "none":
            extra_body["think"] = False
        else:
            extra_body["think"] = True
            extra_body["reasoning_effort"] = OLLAMA_REASONING_EFFORT

        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body=extra_body,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Ollama request failed using model '{LLM_MODEL}': {exc}"
        ) from exc

    content = response.choices[0].message.content
    if not content or not content.strip():
        raise RuntimeError(f"Ollama model '{LLM_MODEL}' returned an empty response")
    return clean_html_math(content.strip())

