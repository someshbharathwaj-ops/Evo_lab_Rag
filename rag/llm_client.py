import time
from openai import OpenAI
from config import LLM_MODEL, MAX_TOKENS, TEMPERATURE, OPENROUTER_API_KEY


_client = None

MAX_RETRIES = 5
RETRY_BASE_DELAY = 30  # seconds (OpenRouter free tier uses ~29s cooldowns)


def _get_client():
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client


def call_llm(prompt: str) -> str:
    """
    Calls OpenRouter API (OpenAI-compatible) with the configured model.
    Retries automatically on 429 rate-limit errors.
    Always returns a string.
    """
    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            err = str(e)
            if "429" in err and attempt < MAX_RETRIES:
                wait = RETRY_BASE_DELAY
                print(f"    [LLM] Rate limited (attempt {attempt}/{MAX_RETRIES}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            return f"LLM error: {err}"
