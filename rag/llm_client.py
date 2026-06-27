from ollama import chat
from ollama import ResponseError
MODEL_NAME = "gemma3:4b"
MAX_TOKENS = 300
TEMPERATURE = 0.0
def call_llm(prompt: str) -> str:
    """
    Calls Ollama using Python SDK.
    Always returns a string.
    """

    try:
        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": TEMPERATURE,
                "num_predict": MAX_TOKENS
            }
        )

        return response["message"]["content"].strip()

    except ResponseError as e:
        return f"LLM error: {e}"

    except Exception as e:
        return f"LLM error: {str(e)}"
