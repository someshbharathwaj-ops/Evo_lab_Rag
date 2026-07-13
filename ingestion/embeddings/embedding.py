"""Shared SentenceTransformer embedding adapter with cloud fallback."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Sequence
import requests

from config import EMBEDDING_BATCH_SIZE, EMBEDDING_MODEL

# Toggle fallback manually or fallback automatically if import/load fails
USE_HF_INFERENCE = os.getenv("USE_HF_INFERENCE", "false").lower() == "true"

_session = requests.Session()


@lru_cache(maxsize=1)
def _get_local_model():
    try:
        from sentence_transformers import SentenceTransformer
        return SentenceTransformer(EMBEDDING_MODEL)
    except Exception as exc:
        # If we cannot load locally, raise exception to trigger API fallback
        raise RuntimeError(f"Local model load failed: {exc}") from exc


def _embed_hf_api(text: str) -> list[float]:
    """Fallback: Query Hugging Face Inference API for embeddings."""
    api_url = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        
    try:
        response = _session.post(api_url, headers=headers, json={"inputs": text}, timeout=15)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                # If nested, unpack it
                if isinstance(result[0], list):
                    return result[0]
                return result
        raise RuntimeError(f"HF API returned status {response.status_code}: {response.text}")
    except Exception as e:
        raise RuntimeError(f"Hugging Face Inference API query failed: {e}") from e


def _embed_texts_hf_api(texts: Sequence[str]) -> list[list[float]]:
    """Fallback: Batch query Hugging Face Inference API."""
    api_url = f"https://api-inference.huggingface.co/models/{EMBEDDING_MODEL}"
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"
        
    try:
        response = _session.post(api_url, headers=headers, json={"inputs": list(texts)}, timeout=30)
        if response.status_code == 200:
            result = response.json()
            if isinstance(result, list):
                return result
        raise RuntimeError(f"HF API returned status {response.status_code}: {response.text}")
    except Exception as e:
        raise RuntimeError(f"Hugging Face Inference API batch query failed: {e}") from e


@lru_cache(maxsize=1)
def _get_embedding_client():
    from config import EMBEDDING_BASE_URL, EMBEDDING_API_KEY
    import openai
    return openai.OpenAI(
        base_url=EMBEDDING_BASE_URL,
        api_key=EMBEDDING_API_KEY,
        timeout=30.0,
    )


def get_embedding_dimension() -> int:
    """Read the actual output dimension from the configured model."""
    from config import EMBEDDING_API_KEY
    if EMBEDDING_API_KEY:
        try:
            return len(embed_text("dimension probe"))
        except Exception as exc:
            print(f"[Embedding] Failed to probe dimension via API: {exc}")

    if not USE_HF_INFERENCE:
        try:
            model = _get_local_model()
            dimension = model.get_sentence_embedding_dimension()
            if dimension:
                return int(dimension)
        except Exception:
            pass
    return len(embed_text("dimension probe"))


def embed_text(text: str) -> list[float]:
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text")

    from config import EMBEDDING_API_KEY, EMBEDDING_MODEL
    if EMBEDDING_API_KEY:
        try:
            client = _get_embedding_client()
            extra_body = {}
            if "nvidia.com" in client.base_url.host:
                extra_body["input_type"] = "query"

            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=[text],
                extra_body=extra_body if extra_body else None,
            )
            return response.data[0].embedding
        except Exception as exc:
            raise RuntimeError(f"NVIDIA/OpenAI embedding failed: {exc}") from exc

    if USE_HF_INFERENCE:
        return _embed_hf_api(text)

    try:
        model = _get_local_model()
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as exc:
        # Automatic fallback if local generation fails (e.g., out of memory on Render)
        print(f"[Embedding] Local generation failed ({exc}). Falling back to HF Inference API...")
        return _embed_hf_api(text)


def embed_texts(texts: Sequence[str]) -> list[list[float]]:
    if not texts:
        return []
    if any(not text or not text.strip() for text in texts):
        raise ValueError("Cannot embed empty text")

    from config import EMBEDDING_API_KEY, EMBEDDING_MODEL
    if EMBEDDING_API_KEY:
        try:
            client = _get_embedding_client()
            extra_body = {}
            if "nvidia.com" in client.base_url.host:
                extra_body["input_type"] = "passage"

            # Batch in chunks of 100 to prevent API timeouts and payload limits
            batch_size = 100
            results = []
            texts_list = list(texts)
            for i in range(0, len(texts_list), batch_size):
                batch = texts_list[i : i + batch_size]
                response = client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch,
                    extra_body=extra_body if extra_body else None,
                )
                results.extend([d.embedding for d in response.data])
            return results
        except Exception as exc:
            raise RuntimeError(f"NVIDIA/OpenAI batch embedding failed: {exc}") from exc

    if USE_HF_INFERENCE:
        return _embed_texts_hf_api(texts)

    try:
        model = _get_local_model()
        embeddings = model.encode(
            list(texts),
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()
    except Exception as exc:
        print(f"[Embedding] Local batch generation failed ({exc}). Falling back to HF Inference API...")
        return _embed_texts_hf_api(texts)
