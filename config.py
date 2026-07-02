"""Environment-backed configuration for ingestion and local RAG."""

from __future__ import annotations

import os

from dotenv import load_dotenv


load_dotenv()


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _get_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc


# Google Gemini API Settings
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-2.5-flash")
MAX_TOKENS = _get_int("MAX_TOKENS", 800)
TEMPERATURE = _get_float("TEMPERATURE", 0.0)
LLM_TIMEOUT = _get_float("LLM_TIMEOUT", 120.0)

# Legacy Ollama settings kept for compatibility
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/")
OLLAMA_REASONING_EFFORT = os.getenv("OLLAMA_REASONING_EFFORT", "none")

# Keep the project's existing embedding model. Its dimension is detected at runtime.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
EMBEDDING_BATCH_SIZE = _get_int("EMBEDDING_BATCH_SIZE", 32)

CHUNK_SIZE = _get_int("CHUNK_SIZE", 800)
CHUNK_OVERLAP = _get_int("CHUNK_OVERLAP", 200)
CHUNKS_PATH = os.getenv("CHUNKS_PATH", os.path.join("data", "chunks", "chunks.json"))

TOP_K = _get_int("TOP_K", 5)
SCORE_THRESHOLD = _get_float("SCORE_THRESHOLD", 0.0)
MAX_CONTEXT_CHARS = _get_int("MAX_CONTEXT_CHARS", 16000)

# DATABASE_URL is the easiest option for Supabase's direct or pooler connection URL.
DATABASE_URL = os.getenv("DATABASE_URL", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = _get_int("DB_PORT", 5432)
DB_NAME = os.getenv("DB_NAME", "rag_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")
DB_SSLMODE = os.getenv("DB_SSLMODE", "prefer")
DB_CONNECT_TIMEOUT = _get_int("DB_CONNECT_TIMEOUT", 10)
DB_SCHEMA = os.getenv("DB_SCHEMA", "public")
DB_TABLE = os.getenv("DB_TABLE", "chunks")


if CHUNK_SIZE <= 0:
    raise ValueError("CHUNK_SIZE must be greater than zero")
if not 0 <= CHUNK_OVERLAP < CHUNK_SIZE:
    raise ValueError("CHUNK_OVERLAP must be at least zero and smaller than CHUNK_SIZE")
if TOP_K <= 0:
    raise ValueError("TOP_K must be greater than zero")
if not -1.0 <= SCORE_THRESHOLD <= 1.0:
    raise ValueError("SCORE_THRESHOLD must be between -1 and 1")
