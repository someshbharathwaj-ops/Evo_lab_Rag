"""Minimal FastAPI wrapper around the existing local RAG pipeline."""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from typing import Any

# The API process must talk to local Ollama. Public traffic should terminate at
# FastAPI/ngrok, never at Ollama itself.
os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434/v1"
os.environ["LLM_API_KEY"] = "ollama"

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

print("Loading configuration...")
from config import LLM_MODEL, OLLAMA_BASE_URL

print("Connecting database...")
from ingestion.vectorstore.pgvector_store import _connection, count_chunks

print("Connecting Ollama...")
from rag.llm_client import _get_client

print("Loading retriever...")
from ingestion.embeddings.embedding import get_embedding_dimension
from rag.rag_pipeline import run_rag


app = FastAPI(title="RAG API", version="1.0.0")


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
    answer: str


@app.on_event("startup")
def startup_log() -> None:
    print("API Ready")


@app.get("/")
def root() -> dict[str, str]:
    return {"service": "RAG API", "status": "running"}


def _check_database_and_pgvector() -> str:
    with _connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT 1")
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        if cur.fetchone() is None:
            raise RuntimeError("pgvector extension is not installed")
    return "ok"


def _check_ollama() -> str:
    tags_url = OLLAMA_BASE_URL.removesuffix("/v1") + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=10) as response:
            if response.status >= 400:
                raise RuntimeError(f"Ollama returned HTTP {response.status}")
            body = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"Ollama returned HTTP {exc.code}: {exc.reason}") from exc
    except Exception as exc:
        raise RuntimeError(f"Ollama is not reachable at {tags_url}: {exc}") from exc

    if LLM_MODEL not in body:
        raise RuntimeError(f"Ollama model '{LLM_MODEL}' is not available")

    _get_client()
    return "ok"


def _check_retriever() -> str:
    dimension = get_embedding_dimension()
    if dimension <= 0:
        raise RuntimeError("embedding model returned an invalid dimension")
    chunks = count_chunks()
    if chunks <= 0:
        raise RuntimeError("vector table is empty; run ingestion before querying")
    return "ok"


@app.get("/health")
def health() -> dict[str, str]:
    checks: dict[str, Any] = {
        "database": _check_database_and_pgvector,
        "ollama": _check_ollama,
        "retriever": _check_retriever,
    }
    result: dict[str, str] = {}
    for name, check in checks.items():
        try:
            result[name] = check()
        except Exception as exc:
            result[name] = str(exc)
    return result


@app.post("/v1/rag/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    try:
        return QueryResponse(answer=run_rag(request.question))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def main() -> None:
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
