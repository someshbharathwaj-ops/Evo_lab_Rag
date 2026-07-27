from contextlib import asynccontextmanager
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag.rag_pipeline import run_rag
from ingestion.vectorstore.pgvector_store import create_table

import threading

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Sequential startup initialization of database schema and model loading
    print("[Startup] Initializing database and embedding configurations...", flush=True)
    try:
        create_table()
        print("[Startup] Database table and embedding model initialized successfully.", flush=True)
    except Exception as exc:
        print(f"[Startup] Warning during database/model initialization: {exc}", flush=True)
    yield

app = FastAPI(title="Evo Lab RAG API", lifespan=lifespan)

origins = [
    "http://localhost:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3000",
    "https://rag-evolab-ui.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https://rag-evolab-ui-.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class SourceInfo(BaseModel):
    source: str
    page: int | None


class QueryResponse(BaseModel):
    response: str
    sources: list[SourceInfo] = []


@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        from rag.pipeline import run_pipeline, NO_CONTEXT_ANSWER
        import os

        result = run_pipeline(query=request.query.strip())

        if not result.answer:
            return QueryResponse(response=result.answer or NO_CONTEXT_ANSWER, sources=[])

        # Deduplicate sources from pipeline result
        unique_sources = []
        seen: set[tuple] = set()
        for s in result.sources:
            src = s.get("source") or "unknown"
            filename = os.path.basename(src)
            pg = s.get("page")
            key = (filename, pg)
            if key not in seen:
                seen.add(key)
                unique_sources.append(SourceInfo(source=filename, page=pg))

        return QueryResponse(response=result.answer, sources=unique_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/debug-env")
def debug_env():
    import os
    from config import (
        EMBEDDING_BASE_URL,
        EMBEDDING_MODEL,
        DB_HOST,
        DB_NAME,
        LLM_MODEL,
        OLLAMA_BASE_URL
    )
    return {
        "EMBEDDING_BASE_URL": EMBEDDING_BASE_URL,
        "EMBEDDING_MODEL": EMBEDDING_MODEL,
        "DB_HOST": DB_HOST,
        "DB_NAME": DB_NAME,
        "LLM_MODEL": LLM_MODEL,
        "OLLAMA_BASE_URL": OLLAMA_BASE_URL,
        "HAS_EMBEDDING_API_KEY": bool(os.getenv("EMBEDDING_API_KEY")),
        "EMBEDDING_API_KEY_LEN": len(os.getenv("EMBEDDING_API_KEY", "")),
        "HAS_LLM_API_KEY": bool(os.getenv("LLM_API_KEY")),
        "LLM_API_KEY_LEN": len(os.getenv("LLM_API_KEY", "")),
    }

@app.get("/clean-db")
def clean_database():
    try:
        from ingestion.vectorstore.pgvector_store import delete_nan_chunks
        deleted = delete_nan_chunks()
        return {
            "status": "success",
            "deleted_rows": deleted,
            "message": f"Cleaned up {deleted} invalid/NaN vector chunks from database",
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Evo Lab RAG Backend API is running"}
