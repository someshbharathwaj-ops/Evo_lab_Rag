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
        from rag.retriever import retrieve
        from rag.prompts import build_context, RAG_PROMPT
        from rag.llm_client import call_llm
        from rag.rag_pipeline import NO_CONTEXT_ANSWER
        import os

        chunks = retrieve(request.query)
        if not chunks:
            return QueryResponse(response=NO_CONTEXT_ANSWER, sources=[])

        context = build_context(chunks)
        answer = call_llm(RAG_PROMPT.format(context=context, question=request.query.strip()))

        unique_sources = []
        seen = set()
        for c in chunks:
            src = c.get("source") or "unknown"
            filename = os.path.basename(src)
            pg = c.get("page")
            key = (filename, pg)
            if key not in seen:
                seen.add(key)
                unique_sources.append(SourceInfo(source=filename, page=pg))

        return QueryResponse(response=answer, sources=unique_sources)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Evo Lab RAG Backend API is running"}
