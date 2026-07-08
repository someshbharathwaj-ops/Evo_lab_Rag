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
    # Automatically initialize/migrate the database table in the background on startup
    def init_db():
        try:
            create_table()
            print("Database table initialized and verified in the background.", flush=True)
        except Exception as exc:
            print(f"Database initialization warning: {exc}", flush=True)

    threading.Thread(target=init_db, daemon=True).start()
    yield

app = FastAPI(title="Evo Lab RAG API", lifespan=lifespan)

# Enable CORS so frontend can call it directly if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

@app.post("/query", response_model=QueryResponse)
def query_endpoint(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        answer = run_rag(request.query)
        return QueryResponse(response=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/")
def read_root():
    return {"message": "Evo Lab RAG Backend API is running"}
