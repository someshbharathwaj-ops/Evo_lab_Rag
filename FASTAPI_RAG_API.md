# FastAPI RAG API Documentation

## Purpose

This service exposes the existing local RAG pipeline through FastAPI so ngrok can
publish the API instead of publishing Ollama directly.

The underlying RAG implementation is unchanged. The API imports and calls the
existing pipeline:

```python
from rag.rag_pipeline import run_rag
```

## Architecture

Before:

```text
User -> run_rag.py -> Retriever -> Supabase PostgreSQL + pgvector
     -> Prompt Builder -> local Ollama -> Answer
```

After:

```text
Internet -> ngrok -> FastAPI -> existing run_rag()
                         -> Retriever
                         -> Supabase PostgreSQL + pgvector
                         -> Prompt Builder
                         -> local Ollama
                         -> Answer
```

Ollama must remain local-only. Expose FastAPI with ngrok on port `8000`; do not
expose Ollama on port `11434`.

## Files Changed

### `api/main.py`

The API wrapper was tightened to:

- force the API process to use local Ollama:
  `http://localhost:11434/v1`
- force the API process to use the local Ollama placeholder API key:
  `ollama`
- expose `GET /`
- expose `GET /health`
- expose `POST /v1/rag/query`
- log complete health-check tracebacks internally
- return a safe failed-health response without exposing traceback details
- verify the configured Ollama model exists in `/api/tags`
- use FastAPI lifespan startup logging instead of deprecated `on_event`
- run on `0.0.0.0:8000` when executed with `python -m api.main`

No RAG, retrieval, embedding, prompt, ingestion, database, or configuration
modules were modified.

## Endpoints

### `GET /`

Returns:

```json
{
  "service": "RAG API",
  "status": "running"
}
```

### `GET /health`

Checks:

- PostgreSQL connectivity
- pgvector extension availability
- retriever readiness through embedding model dimension and stored chunk count
- local Ollama reachability
- configured Ollama model availability

Successful response:

```json
{
  "database": "ok",
  "retriever": "ok",
  "ollama": "ok"
}
```

Failed response:

```json
{
  "status": "failed",
  "component": "database",
  "reason": "failure reason"
}
```

The failed response uses HTTP `503`. Full tracebacks are logged internally by
the server.

### `POST /v1/rag/query`

Request:

```json
{
  "question": "What is Evolutionary Algorithm?"
}
```

Response:

```json
{
  "answer": "Generated answer from the existing RAG pipeline."
}
```

The endpoint does not rebuild retrieval, prompts, embeddings, or database
connections. It delegates to the existing `run_rag()` function.

## Startup

Start Ollama locally:

```powershell
ollama serve
```

Ensure the model exists:

```powershell
ollama pull qwen3
```

Start the API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Or:

```powershell
.\.venv\Scripts\python.exe -m api.main
```

Expected startup logs:

```text
Loading configuration...
Connecting database...
Connecting Ollama...
Loading retriever...
API Ready
```

## Ngrok

Expose FastAPI:

```powershell
ngrok http 8000
```

Do not expose Ollama:

```powershell
ngrok http 11434
```

## Verification Commands

Root endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/
```

Health endpoint:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Query endpoint:

```powershell
$body = @{ question = "What is Evolutionary Algorithm?" } | ConvertTo-Json
Invoke-RestMethod `
  -Uri http://127.0.0.1:8000/v1/rag/query `
  -Method Post `
  -ContentType "application/json" `
  -Body $body
```

Original pipeline compatibility:

```powershell
.\.venv\Scripts\python.exe run_rag.py
```

## Last Verification Results

The latest runtime verification produced:

```json
GET /
{"service":"RAG API","status":"running"}
```

```json
GET /health
{"database":"ok","retriever":"ok","ollama":"ok"}
```

```json
POST /v1/rag/query
{"answer":"An evolutionary algorithm ..."}
```

Local Ollama was verified to include `qwen3:latest`.

The existing vector store was verified with `4806` stored chunks.

## Errors Found During Verification

### FastAPI response model error

Cause:

`/health` returned `dict[str, str] | JSONResponse`, and FastAPI tried to create
a Pydantic response model for that union.

Fix:

`response_model=None` was added to the `/health` route.

### Database sandbox error

Cause:

The sandbox initially blocked the external Supabase PostgreSQL connection.

Fix:

The database check was rerun with approved network access. No code change was
required.

### Existing unit test mismatch

Cause:

`.env` points the core RAG client at an ngrok/OpenRouter-style configuration,
while the existing unit test expects local Ollama credentials. The existing
`rag.llm_client` also passes `default_headers`, which the test does not expect.

Fix:

No code change was applied because the RAG implementation must remain unchanged.

## Backward Compatibility

The original pipeline remains available:

```python
from rag.rag_pipeline import run_rag

answer = run_rag("What is Evolutionary Algorithm?")
```

The following modules were not changed:

- `run_rag.py`
- `rag/rag_pipeline.py`
- `rag/retriever.py`
- `rag/prompts.py`
- `rag/llm_client.py`
- `ingestion/`
- `vectorstores/`
- `config.py`
- database schema and vector store implementation

