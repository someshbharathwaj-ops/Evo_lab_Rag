# Deployment Architecture

## Current system

The current backend is a Python RAG pipeline with these runtime pieces:

```text
PDF files
  -> ingestion.pipeline
  -> PyMuPDF text loader
  -> token splitter
  -> SentenceTransformer embeddings
  -> Supabase PostgreSQL + pgvector
  -> rag.retriever
  -> rag.prompts
  -> local Ollama qwen3
  -> generated answer
```

The repository currently exposes the system through CLI entry points:

- Ingestion: `python -m ingestion.pipeline <pdf paths>`
- Question answering: `python run_rag.py`
- Programmatic RAG call: `rag.rag_pipeline.run_rag(...)`

The next deployment stage should wrap the existing Python modules behind an API
service so a frontend or another backend can call the RAG system through stable
HTTP endpoints.

## Target deployment architecture

```text
Frontend app
  -> HTTPS API calls with API key
  -> Backend API service
     -> RAG pipeline
        -> Supabase PostgreSQL + pgvector
        -> Ollama / hosted LLM endpoint
     -> Ingestion jobs
        -> PDF loader
        -> chunker
        -> embeddings
        -> Supabase PostgreSQL + pgvector
```

Recommended service split:

- API service: HTTP endpoints for question answering, ingestion requests,
  health checks, and document status.
- Database: Supabase PostgreSQL with pgvector enabled.
- LLM service: local Ollama for local/internal deployment, or a hosted
  OpenAI-compatible provider for cloud deployment.
- File storage: local disk for development; object storage such as Supabase
  Storage, S3, or Cloudflare R2 for production PDF uploads.
- Worker process: optional but recommended for large PDFs so ingestion does not
  block frontend requests.

## Deployment steps

### 1. Prepare the environment

Install Python 3.10 or newer on the deployment host.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Create the runtime environment file:

```powershell
Copy-Item .env.example .env
```

Set these required values in `.env`:

```text
DATABASE_URL=<supabase postgres connection string>
OLLAMA_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3
EMBEDDING_MODEL=BAAI/bge-base-en-v1.5
```

For production, also set:

```text
API_KEYS=<comma separated frontend/backend API keys>
ALLOWED_ORIGINS=<comma separated frontend origins>
ENVIRONMENT=production
```

These API settings are planned deployment values. They will need to be read by
the future API service when the HTTP layer is added.

### 2. Prepare Supabase PostgreSQL

Create or select a Supabase project.

Enable pgvector and create the chunks table by running:

```sql
-- Run the contents of db/schema.sql in the Supabase SQL Editor.
```

Verify the schema:

```sql
select count(*) as chunks,
       min(vector_dims(embedding)) as min_dims,
       max(vector_dims(embedding)) as max_dims
from public.chunks;
```

Expected embedding dimension for the default model is `768`.

### 3. Prepare the LLM runtime

For local Ollama deployment:

```powershell
ollama pull qwen3
ollama serve
```

Verify Ollama:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

For cloud deployment, replace `OLLAMA_BASE_URL` and `LLM_MODEL` with an
OpenAI-compatible hosted endpoint and model. Keep secrets in environment
variables only.

### 4. Ingest documents

Place PDFs under `data\raw`.

Run ingestion:

```powershell
.\.venv\Scripts\python.exe -m ingestion.pipeline data\raw\file1.pdf
```

For multiple PDFs:

```powershell
.\.venv\Scripts\python.exe -m ingestion.pipeline data\raw\file1.pdf data\raw\file2.pdf
```

Large PDFs should be ingested as background jobs in production. The API should
accept the upload, return a job ID, and let a worker process chunk and embed the
document asynchronously.

### 5. Verify RAG locally

Run:

```powershell
.\.venv\Scripts\python.exe run_rag.py
```

Expected startup:

```text
Vector store ready (<number> chunks).
```

Ask a test question from the CLI. If the answer works locally, the same
`run_rag(...)` function can be called from the future API endpoint.

### 6. Add the API service layer

Recommended framework: FastAPI.

Planned endpoints:

```text
GET  /health
POST /v1/rag/query
POST /v1/documents/ingest
GET  /v1/documents/jobs/{job_id}
GET  /v1/documents
DELETE /v1/documents/{document_id}
```

Initial endpoint behavior:

```text
GET /health
  Returns service status, database reachability, and LLM reachability.

POST /v1/rag/query
  Accepts a question and optional retrieval settings.
  Calls rag.rag_pipeline.run_rag(...)
  Returns only the generated answer and request metadata.

POST /v1/documents/ingest
  Accepts one or more uploaded PDFs or storage object references.
  Starts ingestion.
  Returns a job ID.

GET /v1/documents/jobs/{job_id}
  Returns ingestion status: queued, running, completed, failed.

GET /v1/documents
  Lists ingested sources and chunk counts.

DELETE /v1/documents/{document_id}
  Removes a document and its chunks.
```

Example query request:

```json
{
  "question": "What does the document conclude?",
  "top_k": 5,
  "score_threshold": 0.2,
  "metadata_filter": {
    "source": "file1.pdf"
  }
}
```

Example query response:

```json
{
  "answer": "The generated answer goes here.",
  "model": "qwen3",
  "top_k": 5
}
```

### 7. Add API key authentication

Use API keys for frontend and backend callers.

Recommended header:

```text
X-API-Key: <key>
```

Security rules:

- Never expose database credentials to the frontend.
- Never expose LLM provider keys to the frontend.
- Store API keys as environment variables or managed platform secrets.
- Use separate keys for frontend, internal backend services, and admin tooling.
- Rotate keys periodically.
- Log key identifiers, not full key values.

Recommended key roles:

```text
frontend_read
  Can call /v1/rag/query.

backend_ingest
  Can call /v1/documents/ingest and job status endpoints.

admin
  Can call document deletion, diagnostics, and maintenance endpoints.
```

### 8. Add CORS and rate limits

Allow only trusted frontend domains:

```text
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

Recommended limits:

- `/v1/rag/query`: per-key request rate limit.
- `/v1/documents/ingest`: stricter file size and request count limit.
- Upload size: enforce a maximum PDF size.
- Timeout: enforce request timeouts for query and ingestion start endpoints.

### 9. Deploy the API service

Recommended deployment options:

- Local/internal server: Windows service, Linux systemd service, or Docker.
- Cloud VM: deploy API, Ollama, and worker on one machine for simple operation.
- Managed app platform: deploy API separately and use hosted LLM instead of
  local Ollama.

For a production API process, run with an ASGI server:

```powershell
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

This command assumes the future API app is placed at `api/main.py`.

### 10. Deploy the frontend

The frontend should call only the API service, never Supabase PostgreSQL or the
LLM service directly.

Frontend environment:

```text
VITE_API_BASE_URL=https://api.your-domain.com
VITE_PUBLIC_CLIENT_ID=<optional non-secret client id>
```

Do not place private API keys in browser JavaScript. If browser access needs a
key, use a restricted public key with tight rate limits, or route requests
through a trusted backend.

### 11. Add monitoring

Track:

- API request count and latency.
- RAG query failures.
- Database connection failures.
- LLM timeout/failure rate.
- Ingestion job duration.
- Chunk count per document.
- Retrieval score distribution.

Minimum logs per query:

```text
request_id
api_key_id
question_length
top_k
score_threshold
duration_ms
status
```

Do not log full user questions or full retrieved context unless explicitly
needed and approved.

### 12. Production readiness checklist

- Supabase schema exists and pgvector is enabled.
- Embedding dimension matches the database vector dimension.
- `.env` or platform secrets are configured.
- Ollama or hosted LLM endpoint is reachable.
- At least one PDF has been ingested.
- `run_rag.py` works locally before API deployment.
- API endpoints require authentication.
- CORS is limited to known frontend domains.
- Upload limits are enforced.
- Large PDF ingestion runs through a background worker.
- Logs avoid leaking secrets, prompts, retrieved context, or full API keys.
- Backups exist for Supabase data.

## Suggested implementation order

1. Keep the current CLI ingestion and RAG flow working.
2. Add `api/main.py` with FastAPI health and query endpoints.
3. Add API key middleware.
4. Add frontend-facing `/v1/rag/query`.
5. Add ingestion upload endpoint.
6. Move ingestion into a background worker for large PDFs.
7. Add job status storage.
8. Add document listing and deletion.
9. Add deployment scripts or Dockerfile.
10. Add CI checks for tests and API startup.

## Notes for this repository

The current architecture is close to API-ready because `run_rag(...)` already
acts as a clean internal boundary. The main missing deployment pieces are:

- HTTP API wrapper.
- API key middleware.
- CORS configuration.
- Background ingestion worker.
- Upload storage strategy.
- Production process manager.
- Observability and rate limiting.
