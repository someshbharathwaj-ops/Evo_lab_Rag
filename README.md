# Local Supabase RAG

A modular PDF question-answering pipeline using the existing
`BAAI/bge-base-en-v1.5` embeddings, Supabase PostgreSQL with pgvector, and a
local Ollama `qwen3` model. Retrieved chunks are private pipeline context; the
CLI displays only the generated answer.

## Architecture

```text
PDF -> PyMuPDF loader -> token chunks -> SentenceTransformer embeddings
    -> Supabase pgvector -> cosine top-k retrieval -> bounded context
    -> grounded prompt -> local Ollama/qwen3 -> answer
```

Responsibilities remain separated:

- `ingestion/loaders/`: PDF extraction and cleanup.
- `ingestion/splitters/`: configured token chunking and deterministic IDs.
- `ingestion/embeddings/`: the shared embedding model and dimension detection.
- `ingestion/vectorstore/`: schema migration, idempotent writes, and search.
- `rag/retriever.py`: query embedding and retrieval.
- `rag/prompts.py`: context deduplication and prompt construction.
- `rag/llm_client.py`: local Ollama adapter.
- `rag/rag_pipeline.py`: answer orchestration.

## Setup

Use Python 3.10 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
```

Set `DATABASE_URL` in `.env` to the URI from Supabase **Connect**. A direct
connection or Supavisor session/transaction pooler URL works; SSL should be
enabled. Alternatively set the individual `DB_*` values. No Supabase HTTP API
key is needed because this application connects to PostgreSQL directly.

Install/start Ollama and fetch the local model:

```powershell
ollama pull qwen3
ollama serve
```

Ollama's OpenAI-compatible URL defaults to `http://localhost:11434/v1`, and
the client uses the fixed placeholder key `ollama`; no real LLM API key is
required.

## Database schema

For a fresh Supabase project, run [db/schema.sql](db/schema.sql) in the
Supabase SQL Editor. Its `vector(768)` matches the default existing embedding
model. The ingestion code also creates or upgrades the table automatically,
detects the actual embedding dimension, and refuses to mix vectors of a
different size.

The database role needs permission to create the `vector` extension, schema,
table, and indexes. If automatic setup lacks extension permission, run the SQL
file as the project owner, then ingest again.

Schema additions include full `jsonb` metadata, an embedding-model audit
column, timestamps, a unique content hash for duplicate-safe upserts, a GIN
metadata index, and an HNSW cosine index.

## Ingest PDFs

```powershell
python -m ingestion.pipeline data\raw\manual.pdf data\raw\paper.pdf
```

The command loads, chunks, batch-embeds, and upserts every document. Repeating
the same ingestion updates matching content rather than adding duplicates.
`data/chunks/chunks.json` remains a human-readable debug artifact without the
large embedding arrays.

## Ask questions

```powershell
python run_rag.py
```

Programmatic usage remains simple:

```python
from rag.rag_pipeline import run_rag

answer = run_rag("What does the document conclude?", top_k=5)
```

Metadata filtering is available through `run_rag(...,
metadata_filter={"source": "..."})`. `SCORE_THRESHOLD` rejects weak cosine
matches, while `MAX_CONTEXT_CHARS` bounds the prompt.

## Verify the system

Check stored vectors in the Supabase SQL Editor:

```sql
select count(*) as chunks,
       min(vector_dims(embedding)) as min_dims,
       max(vector_dims(embedding)) as max_dims
from public.chunks;

select source, page, embedding_model, updated_at
from public.chunks
order by updated_at desc
limit 10;
```

Check Ollama directly:

```powershell
Invoke-RestMethod http://localhost:11434/api/tags
```

The CLI startup line also prints the configured local URL and model. There is
no OpenRouter configuration in the runtime code.

Run the offline unit suite (it mocks the model, database, and LLM):

```powershell
python test_all_components.py
```

## Advanced Multi-Stage RAG Pipeline

The RAG engine supports an optional multi-stage quality control pipeline:

```text
User Query
    ↓
Embedding Search (Retrieval Top-K / Over-fetch Candidate-K)
    ↓
[Optional] Cross-Encoder Reranking (model-agnostic, e.g. RERANKER_MODEL in .env)
    ↓
Context Deduplication & Bounded Assembly
    ↓
Generator LLM (Candidate Answer Generation)
    ↓
[Optional] LLM-as-a-Judge Verification (Factual Grounding & Hallucination Check)
    ↓
[PASS] Final Answer / [FAIL] Corrective Regeneration
```

### Feature Flags & Configuration

All advanced stages can be dynamically enabled or disabled via environment variables:

| Setting | Default | Description |
|---|---|---|
| `ENABLE_RERANKING` | `false` | Enables cross-encoder reranking of retrieved candidates. |
| `RERANKER_MODEL` | `""` | Any HuggingFace Cross-Encoder model ID (e.g. `z-ai/glm` or `cross-encoder/ms-marco-MiniLM-L-6-v2`). |
| `RERANK_CANDIDATE_K` | `20` | Number of initial vector-similarity candidate chunks fetched for reranking. |
| `FINAL_TOP_K` | `5` | Final top-N chunks selected after cross-encoder reranking. |
| `ENABLE_JUDGE` | `false` | Enables LLM-as-a-Judge verification of generated candidate answers. |
| `JUDGE_MODEL` | `LLM_MODEL` | LLM model used for judging correctness (defaults to generator model). |
| `JUDGE_MAX_TOKENS` | `1200` | Token limit for judge structured JSON output. |
| `MAX_REGENERATE_ATTEMPTS` | `2` | Maximum retry attempts if judge rejects an answer. |

### Reranking Details

When `ENABLE_RERANKING=true` and `RERANKER_MODEL` is set, the retriever over-fetches candidates (`RERANK_CANDIDATE_K=20`), and passes pairs of `(query, passage)` to the cross-encoder. The cross-encoder outputs deep semantic relevance scores, allowing fine-grained ranking beyond simple vector similarity.

### LLM-as-a-Judge Verification

When `ENABLE_JUDGE=true`, candidate answers are evaluated against the question and retrieved context for:
- **Groundedness**: All claims must originate from retrieved text.
- **No-Hallucination**: Rejects invented facts or unsupported assertions.
- **Relevance & Completeness**: Ensures the prompt was actually answered.

If the judge rejects an answer (`passed: false`), the pipeline automatically attempts regeneration using a corrective prompt containing the judge's feedback. If still unverified after max attempts, it returns the best available revised answer.

## Remaining limitations

- Image-only/scanned PDFs need OCR (`pytesseract` + `Pillow`) or the image loader.
- Changing embedding models requires a new table or an explicit vector-column migration and re-embedding; mixed dimensions are intentionally rejected.
- Ingestion is synchronous and intended for local/small-batch use.

## Production Deployment

This RAG application is fully production-ready for automated hosting platforms:

### Backend Deployment (Render)

1. Create a new **Web Service** on Render linked to this repository.
2. Set the **Root Directory** to `Evo_lab_Rag`.
3. Set the **Build Command** to `pip install -r requirements.txt`.
4. Set the **Start Command** to `uvicorn main:app --host 0.0.0.0 --port $PORT`.
5. Add the following **Environment Variables** under your Render Service Settings:
   - `DATABASE_URL`: Your Supabase PostgreSQL database URL.
   - `OLLAMA_BASE_URL`: The OpenAI-compatible API base URL (e.g. ngrok tunnel or hosted instance).
   - `LLM_MODEL`: The target LLM model name (e.g., `gemma3:4b`).
   - `USE_HF_INFERENCE`: Set to `true` to use the Hugging Face Inference API fallback for sentence embedding (highly recommended on Render's free tier to avoid Out-Of-Memory limits).
   - `HUGGINGFACE_API_KEY`: (Optional) Your Hugging Face user access token.

### Frontend Deployment (Vercel)

1. Import the repository into your Vercel dashboard.
2. Configure the project to build from the `RAG-evolab-UI` directory.
3. Vercel will automatically auto-configure the build command (`npm run build`).
4. Set the following **Environment Variables** on the Vercel project settings:
   - `NEXT_PUBLIC_API_URL` or `NEXT_PUBLIC_BACKEND_URL`: Point to your production FastAPI Backend URL hosted on Render (e.g., `https://evo-lab-rag-backend.onrender.com`).
   - `BACKEND_TIMEOUT_MS`: Request timeout length in milliseconds (defaults to `120000`).

