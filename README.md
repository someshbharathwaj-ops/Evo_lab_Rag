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

## Configuration

All supported settings are documented in `.env.example`. The main values are:

- `DATABASE_URL` (preferred) or `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`,
  `DB_PASSWORD`, and `DB_SSLMODE`.
- `EMBEDDING_MODEL`, `EMBEDDING_BATCH_SIZE`.
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNKS_PATH`.
- `TOP_K`, `SCORE_THRESHOLD`, `MAX_CONTEXT_CHARS`.
- `OLLAMA_BASE_URL`, `LLM_MODEL`, `MAX_TOKENS`, `TEMPERATURE`, `LLM_TIMEOUT`,
  `OLLAMA_REASONING_EFFORT` (defaults to `none` for responsive grounded Q&A).

## Remaining limitations

- Image-only/scanned PDFs need OCR before this text loader can ingest them.
- Changing embedding models requires a new table or an explicit vector-column
  migration and re-embedding; mixed dimensions are intentionally rejected.
- Ingestion is synchronous and intended for local/small-batch use.
- There is no reranker; quality relies on normalized embeddings, cosine search,
  top-k, thresholding, metadata filters, and context deduplication.
