# Architecture Review Document: RAG-Evolab / Self-Evaluating Educational RAG

## 1. Executive Summary

The repository currently implements a basic local RAG prototype for educational PDF question answering. It can extract PDF text, split it into token-based chunks, store chunks locally as JSON, push embeddings into Weaviate, retrieve relevant chunks, format a RAG prompt, and call a local Ollama model.

Current maturity: prototype / proof of concept.

Strengths:
- Clear separation between ingestion, retrieval, prompting, and LLM call.
- PDF extraction exists using PyMuPDF.
- Token-based chunking is already implemented.
- SentenceTransformer embeddings are used, which aligns with the target architecture.
- Local terminal testing scripts exist.

Weaknesses:
- No FastAPI backend exists yet.
- No frontend exists yet.
- Retrieval is tightly coupled to Weaviate.
- No PostgreSQL / pgvector implementation exists.
- No judge, reflection, scoring, or answer evaluation layer exists.
- Dependency file is incomplete: code uses `fitz`, `tiktoken`, and `ollama`, but `requirements.txt` does not list them.
- `.env` lives under `vectorstores/`, which makes config loading fragile.
- Generated files like `__pycache__` are present in the repo tree.
- Current fallback retrieval can hide real vector store failures.

## 2. Repository Structure

Current tree, excluding `.venv` internals:

```text
RAg-BAckend/
  .gitignore
  README.md
  LICENSE
  requirements.txt
  chek.py
  testrag.py
  testrag2.py
  test_weaviate_connection.py

  data/
    raw/
      1-s2.0-S1877050924021860-main.pdf
      2504.07615v2.pdf
    chunks/
      chunks.json

  ingestion/
    __init__.py
    pipeline.py
    test_ingestion.py
    loaders/
      __init__.py
      loaders.py
    splitters/
      __init__.py
      splitters.py

  querries/
    querries.json

  rag/
    prompts.py
    llm_client.py
    rag_pipeline.py

  vectorstores/
    .env
    retriever.py
    weaviate_store.py

  */__pycache__/
    generated .pyc files
```

File purposes:

- `.gitignore`: currently empty. Exists to define ignored files, but does not yet exclude `.venv`, `.env`, `__pycache__`, chunks, or generated artifacts.
- `README.md`: minimal project description. No setup, architecture, or usage instructions.
- `LICENSE`: project license.
- `requirements.txt`: lists `weaviate-client`, `sentence-transformers`, and `python-dotenv`. Incomplete for current code.
- `chek.py`: diagnostic script for checking PyMuPDF / `fitz`.
- `testrag.py`: single-query terminal smoke test for `run_rag`.
- `testrag2.py`: interactive terminal Q&A loop using `run_rag`.
- `test_weaviate_connection.py`: standalone Weaviate Cloud connection and vector-search test.

Data files:
- `data/raw/1-s2.0-S1877050924021860-main.pdf`: raw educational/research PDF, 10 pages.
- `data/raw/2504.07615v2.pdf`: raw educational/research PDF, 14 pages.
- `data/chunks/chunks.json`: persisted chunk output from ingestion. Useful for debugging/audit but should likely be regenerated, not hand-maintained.
- `querries/querries.json`: evaluation question bank grouped by factual, section-based, explanation, comparison, multi-chunk, negative, and ambiguous queries.

Ingestion:
- `ingestion/__init__.py`: package marker.
- `ingestion/loaders/loaders.py`: PDF loader using PyMuPDF; returns page-level documents with text and metadata.
- `ingestion/loaders/__init__.py`: package marker.
- `ingestion/splitters/splitters.py`: token-based splitter using `tiktoken`; creates UUID chunk IDs and token metadata.
- `ingestion/splitters/__init__.py`: package marker.
- `ingestion/pipeline.py`: orchestrates PDF loading, splitting, local JSON persistence, and upload to vector store.
- `ingestion/test_ingestion.py`: script that ingests the two PDFs from `data/raw`.

RAG:
- `rag/prompts.py`: contains the main RAG prompt template.
- `rag/llm_client.py`: calls Ollama `gemma3:4b` with fixed decoding settings.
- `rag/rag_pipeline.py`: retrieves chunks, builds context, formats prompt, calls LLM.

Vector store:
- `vectorstores/.env`: contains environment variable names for Ollama, Weaviate, Hugging Face/OpenRouter credentials.
- `vectorstores/weaviate_store.py`: Weaviate Cloud client, collection setup, embedding generation, document insertion, vector search.
- `vectorstores/retriever.py`: singleton wrapper around `WeaviateStore`; exposes `retrieve_chunks`, `add_documents_to_store`, and `close_store`.

Generated artifacts:
- `__pycache__/*.pyc`: generated Python bytecode. Should not be part of source architecture.
- `.venv/`: local virtual environment. Should not be committed or treated as application code.

## 3. Current Architecture

Current ingestion pipeline:

```text
PDF file
  ↓
ingestion.pipeline.run_ingestion()
  ↓
loaders.load_pdf()
  ↓
splitters.token_based_splitter()
  ↓
data/chunks/chunks.json
  ↓
vectorstores.retriever.add_documents_to_store()
  ↓
vectorstores.weaviate_store.WeaviateStore
  ↓
Weaviate Cloud collection: DocumentChunks
```

Current retrieval/generation pipeline:

```text
User query
  ↓
rag.rag_pipeline.run_rag()
  ↓
vectorstores.retriever.retrieve_chunks()
  ↓
WeaviateStore.search()
  ↓
SentenceTransformer query embedding
  ↓
Weaviate near_vector search
  ↓
Top chunks
  ↓
RAG_PROMPT.format(context, question)
  ↓
rag.llm_client.call_llm()
  ↓
Ollama gemma3:4b
  ↓
Answer string
```

Current API pipeline:

```text
No FastAPI API currently exists.

Available entrypoints are:
testrag.py         -> single CLI query
testrag2.py        -> interactive CLI Q&A
test_ingestion.py  -> ingestion script
```

## 4. File Dependency Graph

```text
testrag.py
  -> rag/rag_pipeline.py
    -> vectorstores/retriever.py
      -> vectorstores/weaviate_store.py
    -> rag/prompts.py
    -> rag/llm_client.py

testrag2.py
  -> rag/rag_pipeline.py
    -> vectorstores/retriever.py
      -> vectorstores/weaviate_store.py
    -> rag/prompts.py
    -> rag/llm_client.py

ingestion/test_ingestion.py
  -> ingestion/pipeline.py
    -> ingestion/loaders/loaders.py
    -> ingestion/splitters/splitters.py
    -> vectorstores/retriever.py
      -> vectorstores/weaviate_store.py

test_weaviate_connection.py
  -> Weaviate Cloud directly

chek.py
  -> fitz / PyMuPDF only
```

External dependency map:

```text
PyMuPDF / fitz
  used by loaders.py, chek.py

tiktoken
  used by splitters.py

sentence-transformers
  used by weaviate_store.py

weaviate-client
  used by weaviate_store.py, test_weaviate_connection.py

python-dotenv
  used by weaviate_store.py, test_weaviate_connection.py

ollama
  used by rag/llm_client.py
```

## 5. Component Assessment

KEEP:
- `ingestion/loaders/loaders.py`: clean PDF extraction responsibility.
- `ingestion/splitters/splitters.py`: useful token-based chunking, though dependency and model assumptions should be documented.
- `rag/prompts.py`: keep as a prompt module; later add evaluation prompts separately.
- `rag/rag_pipeline.py`: keep the orchestration idea, but extend it for judge/reflection.
- `rag/llm_client.py`: keep conceptually as an LLM adapter, but make model/config injectable.
- `querries/querries.json`: valuable evaluation dataset for testing RAG quality.

MODIFY:
- `ingestion/pipeline.py`: replace Weaviate upload with PostgreSQL + pgvector insertion.
- `vectorstores/retriever.py`: replace Weaviate singleton with pgvector-backed retriever.
- `requirements.txt`: add missing runtime dependencies and remove Weaviate after migration.
- `.gitignore`: should ignore `.venv`, `.env`, `__pycache__`, and generated data where appropriate.
- `README.md`: needs setup, architecture, ingestion, retrieval, API, and demo instructions.

REMOVE:
- `vectorstores/weaviate_store.py`: remove after pgvector replacement is ready.
- `test_weaviate_connection.py`: remove after migration.
- Weaviate dependency from `requirements.txt`.
- Weaviate environment variables from active config.
- `__pycache__` folders from source tree.

FUTURE WORK:
- FastAPI backend.
- Next.js frontend.
- Judge module.
- Reflection engine.
- Metrics logger.
- PostgreSQL schema/migrations.
- Automated tests.

## 6. PostgreSQL Migration Plan

Files currently depending on Weaviate:
- `vectorstores/weaviate_store.py`
- `vectorstores/retriever.py`
- `ingestion/pipeline.py`, indirectly through `add_documents_to_store`
- `rag/rag_pipeline.py`, indirectly through `retrieve_chunks`
- `test_weaviate_connection.py`
- `requirements.txt`
- `vectorstores/.env`

Weaviate currently provides:
- Cloud vector database connection.
- Collection/schema creation.
- Vector object insertion.
- Metadata storage for `text`, `source`, and `page`.
- Near-vector similarity search.
- Batch insertion.

PostgreSQL + pgvector replacement responsibilities:
- PostgreSQL stores chunk text, metadata, source, page, token spans, and embeddings.
- `pgvector` stores `vector(384)` embeddings for `all-MiniLM-L6-v2`.
- A database table replaces Weaviate `DocumentChunks`.
- SQL similarity search replaces Weaviate `near_vector`.
- A new store module should expose the same high-level operations: add chunks, search chunks, close connection.

Current flow:

```text
chunks
  ↓
SentenceTransformer embedding
  ↓
Weaviate batch.add_object()
  ↓
Weaviate near_vector()
```

Target flow:

```text
chunks
  ↓
SentenceTransformer embedding
  ↓
PostgreSQL table insert
  ↓
pgvector similarity query
  ↓
top_k chunks + metadata + similarity score
```

Migration guidance:
- Add a new `postgres_store.py` or `pgvector_store.py`.
- Keep the retriever interface stable: `retrieve_chunks(query, top_k)` and `add_documents_to_store(chunks)`.
- Move embedding generation into a reusable embedding layer, not inside the database adapter.
- Replace Weaviate Cloud config with `DATABASE_URL` or explicit PostgreSQL variables.
- Keep local `chunks.json` only as debug/audit output.
- Do not introduce a complex ORM unless the team already knows it; direct SQL with `psycopg` is enough for a 2-3 day build.

## 7. Proposed Final Architecture

Target architecture:

```text
User
  ↓
Next.js frontend
  ↓
FastAPI backend
  ↓
RAG pipeline
  ↓
Retriever
  ↓
PostgreSQL + pgvector
  ↓
Generator
  ↓
Judge
  ↓
Reflection engine
  ↓
Final answer + metrics
```

Detailed pipeline:

```text
PDF Upload / Ingestion
  ↓
Loader
  ↓
Splitter
  ↓
Embedding Layer
  ↓
PostgreSQL + pgvector

Question Answering
  ↓
FastAPI /ask
  ↓
Retriever
  ↓
Top chunks
  ↓
Generator
  ↓
Draft answer
  ↓
Judge
  ↓
Scores: groundedness, relevance, completeness, confidence
  ↓
If weak: Reflection Engine revises answer once
  ↓
Final Response JSON
```

Recommended final response shape:

```json
{
  "answer": "...",
  "sources": [
    {"source": "file.pdf", "page": 3}
  ],
  "evaluation": {
    "groundedness": 0.8,
    "relevance": 0.9,
    "completeness": 0.7,
    "confidence": 0.75
  },
  "reflected": true
}
```

## 8. Missing Components

Recommended new modules:

- `api/main.py`
  - Purpose: FastAPI app entrypoint.
  - Inputs: HTTP requests.
  - Outputs: JSON responses.
  - Responsibilities: `/ask`, `/ingest`, `/health`.

- `vectorstores/pgvector_store.py`
  - Purpose: PostgreSQL + pgvector implementation.
  - Inputs: chunks, query embeddings.
  - Outputs: stored chunks, retrieved matches.
  - Responsibilities: insert chunks, similarity search, connection handling.

- `rag/embeddings.py`
  - Purpose: central SentenceTransformer embedding layer.
  - Inputs: text/list of texts.
  - Outputs: 384-dimensional vectors.
  - Responsibilities: model loading, encode abstraction.

- `rag/generator.py`
  - Purpose: answer generation using retrieved context.
  - Inputs: question, chunks.
  - Outputs: draft answer.
  - Responsibilities: prompt formatting and LLM call.

- `rag/judge.py`
  - Purpose: evaluate generated answer.
  - Inputs: question, context, answer.
  - Outputs: evaluation scores and critique.
  - Responsibilities: groundedness/relevance/completeness scoring.

- `rag/reflection.py`
  - Purpose: improve weak answers once.
  - Inputs: question, context, draft answer, judge critique.
  - Outputs: revised answer.
  - Responsibilities: single-pass reflection, no multi-agent loop.

- `rag/evaluation_prompts.py`
  - Purpose: prompt templates for judge/reflection.
  - Inputs: variables for evaluation.
  - Outputs: formatted prompts.
  - Responsibilities: keep evaluation prompts separate from RAG prompt.

- `rag/metrics_logger.py`
  - Purpose: record query, answer, scores, latency.
  - Inputs: pipeline result.
  - Outputs: database row or JSON log.
  - Responsibilities: simple observability.

- `db/schema.sql` or `migrations/001_init.sql`
  - Purpose: database schema.
  - Inputs: none at runtime.
  - Outputs: tables/extensions.
  - Responsibilities: `CREATE EXTENSION vector`, chunk table, optional query log table.

- `frontend/`
  - Purpose: Next.js UI.
  - Inputs: user questions/PDF uploads.
  - Outputs: displayed answer, sources, evaluation metrics.
  - Responsibilities: simple educational Q&A interface.

## 9. Team Division

Person A: AI / RAG Engineer
- Owns: `ingestion/`, `rag/`, `vectorstores/pgvector_store.py`, `db/schema.sql`.
- Tasks:
  - Migrate retrieval to pgvector.
  - Extract embedding logic.
  - Build judge prompt and scoring.
  - Build one-pass reflection.
  - Use `querries/querries.json` for manual evaluation.
- Deliverables:
  - Working ingestion to PostgreSQL.
  - Working retrieval from pgvector.
  - RAG output with evaluation metrics.
  - Simple test results for sample questions.

Person B: Backend / Frontend Engineer
- Owns: `api/`, `frontend/`, configuration, docs.
- Tasks:
  - Build FastAPI routes.
  - Connect API to RAG pipeline.
  - Build Next.js question interface.
  - Display answer, sources, metrics.
  - Update README and setup instructions.
- Deliverables:
  - `/ask`, `/ingest`, `/health` API.
  - Usable frontend.
  - Demo-ready local run instructions.

## 10. Three-Day Development Plan

Day 1:
- Objectives:
  - Remove Weaviate from active architecture.
  - Set up PostgreSQL + pgvector schema.
  - Implement pgvector insertion/search.
- Files touched conceptually:
  - `requirements.txt`
  - `vectorstores/pgvector_store.py`
  - `vectorstores/retriever.py`
  - `rag/embeddings.py`
  - `ingestion/pipeline.py`
  - `db/schema.sql`
- Expected outcome:
  - PDFs can be ingested into PostgreSQL.
  - Questions retrieve chunks from pgvector.

Day 2:
- Objectives:
  - Build self-evaluation and reflection.
  - Add FastAPI backend.
- Files touched conceptually:
  - `rag/generator.py`
  - `rag/judge.py`
  - `rag/reflection.py`
  - `rag/evaluation_prompts.py`
  - `rag/rag_pipeline.py`
  - `api/main.py`
- Expected outcome:
  - `/ask` returns answer, sources, scores, and reflection status.

Day 3:
- Objectives:
  - Build simple frontend.
  - Polish docs and demo.
  - Test with query bank.
- Files touched conceptually:
  - `frontend/`
  - `README.md`
  - `querries/querries.json`
  - optional `rag/metrics_logger.py`
- Expected outcome:
  - End-to-end demo: upload/ingest PDFs, ask questions, see answer + evaluation metrics.

## 11. Final Folder Structure

```text
RAg-BAckend/
  README.md                         KEEP/MODIFY
  LICENSE                           KEEP
  requirements.txt                  MODIFY
  .gitignore                        MODIFY

  api/                              NEW
    main.py                         NEW
    schemas.py                      NEW

  db/                               NEW
    schema.sql                      NEW

  data/                             KEEP
    raw/                            KEEP
    chunks/                         KEEP, optional debug output

  ingestion/                        KEEP
    pipeline.py                     MODIFY
    test_ingestion.py               MODIFY
    loaders/
      loaders.py                    KEEP
    splitters/
      splitters.py                  KEEP

  rag/                              KEEP
    prompts.py                      KEEP/MODIFY
    llm_client.py                   MODIFY
    rag_pipeline.py                 MODIFY
    embeddings.py                   NEW
    generator.py                    NEW
    judge.py                        NEW
    reflection.py                   NEW
    evaluation_prompts.py           NEW
    metrics_logger.py               NEW/FUTURE

  vectorstores/                     KEEP/MODIFY
    retriever.py                    MODIFY
    pgvector_store.py               NEW
    weaviate_store.py               REMOVE
    .env                            MOVE/REMOVE from repo

  frontend/                         NEW
    Next.js app                     NEW

  querries/
    querries.json                   KEEP, consider renaming to queries/

  test_weaviate_connection.py       REMOVE
  testrag.py                        KEEP/MODIFY
  testrag2.py                       KEEP/MODIFY
  chek.py                           REMOVE or keep as local diagnostic only

  __pycache__/                      REMOVE
  .venv/                            REMOVE from repo / ignore
```

## 12. Risks and Technical Debt

Scalability risks:
- Current local Ollama call may be slow under concurrent users.
- Loading SentenceTransformer inside store class couples model lifecycle to storage.
- No batching strategy for PostgreSQL inserts yet.
- No indexes currently defined for pgvector.

Design issues:
- RAG pipeline returns only a string, not sources or scores.
- Retriever catches all exceptions and returns mock chunks, which can produce misleading answers.
- Hardcoded model names: `gemma3:4b`, `all-MiniLM-L6-v2`, `gpt-4o-mini`.
- Current code lacks typed response objects.

Deployment issues:
- `.env` is in a subfolder and may contain secrets.
- `.gitignore` is empty.
- Dependencies are incomplete.
- No Docker/PostgreSQL setup instructions.
- No FastAPI server exists yet.

Recommendations:
- Add strict config loading from project root.
- Remove broad exception fallbacks from production path.
- Store source/page/token metadata in PostgreSQL.
- Keep self-evaluation simple: one judge call and one optional reflection pass.
- Avoid multi-agent design for this capstone scope.

## 13. Final Recommendation

KEEP:
- PDF loader.
- Token splitter.
- Basic RAG orchestration.
- Prompt module.
- Ollama/local LLM adapter concept.
- Query evaluation JSON.
- Raw PDFs for demo/testing.

MODIFY:
- Retriever abstraction.
- Ingestion pipeline storage target.
- RAG output format.
- LLM config handling.
- README, requirements, `.gitignore`.
- Terminal test scripts.

REMOVE:
- Weaviate store.
- Weaviate connection test.
- Weaviate dependency.
- Weaviate Cloud assumptions.
- Generated `__pycache__` files.
- Secrets from repo-controlled paths.

POSTPONE:
- Multi-agent research workflow.
- Advanced ranking/reranking.
- Fine-tuning.
- Complex analytics dashboard.
- User authentication.
- Distributed ingestion.
- Sophisticated evaluation frameworks.

Final verdict: migrate to PostgreSQL + pgvector first, then add a simple judge and one-pass reflection layer. That gives the project a credible Self-Evaluating Educational RAG System within 2-3 days without turning it into an overbuilt research platform.
