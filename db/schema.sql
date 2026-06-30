-- Supabase SQL Editor bootstrap for the default BAAI/bge-base-en-v1.5 model.
-- If EMBEDDING_MODEL changes, replace 768 with that model's detected dimension.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS public.chunks (
    chunk_id text PRIMARY KEY,
    text text NOT NULL,
    source text,
    page integer,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    content_hash text UNIQUE,
    embedding_model text NOT NULL DEFAULT '',
    embedding vector(768) NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON public.chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS chunks_metadata_idx
    ON public.chunks USING gin (metadata);
