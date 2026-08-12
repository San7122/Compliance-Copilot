-- Runs automatically on first Postgres container start (docker-entrypoint-initdb.d).
-- Also safe to run manually / idempotently against a local Postgres for dev.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- No ANN index (ivfflat/hnsw) on purpose: with a handful of docs (~50-300 chunks),
-- an exact brute-force cosine scan is sub-millisecond and always fully accurate.
-- An ivfflat index needs roughly `rows / 1000` lists to behave well; at this corpus
-- size any index just adds approximation error for no speed benefit. Add one
-- (e.g. HNSW) once the corpus grows into the tens of thousands of chunks.

CREATE TABLE IF NOT EXISTS query_log (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    citations JSONB NOT NULL,
    confidence TEXT NOT NULL,
    answerable BOOLEAN NOT NULL,
    retrieved_chunk_ids INTEGER[] NOT NULL DEFAULT '{}',
    latency_ms INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
