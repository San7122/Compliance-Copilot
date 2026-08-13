-- Runs automatically on first Postgres container start (docker-entrypoint-initdb.d).
-- Also safe to run manually / idempotently against a local Postgres for dev.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id SERIAL PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    title TEXT NOT NULL,
    -- Identity parsed from the document's front matter. `status` gates whether the
    -- document may be used to answer at all; `entity` records which legal entity its
    -- numbers bind. Both are load-bearing for correctness, not just metadata.
    doc_id TEXT,
    -- policy / entity_policy / procedure / guidance / regulatory
    document_type TEXT,
    entity TEXT,
    version TEXT,
    status TEXT NOT NULL DEFAULT 'current',
    status_note TEXT,
    effective_date TEXT,
    -- SHA-256 of the source file; document identity is (filename, content_hash).
    -- An unchanged file is skipped on re-ingest rather than re-embedded.
    content_hash TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Superseded documents are excluded from retrieval on every query, so this filter is
-- on the hot path for all of them.
-- Status, entity and type are all filter/rank inputs on the query hot path.
CREATE INDEX IF NOT EXISTS documents_status_idx ON documents (status);
CREATE INDEX IF NOT EXISTS documents_entity_idx ON documents (entity);
CREATE INDEX IF NOT EXISTS documents_type_idx ON documents (document_type);

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    heading_path TEXT NOT NULL,
    -- "7.1" / "E.1"; NULL for unnumbered blocks (Definitions, Annexures).
    clause TEXT,
    -- Page the clause starts on, carried through to the citation.
    page INTEGER,
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
    -- Application-level score in [0,1], not a calibrated probability.
    confidence DOUBLE PRECISION NOT NULL,
    grounded BOOLEAN NOT NULL,
    refusal_reason TEXT,
    retrieved_chunk_ids INTEGER[] NOT NULL DEFAULT '{}',
    latency_ms INTEGER,
    -- Zero on refusals that short-circuit before the model is ever called.
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Additive columns for databases created before token/cost logging existed. Note
-- this file only runs automatically on a *fresh* Postgres volume, so these do not
-- migrate a running container -- they exist so the file stays safe to re-run by
-- hand against an existing database (`psql -f init.sql`), which is how an already
-- provisioned volume gets the new columns without being dropped.
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS input_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS output_tokens INTEGER NOT NULL DEFAULT 0;
ALTER TABLE query_log ADD COLUMN IF NOT EXISTS cost_usd DOUBLE PRECISION NOT NULL DEFAULT 0;
