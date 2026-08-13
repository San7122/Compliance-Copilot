# Demo Runbook

A safe, ordered sequence from a clean terminal. Every expected output below was observed
on this machine except where explicitly marked **UNVERIFIED**.

**Before you start:** two keys must be in `.env` — `EMBEDDING_API_KEY` (OpenAI, used for
both embeddings and generation via fallback) and optionally `LLM_API_KEY`. Never show
`.env` on screen.

---

## 1. Start Docker

```bash
cd compliance-copilot
docker compose up -d
```

**Expect:** three containers start; `db` reports healthy before `backend` begins.

**Proves:** the stack comes up through normal compose operation, not a hand-rolled command.

> **Port note:** this machine sets `POSTGRES_HOST_PORT=5434` in `.env` because a native
> PostgreSQL 16 already owns 5432. On a clean machine without that line the database is on
> **5432**. Only the host port differs; the backend always reaches Postgres at `db:5432`.

---

## 2. Verify services

```bash
docker compose ps
curl -s http://localhost:8000/health
docker compose logs backend | grep "Ingestion complete"
```

**Expect:**
- three services `Up`, `db` healthy
- `{"status":"ok"}`
- `Ingestion complete: {'documents': 0, 'chunks': 0, 'skipped_unchanged': 26, ...}`

**Proves:** the API is live **and** ingestion is idempotent — a restart re-processed
nothing. `skipped_unchanged=26` is the whole content-hash design in one line. Worth
pausing on; it is a question interviewers ask.

---

## 3. Verify database and corpus

```bash
docker compose exec db psql -U copilot -d compliance_copilot -c \
"SELECT (SELECT count(*) FROM documents) AS documents,
        (SELECT count(*) FROM chunks) AS chunks,
        (SELECT count(*) FROM chunks WHERE embedding IS NOT NULL) AS embeddings,
        (SELECT extversion FROM pg_extension WHERE extname='vector') AS pgvector;"
```

**Expect:** `26 | 983 | 983 | 0.5.1`

**Proves:** the corpus is fully ingested with real embeddings, in PostgreSQL, with
pgvector installed. Zero null embeddings means no silent partial ingestion.

---

## 4. Open the frontend

```
http://localhost:5173
```

**Expect:** question box, an **Entity** dropdown with four options, and a recent-questions
list.

**Proves:** React + Vite UI wired to the API, with entity selection exposed.

---

## 5. Normal question

**Ask:** `How quickly must a breach be reported to the DPO?`
Entity: *Infer from question*

**Expect (verified):** four (4) hours, citing **NFS-POL-001 clause 7.1, page 3**.

**Proves the headline corpus trap.** The superseded `NFS-POL-001-A` says **eight** hours
and is highly similar text — it is excluded before ranking, so it can never reach the
model. Say this out loud; it is the single best thing in the demo.

To show the exclusion is structural rather than luck:

```bash
docker compose exec db psql -U copilot -d compliance_copilot -tAc \
"SELECT doc_id, status FROM documents WHERE doc_id LIKE 'NFS-POL-001%';"
```

**Expect:** `NFS-POL-001|current` and `NFS-POL-001-A|superseded`.

---

## 6. Entity-specific question

**Ask:** `How long must KYC records be retained?`
Entity: **Northwind Capital Markets**

**Expect (verified):** seven (7) years, citing **NFS-SUB-002 clause 5.1**.

Then repeat with Entity = **Northwind Financial Services (group)**.

**Expect (verified):** five (5) years, citing **NFS-POL-003 clause 5.1**.

**Proves:** entity applicability end to end. Same question, different entity, different
correct figure, different cited document. This was a real bug — the entity arrived via the
API parameter, not the question text, so the model never knew — and the fix was to
propagate the resolved scope into the generation context.

---

## 7. Historical question — **UNVERIFIED END TO END**

**Ask:** `What did the breach policy previously require?`

**Expect:** the superseded requirement (eight hours) from **NFS-POL-001-A**, explicitly
labelled superseded.

⚠️ **Run this once yourself before demoing.** Intent detection and the promotion of
superseded documents are both covered by tests and were confirmed against real corpus text
via the reranker, but this exact question has **never been executed through the full API
with a real LLM**. Everything else in this runbook has. Do not demo it cold.

**Proves (if it behaves):** superseded documents are excluded by *policy*, not deleted —
they remain reachable when the question is explicitly historical.

---

## 8. Unsupported question

**Ask:** `How many days of parental leave are employees entitled to?`

**Expect (verified):** exactly *"I don't know based on the provided documents."* — no
citations, low confidence, refusal shown in the UI.

**Proves:** abstention. The corpus is security/compliance only; this term was confirmed
absent from the full extracted text of all 26 documents. A RAG system that invents an
answer here is the failure mode the whole design exists to prevent.

---

## 9. Show citations

Point at any answer's Sources list.

**Expect:** document title, document ID, entity, clause, page, and the quoted excerpt.

**Proves the strongest design decision.** The model returns only a `chunk_id` and a quote;
every other field is looked up from the stored row. Fabricated citation metadata is
structurally impossible, not merely detected. To make that concrete:

```bash
docker compose exec db psql -U copilot -d compliance_copilot -c \
"SELECT id, question, jsonb_pretty(citations) FROM query_log ORDER BY id DESC LIMIT 1;"
```

**Expect:** the citation JSON in the log matches what the UI displayed, field for field.

---

## 10. PostgreSQL / pgvector in DBeaver

**Connection:** host `localhost` · port **5434** · database `compliance_copilot` · user
`copilot` · password from `POSTGRES_PASSWORD` in `.env` (do not display it).

```sql
-- Confirm the dimension from the data, not the column definition
SELECT DISTINCT vector_dims(embedding) FROM chunks;

-- Chunks with their citation metadata
SELECT d.doc_id, d.document_type, d.entity, d.status,
       c.clause, c.page, left(c.content, 80) AS content
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY d.doc_id, c.chunk_index LIMIT 50;

-- A real pgvector similarity search, no LLM involved
SELECT d.doc_id, c.clause, left(c.content, 60) AS content,
       1 - (c.embedding <=> (SELECT embedding FROM chunks WHERE id =
            (SELECT c2.id FROM chunks c2 JOIN documents d2 ON d2.id = c2.document_id
             WHERE d2.doc_id = 'NFS-POL-001' AND c2.clause = '7.1'))) AS similarity
FROM chunks c JOIN documents d ON d.id = c.document_id
ORDER BY similarity DESC LIMIT 10;

-- Cost and latency per query
SELECT id, left(question, 50) AS question, grounded, confidence,
       input_tokens, output_tokens, round(cost_usd::numeric, 6) AS cost, latency_ms
FROM query_log ORDER BY id DESC LIMIT 10;
```

**Proves:** vectors live inside PostgreSQL (no second datastore), the dimension is 384 as
designed, and similarity search works independently of the LLM.

---

## Optional finale — the evaluation

```bash
cd eval && python eval.py --api-url http://localhost:8000
```

**Expect (verified):** 15/15, all four metrics 100%, exit 0. ~38 seconds, 15 real LLM
calls (a fraction of a cent).

**Proves:** the whole chain under test, and that the suite doubles as a CI gate — it exits
non-zero below `--fail-under`.

⚠️ Costs real API calls and takes ~40s. Run it *before* the meeting and show the output,
or budget the time.

---

## If something fails mid-demo

| Symptom | Check |
|---|---|
| `/health` unreachable | `docker compose logs backend \| tail -20` |
| Every answer refuses | Corpus empty — re-run step 3 |
| 500 with an API-key message | `EMBEDDING_API_KEY` missing from `.env` |
| DB port conflict on startup | Something else owns 5432; set `POSTGRES_HOST_PORT` |
| Answer differs slightly from above | Expected — the model is non-deterministic. Substance should match; wording will not |
