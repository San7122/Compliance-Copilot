# Compliance Copilot

Ask a plain-English question about a corpus of policy documents and get an answer
grounded in those documents, with citations pointing at a specific clause and page — or
an honest refusal when the documents don't answer it.

Built over the Northwind policy pack: **26 PDFs, ~85 pages**, covering group policies,
subsidiary policies, technical standards, operational procedures and a staff handbook.

---

## Architecture

Two pipelines, deliberately separated. Ingestion is offline and writes; querying is
online and reads. Neither can reach into the other: there is no PDF parsing on the query
path and no question answering on the ingestion path.

```
INGESTION (offline)                      QUERY (online)
  documents/*.pdf                          question (+ optional entity)
    → extract       page-aware text          → resolve scope  entity + current/historical intent
    → clean         strip page furniture     → embed          same model as ingestion
    → detect        front matter + clauses   → retrieve       pgvector candidate_k=20
    → chunk         one clause per chunk     → relevance      similarity floor → refuse early
    → embed         hosted API               → applicability  drop non-applicable entities
    → store         Postgres + pgvector      → rerank         explainable signals → top_k=5
                                             → generate       LLM sees evidence only
                                             → validate       Pydantic
                                             → map citations  chunk_id → DB record
                                             → respond        + retrieval metadata
```

The one-paragraph version: *ingestion extracts text from PDFs, preserves clause/page/entity
metadata, chunks at clause boundaries, embeds, and stores chunks and vectors in Postgres
using pgvector. Querying embeds the question with the same model, retrieves the most
similar chunks, checks whether sufficient evidence exists before calling the LLM,
validates the structured response, and maps citations back to the actual retrieved
records.*

### Project structure

```
backend/app/
  main.py                    app assembly only
  config.py                  all tunables, env-configurable
  api/routes/                health.py · query.py · ingestion.py
  ingestion/                 extractor.py · chunker.py · embedder.py · pipeline.py
  query/pipeline.py          QueryPipeline
  query/applicability.py     entity scope + current/historical intent
  query/reranker.py          explainable reranking signals
  retrieval.py               pgvector candidates + status filter
  generation → llm.py        prompts + forced tool-use structured output
  grounding  → verification.py
  citations/mapper.py        chunk_id → citation (the DB is the source of truth)
  confidence.py              application-level score
  models.py · db.py · init.sql · schemas.py
backend/tests/               ingestion/ · retrieval/ · query/ · api/   (180 tests)
eval/                        eval.py · questions.yaml
frontend/src/                components/ · services/
docs/                        the 26-PDF corpus
DESIGN.md                    why it is built this way
CURRENT_STATUS.md            gap analysis against the target architecture
```

---

## The corpus is the interesting part

The document set is built with traps, and handling them is most of the work. Each is
handled structurally rather than by asking the model nicely.

| Trap in the corpus | Why it's dangerous | How it's handled |
|---|---|---|
| **`NFS-POL-001-A` is SUPERSEDED** — says "do not rely on this version", and gives an **8-hour** breach reporting window against the current policy's **4 hours** | It's fluent, on-topic and highly retrievable. Cosine similarity loves it. | Status is parsed at ingest and excluded from retrieval **by default**, before ranking — but remains reachable for explicitly historical questions ("what did it previously require?"), where it is promoted and labelled `SUPERSEDED` in both context and citation |
| **Three legal entities** — group, Capital Markets, Payments (Singapore) — with near-identical policies and deliberately different numbers (KYC retention 5 vs **7** years; high-risk re-verification 12 vs **6** months) | The documents look the same apart from the figures, so a merged answer is confidently wrong for both entities | Entity is parsed per document, filtered on (other subsidiaries are dropped outright) and **reranked above group policy** when the question is entity-scoped — before any document-type consideration. `POST /api/query` takes an optional `entity`; otherwise it is inferred from the question |
| **`NFS-POL-011` governs retention conflicts** (clause 1.2), and its Annexure B works an example: policy says 7 years, schedule says **8**, so 8 applies | The most similar chunk gives the wrong answer | Precedence is stated in the prompt; the governing clause is itself retrievable and citable |
| **`NFS-GUID-001` is a handbook summary**, not policy, and paraphrases loosely | Reads like policy | Classified as `document_type: guidance` and demoted by a small reranking weight — enough to lose a tie against an equally relevant policy, not enough to lose to a marginally relevant one |
| **Lettered clauses** (`E.n`, `C.n`, `R.n` — 325 of them) and unnumbered sections | Matching only `\d+\.\d+` silently glues them onto the previous clause | Both are recognised. Fixing this took the corpus from 571 to **983 chunks** — the first version was losing or misattributing ~40% of it |

---

## Design decisions

**Chunk at the clause, not at a token window.** Every document numbers its clauses, and
the corpus notes say those numbers make good citation targets. A clause is one
self-contained obligation — the right unit both to retrieve and to cite. "NFS-POL-001
clause 7.1, p. 3" is checkable; "somewhere in the Data Protection Policy" is not. Clauses
here are short (longest ~170 words), so they aren't windowed further; splitting one would
sever an obligation from its own conditions. The markdown path retains heading-based
chunking with configurable size/overlap for non-PDF sources.

**The LLM never authors citation metadata.** It returns a `chunk_id` and a quote; the
backend looks that ID up and builds the citation from the stored record. Fabricated
metadata is structurally impossible, not merely detected. Two checks remain: the
`chunk_id` must be one actually shown to the model this turn, and the quote must appear
in *that specific chunk* — verifying against the whole context would let a quote from
chunk A be attributed to chunk B, producing a citation that points at the wrong clause
while quoting real text.

**Three distinct refusal reasons**, because they mean different things operationally:
`no_relevant_evidence` (nothing cleared the floor — the model is never called, so it's
free), `model_declined` (evidence existed but didn't answer), and `citations_unverified`
(an answer was drafted but nothing it quoted survived checking — withheld, because a
confident answer with unverifiable support is worse than none).

**Confidence is a score, not a probability.** It combines retrieval relevance,
groundedness, citation coverage and the model's self-assessment, weighted so measured
signals outrank self-report. It is explicitly *not* calibrated — no labelled dataset was
used to fit it — and an answer with no verified citation is capped at 0.2 regardless of
how relevant the retrieved text looked.

**Ingestion is idempotent by content hash.** Identity is `(filename, sha256)`. Unchanged
documents are skipped without being re-embedded; a changed document replaces only its own
chunks; new documents are simply added. Adding five PDFs to an indexed corpus of 26 costs
five embed calls, not 31. One malformed document fails in isolation and is reported
rather than aborting the run.

**Hosted embeddings, 384 dimensions.** `text-embedding-3-small` supports a `dimensions`
parameter, so its output is reduced to the 384 the schema already used — **no vector
column migration and no re-embedding forced by the provider switch**. The returned width
is asserted on every response rather than assumed. Documents and queries go through one
code path, because embedding them with different models puts the vectors in different
spaces and degrades retrieval silently rather than erroring.

**Brute-force vector search, no ANN index.** At ~1,000 chunks an exact scan is
sub-millisecond and always correct. An earlier `ivfflat` index with `lists=100` over 51
rows was silently dropping most results — removed rather than tuned.

---

## Running it

### Setup

```bash
cp .env.example .env
```

Two keys are required, for two different purposes:

- `ANTHROPIC_API_KEY` — answer generation (Claude Haiku 4.5)
- `EMBEDDING_API_KEY` (or `OPENAI_API_KEY`) — embeddings, for documents *and* queries

```bash
docker compose up -d --build
```

The backend runs ingestion on start, then serves the API.

> If your machine already runs Postgres on 5432, set `POSTGRES_HOST_PORT=5433` in `.env`.
> Host port only — the backend always reaches the database at `db:5432` inside the
> compose network.

### Ingestion

Ingestion runs automatically at container start. To run it explicitly:

```bash
# inside the container
docker compose exec backend python -m scripts.run_ingest

# or over HTTP
curl -X POST http://localhost:8000/api/ingest
```

Both are safe to re-run — unchanged documents are skipped.

### Querying

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "How quickly must a personal data breach be reported to the DPO?"}'
```

Frontend: <http://localhost:5173> · API docs: <http://localhost:8000/docs>

### Tests

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests -q          # 180 tests, ~6s
```

No database, no API keys and no network are needed: the suite covers extraction,
chunking, metadata, idempotency, retrieval filtering, entity applicability, superseded /
historical handling, reranking, grounding, citation mapping, confidence, abstention, the
query pipeline and the HTTP contract, with retrieval and generation stubbed at their
boundaries.

| Area | Tests |
|---|---|
| `tests/ingestion` | 45 |
| `tests/retrieval` | 6 |
| `tests/query` | 105 |
| `tests/api` | 24 |

### Evaluation

```bash
cd eval
pip install -r requirements.txt
python eval.py --api-url http://localhost:8000
```

Exits non-zero if any metric falls below `--fail-under`, so it works as a CI gate. It
calls the same `/api/query` the app uses — no RAG logic is duplicated — and verifies
citation excerpts against the PDFs on disk rather than trusting the API's own metadata.

---

## API

`POST /api/query`

```json
{
  "question": "How long must KYC records be retained?",
  "entity": "Northwind Capital Markets Ltd"
}
```

`entity` is optional. When supplied it wins over anything inferred from the question;
when omitted the entity is inferred from the question text and falls back to group scope.

```json
{
  "answer": "Within four (4) hours of discovery...",
  "citations": [
    {
      "chunk_id": 183,
      "document": "Data Protection and Privacy Policy",
      "document_id": "NFS-POL-001",
      "entity": "Northwind Financial Services Pvt. Ltd.",
      "section": "Data Protection and Privacy Policy > 7. Breach Notification > Clause 7.1",
      "clause": "7.1",
      "page": 3,
      "excerpt": "must be reported to the Data Protection Officer within four (4) hours"
    }
  ],
  "confidence": 0.86,
  "grounded": true,
  "refusal_reason": null,
  "retrieval": {
    "entity_scope": "Northwind Capital Markets Ltd",
    "entity_source": "explicit",
    "intent": "current",
    "superseded_included": false,
    "candidates_considered": 20,
    "evidence_used": 5,
    "ranking": ["NFS-SUB-002 clause 5.1 => 1.183 (similarity+0.780, entity_specific+0.350, authority+0.050, clause_specificity+0.030)"]
  }
}
```

Also: `GET /health`, `GET /api/history?limit=`, `POST /api/ingest`.

---

## What is and isn't verified

**Verified by running it:** all 26 PDFs parse, with exactly 1 superseded document, 3
entities and 5 document types detected; 983 clause-level chunks with page numbers and
clause references; entity scoping (an NCM-scoped question selects `NFS-SUB-002`'s 7-year
figure, an unscoped one selects the group's 5-year figure); superseded handling (a current
question selects the 4-hour clause, a historical one selects the superseded 8-hour clause
and labels it); cross-subsidiary isolation;
ingestion idempotency (unchanged documents skipped, additions isolated, one bad file
doesn't abort the run); the full HTTP contract including every refusal path; 129 tests
passing in ~6 seconds.

**Not verified — and this matters:**

- **The stack has not been run since the embedding provider changed.** The switch to
  hosted embeddings removed torch, but the machine ran out of disk mid-rebuild, so the
  new image has not been built. Ingestion against a live database with the new embedder
  is unexercised.
- **No live LLM or embedding calls have been made.** There is no `ANTHROPIC_API_KEY` and
  no embedding key in `.env`, so generation, real citation behaviour and the eval numbers
  are all unconfirmed.
- **The frontend has not been rebuilt** since the component split (same disk constraint).

## Eval results

**There are none yet, and none are invented here.** The harness runs correctly against a
live API — it reaches the service, scores every question, prints all four metrics and
exits non-zero on failure — but every question currently fails at the same point: no API
key is configured. That is a configuration state, not a defect.

The question set (`eval/questions.yaml`) covers what the corpus is built to test:
questions with a clear single answer, the group-vs-subsidiary conflict, the
policy-vs-retention-schedule precedence case, and four questions the documents genuinely
don't answer (each term confirmed absent from the full extracted text).

Refusal accuracy, retrieval hit rate, citation correctness and answer correctness are all
**unmeasured**. Run the commands above with real keys to produce them.

---

## Migration required

The schema changed (page, content hash, float confidence, `grounded`, `refusal_reason`).
`init.sql` runs only on a fresh volume, so:

```bash
docker compose down -v && docker compose up -d --build
```

This drops the local dev database and re-ingests. There is no Alembic: for a schema that
isn't evolving under production traffic, a single reliable `init.sql` plus a documented
recreate is the smaller, more honest tool. Real production use would need real migrations.

---

## What I'd change for production

- Real auth and per-user rate limiting.
- Alembic once the schema evolves against data that can't be dropped.
- A reranker between retrieval and generation once the corpus outgrows a few thousand
  chunks; an HNSW index at tens of thousands.
- Document versioning so a citation can survive the policy being updated — the corpus
  already models this with `NFS-POL-001` / `-A`.
- Move ingestion fully off container start to an explicitly triggered job (`/api/ingest`
  exists; boot-time ingestion still runs).
- Aggregate the per-query cost and latency already recorded, and alert on drift —
  especially a rising `citations_unverified` rate, which is the signal that generation
  quality is slipping.
- Sample real queries for human review of citation faithfulness. It is the failure mode
  that matters most here and the hardest to catch automatically.

---

## Note on AI tool usage

I used Claude in two phases: the initial codebase via the chat interface from a detailed
spec, then a second pass with Claude Code — with Docker available — to move onto the real
PDF corpus, add the test suite, and restructure into the two pipelines.

Three bugs are worth recording, because all three were found by *running* the code rather
than reading it, and none looked wrong in review:

1. **An oversized `ivfflat` index** silently dropping most search results — caught by a
   query returning 1 row instead of 5.
2. **Clause parsing that matched only `\d+\.\d+`**, silently gluing 325 lettered clauses
   and every unnumbered section onto their preceding clause. Chunk count went from 571 to
   983 once fixed; before that, ~40% of the corpus was lost or misattributed.
3. **A `dict.get(value, 0)` in confidence blending** that mapped an unrecognised value to
   the lowest level. It reads as the safe choice, but it *repaired* off-schema model
   output into a schema-valid value, turning a loud 500 into a plausible-looking 200. An
   existing test caught it. The fix was to pass unrecognised values through so validation
   still rejects them — and the same lesson drove the `ModelAnswer` schema, which parses
   the model's response instead of reading it field-by-field with defaults.

The common thread: a normalisation or convenience default placed upstream of a validator
quietly weakens that validator, and each piece looks correct on its own. That is the
argument for running the code, and for keeping the test suite cheap enough that running
it is never a decision.
