# Compliance Copilot

A small internal tool: ask a plain-English question about a set of policy documents and get
back an answer grounded in those documents, with citations — or an honest "not in the docs"
refusal if it isn't there.

Built as a scoped ~5-6 hour exercise, not a production system.

---

## What I built and what I left out, and why

**Built, all core requirements:**
- Docker Compose stack: Postgres (with pgvector) + FastAPI backend + React (Vite) frontend.
- Ingestion pipeline: 5 fabricated sample policy docs in `/docs`, chunked by markdown
  heading with token-window sub-splitting for long sections, embedded, and stored in Postgres
  alongside document/section metadata.
- Retrieval: pgvector cosine similarity search over chunk embeddings.
- Generation: Claude Haiku, called with a forced tool-use schema so the response is
  reliably structured JSON, not parsed out of free text.
- Refusal logic: two layers — (1) a similarity floor that filters out irrelevant chunks
  before the LLM ever sees them, and (2) the LLM itself is instructed to set
  `answerable: false` if the surviving chunks don't actually answer the question. If layer
  1 filters out everything, we skip the LLM call entirely and return the refusal shape
  directly (cheaper, and removes any chance of the model improvising with zero context).
- Frontend: one page — input box, submit, confidence badge, citation list.
- Query history: every query + structured response logged to `query_log`, exposed via
  `GET /history`.
- Eval script: 10 questions (7 answerable, 3 deliberately unanswerable) scored for
  retrieval hit rate, citation correctness (checked against the actual doc text on disk,
  not just "did it return something"), and refusal accuracy.

**Left out, deliberately:**
- All optional extras (function calling / human review flow / deployment / cost logging).
  Per your instructions, I skipped these entirely to keep the core solid rather than
  spreading thin.
- Alembic / migrations framework — a single `init.sql` run via Postgres's
  `docker-entrypoint-initdb.d` hook is enough for a schema that isn't going to evolve
  during this exercise. Real production use would need real migrations (see below).
- Auth, rate limiting, polished UI — explicitly out of scope per the assignment.
- Streaming responses — the query volume here doesn't need it, and it would complicate
  the structured-output tool-use flow for no real benefit at this scale.

---

## Tradeoffs

**Chunking & retrieval:** Documents are split on `#`/`##`/`###` headings first (this
mirrors how a compliance reader actually navigates a policy — "which section" matters
more than "which 500-word blob"), and any section body longer than ~500 tokens is further
split into overlapping ~500-token windows (~50-token overlap) so a fact near a boundary
isn't cut off from its context. Token count is approximated as `words / 0.75` rather than
a real tokenizer — good enough at this corpus size, would swap for `tiktoken` at scale.
Retrieval is a brute-force pgvector cosine scan with no ANN index (ivfflat/HNSW): at
~50-300 chunks, brute force is sub-millisecond and always exact. I actually built this
with an `ivfflat` index initially, `lists=100`, and caught during testing that it was
badly oversized for a corpus this small — pgvector's own docs recommend roughly
`rows / 1000` lists, so 100 lists over 51 rows meant most lists had 0-1 vectors, and
retrieval was silently dropping relevant chunks (visible as a "low recall" NOTICE at
index-creation time, and confirmed when a query that should have returned 5 rows returned
1). I removed the index rather than tuning it, since brute force is strictly better at
this scale. A relevance floor (`MIN_SIMILARITY`) then drops chunks the LLM shouldn't be
trusted to reason from at all.

**Model choice:** Claude Haiku (`claude-haiku-4-5-20251001`) for generation — cheap, fast,
and more than capable of grounded extraction + light reasoning over ~5 short chunks.
Embeddings use a local `sentence-transformers` model (`all-MiniLM-L6-v2`, 384-dim) instead
of a paid embedding API — free, no second API key, and fully sufficient for a corpus this
small. Tradeoff: slightly lower embedding quality than e.g. OpenAI's
`text-embedding-3-small`, and it adds ~90MB + torch to the backend image. For a 4-6 doc
corpus this is a non-issue; I'd revisit at real scale.

**Structured output:** enforced via Anthropic's tool-use with `tool_choice` forced to a
single `submit_answer` tool whose `input_schema` matches the required response schema
exactly. This means the API itself validates the shape — no regex, no markdown-fence
stripping, no "the model added a preamble before the JSON" failure mode.

**"Not in the docs" handling:** two-layer refusal (described above under chunking &
retrieval) rather than relying on the LLM alone. Layer 1 is deterministic and free; layer
2 catches cases where chunks are topically similar but don't actually contain the answer
(e.g. a question about PTO carryover retrieving the Code of Conduct's "annual
acknowledgment" section by loose semantic similarity, but not answering the question).

---

## How to run it

```bash
git clone <this-repo>
cd compliance-copilot
cp .env.example .env
# edit .env and set ANTHROPIC_API_KEY to a real key
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8000 (docs at http://localhost:8000/docs)
- On first boot, the backend container runs ingestion automatically (`python -m
  scripts.run_ingest`) before starting the API, so the docs in `/docs` are loaded and
  embedded before the frontend can query anything. Re-running `docker compose up` re-runs
  ingestion idempotently (it clears and re-embeds each document's chunks rather than
  duplicating them).

To run the eval script once the stack is up:

```bash
cd eval
pip install -r requirements.txt
python eval.py --api-url http://localhost:8000
```

### A note on how I verified this

I don't have Docker available in the environment I built this in, so I could not literally
run `docker compose up --build` myself. Two things I could not fully verify as a result:

1. **The Docker build itself** — the Dockerfile, compose wiring, and startup command are
   correct by inspection and match a pattern I'd normally trust, but I have not watched a
   real `docker compose up --build` succeed.
2. **Live LLM calls** — I did not have an Anthropic API key available to exercise real
   `/query` calls end-to-end.

What I *did* verify directly, outside Docker, against a real local Postgres+pgvector
install and a real Python venv with the actual project dependencies:
- `init.sql` runs cleanly and creates the `vector` extension + all three tables.
- The chunking logic (`chunk_document`) run against the real 5 sample docs, producing 51
  correctly-headed chunks — this run caught and fixed a real bug (heading paths were
  duplicating the document title, e.g. "X > X > Section").
- Writing chunks + embeddings into Postgres and querying them back with pgvector's cosine
  distance operator — this run caught and fixed the oversized-`ivfflat`-index bug above.
- The FastAPI app boots for real; `/health` and `/history` respond correctly over HTTP.
- `/query` correctly returns a clean `500` (not a crash, not a bad guess) when it can't
  produce an answer — in my sandbox this triggered because the embedding model download
  (`huggingface.co`) is network-blocked there, not because of a code bug; in a normal
  Docker build with full internet access, the model is pre-downloaded at build time (see
  `Dockerfile`), so this isn't expected to be an issue for you.
- The frontend installs and builds cleanly (`npm run build`) with no errors.
- `eval.py` runs against a live server and produces a correctly-formatted report (see
  below) — the mechanics of the harness are confirmed even though the actual pass rates
  couldn't be captured in my sandbox.

I'd treat first boot on your machine as the real test of the Docker wiring and the live
LLM path — I expect it to work, but I want to be upfront that I couldn't confirm it myself.

---

## Eval results

I could not produce genuine pass/fail numbers in my build environment, for the reasons
above (no Docker, no Anthropic API key, and the embedding model's one-time download is
blocked by my sandbox's network policy). Here is the actual output from running `eval.py`
against a live instance of the API, to show the harness itself works end-to-end — every
row fails at the same point (the embedding call inside `/query`, per the note above), not
because of a bug in the eval script:

```
ID   Answerable? Refusal Retrieval Citation  Question
----------------------------------------------------------------------------------------------------
q1   ERROR       FAIL    -         -         How long do we keep financial and billing records?  (500 Server Error...)
q2   ERROR       FAIL    -         -         Is multi-factor authentication required for employ  (500 Server Error...)
q3   ERROR       FAIL    -         -         How quickly must we notify customers after a confi  (500 Server Error...)
q4   ERROR       FAIL    -         -         What's the maximum value of a gift an employee can  (500 Server Error...)
q5   ERROR       FAIL    -         -         Can employees use personal AI tools to process cus  (500 Server Error...)
q6   ERROR       FAIL    -         -         How often are production system access rights revi  (500 Server Error...)
q7   ERROR       FAIL    -         -         What is the company's policy on remote work stipen  (500 Server Error...)
q8   ERROR       FAIL    -         -         What's the maximum PTO an employee can carry over   (500 Server Error...)
q9   ERROR       FAIL    -         -         Does the company reimburse employees for home inte  (500 Server Error...)
q10  ERROR       FAIL    -         -         How long are employee personnel records kept after  (500 Server Error...)
----------------------------------------------------------------------------------------------------
Refusal accuracy:     0%  (0/10)
Retrieval hit rate:   0%  (0/0 answerable questions)
Citation correctness: 0%  (0/0 questions with citations)
```

**Please re-run this yourself** after `docker compose up --build` with a real
`ANTHROPIC_API_KEY` in `.env` — I expect real numbers in the 80-100% range given the
corpus is small, the questions are direct, and the two-layer refusal logic was designed
and independently retrieval-tested (see "How I verified this" above) to be conservative
about false positives. I'd genuinely like to see the real numbers and would fix anything
that doesn't look right.

---

## What I'd change for production, and how I'd monitor quality over time

**Production changes:**
- Real auth (even just an internal SSO gate) and per-user rate limiting.
- Real migrations (Alembic) instead of a single `init.sql`.
- A proper document ingestion job that handles updates/deletions of source docs, PDF
  input (not just markdown), and versioning — so a citation can point to "Data Retention
  Policy v3, Section 2.2" and survive the policy being updated later.
- A reranker (e.g. a cross-encoder) between retrieval and generation once the corpus
  grows past a few hundred chunks — pure cosine similarity on a small embedding model
  starts missing nuance at scale.
- Move off `docker-entrypoint-initdb.d`-style ingestion-on-boot to a separate,
  explicitly-triggered ingestion job, so a backend restart doesn't imply an ingestion run.
- Swap the local embedding model for a hosted one only if latency/throughput at scale
  demands it — otherwise it's genuinely the cheaper and simpler option.
- Add a real ANN index (HNSW, which pgvector now supports and which degrades more
  gracefully than ivfflat) once chunk count is in the tens of thousands.

**Monitoring quality over time:**
- Run the eval script (extended to 50-100 questions, covering edge cases like partially-
  answerable questions and questions that span multiple documents) as a CI gate on every
  change to prompts, chunking, or the embedding model.
- Log retrieval similarity scores and confidence levels per query (the `query_log` table
  already has the shape for this) and alert on drift — e.g. a rising rate of `low`
  confidence answers, or `answerable: false` rates that spike after a doc corpus update.
- Sample a percentage of real production queries for periodic human review, specifically
  checking citation faithfulness (does the excerpt actually say what the answer claims it
  says) — this is the failure mode that matters most in a compliance context and is the
  hardest to catch automatically.
- Track latency and token cost per query (the `latency_ms` column is already there;
  token cost would be a small addition using the `usage` field on the Anthropic response)
  to catch cost or performance regressions early.

---

## Note on AI tool usage

I used Claude (via this interface, not Claude Code specifically) to write the full
codebase from a detailed spec I gave it up front — file tree, chunking strategy, schema,
and constraints were all specified before code was written, and I had it build
checkpoint-by-checkpoint rather than dumping the whole repo at once.

Where it helped: fast, consistent scaffolding across backend/frontend/eval in one pass;
the forced-tool-use structured output pattern was implemented correctly on the first try;
the chunking logic (heading-path tracking, overlap windows) was correct on the first pass
too.

Where it needed correction, caught by actually running the code rather than trusting it
by inspection: an oversized `ivfflat` vector index that was silently dropping most search
results (real recall bug, only caught by running a query and getting 1 row back instead
of 5), and a heading-path bug that duplicated the document title in every citation's
section field. Both were real bugs that inspection alone likely would have missed, which
is the main argument for the "run it, don't just read it" verification approach used
throughout this build (see the "How I verified this" section above for what was and
wasn't possible to run in this particular environment).
