# Current Status — Verified Baseline

**As of 2026-08-13.** Every figure below was verified by running the system, not inferred
from the code. Where something is unverified or unstable, it is labelled as such.

| | |
|---|---|
| Evaluation | **15/15 questions pass** |
| Refusal accuracy | **100%** (15/15) |
| Retrieval hit rate | **100%** (11/11 answerable) |
| Citation correctness | **100%** (15/15) |
| Answer correctness | **100%** (11/11) |
| Test suite | **229 passing**, 0 failing, ~5s |
| Corpus | 26 documents · 983 chunks · 983 embeddings · 870 numbered clauses |
| Stack | db + backend + frontend running under normal `docker compose up -d` |

> An earlier revision of this file was a gap analysis written *before* the work was done.
> It has been replaced: several items it listed as missing are now built and verified.
> The history is preserved in git and summarised under "How this baseline was reached".

---

## 1. Architecture

Two pipelines, deliberately separated. Ingestion is offline and writes; querying is
online and reads. Neither can reach into the other — there is no PDF parsing on the query
path and no question answering on the ingestion path.

```
INGESTION (offline)                    QUERY (online)
  documents/*.pdf                        question (+ optional entity)
    → extract      page-aware text        → resolve scope   entity + current/historical
    → clean        strip page furniture   → embed           same model as ingestion
    → detect       front matter + clauses → retrieve        pgvector, candidate_k=20
    → chunk        one clause per chunk   → relevance       similarity floor → refuse early
    → embed        hosted API             → applicability   drop non-applicable entities
    → store        Postgres + pgvector    → rerank          explainable signals → top_k=5
                                          → governing       retention questions only
                                          → generate        LLM sees evidence only
                                          → validate        Pydantic
                                          → map citations   chunk_id → DB record
                                          → respond         + retrieval metadata
```

**Module map**

```
app/main.py                  assembly only
app/config.py                every tunable, env-configurable
app/api/routes/              health.py · query.py · ingestion.py
app/ingestion/               extractor.py · chunker.py · embedder.py · pipeline.py
app/query/                   pipeline.py · applicability.py · reranker.py · governing.py
app/retrieval.py             pgvector search + status/document filters
app/llm.py                   prompts + forced function-calling structured output
app/verification.py          excerpt grounding
app/citations/mapper.py      chunk_id → citation (DB is the source of truth)
app/confidence.py            application-level score
```

## 2. Ingestion pipeline

`document → extract → clean → section/clause detection → chunk → embed → store`

- **PDF extraction** with page numbers preserved from extraction through to citation.
- **Front-matter metadata**: `doc_id`, `document_type`, `entity`, `version`, `status`,
  `status_note`, `effective_date`. All 26 documents parse with no nulls.
- **Page furniture removed** (running header, disclaimer, page marker) — verified absent
  from every chunk.
- **Scanned-PDF detection**: a PDF with no extractable text raises rather than silently
  producing zero chunks.
- **Idempotent by content hash**: identity is `(filename, sha256)`. Unchanged documents
  are skipped without re-embedding; a changed document replaces only its own chunks; new
  documents are added. Verified: a full re-run reports `skipped_unchanged=26`,
  `documents=0`, counts unchanged, in ~2s versus ~31s for the initial run.
- **Failure isolation**: one malformed document is reported and skipped without aborting
  the run or committing partial state.
- **Batch embedding**: 128 inputs per request.

**Document type distribution:** policy 16 · entity_policy 4 · procedure 4 · guidance 1 ·
regulatory 1.

## 3. Retrieval pipeline

`candidate_k=20 → relevance floor (0.25) → applicability → rerank → top_k=5`

`candidate_k` and `top_k` are separate on purpose. Fetching exactly the chunks you intend
to use means any filter can only *shrink* the evidence; oversampling lets a filtered-out
chunk be replaced by the next applicable one.

**No ANN index**, deliberately. At ~1,000 chunks an exact brute-force cosine scan is
sub-millisecond and always correct. An earlier `ivfflat` index with `lists=100` over 51
rows was silently dropping most results and was removed rather than tuned. HNSW becomes
appropriate in the tens of thousands of chunks.

## 4. Entity applicability

Northwind is three legal entities (NFS group 22 docs · NCM 3 · NPS 1) whose policies
carry deliberately different figures.

- Entity is resolved from an explicit API parameter, or inferred from the question text,
  or left unscoped (group default). An explicit value always wins.
- **Other subsidiaries are filtered out entirely** — a Payments (Singapore) figure is not
  weak evidence about Capital Markets, it is not evidence about it at all.
- Group documents remain applicable alongside the scoped entity, because group policy
  applies wherever the subsidiary has nothing of its own.
- **The resolved scope is passed into the generation context.** This closed a real bug:
  with the entity selector set to Capital Markets, retrieval and reranking correctly
  ranked `NFS-SUB-002` first, but the model still led with the group's five years because
  the question text named no subsidiary. Verified fixed — NCM now answers seven years.

## 5. Reranking

A weighted sum of named metadata signals, each recorded and returned in the API's
retrieval metadata. No ML reranker: a cross-encoder reads the same text that cosine
similarity already read, and in this corpus the deciding information is metadata, not
wording.

**Ordering of concerns:** applicability first (entity match +0.35, group fallback +0.05),
then status (superseded −0.60 for current questions, +0.30 when historical), then document
authority *last and gently* (policy +0.05, guidance −0.10), plus clause specificity +0.03.

**There is deliberately no global `POLICY > SOP > HANDBOOK` rule**, and the corpus proves
why: `NFS-POL-009 §3.5` defers to `NFS-REG-001` for the only statement of the change-freeze
window, so a universal type hierarchy would demote the sole correct source. Guidance is
demoted by a *small* weight — enough to lose a tie against an equally relevant policy, not
enough to lose to a marginally relevant one.

## 6. Governing-document retrieval

**Corpus-specific rule, and labelled as such in the code.** For retention questions only,
two targeted lookups against `NFS-POL-011` (Records Retention Schedule):

1. **§1.2 by identity** — the clause stating the schedule governs. Fetched by clause
   number, not similarity, because a rule of precedence scores poorly against the
   questions that need it: measured at **rank 39 corpus-wide and rank 6 within its own
   document**, behind three chunks about destruction logging.
2. **The applicable period by document-scoped similarity** — `retrieve(doc_id=...,
   limit=2)`, reusing the existing vector path, returning both annexures.

Detection is narrow: explicit retention vocabulary, or "how long" *paired with* a storage
verb. "How long does the company have to respond…" is correctly excluded. Non-retention
questions short-circuit before either query — verified by a test asserting zero calls.

A failed lookup is logged and ignored rather than failing the request; supplementary
evidence must not take down an answer the normal path already produced.

## 7. LLM generation

- **Provider: OpenAI.** Model `gpt-4o-mini`, chosen for function-calling support at the
  cheapest tier; the task is grounded extraction over ~5 short clauses.
- Structured output is enforced by **forced function calling**, not by asking for JSON.
  Arguments arrive as a JSON string and are `json.loads`-ed; malformed payloads raise
  rather than degrading into a plausible refusal.
- Every response is parsed into a Pydantic `ModelAnswer` before use — reading fields
  defensively would silently accept off-schema values (`bool("not-a-bool")` is `True`).
- Pricing constants (0.15 / 0.60 per MTok) were verified against OpenAI's published
  pricing, not guessed. They will drift if rates change.
- Token usage and indicative cost are logged per query; refusals that short-circuit
  before the API call record zero.

## 8. Grounding

Every excerpt is verified against **the specific chunk it cites**, not the whole context.
Verifying against the concatenation would let a quote from chunk A be attributed to chunk
B — a citation pointing at the wrong clause while quoting genuine corpus text, which
survives a casual read. Matching is fuzzy enough to tolerate whitespace and truncation,
strict enough to reject invention.

## 9. Citation mapping

**The model returns a `chunk_id` and a quote. Nothing else.** Document title, ID, version,
entity, section, clause, page and status are looked up from the stored record. There is no
field for the model to fabricate because it never supplies one.

Two checks remain: the `chunk_id` must be one shown to the model this turn (an invented
integer cannot address an arbitrary row), and the quote must appear in that chunk. If a
model claims an answer but no citation survives, the answer is withheld.

## 10. Abstention / refusal

Three distinct reasons, because they mean different things operationally:

| Reason | Meaning | Model called? |
|---|---|---|
| `no_relevant_evidence` | Nothing cleared the similarity floor | No — free, cannot be argued out of |
| `model_declined` | Evidence existed but did not answer | Yes |
| `citations_unverified` | Answer drafted, support failed verification | Yes |

Refusal wording is exactly *"I don't know based on the provided documents."* Verified end
to end: `grounded=false`, `refusal_reason` set, zero citations, confidence ≤ 0.2.

## 11. Database / pgvector

| | |
|---|---|
| PostgreSQL | 15.4, healthy, host port **5434** |
| pgvector | **0.5.1** |
| `chunks.embedding` | **`vector(384)`** — confirmed from stored data via `vector_dims()` |
| Rows | documents 26 · chunks 983 · non-null embeddings 983 · null 0 |
| Clause coverage | 870 chunks carry a clause reference; all 983 carry a page |
| Indexes | `documents_status_idx`, `documents_entity_idx`, `documents_type_idx` |
| Vector index | None (deliberate at this scale — see §3) |

Embeddings verified genuine, not placeholders: 152 distinct L2 norms across a 200-row
sample.

## 12. Evaluation results

`python eval.py --api-url http://localhost:8000` — 15 questions, real retrieval and real
OpenAI calls, ~38s.

```
Refusal accuracy:     100%  (15/15)
Retrieval hit rate:   100%  (11/11 answerable questions)
Citation correctness: 100%  (15/15 questions with citations)
Answer correctness:   100%  (11/11 questions with expected keywords)
PASS: all metrics >= 80%
```

The set covers what the corpus is built to test: questions with a clear single answer, the
current-vs-superseded trap, the group-vs-subsidiary conflict, the policy-vs-schedule
precedence case, and four questions the documents genuinely do not answer (each term
confirmed absent from the full extracted text).

## 13. Test results

**229 passing, 0 failing, 0 skipped, ~5s.** No database, API keys or network required —
retrieval and generation are stubbed at their boundaries.

| Area | Tests |
|---|---|
| `tests/ingestion` | 45 |
| `tests/retrieval` | 6 |
| `tests/query` | 154 |
| `tests/api` | 24 |

Covering extraction, metadata, clause chunking, page tracking, content-hash idempotency,
relevance filtering, entity applicability, superseded/historical handling, reranking
signals, governing-document selection, grounding, citation mapping, confidence,
abstention, structured-output validation, the query pipeline and the HTTP contract.

## 14. Known limitations

1. **Unscoped subsidiary-vs-group ranking can vary.** With no entity specified, a
   subsidiary clause can out-rank group policy on similarity (measured: `NFS-SUB-003`
   0.804 vs `NFS-POL-001` 0.792). The prompt instructs the model to lead with the group
   position when no scope is given, but ranking order still influences which figure it
   reaches for. Not addressed.

2. **q11's pass is NOT a verified fix.** "How often must high-risk customers be
   re-verified?" scores PASS, but its answer still *leads* with Capital Markets' six
   months on an unscoped question, then mentions the group's twelve. It passes only
   because the eval's any-keyword match and cited-document check are both satisfied.
   Nothing in the codebase was changed to address it — the governing rule provably does
   not fire for this question. **Treat this as model variance; a re-run may flip it back.**

3. **Governing-document retrieval is intentionally corpus-specific.** It hardcodes
   `NFS-POL-011` and clause `1.2`. This is justified by the corpus — that document
   declares itself governing in its own §1.2 — but it is **not a general RAG technique**.
   The general version is a `governing` flag set during ingestion, which would require a
   schema change.

4. **LLM generation uses OpenAI** (`gpt-4o-mini`), not Anthropic. The provider swap is
   contained to `app/llm.py`; everything downstream consumes the provider-neutral
   `Generation` dataclass. `LLM_API_KEY` is unset and falls back to `EMBEDDING_API_KEY`,
   since both run on the same OpenAI account.

5. **Confidence is not a calibrated probability.** It is a bounded application-level score
   combining retrieval relevance, groundedness, citation coverage and the model's
   self-assessment. No labelled dataset was used to fit the weights.

6. **The test suite stubs the LLM entirely.** All 229 tests pass whether or not the real
   API integration is correct. This gap is real and has bitten twice — a provider-swap
   parsing difference and a prompt regression were both caught only by live calls, never
   by tests. There are no integration tests against a real Postgres, and no frontend tests.

7. **Effective dates are extracted and stored for all 26 documents but unused** in
   filtering or ranking.

8. **`eval.py`'s "retrieval hit rate" measures citation of the expected document**, not
   whether it was retrieved. A retrieved-but-uncited document is scored as a retrieval
   failure.

9. **Token approximation in markdown chunking** uses `words / 0.75` rather than a real
   tokenizer. Unused by the PDF corpus.

10. **No auth, no rate limiting, CORS is `*`.** Explicitly out of scope for the exercise.

11. **Pricing constants will drift** from OpenAI's real rates unless updated.

---

## How this baseline was reached

Four defects were found by *running* the system; none were visible by reading it.

1. **Oversized `ivfflat` index** silently dropping most search results — caught by a query
   returning 1 row instead of 5.
2. **Clause parsing matching only `\d+\.\d+`**, silently gluing 325 lettered clauses and
   every unnumbered section onto the preceding clause. Chunk count 571 → 983 once fixed;
   before that, ~40% of the corpus was lost or misattributed, and no test failed.
3. **A `dict.get(value, 0)` in confidence blending** that mapped an unrecognised value to
   the lowest level — it *repaired* off-schema model output into a schema-valid value,
   turning a loud 500 into a plausible 200.
4. **Entity scope never reaching the model**, and **the governing rule never reaching the
   model** — both produced fluent, well-cited, wrong answers.

The common thread: a normalisation or convenience default placed upstream of a validator
quietly weakens that validator, and each piece looks correct on its own.
