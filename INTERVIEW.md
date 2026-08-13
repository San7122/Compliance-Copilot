# Interview Preparation

Everything below describes what is actually implemented. No aspirational architecture.

---

# Part 1 — Architecture Summary

### A. Problem
Answer plain-English questions about a policy corpus with citations traceable to a
specific clause and page, and refuse honestly when the documents don't cover it. The hard
part is not retrieval — it is that near-identical documents differ only in metadata.

### B. Corpus — `docs/`, `CORPUS_README.md`
26 PDFs, ~85 pages, from a fictional financial group. Built with deliberate traps: one
**superseded** policy (`NFS-POL-001-A`, 8-hour breach window vs the current 4), **three
legal entities** with different figures for the same obligations, a **governing** retention
schedule (`NFS-POL-011` §1.2), and a **handbook** that paraphrases policy loosely.

### C. Ingestion pipeline — `app/ingestion/pipeline.py` (`IngestionPipeline`)
`discover → hash → extract → clean → detect → chunk → embed → store`. Offline, writes
only. Runs via `scripts/run_ingest.py` or `POST /api/ingest`.

### D. Metadata strategy — `app/ingestion/extractor.py`
Parsed from front matter: `doc_id`, `document_type`, `entity`, `version`, `status`,
`status_note`, `effective_date`, plus `page` and `clause` per chunk. **Metadata is
load-bearing, not decoration** — `status` and `entity` decide whether a document may be
used at all. `classify_document()` maps the ID series (POL/SUB/SOP/GUID/REG) to a type;
unrecognised series stay `unknown` rather than being guessed.

### E. Chunking strategy — `app/ingestion/chunker.py` (`split_into_clauses`)
One chunk per numbered clause — 983 chunks, 870 with a clause reference. Handles lettered
clauses (`E.n`/`C.n`/`R.n`, 325 of them) and unnumbered sections. No token windowing on
the PDF path: the longest clause is ~170 words, so windowing would only sever an
obligation from its conditions. Markdown path retains heading chunking with configurable
size/overlap for non-PDF sources.

### F. Embedding strategy — `app/ingestion/embedder.py` (`EmbeddingService`)
OpenAI `text-embedding-3-small` at **384 dimensions** via the `dimensions` parameter.
Documents and queries route through one `_embed` call so they cannot drift apart. The
returned width is asserted on every response rather than assumed.

### G. PostgreSQL + pgvector — `app/models.py`, `app/init.sql`
One datastore for rows and vectors. `chunks.embedding vector(384)`; indexes on
`documents.status`, `.entity`, `.document_type`. **No ANN index** — see Q6.

### H. Retrieval — `app/retrieval.py` (`retrieve`, `filter_by_relevance`)
`candidate_k=20` → similarity floor `0.25` → applicability → rerank → `top_k=5`. Status
filtering happens in SQL *before* ranking. An optional `doc_id` filter exists solely for
governing-document lookups.

### I. Entity applicability — `app/query/applicability.py`
`resolve_scope()` resolves entity (explicit API value wins over inferred) and intent
(current vs historical). `applies_to()` admits the scoped entity plus group and excludes
other subsidiaries outright. The resolved scope is passed into generation via
`llm.scope_directive()`.

### J. Reranking — `app/query/reranker.py`
Weighted sum of named signals, each returned in the API response. Entity match +0.35,
group fallback +0.05, superseded −0.60 (or +0.30 when historical), policy +0.05, guidance
−0.10, clause +0.03. **No ML reranker** — see Q11.

### K. Governing-document retrieval — `app/query/governing.py`
Corpus-specific. For retention questions only: `fetch_clause()` retrieves §1.2 **by
identity**, and a document-scoped `retrieve(doc_id=...)` returns the annexures.

### L. LLM generation — `app/llm.py` (`generate_answer`)
OpenAI `gpt-4o-mini`. Structured output enforced by forced function calling
(`tool_choice`), arguments `json.loads`-ed, validated into `schemas.ModelAnswer`.

### M. Grounding — `app/verification.py` (`excerpt_is_grounded`)
Fuzzy sliding-window match of the quote against **the specific cited chunk**.

### N. Citation mapping — `app/citations/mapper.py` (`map_citations`)
The model returns `chunk_id` + quote only; all other fields come from the stored row.

### O. Abstention — `app/query/pipeline.py`
Three reasons: `no_relevant_evidence`, `model_declined`, `citations_unverified`. Wording:
*"I don't know based on the provided documents."*

### P. Evaluation — `eval/eval.py`, `eval/questions.yaml`
15 questions (11 answerable, 4 not). Scores refusal accuracy, retrieval hit rate, citation
correctness, answer correctness. Verifies excerpts against the PDFs on disk, not the API's
own claims. Exits non-zero below `--fail-under`.

### Q. Main limitations
Unscoped subsidiary-vs-group ranking varies; q11's pass is not a verified fix; governing
retrieval is corpus-specific; tests stub the LLM entirely; confidence is uncalibrated.
Full list in `CURRENT_STATUS.md` §14.

---

# Part 2 — 20 Interview Questions

### Q1. Why PostgreSQL + pgvector rather than a dedicated vector database?
**Answer:** The metadata *is* the hard part of this problem, not the vectors. Entity,
status and document type decide correctness, and they need to be filtered and joined
alongside embeddings. One datastore gives that in a single query with transactional
consistency.
**Deeper:** A dedicated vector DB would mean syncing document metadata into two systems
and reconciling them. At ~1,000 chunks the vector workload is trivial; the relational
workload is what matters. Status filtering happens in the same `WHERE` as the vector scan.
**Point to:** `app/retrieval.py:retrieve`, `app/init.sql`

### Q2. Why 384 dimensions?
**Answer:** The schema already used `vector(384)` from an earlier local model.
`text-embedding-3-small` supports a `dimensions` parameter, so I pinned it to 384 and the
provider switch needed **no migration and no re-embedding**.
**Deeper:** Matryoshka truncation costs some representational fidelity, but at a
1,000-chunk corpus retrieval is not precision-limited. The width is asserted on every
response, so a silent mismatch fails loudly instead of writing bad vectors.
**Point to:** `app/ingestion/embedder.py:_embed`, `config.embedding_dimensions`

### Q3. Why clause-aware chunking rather than fixed-size with overlap?
**Answer:** Every document numbers its clauses, and a clause is exactly one self-contained
obligation — the right unit both to retrieve and to cite. "NFS-POL-001 clause 7.1, p.3" is
checkable; "somewhere in the policy" is not.
**Deeper:** Measured: longest clause ~170 words, well inside a 500-token budget, so
windowing would almost never fire and would sever obligations from their conditions when
it did. Markdown sources still use heading chunking with configurable windows.
**Point to:** `app/ingestion/chunker.py:split_into_clauses`

### Q4. Why is metadata so central?
**Answer:** Because in this corpus the group and subsidiary versions of a policy are
written to look identical apart from the numbers. Cosine similarity cannot tell them
apart — they score within noise of each other. Only metadata can.
**Deeper:** It drives three separate mechanisms: applicability filtering (which documents
may be used), reranking (which rank highest), and citation generation (what the user
sees). Same fields, three jobs.
**Point to:** `app/ingestion/extractor.py:parse_metadata`

### Q5. Why `candidate_k=20` and `top_k=5`?
**Answer:** They answer different questions. `candidate_k` is how much evidence I consider;
`top_k` is how much the model sees. If you fetch exactly what you'll use, any filter can
only *shrink* the evidence — drop an inapplicable chunk and you send four, with nothing
promoted to replace it.
**Deeper:** Oversampling makes filtering free. `top_k=5` is a context-cost and focus
choice, not a retrieval limit. Both are configurable.
**Point to:** `config.candidate_k`, `config.top_k`, `app/query/pipeline.py:retrieve_evidence`

### Q6. Why no ANN index at this size?
**Answer:** At ~1,000 chunks a brute-force exact scan is sub-millisecond and always
correct. An ANN index can only add approximation error.
**Deeper:** I had an `ivfflat` index with `lists=100` over 51 rows and it was silently
dropping most results — pgvector recommends roughly `rows/1000` lists, so most lists held
0–1 vectors. I removed it rather than tuned it. HNSW becomes appropriate in the tens of
thousands.
**Point to:** `app/init.sql` (comment explaining the omission)

### Q7. Why does entity applicability come before reranking?
**Answer:** Because applicability is a question of *whether* a document may be used, and
ranking is a question of *how relevant* it is. A Payments (Singapore) figure isn't weak
evidence about Capital Markets — it isn't evidence about it at all.
**Deeper:** `NFS-SUB-001`'s own status line says staff "must follow this document, not the
group policy of the same name". If authority ranked first, group and subsidiary tie on
document type and the group's text wins on similarity — wrong for that reader.
**Point to:** `app/query/applicability.py:applies_to`, `app/query/reranker.py:score_chunk`

### Q8. Why no global `POLICY > SOP > HANDBOOK` hierarchy?
**Answer:** Because the corpus disproves it. `NFS-POL-009 §3.5` says the change freeze
applies during periods *"listed in NFS-REG-001"* — the only statement of the actual
3-business-day window lives in the regulatory document. A universal hierarchy would demote
the sole correct source, against the policy's own instruction.
**Deeper:** Guidance is demoted by a *small* weight (−0.10) — enough to lose a tie against
an equally relevant policy, not enough to lose to a marginally relevant one. Tests pin
both directions.
**Point to:** `app/query/reranker.py:_authority_adjustment`, `tests/query/test_reranker.py`

### Q9. Why are superseded documents excluded rather than deleted?
**Answer:** They're excluded from *current* questions because they don't state current
obligations — `NFS-POL-001-A` says so on its own front page. They're not deleted because
the corpus keeps them for audit reference, and "what did we previously require?" is a
legitimate question.
**Deeper:** Filtering happens in SQL *before* ranking, so the superseded text can't consume
top-k slots. That matters: it's fluent, on-topic and highly retrievable, with an 8-hour
window against the current 4.
**Point to:** `app/retrieval.py:retrieve`, `config.rerank_superseded_penalty`

### Q10. How do historical questions work?
**Answer:** `detect_intent()` classifies the question; historical intent sets
`include_superseded=True`, and the reranker *promotes* superseded text (+0.30) instead of
penalising it. Same signal, read differently by intent.
**Deeper:** The marker list is deliberately conservative. `"used to"` is a marker;
`"use to"` is **not**, because it appears in ordinary present-tense questions ("what
encryption do we *use to* protect data?"). A false positive there would admit superseded
policy into a current-requirements answer. There's a test for exactly that.
**Point to:** `app/query/applicability.py:detect_intent`, `tests/query/test_applicability.py`
⚠️ *Honest caveat: verified at component level and on real corpus text, but never run
end-to-end through the API with a live LLM.*

### Q11. Why is reranking metadata-based rather than a cross-encoder?
**Answer:** A cross-encoder reads the same text cosine similarity already read. In this
corpus the deciding information is metadata, not wording — so it wouldn't help, while
adding a model, a dependency, latency, and the inability to explain a ranking.
**Deeper:** Every contribution is a named signal summed to a score and returned in the API
response, so a wrong ranking is diagnosable. That's how I found the q9 defect: I could see
the governing chunk at 0.737 sitting 0.061 below the cut.
**Point to:** `app/query/reranker.py:ScoredChunk.explain`

### Q12. Why do citations come from the database rather than the LLM?
**Answer:** So fabricated citations are structurally impossible rather than detected after
the fact. The model returns a `chunk_id` and a quote; document, ID, version, entity,
clause, page and status are looked up. There's no field for it to invent.
**Deeper:** Two checks remain — the `chunk_id` must be one shown this turn (an invented
integer can't address an arbitrary row), and the quote must appear in *that* chunk. If a
model claims an answer and no citation survives, the answer is withheld.
**Point to:** `app/citations/mapper.py:map_citations`

### Q13. How does grounding verification work, and why per-chunk?
**Answer:** Fuzzy sliding-window match of the quote against the cited chunk's stored text.
Fuzzy because models normalise whitespace and trim clauses; strict enough to reject
invention.
**Deeper:** Verifying against the whole context would let a quote lifted from chunk A be
attributed to chunk B — a citation pointing at the wrong clause while quoting *genuine*
corpus text. That survives a casual read, which makes it worse than an obvious error.
There's a test pairing exactly that case with its control.
**Point to:** `app/verification.py:excerpt_is_grounded`, `tests/query/test_citation_mapper.py`

### Q14. How does abstention work?
**Answer:** Three distinct reasons. `no_relevant_evidence` — nothing cleared the similarity
floor, so the model is never called. `model_declined` — evidence existed but didn't answer.
`citations_unverified` — an answer was drafted but nothing it quoted survived checking.
**Deeper:** They're separate because they mean different things operationally: a rising
`citations_unverified` rate is the clearest early signal that generation quality is
slipping. Collapsing them into one "I don't know" would hide which is firing.
**Point to:** `app/query/pipeline.py:run`, `_refusal`

### Q15. How is structured output enforced?
**Answer:** Forced function calling — `tool_choice` pinned to a single `submit_answer`
function whose parameters are the response schema. The API constrains the shape; there's no
regex or markdown-fence stripping.
**Deeper:** OpenAI returns arguments as a **JSON string**, not an object, so it's
`json.loads`-ed, and malformed JSON raises rather than degrading into a plausible refusal.
The result is then parsed into a Pydantic `ModelAnswer` — reading fields defensively would
silently accept off-schema values, since `bool("not-a-bool")` is `True`.
**Point to:** `app/llm.py:ANSWER_SCHEMA`, `app/schemas.py:ModelAnswer`

### Q16. Why OpenAI for generation?
**Answer:** Pragmatism — I had an OpenAI key and no Anthropic key. It was originally
Anthropic Claude Haiku, and the swap touched three files because everything downstream
consumes a provider-neutral `Generation` dataclass.
**Deeper:** `gpt-4o-mini` because the task is grounded extraction over five short clauses
and it supports function calling at the cheapest tier ($0.15/$0.60 per MTok, verified
against OpenAI's published pricing rather than from memory). Real measured cost: ~$0.0003
per query.
**Point to:** `app/llm.py`, `config.llm_model`

### Q17. How does idempotent ingestion work?
**Answer:** Document identity is `(filename, sha256)`. Unchanged files are skipped without
re-embedding; a changed file replaces only its own chunks; new files are added.
**Deeper:** Filename alone can't distinguish "same document" from "edited document", so a
filename-only design must re-embed everything every run. Measured: full ingest 31s, re-run
2s with `skipped_unchanged=26`. One malformed document fails in isolation and is reported
rather than aborting the run.
**Point to:** `app/ingestion/pipeline.py:content_hash`, `IngestionPipeline.run`

### Q18. What happens at 100,000+ chunks?
**Answer:** The architecture holds; the parameters move. Add an HNSW index, push metadata
filtering into SQL *before* the vector scan, and parallelise ingestion per document — safe
because documents are independent and identity is content-hashed.
**Deeper:** Reranking matters *more* as `candidate_k` grows, and it stays cheap because the
signals are metadata lookups. The first thing to break is the exact scan; the indexes on
status/entity/type are already in place to support pre-filtering.
**Point to:** `DESIGN.md` §11

### Q19. What are the current limitations?
**Answer:** Unscoped questions can rank a subsidiary above group policy; q11's eval pass is
model variance, not a verified fix; governing retrieval is corpus-specific; the test suite
stubs the LLM entirely; confidence is uncalibrated.
**Deeper:** The test gap is the one I'd fix first — 229 tests pass whether or not the real
API integration works, and it has already missed two real defects (a provider-swap parsing
difference and a prompt regression), both caught only by live calls.
**Point to:** `CURRENT_STATUS.md` §14

### Q20. What would you change for production?
**Answer:** Auth and rate limiting; Alembic once the schema evolves against undroppable
data; integration tests against real Postgres and a recorded LLM; a `governing` metadata
flag set at ingest to replace the hardcoded document ID.
**Deeper:** Operationally: aggregate the per-query cost and latency already recorded, and
alert on a rising `citations_unverified` rate. Sample real queries for human review of
citation faithfulness — the failure mode that matters most here and the hardest to catch
automatically.
**Point to:** `README.md` "What I'd change for production"

---

# Part 3 — Failure Stories

All four were found by **running** the code. None was visible by reading it.

### 1. Oversized `ivfflat` index
- **Symptom:** a query that should return 5 rows returned 1.
- **Root cause:** `lists=100` over 51 rows. pgvector suggests roughly `rows/1000` lists, so
  nearly every list held 0–1 vectors and the probe missed most of them.
- **Discovery:** running a real query and counting rows. A "low recall" NOTICE at index
  creation was the clue.
- **Fix:** removed the index entirely.
- **Why correct:** at this corpus size exact brute-force scan is sub-millisecond and always
  accurate — an index could only add approximation error.
- **Lesson:** an ANN index is a scale optimisation. Below scale it is pure downside.

### 2. Clause parsing
- **Symptom:** none visible. Every chunk looked well-formed and no test failed.
- **Root cause:** the pattern matched only `\d+\.\d+`. All 325 lettered clauses
  (`E.n`/`C.n`/`R.n`) and every unnumbered section were silently appended to whichever
  numbered clause preceded them.
- **Discovery:** reading extracted output against the source PDF — not from a failure.
- **Fix:** recognise lettered clauses and named sections; drop revision-history tables.
- **Why correct:** chunk count went 571 → 983 and the longest chunk fell from 631 to 167
  words. Roughly 40% of the corpus had been lost or misattributed.
- **Lesson:** silent data corruption is invisible to tests that only assert well-formedness.
  Look at the actual output.

### 3. Confidence normalisation
- **Symptom:** a malformed model response started returning HTTP 200 with a plausible
  answer instead of a loud 500.
- **Root cause:** I added `dict.get(value, 0)` to map an unrecognised confidence to the
  lowest level. That reads as the safe choice — but it *repaired* off-schema output into a
  schema-valid value, defeating the response validation downstream.
- **Discovery:** an existing test asserting the 500.
- **Fix:** pass unrecognised values through untouched so validation still rejects them.
  Later generalised into the `ModelAnswer` Pydantic model.
- **Why correct:** validation should reject invalid input, not silently accept a fallback.
- **Lesson:** **a normalisation step placed upstream of a validator quietly weakens that
  validator.** Every piece looks correct on its own.

### 4. Entity scope and governing evidence
- **Symptom:** two fluent, well-cited, wrong answers. With the entity selector on Capital
  Markets, "how long must KYC records be retained?" answered the group's 5 years instead of
  NCM's 7. And "how long must financial transaction records be retained?" answered 7 years
  when the governing schedule says 8.
- **Root cause:** *both were information the pipeline had and the model didn't.* Retrieval
  and reranking correctly ranked `NFS-SUB-002` first, but the prompt only treated a
  subsidiary as selected if the question text named one — and the entity had come from the
  API parameter. Separately, the governing rule `NFS-POL-011 §1.2` ranked **39th
  corpus-wide**, so it never reached the context.
- **Discovery:** the real 15-question evaluation, then reading the ranking explanations the
  reranker already emits.
- **Fix:** propagate the resolved scope into the generation context; for retention
  questions, fetch §1.2 **by identity** (not similarity) plus the annexures by
  document-scoped search.
- **Why correct:** measured, not assumed. My first attempt supplied only the annexure and
  changed nothing — the model saw 8 years beside 7 and had no stated reason to prefer
  either. Adding the precedence *rule* moved q9 from FAIL to PASS with the model explicitly
  reasoning *"takes precedence since it is longer"*.
- **Lesson:** ranking correctly is not the same as communicating. And when a fix doesn't
  move the metric, that is information — the first attempt failed because I'd traded
  capability for a smaller blast radius, and selecting only from existing candidates could
  never reach a chunk ranked 39th.
