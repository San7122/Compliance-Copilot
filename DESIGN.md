# Design

Why the system is built this way. Focused on the decisions that were genuinely contested
— where a reasonable alternative existed and was rejected for a reason.

---

## 1. Two pipelines, no shared state

Ingestion is offline and writes; querying is online and reads. They share model
definitions and an embedding service, nothing else. There is no PDF parsing reachable
from a request and no question answering reachable from ingestion.

This isn't tidiness. The two have opposite operational profiles: ingestion is slow,
batched, idempotent and re-runnable; querying is latency-sensitive and per-request.
Mixing them produces a system where a slow document parse can block a user query, and
where "why was this document indexed like that?" and "why was this answer given?" are
tangled in the same call stack.

---

## 2. Chunking: clause-level, not fixed-size

**Decision:** one chunk per numbered clause. No token windowing on the PDF path.

The instinct for policy documents is fixed-size chunks with overlap. That was rejected
after looking at the actual corpus:

- Every document numbers its clauses (`7.1`, `E.1`), and the corpus notes state these
  "make good citation targets".
- A clause is exactly one self-contained obligation. Fixed windows cut obligations in
  half and merge unrelated ones, and then the citation can only say "somewhere around
  here".
- Measured: the longest clause in the corpus is ~170 words, well inside a 500-token
  budget. Windowing would therefore almost never fire, and where it did it would split
  an obligation from its own conditions.

**What windowing is still used for:** the markdown ingestion path, retained for non-PDF
sources where sections are prose of arbitrary length. `chunk_max_tokens` and
`chunk_overlap_tokens` remain configurable there.

**What this cost:** clause detection has to be right, and initially it wasn't. Matching
only `\d+\.\d+` silently appended all 325 lettered clauses (`E.n`/`C.n`/`R.n`) and every
unnumbered section onto whichever clause preceded them. Chunk count went from 571 to 983
once fixed — roughly 40% of the corpus had been lost or misattributed, and *no test
failed*, because every chunk still looked well-formed. It was found by reading extracted
output against the source PDF.

**Page numbers** are carried from extraction through to the citation. A clause reference
tells a reader which rule; a page tells them where to look.

---

## 3. Retrieval: oversample, filter, rerank

```
candidate_k=20  →  relevance floor  →  applicability filter  →  rerank  →  top_k=5
```

`candidate_k` and `top_k` are deliberately separate. Fetching exactly the five chunks
you intend to use means any filter can only *shrink* the evidence — drop an inapplicable
chunk and you send four, with nothing promoted to replace it. Oversampling means
filtering costs nothing.

**No ANN index.** At ~1,000 chunks a brute-force exact scan is sub-millisecond and always
correct. An earlier `ivfflat` index with `lists=100` over 51 rows was silently dropping
most results (pgvector recommends roughly `rows/1000` lists). It was removed rather than
tuned, because at this size an index can only add approximation error. At tens of
thousands of chunks, HNSW.

---

## 4. Reranking: explainable signals, no ML

**Decision:** weighted sum of named metadata signals. No cross-encoder.

Cosine similarity cannot resolve this corpus's central difficulty. The group and
subsidiary versions of a policy are written to look identical apart from their numbers,
so they score within noise of each other. The deciding information is metadata — entity,
status, document type — not text. A cross-encoder reranker reads the *same text* and so
would not help, while adding a model, a dependency, latency, and the inability to answer
"why was this chunk chosen?".

Every contribution is a named signal, summed to the score, and returned in the API's
retrieval metadata. A wrong ranking is diagnosable.

### Ordering of concerns

**1. Entity applicability — dominant.** A document that binds the reader's entity
outranks one that doesn't, *before* any consideration of document type. The corpus states
that subsidiary staff "must follow this document, not the group policy of the same name",
so entity is not a tiebreaker; it is the primary question. Group documents remain in play
below entity-specific ones, because group policy applies wherever the subsidiary has
nothing of its own. Other subsidiaries are filtered out entirely — a Payments (Singapore)
figure is not a weak fact about Capital Markets, it is not a fact about it at all.

**2. Status — strong, and intent-dependent.** Superseded text is heavily penalised for
current questions and *promoted* for historical ones. Same signal, read differently.

**3. Document authority — last, and gentle.** Guidance is nudged down; policy nudged up.

### Why there is no `POLICY > SOP > HANDBOOK` rule

That ordering is tempting and wrong. Large parts of this corpus have their true answer in
a procedure (`NFS-SOP-003` for data subject requests) or a regulatory calendar
(`NFS-REG-001` for change freezes). A universal type hierarchy would systematically
demote the correct source for those questions. The handbook is demoted by a *small*
weight — enough to lose a tie against an equally relevant policy, not enough to lose to a
marginally relevant one. Tests pin both directions.

---

## 5. Superseded documents: excluded by default, reachable by intent

`NFS-POL-001-A` states on its front page: "Do not rely on this version for current
obligations." It also contains fluent, on-topic, highly retrievable text with *different
numbers* (an 8-hour breach reporting window against the current 4 hours).

Three options were considered:

1. **Don't ingest it.** Rejected: the corpus retains it deliberately for audit reference,
   and "what did we previously require?" becomes permanently unanswerable.
2. **Ingest and filter unconditionally.** This is what the system did before this
   revision. Correct for current questions, but it makes historical questions impossible
   *by construction* rather than by policy.
3. **Ingest, filter by query intent.** Chosen.

Intent detection is rule-based and inspectable. A model could infer it, but that puts a
non-deterministic step in front of a correctness-critical filter.

The marker list is conservative for a specific reason: `"use to"` is deliberately **not**
a historical marker, because it appears in ordinary present-tense questions ("what
encryption do we *use to* protect customer data?"). A false positive there would admit
superseded policy into an answer about current requirements — the exact failure being
guarded against. `"used to"` is a marker; `"use to"` is not, and there is a test for it.

When superseded text is used, it is labelled `Status: SUPERSEDED` in the context, the
prompt requires the answer to say so, and the citation carries the status.

---

## 6. Citations: the database is the source of truth

**The model returns a `chunk_id` and a quote. Nothing else.**

Document title, ID, version, entity, section, clause and page are all looked up from the
retrieved record. There is no field for the model to fabricate, because it never supplies
one. This is structural prevention rather than post-hoc detection.

Two checks remain:

1. **The `chunk_id` must be one shown to the model this turn.** A model can emit a
   plausible integer for a chunk it never saw; only IDs in the current retrieved set are
   accepted, so an invented ID cannot address an arbitrary database row.
2. **The quote must appear in *that specific chunk*.** Verifying against the whole
   context would allow a quote from chunk A to be attached to chunk B — a citation
   pointing at the wrong clause while quoting genuine corpus text, which survives a
   casual read.

If a model claims an answer but no citation survives, the answer is withheld and becomes
a `citations_unverified` refusal. A confident answer with unverifiable support is worse
than no answer here.

---

## 7. Confidence: a score, not a probability

Combines retrieval relevance, groundedness, citation coverage and the model's
self-assessment, weighted so measured signals (0.65) outrank self-report (0.35). Capped at
0.2 when nothing is verifiably grounded.

It is **not calibrated**. No labelled dataset was used to fit these weights, so it must
not be read as "86% likely correct". The UI shows a qualitative band alongside the number
for this reason.

---

## 8. Abstention

Three distinct refusal reasons, because they mean different things operationally:

| Reason | Meaning | Model called? |
|---|---|---|
| `no_relevant_evidence` | Nothing cleared the similarity floor | No — free, and cannot be argued out of |
| `model_declined` | Evidence existed but did not answer | Yes |
| `citations_unverified` | Answer drafted, support failed verification | Yes |

Collapsing them into one "I don't know" would hide which is firing — and a rising
`citations_unverified` rate is the clearest early signal that generation quality is
slipping.

---

## 9. Embeddings: hosted, 384 dimensions

`text-embedding-3-small` with `dimensions=384`. The dimension parameter was the deciding
factor: it matches the existing `vector(384)` column, so switching providers required **no
schema migration and no forced re-embedding**. The returned width is asserted on every
response rather than assumed.

Replacing local `sentence-transformers` removed torch and took the image from ~3.3GB to a
few hundred MB. Documents and queries share one code path — embedding them with different
models puts the vectors in different spaces and degrades retrieval *silently*.

---

## 10. Idempotent ingestion

Identity is `(filename, sha256)`. Unchanged documents are skipped without re-embedding; a
changed document replaces only its own chunks; new documents are added. Adding five PDFs
to an indexed corpus of 26 costs five embed calls, not 31.

Filename alone cannot distinguish "same document" from "edited document", so a
filename-only design must re-embed everything on every run to stay correct. One malformed
document fails in isolation and is reported, rather than aborting the run and leaving the
corpus partially indexed.

---

## 11. Scaling

The architecture is unchanged at larger corpus sizes; the parameters move:

- **HNSW index** on the embedding column once chunks reach tens of thousands.
- **Metadata pre-filtering** in SQL (entity, status) before vector search, once the
  candidate set is large enough that post-filtering wastes work. Indexes on `status`,
  `entity` and `document_type` are already in place.
- **Batch embedding** already implemented (128 per request).
- **Parallel ingestion** per document — safe, since documents are independent and
  identity is content-hashed.
- **Reranking** matters more as `candidate_k` grows; the signal set stays the same.
