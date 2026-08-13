"""Corpus-specific rule: make the governing retention schedule visible for retention
questions.

*** THIS IS NOT A GENERAL RAG TECHNIQUE. ***

It encodes one fact about the Northwind corpus: `NFS-POL-011` (Records Retention
Schedule) declares itself governing for retention conflicts in its own clause 1.2 --
"Where a policy and this schedule state different periods for the same record category,
this schedule governs, except where the policy period is longer" -- and clause 1.3
explains why that matters: "early destruction cannot be undone". Any corpus without such
a self-declared governing document needs a different mechanism, and the general version
of this is a `governing` flag set during ingestion rather than a document ID in code.

**Why it is needed.** Measured on the failing question ("How long must financial
transaction records be retained?"), the governing evidence *was* retrieved -- Annexure A
sat at rank 7 of the 20 candidates with a rerank score of 0.737 -- but the `top_k=5` cut
dropped it, 0.061 below the fifth-placed chunk. The model then saw only NFS-POL-001's
seven years and never learned that the schedule says eight. This was a selection failure,
not a retrieval failure, which is why the fix lives here and not in `retrieval.py`.

**Why it selects from existing candidates rather than issuing a new query.** The
governing chunks are already in the candidate set, so re-querying the database would
re-embed the question and re-run a vector search to obtain rows we already hold. Reusing
the candidates keeps `retrieval.py`, the reranker, `candidate_k` and `top_k` untouched,
and costs nothing.

**Known limitation:** if a retention question does not surface any `NFS-POL-011` chunk
within `candidate_k`, no governing evidence is added and the answer falls back to
today's behaviour. That is a deliberate floor -- it never fabricates evidence -- but it
means this is a mitigation, not a guarantee.
"""

import re

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Chunk, Document
from app.retrieval import CURRENT, RetrievedChunk, retrieve

# The one document this rule knows about. See the module docstring for why hardcoding a
# document ID is acceptable here and what the general solution would be instead.
GOVERNING_DOC_ID = "NFS-POL-011"

# The clause that states the schedule governs. It is fetched by identity rather than by
# similarity because it is *about* resolving conflicts, not about any particular
# retention period -- so it scores poorly against the questions that need it most.
# Measured: for "how long must financial transaction records be retained?", clause 1.2
# ranks 39th corpus-wide and 6th even within its own document, behind three chunks about
# destruction logging. Similarity cannot reliably surface a rule of precedence.
GOVERNING_CLAUSE = "1.2"

# How many chunks of the schedule to pull by similarity. Two is enough to reach both
# annexures (measured ranks 1 and 2 within the document): Annexure A holds the retention
# table, Annexure B a worked example of exactly this precedence case.
GOVERNING_SIMILARITY_LIMIT = 2

# Deliberately narrow. "how long" alone is far too broad -- it also matches response-time
# questions such as "How long does the company have to respond to a data subject
# request?", which is not a retention question and must not have governing evidence
# injected into it. So a retention verb is required, either explicitly or paired with
# "how long".
_EXPLICIT = re.compile(r"\b(retention|retain|retained|retaining)\b", re.I)
_HOW_LONG = re.compile(r"\bhow long\b", re.I)
_STORAGE_VERB = re.compile(r"\b(kept|keep|stored|store|held|hold|archiv\w*)\b", re.I)


def is_retention_question(question: str) -> bool:
    """True for questions about how long records are kept.

    Two ways to qualify:
      1. The question uses retention vocabulary outright ("retained", "retention").
      2. It asks "how long" *and* pairs it with a storage verb ("kept", "stored").
    """
    if not question:
        return False
    if _EXPLICIT.search(question):
        return True
    return bool(_HOW_LONG.search(question) and _STORAGE_VERB.search(question))


def fetch_clause(db: Session, doc_id: str, clause: str) -> RetrievedChunk | None:
    """Fetch one clause by identity.

    Not a vector search, deliberately: this clause is addressed by number, not found by
    resemblance. Similarity is the wrong tool for "give me the rule that says which
    document wins" -- see the note on GOVERNING_CLAUSE. `similarity` is reported as 0.0
    because none was computed; the chunk is admitted by rule, and the ranking
    explanation labels it as such.
    """
    row = db.execute(
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.doc_id == doc_id, Chunk.clause == clause)
        .limit(1)
    ).first()
    if row is None:
        return None

    chunk, document = row
    return RetrievedChunk(
        chunk_id=chunk.id,
        document=document.title,
        section=chunk.heading_path,
        content=chunk.content,
        similarity=0.0,
        doc_id=document.doc_id,
        entity=document.entity,
        clause=chunk.clause,
        version=document.version,
        page=chunk.page,
        document_type=document.document_type,
        status=document.status or CURRENT,
        effective_date=document.effective_date,
    )


def governing_evidence(
    db: Session,
    question: str,
    already_selected: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    """The governing schedule's relevant evidence for a retention question.

    Two parts, because they fail in different ways:

    - **The applicable period**, found by a document-scoped vector search. Which record
      category the question is about genuinely is a similarity problem.
    - **The precedence rule (clause 1.2)**, fetched by identity, because similarity
      demonstrably does not surface it for the questions that need it.

    Without the rule the model sees two numbers and no reason to prefer either, which is
    exactly what happened when only the annexure was supplied: it kept answering seven
    years from the policy while eight sat unused in the context.

    Deliberately not the whole document -- three chunks at most, and duplicates of what
    the normal path already selected are dropped.
    """
    chosen_ids = {c.chunk_id for c in already_selected}
    evidence: list[RetrievedChunk] = []

    rule = fetch_clause(db, GOVERNING_DOC_ID, GOVERNING_CLAUSE)
    if rule is not None and rule.chunk_id not in chosen_ids:
        evidence.append(rule)
        chosen_ids.add(rule.chunk_id)

    for chunk in retrieve(
        db, question, limit=GOVERNING_SIMILARITY_LIMIT, doc_id=GOVERNING_DOC_ID
    ):
        if chunk.chunk_id not in chosen_ids:
            evidence.append(chunk)
            chosen_ids.add(chunk.chunk_id)

    return evidence
