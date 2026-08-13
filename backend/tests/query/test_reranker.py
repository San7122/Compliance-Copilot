"""Reranking decides which evidence the model actually sees.

The corpus is written so that the group and subsidiary versions of a policy score almost
identically on cosine similarity — the deciding information is metadata. These tests pin
the *ordering of concerns*: applicability first, status second, document authority last
and only as a tiebreaker.

The specific thing being guarded against is a universal `POLICY > SOP > HANDBOOK` rule.
That would be simple, defensible-sounding, and wrong: plenty of correct answers in this
corpus live in a procedure or a regulatory calendar.
"""

from app.ingestion.extractor import (
    GUIDANCE,
    POLICY,
    PROCEDURE,
    REGULATORY,
    SUPERSEDED,
    ENTITY_POLICY,
)
from app.query.applicability import GROUP_ENTITY, resolve_scope
from app.query.reranker import rerank, score_chunk
from app.retrieval import RetrievedChunk

CAPITAL_MARKETS = "Northwind Capital Markets Ltd"


def chunk(
    chunk_id=1,
    similarity=0.6,
    entity=GROUP_ENTITY,
    document_type=POLICY,
    status="current",
    clause="5.1",
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Know Your Customer and Anti-Money-Laundering Policy",
        section="... > Clause 5.1",
        content="KYC records are retained for a minimum of five (5) years.",
        similarity=similarity,
        doc_id="NFS-POL-003",
        entity=entity,
        clause=clause,
        document_type=document_type,
        status=status,
    )


def ordered_ids(chunks, scope):
    return [s.chunk.chunk_id for s in rerank(chunks, scope, limit=10)]


# --- entity applicability dominates -------------------------------------------------


def test_entity_specific_document_outranks_group_even_with_lower_similarity():
    """The whole point: NCM staff follow their own document, not the group policy."""
    scope = resolve_scope("What is the KYC retention period for Northwind Capital Markets?")
    group = chunk(chunk_id=1, similarity=0.75, entity=GROUP_ENTITY)
    entity_doc = chunk(
        chunk_id=2, similarity=0.60, entity=CAPITAL_MARKETS, document_type=ENTITY_POLICY
    )

    assert ordered_ids([group, entity_doc], scope)[0] == 2


def test_group_document_still_ranks_when_entity_scoped():
    """Group policy applies wherever the subsidiary has nothing of its own."""
    scope = resolve_scope("Northwind Capital Markets encryption requirement?")
    group = chunk(chunk_id=1, similarity=0.7, entity=GROUP_ENTITY)

    assert ordered_ids([group], scope) == [1]


def test_no_entity_scope_means_no_entity_bonus():
    scope = resolve_scope("How long are KYC records kept?")
    scored = score_chunk(chunk(entity=CAPITAL_MARKETS), scope)

    assert "entity_specific" not in scored.signals


# --- status -------------------------------------------------------------------------


def test_superseded_is_pushed_below_current_for_current_questions():
    scope = resolve_scope("How quickly must a breach be reported?")
    current = chunk(chunk_id=1, similarity=0.60)
    superseded = chunk(chunk_id=2, similarity=0.95, status=SUPERSEDED)

    # Higher similarity must not rescue withdrawn policy.
    assert ordered_ids([current, superseded], scope)[0] == 1


def test_superseded_is_promoted_for_historical_questions():
    """The same signal, read differently by intent."""
    scope = resolve_scope("What did the breach policy previously require?")
    current = chunk(chunk_id=1, similarity=0.70)
    superseded = chunk(chunk_id=2, similarity=0.70, status=SUPERSEDED)

    assert ordered_ids([current, superseded], scope)[0] == 2


def test_superseded_penalty_is_recorded_as_a_named_signal():
    scope = resolve_scope("current requirement?")
    scored = score_chunk(chunk(status=SUPERSEDED), scope)

    assert "superseded_penalty" in scored.signals
    assert scored.signals["superseded_penalty"] < 0


# --- document authority is a tiebreaker, not a hierarchy -----------------------------


def test_guidance_is_demoted_against_equally_similar_policy():
    scope = resolve_scope("What is the encryption requirement?")
    policy = chunk(chunk_id=1, similarity=0.70, document_type=POLICY)
    handbook = chunk(chunk_id=2, similarity=0.70, document_type=GUIDANCE)

    assert ordered_ids([policy, handbook], scope)[0] == 1


def test_a_clearly_more_relevant_procedure_still_beats_a_marginal_policy():
    """Guards against a universal POLICY > SOP ranking."""
    scope = resolve_scope("What are the steps for handling a data subject request?")
    marginal_policy = chunk(chunk_id=1, similarity=0.40, document_type=POLICY)
    relevant_sop = chunk(chunk_id=2, similarity=0.85, document_type=PROCEDURE)

    assert ordered_ids([marginal_policy, relevant_sop], scope)[0] == 2


def test_a_relevant_regulatory_document_is_not_demoted_below_policy():
    scope = resolve_scope("When does the change freeze start?")
    policy = chunk(chunk_id=1, similarity=0.45, document_type=POLICY)
    calendar = chunk(chunk_id=2, similarity=0.80, document_type=REGULATORY)

    assert ordered_ids([policy, calendar], scope)[0] == 2


def test_unknown_document_type_is_neutral_not_promoted():
    scope = resolve_scope("anything")
    scored = score_chunk(chunk(document_type=None), scope)

    assert "authority" not in scored.signals


# --- specificity and explainability ---------------------------------------------------


def test_numbered_clause_gets_a_small_specificity_bonus():
    scope = resolve_scope("anything")

    with_clause = score_chunk(chunk(chunk_id=1, clause="5.1"), scope)
    without = score_chunk(chunk(chunk_id=2, clause=None), scope)

    assert with_clause.score > without.score


def test_every_contribution_is_named_and_sums_to_the_score():
    scope = resolve_scope("Northwind Capital Markets KYC retention?")
    scored = score_chunk(
        chunk(entity=CAPITAL_MARKETS, document_type=ENTITY_POLICY), scope
    )

    assert abs(sum(scored.signals.values()) - scored.score) < 1e-9
    assert "similarity" in scored.signals


def test_explanation_is_human_readable():
    scope = resolve_scope("anything")
    text = score_chunk(chunk(), scope).explain()

    assert "NFS-POL-003" in text
    assert "similarity" in text


def test_rerank_limits_to_the_evidence_budget():
    scope = resolve_scope("anything")
    chunks = [chunk(chunk_id=i, similarity=0.5 + i / 100) for i in range(10)]

    assert len(rerank(chunks, scope, limit=3)) == 3
