"""Rerank candidate chunks using signals a human can audit.

Vector similarity alone cannot answer "which of these two near-identical clauses
actually binds this reader?" — in this corpus the group and subsidiary versions of a
policy are written to look the same apart from the numbers, so they score almost
identically. The deciding information is metadata, not text.

**Ordering of concerns, deliberately:**

1. **Applicability first.** A document that applies to the reader's entity outranks one
   that does not, before any consideration of what kind of document it is.
2. **Status next.** Superseded text is heavily penalised for current questions, and
   *promoted* for historical ones — the same signal, read differently by intent.
3. **Authority last, and gently.** Guidance is a summary that "paraphrases the real
   policies loosely" by its own admission, so it is nudged below authoritative policy.
   This is a tiebreaker with a small weight, **not** a universal
   `POLICY > SOP > HANDBOOK` order: a genuinely more relevant, applicable SOP clause
   still beats a marginally relevant policy clause. Hardcoding a global ranking would
   break the many questions whose true answer lives in a procedure or a calendar.

No ML reranker. A cross-encoder would add a model, a dependency and latency to reorder
~20 short clauses, and would make "why was this chosen?" unanswerable — the opposite of
what a compliance tool needs. A weighted sum of named signals is inspectable, testable,
and tunable without retraining.
"""

from dataclasses import dataclass, field

from app.config import settings
from app.ingestion.extractor import ENTITY_POLICY, GUIDANCE, POLICY, SUPERSEDED
from app.query.applicability import GROUP_ENTITY, QueryScope
from app.retrieval import RetrievedChunk


@dataclass
class ScoredChunk:
    chunk: RetrievedChunk
    score: float
    signals: dict[str, float] = field(default_factory=dict)

    def explain(self) -> str:
        contributions = ", ".join(f"{name}{value:+.3f}" for name, value in self.signals.items())
        source = self.chunk.doc_id or self.chunk.document
        return f"{source} clause {self.chunk.clause or '-'} => {self.score:.3f} ({contributions})"


def _authority_adjustment(document_type: str | None) -> float:
    """Small nudge by document kind. Guidance is demoted; nothing is promoted above
    applicability, and unknown types are left neutral rather than guessed at."""
    if document_type == GUIDANCE:
        return -settings.rerank_guidance_penalty
    if document_type in (POLICY, ENTITY_POLICY):
        return settings.rerank_policy_bonus
    return 0.0


def score_chunk(chunk: RetrievedChunk, scope: QueryScope) -> ScoredChunk:
    signals: dict[str, float] = {"similarity": chunk.similarity}

    # --- entity applicability (dominant signal) --------------------------------
    if scope.entity is not None:
        if chunk.entity == scope.entity and scope.is_entity_scoped:
            # The reader's own entity document; the corpus states staff follow this
            # rather than the group policy of the same name.
            signals["entity_specific"] = settings.rerank_entity_match_bonus
        elif chunk.entity == GROUP_ENTITY:
            # Group policy still applies wherever the entity has nothing of its own,
            # so it stays in play — just below an entity-specific match.
            signals["group_fallback"] = settings.rerank_group_fallback_bonus

    # --- status -----------------------------------------------------------------
    is_superseded = chunk.status == SUPERSEDED
    if is_superseded:
        if scope.intent == "historical":
            signals["historical_match"] = settings.rerank_historical_bonus
        else:
            signals["superseded_penalty"] = -settings.rerank_superseded_penalty
    elif scope.intent == "historical":
        # Current text is still useful context for a historical question (it is what the
        # old requirement was replaced *by*), but it is not what was asked for.
        signals["current_when_historical"] = -settings.rerank_historical_bonus / 2

    # --- authority ---------------------------------------------------------------
    authority = _authority_adjustment(chunk.document_type)
    if authority:
        signals["authority"] = authority

    # --- specificity -------------------------------------------------------------
    if chunk.clause:
        # A numbered clause is a single citable obligation; unnumbered blocks
        # (definitions, annexures) are usually context rather than the requirement.
        signals["clause_specificity"] = settings.rerank_clause_bonus

    return ScoredChunk(chunk=chunk, score=sum(signals.values()), signals=signals)


def rerank(chunks: list[RetrievedChunk], scope: QueryScope, limit: int | None = None) -> list[ScoredChunk]:
    """Score, sort, and cut to the evidence actually sent to the model."""
    scored = sorted(
        (score_chunk(chunk, scope) for chunk in chunks),
        key=lambda s: s.score,
        reverse=True,
    )
    return scored[: (limit or settings.top_k)]
