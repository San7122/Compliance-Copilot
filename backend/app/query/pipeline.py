"""Online query pipeline.

    question -> validate -> embed -> retrieve -> relevance/grounding
             -> LLM -> structured output -> citation mapping -> response

Orchestration only. Retrieval, generation, grounding and citation mapping each live in
their own module; this file makes the order visible and owns the decisions *between*
stages — chiefly when to refuse. It contains no PDF parsing, no chunking and no
ingestion: those belong to the offline pipeline and are never reachable from a request.

There are three distinct ways to refuse, and keeping them separate matters because they
mean different things operationally:

1. `no_relevant_evidence` — nothing cleared the similarity floor. The model is never
   called, so this refusal is free and cannot be argued out of.
2. `model_declined` — evidence existed, but the model judged it insufficient.
3. `citations_unverified` — the model answered, but nothing it quoted survived checking
   against the chunk it cited. A confident answer whose support cannot be verified is
   worse than no answer in a compliance setting, so it is downgraded to a refusal
   rather than shown with its citations quietly removed.
"""

import logging
import time
from dataclasses import dataclass, field

from pydantic import ValidationError
from sqlalchemy.orm import Session

from app.citations.mapper import map_citations
from app.confidence import compute_confidence
from app.llm import generate_answer
from app.query.applicability import QueryScope, applies_to, resolve_scope
from app.query.governing import governing_evidence, is_retention_question
from app.query.reranker import ScoredChunk, rerank
from app.retrieval import RetrievedChunk, filter_by_relevance, retrieve
from app.schemas import AnswerResponse, ModelAnswer, RetrievalMetadata

logger = logging.getLogger(__name__)

NO_EVIDENCE_ANSWER = "I don't know based on the provided documents."
UNVERIFIED_ANSWER = (
    "The documents do not clearly support an answer to this question. An answer was "
    "drafted, but its supporting quotes could not be verified against the source "
    "documents, so it was withheld."
)


class QuestionInvalid(ValueError):
    """The question itself is unusable (empty/oversized)."""


class RetrievalUnavailable(RuntimeError):
    """Search could not run — an availability problem, not a bad request."""


class MalformedAnswer(RuntimeError):
    """The model returned something off-schema; fail loudly rather than interpret it."""


@dataclass
class QueryResult:
    """The response plus what the caller needs for logging."""

    response: AnswerResponse
    retrieved_chunk_ids: list[int] = field(default_factory=list)
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    rejected_citations: list[dict] = field(default_factory=list)
    scope: QueryScope | None = None
    ranked: list[ScoredChunk] = field(default_factory=list)


class QueryPipeline:
    def __init__(self, db: Session, max_question_chars: int = 2000):
        self.db = db
        self.max_question_chars = max_question_chars

    # --- stages ---------------------------------------------------------------

    def validate_question(self, question: str) -> str:
        cleaned = (question or "").strip()
        if not cleaned:
            raise QuestionInvalid("question must not be empty")
        if len(cleaned) > self.max_question_chars:
            raise QuestionInvalid(
                f"question must be {self.max_question_chars} characters or fewer"
            )
        return cleaned

    def retrieve_evidence(
        self, question: str, scope: QueryScope
    ) -> tuple[list[RetrievedChunk], list[ScoredChunk]]:
        """Candidates -> relevance floor -> applicability -> rerank -> evidence.

        Returns (candidates, ranked evidence). Filtering happens on the oversampled
        candidate set, so dropping an inapplicable chunk promotes the next applicable
        one rather than shrinking the evidence.
        """
        try:
            candidates = retrieve(
                self.db, question, include_superseded=scope.include_superseded
            )
        except Exception as exc:
            logger.exception("retrieval failed for question=%r", question)
            raise RetrievalUnavailable(str(exc)) from exc

        relevant = filter_by_relevance(candidates)
        applicable = [c for c in relevant if applies_to(c.entity, scope)]
        ranked = rerank(applicable, scope)

        # Corpus-specific: retention questions additionally get the governing schedule
        # (NFS-POL-011), which declares itself governing for retention conflicts but can
        # fall below the top_k cut. Non-retention questions take exactly the path above
        # -- `is_retention_question` short-circuits and nothing is appended.
        # See app/query/governing.py for why this is not a general technique.
        if is_retention_question(question):
            selected = [s.chunk for s in ranked]
            try:
                extra = governing_evidence(self.db, question, selected)
            except Exception:
                # Supplementary evidence must never take down an answer that the normal
                # path has already produced. Log and continue with what we have.
                logger.exception("governing-document lookup failed for question=%r", question)
                extra = []
            for chunk in extra:
                ranked.append(
                    ScoredChunk(
                        chunk=chunk,
                        score=0.0,
                        # Zero-scored and named, so the ranking explanation shows this
                        # chunk was admitted by rule rather than earned on similarity.
                        signals={"governing_document": 0.0},
                    )
                )

        return candidates, ranked

    # --- orchestration --------------------------------------------------------

    def run(self, question: str, entity: str | None = None) -> QueryResult:
        started = time.time()
        question = self.validate_question(question)

        scope = resolve_scope(question, entity)
        candidates, ranked = self.retrieve_evidence(question, scope)
        relevant = [s.chunk for s in ranked]

        logger.info(
            "query scope: %s | candidates=%d applicable=%d", scope.explain(), len(candidates), len(ranked)
        )

        if not relevant:
            # Refuse without calling the model: it has nothing to ground an answer in,
            # and asking anyway is both a cost and an invitation to improvise.
            return self._refusal(
                NO_EVIDENCE_ANSWER, "no_relevant_evidence", scope, candidates, ranked, started
            )

        # The resolved scope goes to the model too: filtering and ranking already use it,
        # but the model cannot infer a selector choice from the question text.
        generation = generate_answer(question, relevant, scope=scope)

        # Validate before interpreting. Reading these fields defensively (bool(...),
        # .get(...)) would quietly accept off-schema values -- `bool("not-a-bool")` is
        # True -- turning a malformed response into a confident-looking answer.
        try:
            content = ModelAnswer(**generation.content)
        except ValidationError as exc:
            logger.exception("model returned an off-schema response: %r", generation.content)
            raise MalformedAnswer(str(exc)) from exc

        citations, rejected = map_citations(
            [c.model_dump() for c in content.citations], relevant
        )
        if rejected:
            logger.warning(
                "dropped %d unverifiable citation(s) for question=%r", len(rejected), question
            )

        if not content.answerable:
            return self._refusal(
                content.answer or NO_EVIDENCE_ANSWER,
                "model_declined",
                scope,
                candidates,
                ranked,
                started,
                generation=generation,
            )

        if not citations:
            return self._refusal(
                UNVERIFIED_ANSWER,
                "citations_unverified",
                scope,
                candidates,
                ranked,
                started,
                generation=generation,
                rejected=rejected,
            )

        confidence = compute_confidence(
            model_confidence=content.confidence,
            chunks=relevant,
            citations=citations,
            grounded=True,
        )

        return QueryResult(
            response=AnswerResponse(
                answer=content.answer,
                citations=citations,
                confidence=confidence,
                grounded=True,
                refusal_reason=None,
                retrieval=_retrieval_metadata(scope, candidates, ranked),
            ),
            retrieved_chunk_ids=[c.chunk_id for c in relevant],
            latency_ms=_elapsed_ms(started),
            input_tokens=generation.input_tokens,
            output_tokens=generation.output_tokens,
            cost_usd=generation.cost_usd,
            rejected_citations=rejected,
            scope=scope,
            ranked=ranked,
        )

    def _refusal(
        self,
        answer: str,
        reason: str,
        scope: QueryScope,
        candidates: list[RetrievedChunk],
        ranked: list[ScoredChunk],
        started: float,
        generation=None,
        rejected: list[dict] | None = None,
    ) -> QueryResult:
        relevant = [s.chunk for s in ranked]
        confidence = compute_confidence(
            model_confidence="low", chunks=relevant, citations=[], grounded=False
        )
        return QueryResult(
            response=AnswerResponse(
                answer=answer,
                citations=[],
                confidence=confidence,
                grounded=False,
                refusal_reason=reason,
                retrieval=_retrieval_metadata(scope, candidates, ranked),
            ),
            retrieved_chunk_ids=[c.chunk_id for c in relevant],
            latency_ms=_elapsed_ms(started),
            input_tokens=generation.input_tokens if generation else 0,
            output_tokens=generation.output_tokens if generation else 0,
            cost_usd=generation.cost_usd if generation else 0.0,
            rejected_citations=rejected or [],
            scope=scope,
            ranked=ranked,
        )



def _retrieval_metadata(scope, candidates, ranked) -> RetrievalMetadata:
    """What the retrieval stage considered, so an answer can be audited without logs."""
    return RetrievalMetadata(
        entity_scope=scope.entity,
        entity_source=scope.entity_source,
        intent=scope.intent,
        superseded_included=scope.include_superseded,
        candidates_considered=len(candidates),
        evidence_used=len(ranked),
        ranking=[s.explain() for s in ranked],
    )


def _elapsed_ms(started: float) -> int:
    return int((time.time() - started) * 1000)
