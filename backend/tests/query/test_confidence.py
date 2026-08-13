"""Confidence is an application-level score, and these tests pin its *ordering*.

The exact weights are a judgement call and may be tuned; what must not change is the
direction of each signal. Over-confidence is the expensive failure for a compliance
tool, so the invariants worth locking down are: an answer with no verified citation can
never score well, weaker retrieval never scores above stronger retrieval, and the
model's own opinion never outweighs the measured evidence on its own.
"""

import pytest

from app.confidence import compute_confidence, evidence_ceiling, relevance_component
from app.config import settings
from app.retrieval import RetrievedChunk

HIGH = settings.confidence_high_similarity
MEDIUM = settings.confidence_medium_similarity
FLOOR = settings.min_similarity

CITATION = {"chunk_id": 1, "excerpt": "..."}


def chunk(similarity: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        document="Data Protection and Privacy Policy",
        section="... > Clause 7.1",
        content="text",
        similarity=similarity,
    )


def score(similarity=HIGH + 0.1, model="high", citations=1, grounded=True) -> float:
    return compute_confidence(
        model_confidence=model,
        chunks=[chunk(similarity)],
        citations=[CITATION] * citations,
        grounded=grounded,
    )


# --- relevance component --------------------------------------------------------


def test_relevance_is_zero_at_the_floor_and_one_at_the_high_threshold():
    assert relevance_component([chunk(FLOOR)]) == pytest.approx(0.0)
    assert relevance_component([chunk(HIGH)]) == pytest.approx(1.0)


def test_relevance_is_clamped_outside_the_thresholds():
    assert relevance_component([chunk(FLOOR - 0.2)]) == 0.0
    assert relevance_component([chunk(1.0)]) == 1.0


def test_relevance_uses_the_best_chunk():
    assert relevance_component([chunk(FLOOR), chunk(HIGH)]) == pytest.approx(1.0)


def test_no_chunks_has_no_relevance():
    assert relevance_component([]) == 0.0


# --- qualitative ceiling (retained for logs/explanation) ------------------------


def test_evidence_ceiling_tracks_the_thresholds():
    assert evidence_ceiling([chunk(HIGH + 0.1)]) == "high"
    assert evidence_ceiling([chunk(MEDIUM + 0.01)]) == "medium"
    assert evidence_ceiling([chunk(MEDIUM - 0.05)]) == "low"
    assert evidence_ceiling([]) == "low"


# --- compute_confidence ---------------------------------------------------------


def test_score_is_always_within_bounds():
    for similarity in (0.0, FLOOR, MEDIUM, HIGH, 1.0):
        for grounded in (True, False):
            value = score(similarity=similarity, grounded=grounded, citations=1 if grounded else 0)
            assert 0.0 <= value <= 1.0


def test_ungrounded_answer_is_capped_low_even_with_perfect_retrieval():
    """The whole point: strong-looking retrieval must not rescue unverifiable support."""
    assert score(similarity=1.0, model="high", citations=0, grounded=False) <= 0.2


def test_grounded_answer_with_strong_evidence_scores_high():
    assert score(similarity=HIGH + 0.1, model="high", citations=3) >= 0.8


def test_weaker_retrieval_scores_lower_than_stronger_retrieval():
    assert score(similarity=MEDIUM) < score(similarity=HIGH + 0.1)


def test_model_self_doubt_lowers_the_score():
    assert score(model="low") < score(model="high")


def test_model_confidence_alone_cannot_produce_a_high_score():
    """A confident model on barely-relevant evidence must not clear a high bar."""
    assert score(similarity=FLOOR, model="high", citations=1) < 0.7


def test_more_corroboration_helps_but_saturates():
    one, three, five = score(citations=1), score(citations=3), score(citations=5)

    assert one < three
    assert three == five  # coverage saturates at three independent clauses


def test_unknown_model_label_is_treated_conservatively():
    assert score(model="extremely-high") < score(model="high")


def test_score_is_rounded_for_display():
    value = score()
    assert value == round(value, 2)
