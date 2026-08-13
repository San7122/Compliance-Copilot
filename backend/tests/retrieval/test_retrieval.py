"""The relevance floor is refusal layer 1 — the deterministic half.

If it lets noise through, the LLM gets handed irrelevant context and is invited to
improvise. If it's too aggressive, real answers get refused. These tests pin the
boundary behaviour rather than the exact threshold value, so tuning
`MIN_SIMILARITY` in config doesn't break the suite.
"""

from app.config import settings
from app.retrieval import RetrievedChunk, filter_by_relevance


def chunk(similarity: float, content: str = "text") -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=1,
        document="Data Retention Policy",
        section="Data Retention Policy > 2. Retention Periods",
        content=content,
        similarity=similarity,
    )


def test_keeps_chunks_above_the_floor():
    floor = settings.min_similarity
    chunks = [chunk(floor + 0.2), chunk(floor + 0.1)]
    assert len(filter_by_relevance(chunks)) == 2


def test_drops_chunks_below_the_floor():
    floor = settings.min_similarity
    assert filter_by_relevance([chunk(floor - 0.01)]) == []


def test_floor_is_inclusive():
    """A chunk sitting exactly on the threshold is kept, not dropped."""
    assert len(filter_by_relevance([chunk(settings.min_similarity)])) == 1


def test_mixed_relevance_keeps_only_the_good_ones():
    floor = settings.min_similarity
    chunks = [chunk(floor + 0.3, "relevant"), chunk(floor - 0.2, "noise")]

    survivors = filter_by_relevance(chunks)

    assert [c.content for c in survivors] == ["relevant"]


def test_all_irrelevant_returns_empty_so_the_llm_is_skipped():
    """The empty list is what tells /query to refuse without calling the model."""
    floor = settings.min_similarity
    assert filter_by_relevance([chunk(floor - 0.2), chunk(floor - 0.3)]) == []


def test_empty_input_is_not_an_error():
    assert filter_by_relevance([]) == []
