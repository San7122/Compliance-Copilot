"""Application-level confidence score.

**This is not a calibrated probability.** It is a bounded, explainable score in [0, 1]
combining the signals the system actually has. It should be read as "how much
corroboration does this answer have", not "there is an 82% chance this is correct".
Presenting it as a probability would imply a calibration exercise that has not been
done — there is no labelled dataset here against which these weights were fitted.

Four signals feed it:

- **Retrieval relevance** — how similar the best matching clause was to the question.
- **Groundedness** — whether citations survived verification against their own chunks.
- **Citation availability** — whether there is any support at all.
- **Model self-assessment** — the model's own high/medium/low judgement, which is
  informative but self-reported, so it is weighted below the measured signals.

The ordering principle throughout: measured evidence outranks the model's opinion of
itself. An answer the model calls "high" that rests on a barely-relevant chunk with no
verifiable citation should not score highly, because over-confidence is the expensive
direction to be wrong in for a compliance tool.
"""

from app.config import settings
from app.retrieval import RetrievedChunk

_MODEL_SELF_ASSESSMENT = {"high": 1.0, "medium": 0.6, "low": 0.3}

# Weights sum to 1.0. Measured signals (relevance + coverage) carry 0.65 between them.
_W_RELEVANCE = 0.50
_W_MODEL = 0.35
_W_COVERAGE = 0.15

# An answer with no verified citation is reported at or below this, whatever else
# looked promising.
_UNGROUNDED_CEILING = 0.2

_LEVELS = ("low", "medium", "high")
_RANK = {level: i for i, level in enumerate(_LEVELS)}


def relevance_component(chunks: list[RetrievedChunk]) -> float:
    """Best similarity, rescaled so the configured thresholds anchor 0 and 1."""
    if not chunks:
        return 0.0

    top = max(c.similarity for c in chunks)
    floor = settings.min_similarity
    ceiling = settings.confidence_high_similarity
    if ceiling <= floor:  # misconfiguration; degrade rather than divide by zero
        return 1.0 if top >= ceiling else 0.0
    return _clamp((top - floor) / (ceiling - floor))


def evidence_ceiling(chunks: list[RetrievedChunk]) -> str:
    """Highest label the retrieved evidence alone can justify.

    Retained as the qualitative view of the same signal — useful for logs and for
    explaining a score without showing arithmetic.
    """
    if not chunks:
        return "low"

    top = max(c.similarity for c in chunks)
    if top >= settings.confidence_high_similarity:
        return "high"
    if top >= settings.confidence_medium_similarity:
        return "medium"
    return "low"


def compute_confidence(
    *,
    model_confidence: str | None,
    chunks: list[RetrievedChunk],
    citations: list[dict],
    grounded: bool,
) -> float:
    """Combine the available signals into a score in [0, 1]."""
    relevance = relevance_component(chunks)

    if not grounded or not citations:
        # No verifiable support: the retrieved text may have looked relevant, but
        # nothing survived checking, so the score is capped low rather than dropped to
        # zero (retrieval strength is still weak evidence about the corpus).
        return round(min(_UNGROUNDED_CEILING, relevance), 2)

    model = _MODEL_SELF_ASSESSMENT.get(model_confidence or "", 0.3)
    # Corroboration saturates quickly: three independent clauses is plenty, and beyond
    # that extra citations tend to be restatements rather than new evidence.
    coverage = min(len(citations), 3) / 3

    score = _W_RELEVANCE * relevance + _W_MODEL * model + _W_COVERAGE * coverage
    return round(_clamp(score), 2)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))
