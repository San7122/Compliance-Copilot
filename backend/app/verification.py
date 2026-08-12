"""Runtime check that every citation the model returns is actually grounded.

The eval script verifies citations against the docs on disk after the fact, but that
does nothing for a real user: a hallucinated excerpt still reaches them. This module
runs the same check inline, against the exact chunks the model was given, and drops
any citation whose excerpt doesn't appear in them.

Verifying against the retrieved chunks (rather than the full document) is the stricter
choice on purpose -- the model must quote from what it was actually shown, not from
something it happens to remember about the corpus.
"""

import difflib
import re

from app.config import settings
from app.retrieval import RetrievedChunk

UNVERIFIED_REFUSAL = {
    "answer": (
        "The documents do not clearly support an answer to this question. "
        "(An answer was drafted, but its supporting quotes could not be verified "
        "against the source documents, so it was withheld.)"
    ),
    "citations": [],
    "confidence": "low",
    "answerable": False,
}


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def excerpt_is_grounded(excerpt: str, source: str, threshold: float) -> bool:
    """Does something close to `excerpt` actually appear in `source`?

    Fuzzy rather than exact: the model may normalize whitespace, trim a trailing
    clause, or fix a stray character when quoting. A sliding window with
    SequenceMatcher tolerates that without tolerating an invented quote.
    """
    norm_excerpt = _normalize(excerpt)
    if not norm_excerpt:
        return False
    if norm_excerpt in source:
        return True

    window = len(norm_excerpt)
    step = max(1, window // 4)
    for start in range(0, max(1, len(source) - window), step):
        candidate = source[start : start + window + step]
        if difflib.SequenceMatcher(None, norm_excerpt, candidate).ratio() >= threshold:
            return True
    return False


def enforce_citation_grounding(result: dict, chunks: list[RetrievedChunk]) -> tuple[dict, list[dict]]:
    """Drop citations that aren't grounded in the retrieved chunks.

    Returns (result, rejected). If the model claimed the question was answerable but
    none of its citations survive, we downgrade to the refusal shape: a confident-
    sounding answer with no verifiable support is exactly the failure mode this tool
    exists to prevent, and in a compliance context it is worse than saying nothing.
    """
    citations = result.get("citations") or []
    if not citations:
        return result, []

    # The model saw these chunks concatenated; verify against the same text.
    source = _normalize("\n".join(c.content for c in chunks))

    verified = []
    rejected = []
    for citation in citations:
        excerpt = citation.get("excerpt", "")
        if excerpt_is_grounded(excerpt, source, settings.citation_min_match):
            verified.append(citation)
        else:
            rejected.append(citation)

    if not rejected:
        return result, []

    if result.get("answerable") and not verified:
        return dict(UNVERIFIED_REFUSAL), rejected

    result = dict(result)
    result["citations"] = verified
    # Some support was fabricated, so the rest earns less trust than the model claimed.
    if result.get("confidence") == "high":
        result["confidence"] = "medium"
    return result, rejected
