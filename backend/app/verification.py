"""Grounding check: does a quoted excerpt actually appear in the text it claims to?

This is the primitive the citation mapper builds on. It answers one narrow question --
"is this quote really in that chunk?" -- and deliberately knows nothing about documents,
sections or pages, because those now come from stored records rather than from anything
the model said.

Matching is fuzzy rather than exact. A model quoting a clause will often normalise
whitespace, drop a trailing sub-clause, or fix a stray character; treating those as
fabrication would reject honest citations. A sliding window with `SequenceMatcher`
tolerates that much drift without tolerating an invented quote.
"""

import difflib
import re


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def excerpt_is_grounded(excerpt: str, source: str, threshold: float) -> bool:
    """Does something close to `excerpt` appear in `source`?

    `source` is expected to be already normalised (see `normalize`).
    """
    norm_excerpt = normalize(excerpt)
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
