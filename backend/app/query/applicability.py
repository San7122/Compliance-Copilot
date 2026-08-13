"""Work out *which documents can apply* before working out which are most similar.

Two questions are answered here, and both are about scope rather than relevance:

1. **Which legal entity is being asked about?** Northwind is a group of separate
   companies whose policies deliberately carry different numbers. A question naming
   Capital Markets must not be answered from group policy where Capital Markets has its
   own document, and a group question must never be answered from a subsidiary's.

2. **Is this a question about current obligations, or about history?** Superseded policy
   is not a source of current requirements — but it is the only source for "what did we
   require before?", so deleting it or filtering it unconditionally makes a legitimate
   question unanswerable. Intent decides which corpus is in scope.

Both are deliberately rule-based and inspectable. A model could infer entity and intent,
but that puts a non-deterministic step in front of a correctness-critical filter, and
makes "why did it use that document?" unanswerable. Everything here reports *why* it
decided, so a wrong scope is diagnosable rather than mysterious.
"""

import re
from dataclasses import dataclass, field

GROUP_ENTITY = "Northwind Financial Services Pvt. Ltd."

# Aliases people actually type, mapped to the entity string stored on the document.
_ENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "Northwind Capital Markets Ltd": (
        "northwind capital markets",
        "capital markets",
        "ncm",
    ),
    "Northwind Payments (Singapore) Pte Ltd": (
        "northwind payments",
        "payments singapore",
        "singapore",
        "nps",
    ),
    GROUP_ENTITY: (
        "northwind financial services",
        "group policy",
        "the group",
    ),
}

# Phrasing that signals a question about the past rather than current obligations.
_HISTORICAL_MARKERS = (
    "previously",
    "used to",
    "prior version",
    "previous version",
    "earlier version",
    "old policy",
    "older policy",
    "superseded",
    "before it was changed",
    "before the change",
    "historical",
    "history of",
    "what did it say before",
    "at the time",
)

# Present-tense obligation language. Only consulted to break a tie when a historical
# marker also appears, e.g. "is the superseded policy still current?".
_CURRENT_MARKERS = ("currently", "right now", "today", "at present")


@dataclass
class QueryScope:
    """The resolved scope of a question, with the reasoning that produced it."""

    entity: str | None = None
    entity_source: str = "unspecified"  # explicit | inferred | unspecified
    include_superseded: bool = False
    intent: str = "current"  # current | historical
    matched_terms: list[str] = field(default_factory=list)

    @property
    def is_entity_scoped(self) -> bool:
        return self.entity is not None and self.entity != GROUP_ENTITY

    def explain(self) -> str:
        parts = [f"intent={self.intent}"]
        parts.append(f"entity={self.entity or 'group (default)'} ({self.entity_source})")
        if self.matched_terms:
            parts.append(f"matched={','.join(self.matched_terms)}")
        return "; ".join(parts)


def resolve_entity(question: str, explicit: str | None = None) -> tuple[str | None, str, list[str]]:
    """Return (entity, source, matched_terms).

    An explicit API value always wins: the caller knows which company the user belongs
    to, and that is better information than anything inferable from phrasing.
    """
    if explicit and explicit.strip():
        return _canonical_entity(explicit.strip()), "explicit", []

    haystack = question.lower()
    # Longest alias first, so "northwind capital markets" is preferred over "ncm" and a
    # short alias can't shadow a more specific one.
    for entity, aliases in _ENTITY_ALIASES.items():
        for alias in sorted(aliases, key=len, reverse=True):
            if re.search(rf"\b{re.escape(alias)}\b", haystack):
                return entity, "inferred", [alias]
    return None, "unspecified", []


def _canonical_entity(value: str) -> str:
    """Accept a stored entity string, or any known alias, and return the stored form."""
    lowered = value.lower()
    for entity, aliases in _ENTITY_ALIASES.items():
        if lowered == entity.lower() or lowered in aliases:
            return entity
    for entity, aliases in _ENTITY_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return entity
    # Unrecognised entity: pass it through rather than silently mapping it to the group,
    # so a typo yields "nothing applicable" rather than a confidently group-scoped answer.
    return value


def detect_intent(question: str) -> tuple[str, list[str]]:
    """Return ("current"|"historical", matched markers)."""
    haystack = question.lower()
    matched = [marker for marker in _HISTORICAL_MARKERS if marker in haystack]
    if not matched:
        return "current", []
    # "Is the superseded policy still current?" is a question about now.
    if any(marker in haystack for marker in _CURRENT_MARKERS):
        return "current", matched
    return "historical", matched


def resolve_scope(question: str, entity: str | None = None) -> QueryScope:
    entity_value, source, entity_terms = resolve_entity(question, entity)
    intent, intent_terms = detect_intent(question)

    return QueryScope(
        entity=entity_value,
        entity_source=source,
        intent=intent,
        # Superseded documents become reachable only for historical questions, and are
        # labelled as superseded in the context when they are.
        include_superseded=(intent == "historical"),
        matched_terms=entity_terms + intent_terms,
    )


def applies_to(chunk_entity: str | None, scope: QueryScope) -> bool:
    """Can a chunk from this entity be used to answer within this scope?

    - No entity resolved: everything is in scope; ranking decides.
    - Entity resolved: that entity's own documents, plus group documents, which apply
      except where the entity has its own document on the topic. Ranking (not filtering)
      resolves that overlap, because "has its own document on this topic" is a semantic
      judgement, not something a WHERE clause can express.
    - Other subsidiaries are excluded outright: a Payments (Singapore) figure is simply
      not a fact about Capital Markets, however similar the text.
    """
    if scope.entity is None or chunk_entity is None:
        return True
    if chunk_entity == scope.entity:
        return True
    return chunk_entity == GROUP_ENTITY
