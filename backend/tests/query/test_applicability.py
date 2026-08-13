"""Scope resolution decides which documents may be used at all.

Getting this wrong produces the two most dangerous answers this corpus can generate: a
subsidiary's figure presented as a group requirement, and superseded text presented as a
current obligation. Both are fluent, well-cited and wrong.
"""

import pytest

from app.query.applicability import (
    GROUP_ENTITY,
    applies_to,
    detect_intent,
    resolve_entity,
    resolve_scope,
)

CAPITAL_MARKETS = "Northwind Capital Markets Ltd"
PAYMENTS = "Northwind Payments (Singapore) Pte Ltd"


# --- entity resolution ------------------------------------------------------------


@pytest.mark.parametrize(
    "question,expected",
    [
        ("What are the KYC rules for Northwind Capital Markets?", CAPITAL_MARKETS),
        ("capital markets retention period?", CAPITAL_MARKETS),
        ("What does NCM require?", CAPITAL_MARKETS),
        ("Northwind Payments retention?", PAYMENTS),
        ("What applies in Singapore?", PAYMENTS),
        ("What is the group policy on retention?", GROUP_ENTITY),
    ],
)
def test_entity_is_inferred_from_the_question(question, expected):
    entity, source, _ = resolve_entity(question)

    assert entity == expected
    assert source == "inferred"


def test_no_entity_mentioned_leaves_scope_open():
    entity, source, _ = resolve_entity("How long are KYC records kept?")

    assert entity is None
    assert source == "unspecified"


def test_explicit_entity_overrides_the_question_text():
    """The caller knows which company the user belongs to; phrasing is weaker evidence."""
    entity, source, _ = resolve_entity("What is the group policy?", explicit="NCM")

    assert entity == CAPITAL_MARKETS
    assert source == "explicit"


def test_explicit_entity_accepts_the_stored_form():
    entity, _, _ = resolve_entity("anything", explicit=CAPITAL_MARKETS)

    assert entity == CAPITAL_MARKETS


def test_unrecognised_entity_is_not_silently_treated_as_the_group():
    """A typo should yield 'nothing applicable', not a confident group-scoped answer."""
    entity, _, _ = resolve_entity("anything", explicit="Northwind Atlantis Ltd")

    assert entity == "Northwind Atlantis Ltd"
    assert entity != GROUP_ENTITY


# --- applicability ----------------------------------------------------------------


def test_entity_documents_and_group_documents_are_both_applicable():
    scope = resolve_scope("What does Northwind Capital Markets require?")

    assert applies_to(CAPITAL_MARKETS, scope) is True
    assert applies_to(GROUP_ENTITY, scope) is True


def test_a_different_subsidiary_is_never_applicable():
    """A Payments (Singapore) figure is simply not a fact about Capital Markets."""
    scope = resolve_scope("What does Northwind Capital Markets require?")

    assert applies_to(PAYMENTS, scope) is False


def test_unscoped_question_admits_everything():
    scope = resolve_scope("How long are KYC records kept?")

    for entity in (GROUP_ENTITY, CAPITAL_MARKETS, PAYMENTS):
        assert applies_to(entity, scope) is True


# --- intent -------------------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "What did the breach policy previously require?",
        "What did we used to require for breach reporting?",
        "What was in the prior version of the data protection policy?",
        "Show me the superseded retention rule",
        "What is the history of the breach notification window?",
    ],
)
def test_historical_questions_are_detected(question):
    intent, matched = detect_intent(question)

    assert intent == "historical"
    assert matched


@pytest.mark.parametrize(
    "question",
    [
        "How quickly must a breach be reported?",
        "What is the retention period for KYC records?",
        # "use to" is deliberately NOT a historical marker: it appears in ordinary
        # present-tense questions like this one, and a false positive here would admit
        # superseded policy into an answer about current requirements.
        "What encryption do we use to protect customer data?",
        "Which systems do we use to monitor transactions?",
    ],
)
def test_ordinary_questions_are_current(question):
    assert detect_intent(question)[0] == "current"


def test_present_tense_marker_wins_over_a_historical_word():
    """'Is the superseded policy still current?' is a question about now."""
    assert detect_intent("Is the superseded policy currently in force?")[0] == "current"


# --- scope ---------------------------------------------------------------------------


def test_current_scope_excludes_superseded_documents():
    scope = resolve_scope("How quickly must a breach be reported?")

    assert scope.intent == "current"
    assert scope.include_superseded is False


def test_historical_scope_admits_superseded_documents():
    """Superseded policy is the only source for what was previously required."""
    scope = resolve_scope("What did the breach policy previously require?")

    assert scope.intent == "historical"
    assert scope.include_superseded is True


def test_scope_explains_itself():
    scope = resolve_scope("What did Northwind Capital Markets previously require?")

    explanation = scope.explain()
    assert "historical" in explanation
    assert "Capital Markets" in explanation


def test_group_scope_is_not_treated_as_entity_scoped():
    """Group is the default, so it must not trigger entity-specific promotion."""
    scope = resolve_scope("What is the group policy on retention?")

    assert scope.entity == GROUP_ENTITY
    assert scope.is_entity_scoped is False
