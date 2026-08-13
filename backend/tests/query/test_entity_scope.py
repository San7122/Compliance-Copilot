"""Regression tests for entity-scope propagation into the generation context.

The bug these exist for was real and observed against the live API. Asked "How long
must KYC records be retained?" with the entity selector set to Capital Markets,
retrieval and reranking did everything right -- NFS-SUB-002 (seven years) was ranked
first with the entity bonus applied -- and the answer still led with the group's five
years, relegating the correct figure to an aside.

The model was not at fault. It was following the prompt as written, which treated a
subsidiary as selected only when the question text named one. The entity had arrived
through the API parameter, so from inside the prompt no subsidiary had been chosen.

The fix carries the resolved scope into the user message. These tests assert the
propagation, because that is the deterministic part; whether the model then leads with
the right figure is verified by a real API call, not by a unit test.
"""

import pytest

from app.llm import scope_directive
from app.query.applicability import resolve_scope

@pytest.fixture(autouse=True)
def _no_governing_lookup(monkeypatch):
    """These tests pass db=None and use retention-shaped questions; the governing
    lookup is out of scope for what they assert, so it is stubbed to nothing."""
    monkeypatch.setattr("app.query.governing.fetch_clause", lambda db, d, c: None)
    monkeypatch.setattr("app.query.governing.retrieve", lambda db, q, **kw: [])


GROUP = "Northwind Financial Services Pvt. Ltd."
NCM = "Northwind Capital Markets Ltd"
NPS = "Northwind Payments (Singapore) Pte Ltd"

KYC_QUESTION = "How long must KYC records be retained?"


# --- TEST 1: entity supplied through the API, not named in the question ---------------


def test_api_supplied_entity_reaches_the_prompt():
    """The exact failing case: NCM selected, question silent about it."""
    scope = resolve_scope(KYC_QUESTION, entity=NCM)
    directive = scope_directive(scope)

    assert "ANSWERING SCOPE" in directive
    assert NCM in directive
    # It must say the question need not name the entity -- that was the missing signal.
    assert "does not have to name it" in directive
    assert "Lead with the requirement that binds it" in directive


def test_api_supplied_entity_is_marked_explicit():
    scope = resolve_scope(KYC_QUESTION, entity=NCM)

    assert scope.entity == NCM
    assert scope.entity_source == "explicit"


# --- TEST 2: group selected explicitly ------------------------------------------------


def test_explicit_group_scope_is_propagated():
    scope = resolve_scope(KYC_QUESTION, entity=GROUP)
    directive = scope_directive(scope)

    assert GROUP in directive
    assert "ANSWERING SCOPE" in directive
    # A group selection must not smuggle a subsidiary into the scope line.
    assert NCM not in directive
    assert NPS not in directive


# --- TEST 3: entity named in the question AND supplied through the API ----------------


def test_question_named_entity_plus_api_entity_agree():
    scope = resolve_scope(
        "How long must KYC records be retained for Northwind Capital Markets?", entity=NCM
    )
    directive = scope_directive(scope)

    assert scope.entity == NCM
    assert scope.entity_source == "explicit"  # the API value wins when both are present
    assert NCM in directive


def test_entity_named_only_in_the_question_is_still_scoped():
    """Inference still establishes scope; it's just labelled differently."""
    scope = resolve_scope("How long must KYC records be retained for Capital Markets?")
    directive = scope_directive(scope)

    assert scope.entity == NCM
    assert scope.entity_source == "inferred"
    assert NCM in directive
    assert "named in the question" in directive


# --- TEST 4: unscoped behaviour must be unchanged -------------------------------------


def test_no_entity_produces_no_scope_directive():
    """The previously-correct unscoped path must not acquire a scope line."""
    scope = resolve_scope(KYC_QUESTION)

    assert scope.entity is None
    assert scope_directive(scope) == ""


def test_none_scope_produces_no_directive():
    assert scope_directive(None) == ""


# --- TEST 5: historical + entity combined ---------------------------------------------


def test_historical_intent_and_entity_scope_coexist():
    scope = resolve_scope(
        "What did the breach policy previously require?", entity=NCM
    )
    directive = scope_directive(scope)

    assert scope.intent == "historical"
    assert scope.include_superseded is True
    assert scope.entity == NCM
    assert NCM in directive  # scope survives alongside historical intent


# --- the directive must not degenerate into "trust the top chunk" ---------------------


def test_directive_does_not_tell_the_model_to_trust_ranking():
    """Ranking order is not evidence about which entity binds the reader.

    Telling the model to prefer the top-ranked chunk would make answers follow retrieval
    noise whenever scores are close -- which is precisely the near-tie that produced the
    original bug.
    """
    directive = scope_directive(resolve_scope(KYC_QUESTION, entity=NCM))

    lowered = directive.lower()
    for phrase in ("top-ranked", "top ranked", "first chunk", "highest score", "most similar"):
        assert phrase not in lowered


def test_directive_preserves_the_fallback_when_entity_has_no_document():
    """Scoping must not force an answer where the entity has nothing on the topic."""
    directive = scope_directive(resolve_scope(KYC_QUESTION, entity=NCM))

    assert "no applicable document" in directive
    assert "group policy" in directive


# --- propagation through the pipeline --------------------------------------------------


def test_pipeline_passes_resolved_scope_to_generation(monkeypatch):
    from app.llm import Generation
    from app.query.pipeline import QueryPipeline
    from app.retrieval import RetrievedChunk

    captured = {}

    def fake_retrieve(db, question, **kwargs):
        return [
            RetrievedChunk(
                chunk_id=1,
                document="KYC and AML Policy",
                section="... > Clause 5.1",
                content="KYC records are retained for a minimum of seven (7) years.",
                similarity=0.8,
                doc_id="NFS-SUB-002",
                entity=NCM,
                clause="5.1",
                page=2,
            )
        ]

    def fake_generate(question, relevant, scope=None):
        captured["scope"] = scope
        return Generation(
            content={
                "answer": "Seven years.",
                "citations": [{"chunk_id": 1, "excerpt": "seven (7) years"}],
                "confidence": "high",
                "answerable": True,
            },
            input_tokens=10,
            output_tokens=5,
        )

    monkeypatch.setattr("app.query.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("app.query.pipeline.generate_answer", fake_generate)

    QueryPipeline(db=None).run(KYC_QUESTION, entity=NCM)

    assert captured["scope"] is not None
    assert captured["scope"].entity == NCM
    assert captured["scope"].entity_source == "explicit"


@pytest.mark.parametrize("entity,expected", [(NCM, NCM), (GROUP, GROUP), (NPS, NPS)])
def test_each_entity_scopes_to_itself(entity, expected):
    assert resolve_scope(KYC_QUESTION, entity=entity).entity == expected
