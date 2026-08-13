"""Tests for the corpus-specific governing-schedule rule.

Two things matter equally. The rule must fire for retention questions -- otherwise the
answer under-states a retention period, and NFS-POL-011 clause 1.3 spells out why that
direction is the dangerous one ("early destruction cannot be undone"). And it must *not*
fire for anything else, because injecting a retention schedule into an unrelated question
adds distracting evidence to a path that already answers correctly.

The detection tests lean towards false positives, especially the "how long" trap: "How
long does the company have to respond to a data subject request?" reads like a retention
question and is not one.

The evidence tests pin *both* halves. An earlier version supplied only the annexure --
the model then saw eight years sitting beside the policy's seven and had no stated reason
to prefer either, so it kept answering seven. The precedence rule is what makes the
number usable.
"""

import pytest

from app.query.governing import (
    GOVERNING_CLAUSE,
    GOVERNING_DOC_ID,
    governing_evidence,
    is_retention_question,
)
from app.retrieval import RetrievedChunk


def chunk(chunk_id, doc_id=GOVERNING_DOC_ID, clause=None, similarity=0.7, content="text"):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Records Retention Schedule",
        section="...",
        content=content,
        similarity=similarity,
        doc_id=doc_id,
        clause=clause,
    )


RULE = chunk(457, clause=GOVERNING_CLAUSE, similarity=0.0,
             content="Where a policy and this schedule state different periods ... this schedule governs")
ANNEXURE_A = chunk(483, content="Financial transaction records 8 years ... Schedule governs")
ANNEXURE_B = chunk(484, content="Worked example: policy 7 years, schedule 8 years, applied 8 years")


@pytest.fixture
def stub_sources(monkeypatch):
    """Stub the two evidence sources the helper uses."""
    state = {"rule": RULE, "similar": [ANNEXURE_A, ANNEXURE_B]}

    monkeypatch.setattr(
        "app.query.governing.fetch_clause", lambda db, doc_id, clause: state["rule"]
    )
    monkeypatch.setattr(
        "app.query.governing.retrieve", lambda db, q, **kw: list(state["similar"])
    )
    return state


# --- detection: must fire ------------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        "How long must financial transaction records be retained?",
        "How long must KYC records be retained?",
        "What is the retention period for consent records?",
        "How long do we retain security event logs?",
        "How long must records relating to a filed suspicious transaction report be kept?",
        "How long are personnel files stored?",
    ],
)
def test_retention_questions_are_detected(question):
    assert is_retention_question(question) is True


# --- detection: must NOT fire --------------------------------------------------------


@pytest.mark.parametrize(
    "question",
    [
        # The trap: "how long" with no storage verb. A response-time question, which
        # currently answers correctly without governing evidence.
        "How long does the company have to respond to a verified data subject request?",
        "How long before a regulatory submission deadline does the change freeze start?",
        "How quickly must a breach be reported to the DPO?",
        "When is multi-factor authentication required?",
        "What is the minimum password length for privileged accounts?",
        "How often must high-risk customers be re-verified?",
        "What key length is required for symmetric encryption?",
    ],
)
def test_non_retention_questions_are_not_detected(question):
    assert is_retention_question(question) is False


def test_empty_question_is_not_retention():
    assert is_retention_question("") is False
    assert is_retention_question(None) is False


# --- TEST B: the context must contain BOTH the rule and the annexure -------------------


def test_evidence_contains_both_the_precedence_rule_and_the_annexure(stub_sources):
    evidence = governing_evidence(db=None, question="How long are records retained?",
                                  already_selected=[])

    ids = [c.chunk_id for c in evidence]
    assert 457 in ids, "the §1.2 precedence rule must be present"
    assert 483 in ids, "the annexure holding the applicable period must be present"


def test_precedence_rule_is_fetched_by_clause_not_similarity(stub_sources):
    """Similarity ranks §1.2 sixth even within its own document; identity is reliable."""
    evidence = governing_evidence(db=None, question="How long are records retained?",
                                  already_selected=[])

    rule = next(c for c in evidence if c.chunk_id == 457)
    assert rule.clause == GOVERNING_CLAUSE


def test_evidence_is_bounded_not_the_whole_document(stub_sources):
    evidence = governing_evidence(db=None, question="How long are records retained?",
                                  already_selected=[])

    assert len(evidence) <= 3
    assert all(c.doc_id == GOVERNING_DOC_ID for c in evidence)


def test_duplicates_of_already_selected_chunks_are_dropped(stub_sources):
    """The context should gain evidence, not repetition."""
    evidence = governing_evidence(db=None, question="How long are records retained?",
                                  already_selected=[ANNEXURE_A])

    ids = [c.chunk_id for c in evidence]
    assert 483 not in ids
    assert 457 in ids


def test_missing_precedence_clause_degrades_gracefully(stub_sources):
    """If the clause can't be found, still supply what we have rather than failing."""
    stub_sources["rule"] = None

    evidence = governing_evidence(db=None, question="How long are records retained?",
                                  already_selected=[])

    assert [c.chunk_id for c in evidence] == [483, 484]


def test_no_governing_evidence_at_all_returns_empty(stub_sources):
    stub_sources["rule"] = None
    stub_sources["similar"] = []

    assert governing_evidence(db=None, question="x", already_selected=[]) == []


# --- pipeline integration ---------------------------------------------------------------


def _pipeline_chunks(monkeypatch, question, candidates, stub_governing=True):
    """Run the pipeline with stubs and return the chunks handed to the model."""
    from app.llm import Generation
    from app.query.pipeline import QueryPipeline

    seen = {}

    def fake_generate(q, relevant, scope=None):
        seen["chunks"] = list(relevant)
        return Generation(
            content={
                "answer": "ok",
                "citations": [{"chunk_id": relevant[0].chunk_id, "excerpt": relevant[0].content}],
                "confidence": "high",
                "answerable": True,
            },
            input_tokens=1,
            output_tokens=1,
        )

    monkeypatch.setattr("app.query.pipeline.retrieve", lambda db, q, **kw: candidates)
    monkeypatch.setattr("app.query.pipeline.generate_answer", fake_generate)
    if stub_governing:
        monkeypatch.setattr("app.query.governing.fetch_clause", lambda db, d, c: RULE)
        monkeypatch.setattr(
            "app.query.governing.retrieve", lambda db, q, **kw: [ANNEXURE_A, ANNEXURE_B]
        )

    QueryPipeline(db=None).run(question)
    return seen["chunks"]


def _many(n):
    return [
        chunk(i + 1, doc_id="NFS-POL-001", clause=f"5.{i}", similarity=0.80 - i * 0.01,
              content=f"policy clause {i}")
        for i in range(n)
    ]


def test_retention_question_gains_rule_and_annexure(monkeypatch):
    """The q9 shape: governing evidence sits below the top_k cut and must still arrive."""
    chunks = _pipeline_chunks(
        monkeypatch, "How long must financial transaction records be retained?", _many(6)
    )

    ids = [c.chunk_id for c in chunks]
    assert 457 in ids  # precedence rule
    assert 483 in ids  # annexure with the applicable period


def test_non_retention_question_is_unchanged(monkeypatch):
    """Byte-for-byte the old behaviour: exactly top_k chunks, no governing evidence."""
    from app.config import settings

    chunks = _pipeline_chunks(
        monkeypatch, "How quickly must a breach be reported to the DPO?", _many(6)
    )

    assert len(chunks) == settings.top_k
    assert all(c.doc_id != GOVERNING_DOC_ID for c in chunks)


def test_governing_sources_are_not_touched_for_non_retention_questions(monkeypatch):
    """The gate must short-circuit before any extra query is issued."""
    calls = []
    monkeypatch.setattr(
        "app.query.governing.fetch_clause",
        lambda db, d, c: calls.append("fetch") or RULE,
    )
    monkeypatch.setattr(
        "app.query.governing.retrieve",
        lambda db, q, **kw: calls.append("retrieve") or [],
    )

    _pipeline_chunks(
        monkeypatch, "When is multi-factor authentication required?", _many(6),
        stub_governing=False,
    )

    assert calls == []


def test_retention_question_adds_a_bounded_amount(monkeypatch):
    from app.config import settings

    chunks = _pipeline_chunks(monkeypatch, "How long are records retained?", _many(6))

    assert len(chunks) <= settings.top_k + 3
