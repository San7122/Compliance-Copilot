"""The query pipeline's job is deciding *when not to answer*.

Retrieval and generation are tested elsewhere; what's specific to this layer is the
refusal logic and the ordering of stages. Three refusal reasons exist because they mean
different things operationally — no evidence retrieved, the model declining, and support
that failed verification — and conflating them would hide which one is firing.
"""

import pytest

from app.llm import Generation
from app.query.pipeline import (
    MalformedAnswer,
    QueryPipeline,
    QuestionInvalid,
)
from app.retrieval import RetrievedChunk

@pytest.fixture(autouse=True)
def _no_governing_lookup(monkeypatch):
    """These tests pass db=None and use retention-shaped questions; the governing
    lookup is out of scope for what they assert, so it is stubbed to nothing."""
    monkeypatch.setattr("app.query.governing.fetch_clause", lambda db, d, c: None)
    monkeypatch.setattr("app.query.governing.retrieve", lambda db, q, **kw: [])


BREACH_TEXT = (
    "Any suspected personal data breach must be reported to the Data Protection "
    "Officer within four (4) hours of discovery."
)


def chunk(chunk_id=7, similarity=0.8, content=BREACH_TEXT) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Data Protection and Privacy Policy",
        section="... > Clause 7.1",
        content=content,
        similarity=similarity,
        doc_id="NFS-POL-001",
        clause="7.1",
        page=3,
    )


def model_answer(**overrides):
    base = {
        "answer": "Within four hours.",
        "citations": [{"chunk_id": 7, "excerpt": "within four (4) hours of discovery"}],
        "confidence": "high",
        "answerable": True,
    }
    base.update(overrides)
    return base


@pytest.fixture
def wire(monkeypatch):
    """Install retrieval + generation stubs; return a setter for per-test behaviour."""

    state = {"chunks": [chunk()], "content": model_answer(), "llm_error": None, "calls": []}

    def fake_retrieve(db, question, **kwargs):
        return state["chunks"]

    def fake_generate(question, relevant, **kwargs):
        state["calls"].append(question)
        if state["llm_error"]:
            raise state["llm_error"]
        return Generation(content=dict(state["content"]), input_tokens=10, output_tokens=5)

    monkeypatch.setattr("app.query.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("app.query.pipeline.generate_answer", fake_generate)
    return state


def run(question="How quickly must a breach be reported?"):
    return QueryPipeline(db=None).run(question)


# --- validation -----------------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t", None])
def test_blank_questions_are_rejected(wire, question):
    with pytest.raises(QuestionInvalid):
        QueryPipeline(db=None).run(question)


def test_oversized_question_is_rejected(wire):
    with pytest.raises(QuestionInvalid):
        run("x" * 5000)


def test_question_is_trimmed_before_use(wire):
    run("   breach reporting?   ")

    assert wire["calls"] == ["breach reporting?"]


# --- refusal: no relevant evidence ----------------------------------------------


def test_below_the_floor_refuses_without_calling_the_model(wire):
    wire["chunks"] = [chunk(similarity=0.01)]

    result = run()

    assert wire["calls"] == []  # the model was never invoked
    assert result.response.grounded is False
    assert result.response.refusal_reason == "no_relevant_evidence"
    assert result.cost_usd == 0.0


def test_no_chunks_at_all_refuses(wire):
    wire["chunks"] = []

    result = run()

    assert result.response.refusal_reason == "no_relevant_evidence"
    assert result.response.citations == []


# --- refusal: model declined ------------------------------------------------------


def test_model_declining_is_distinguished_from_no_evidence(wire):
    wire["content"] = model_answer(answerable=False, citations=[], confidence="low")

    result = run()

    assert result.response.refusal_reason == "model_declined"
    assert wire["calls"]  # the model WAS called, unlike the case above


# --- refusal: citations could not be verified -------------------------------------


def test_unverifiable_support_becomes_a_refusal(wire):
    wire["content"] = model_answer(
        citations=[{"chunk_id": 7, "excerpt": "an obligation that appears nowhere"}]
    )

    result = run()

    assert result.response.refusal_reason == "citations_unverified"
    assert result.response.citations == []
    assert result.rejected_citations  # recorded, not silently dropped


def test_citation_for_an_unretrieved_chunk_becomes_a_refusal(wire):
    wire["content"] = model_answer(
        citations=[{"chunk_id": 999, "excerpt": "within four (4) hours of discovery"}]
    )

    result = run()

    assert result.response.refusal_reason == "citations_unverified"


# --- success ----------------------------------------------------------------------


def test_grounded_answer_carries_verified_citations(wire):
    result = run()

    assert result.response.grounded is True
    assert result.response.refusal_reason is None
    assert result.response.citations[0].chunk_id == 7
    assert result.response.citations[0].page == 3
    assert result.response.confidence > 0.2


def test_partial_verification_keeps_the_answer_with_the_good_citation(wire):
    wire["content"] = model_answer(
        citations=[
            {"chunk_id": 7, "excerpt": "within four (4) hours of discovery"},
            {"chunk_id": 7, "excerpt": "an invented clause"},
        ]
    )

    result = run()

    assert result.response.grounded is True
    assert len(result.response.citations) == 1
    assert len(result.rejected_citations) == 1


def test_telemetry_is_reported_for_logging(wire):
    result = run()

    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.cost_usd > 0
    assert result.retrieved_chunk_ids == [7]
    assert result.latency_ms >= 0


# --- malformed model output --------------------------------------------------------


def test_offschema_response_raises_rather_than_being_interpreted(wire):
    """`bool("not-a-bool")` is True -- reading defensively would hide this."""
    wire["content"] = model_answer(answerable="not-a-bool")

    with pytest.raises(MalformedAnswer):
        run()


def test_missing_required_field_raises(wire):
    wire["content"] = {"answer": "hi"}

    with pytest.raises(MalformedAnswer):
        run()


def test_invalid_confidence_label_raises(wire):
    wire["content"] = model_answer(confidence="extremely-high")

    with pytest.raises(MalformedAnswer):
        run()


# --- scope: entity and history end-to-end through the pipeline -----------------------


def entity_chunk(chunk_id, entity, similarity=0.7, status="current", content=BREACH_TEXT):
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="KYC and AML Policy",
        section="... > Clause 5.1",
        content=content,
        similarity=similarity,
        doc_id="NFS-POL-003",
        clause="5.1",
        page=2,
        entity=entity,
        status=status,
    )


GROUP = "Northwind Financial Services Pvt. Ltd."
NCM = "Northwind Capital Markets Ltd"
NPS = "Northwind Payments (Singapore) Pte Ltd"


def test_another_subsidiarys_document_is_never_used(wire):
    """A Payments (Singapore) figure is not a fact about Capital Markets."""
    wire["chunks"] = [entity_chunk(1, NPS)]

    result = QueryPipeline(db=None).run("KYC retention?", entity="NCM")

    assert result.response.refusal_reason == "no_relevant_evidence"
    assert result.response.retrieval.entity_scope == NCM


def test_entity_document_is_preferred_over_group(wire):
    wire["chunks"] = [entity_chunk(1, GROUP, similarity=0.85), entity_chunk(2, NCM, similarity=0.6)]

    result = QueryPipeline(db=None).run("KYC retention?", entity="NCM")

    assert result.retrieved_chunk_ids[0] == 2


def test_explicit_entity_is_reported_in_retrieval_metadata(wire):
    result = QueryPipeline(db=None).run("KYC retention?", entity="NCM")

    assert result.response.retrieval.entity_source == "explicit"
    assert result.response.retrieval.entity_scope == NCM


def test_current_question_does_not_request_superseded_documents(wire):
    seen = {}

    def recording_retrieve(db, question, **kwargs):
        seen.update(kwargs)
        return wire["chunks"]

    import app.query.pipeline as pipeline_module

    original = pipeline_module.retrieve
    pipeline_module.retrieve = recording_retrieve
    try:
        run("How quickly must a breach be reported?")
    finally:
        pipeline_module.retrieve = original

    assert seen.get("include_superseded") is False


def test_historical_question_requests_superseded_documents(wire):
    seen = {}

    def recording_retrieve(db, question, **kwargs):
        seen.update(kwargs)
        return wire["chunks"]

    import app.query.pipeline as pipeline_module

    original = pipeline_module.retrieve
    pipeline_module.retrieve = recording_retrieve
    try:
        result = run("What did the breach policy previously require?")
    finally:
        pipeline_module.retrieve = original

    assert seen.get("include_superseded") is True
    assert result.response.retrieval.intent == "historical"
    assert result.response.retrieval.superseded_included is True


def test_abstention_uses_the_specified_wording(wire):
    wire["chunks"] = []

    result = run("What is the parental leave entitlement?")

    assert result.response.answer == "I don't know based on the provided documents."
    assert result.response.grounded is False


def test_retrieval_metadata_reports_ranking(wire):
    result = run()

    assert result.response.retrieval.candidates_considered >= 1
    assert result.response.retrieval.evidence_used >= 1
    assert result.response.retrieval.ranking  # human-readable per-chunk explanations
