"""End-to-end tests of the HTTP contract, with retrieval and the LLM stubbed.

The point isn't to test Claude or pgvector — it's to pin the API surface: that the
response shape is stable, that refusals carry a machine-readable reason, that citations
are built from retrieved records rather than model output, that upstream failures map to
meaningful status codes, and that a malformed model response is not written to the query
log.
"""

import openai
import pytest
from fastapi.testclient import TestClient

from app.db import get_db
from app.llm import Generation
from app.main import app
from app.retrieval import RetrievedChunk

BREACH_TEXT = (
    "Any suspected personal data breach must be reported to the Data Protection "
    "Officer within four (4) hours of discovery."
)


class FakeSession:
    """Just enough Session for the routes: add/commit, and a stubbed history query."""

    def __init__(self, rows=None):
        self.added = []
        self.committed = False
        self._rows = rows or []

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed = True

    def rollback(self):
        pass

    def query(self, *_):
        return self

    def order_by(self, *_):
        return self

    def limit(self, n):
        return self

    def all(self):
        return self._rows


@pytest.fixture
def session():
    fake = FakeSession()
    app.dependency_overrides[get_db] = lambda: fake
    yield fake
    app.dependency_overrides.clear()


@pytest.fixture
def client(session):
    return TestClient(app)


def retrieved(chunk_id=7, content=BREACH_TEXT, similarity=0.8, page=3) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Data Protection and Privacy Policy",
        section="Data Protection and Privacy Policy > 7. Breach Notification > Clause 7.1",
        content=content,
        similarity=similarity,
        doc_id="NFS-POL-001",
        entity="Northwind Financial Services Pvt. Ltd.",
        clause="7.1",
        version="2.1",
        page=page,
    )


def answer(citations=None, answerable=True, confidence="high"):
    return {
        "answer": "Within four hours of discovery.",
        "citations": citations
        if citations is not None
        else [{"chunk_id": 7, "excerpt": "within four (4) hours of discovery"}],
        "confidence": confidence,
        "answerable": answerable,
    }


def stub(monkeypatch, *, chunks=None, content=None, retrieve_error=None, llm_error=None,
         usage=(1000, 200)):
    def fake_retrieve(db, question, **kwargs):
        if retrieve_error:
            raise retrieve_error
        return chunks if chunks is not None else [retrieved()]

    def fake_generate(question, relevant, **kwargs):
        if llm_error:
            raise llm_error
        return Generation(
            content=dict(content if content is not None else answer()),
            input_tokens=usage[0],
            output_tokens=usage[1],
        )

    monkeypatch.setattr("app.query.pipeline.retrieve", fake_retrieve)
    monkeypatch.setattr("app.query.pipeline.generate_answer", fake_generate)


# --- happy path ----------------------------------------------------------------


def test_health_needs_no_database():
    assert TestClient(app).get("/health").json() == {"status": "ok"}


def test_grounded_answer_returns_the_full_contract(client, session, monkeypatch):
    stub(monkeypatch)

    r = client.post("/api/query", json={"question": "How quickly must a breach be reported?"})

    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "answer",
        "citations",
        "confidence",
        "grounded",
        "refusal_reason",
        "retrieval",
    }
    # Retrieval metadata makes the selection auditable without reading server logs.
    assert body["retrieval"]["intent"] == "current"
    assert body["retrieval"]["evidence_used"] >= 1
    assert body["grounded"] is True
    assert body["refusal_reason"] is None
    assert 0.0 <= body["confidence"] <= 1.0
    assert session.committed


def test_citation_is_built_from_the_retrieved_record(client, session, monkeypatch):
    """The model supplied only chunk_id + excerpt; everything else is looked up."""
    stub(monkeypatch)

    citation = client.post("/api/query", json={"question": "breach?"}).json()["citations"][0]

    assert citation["chunk_id"] == 7
    assert citation["document"] == "Data Protection and Privacy Policy"
    assert citation["clause"] == "7.1"
    assert citation["page"] == 3
    assert citation["document_id"] == "NFS-POL-001"


def test_model_cannot_override_citation_metadata(client, session, monkeypatch):
    stub(
        monkeypatch,
        content=answer(
            citations=[
                {
                    "chunk_id": 7,
                    "excerpt": "within four (4) hours of discovery",
                    "document": "Fabricated Policy",
                    "page": 99,
                }
            ]
        ),
    )

    citation = client.post("/api/query", json={"question": "breach?"}).json()["citations"][0]

    assert citation["document"] == "Data Protection and Privacy Policy"
    assert citation["page"] == 3


# --- refusals -------------------------------------------------------------------


def test_no_relevant_evidence_refuses_without_calling_the_model(client, session, monkeypatch):
    called = []

    def exploding_generate(question, relevant, **kwargs):
        called.append(question)
        raise AssertionError("the model must not be called with zero relevant chunks")

    monkeypatch.setattr(
        "app.query.pipeline.retrieve", lambda db, q, **kwargs: [retrieved(similarity=0.01)]
    )
    monkeypatch.setattr("app.query.pipeline.generate_answer", exploding_generate)

    body = client.post("/api/query", json={"question": "unrelated"}).json()

    assert called == []
    assert body["grounded"] is False
    assert body["refusal_reason"] == "no_relevant_evidence"
    assert body["citations"] == []


def test_model_declining_is_reported_as_such(client, session, monkeypatch):
    stub(monkeypatch, content=answer(citations=[], answerable=False, confidence="low"))

    body = client.post("/api/query", json={"question": "not covered"}).json()

    assert body["grounded"] is False
    assert body["refusal_reason"] == "model_declined"


def test_unverifiable_citations_downgrade_to_a_refusal(client, session, monkeypatch):
    """A confident answer whose support can't be verified must not reach the user."""
    stub(
        monkeypatch,
        content=answer(citations=[{"chunk_id": 7, "excerpt": "an invented obligation"}]),
    )

    body = client.post("/api/query", json={"question": "breach?"}).json()

    assert body["grounded"] is False
    assert body["refusal_reason"] == "citations_unverified"
    assert body["citations"] == []


def test_refusals_score_low_confidence(client, session, monkeypatch):
    stub(monkeypatch, content=answer(citations=[], answerable=False))

    assert client.post("/api/query", json={"question": "x"}).json()["confidence"] <= 0.2


# --- input validation -----------------------------------------------------------


@pytest.mark.parametrize("question", ["", "   ", "\n\t"])
def test_blank_question_is_rejected(client, session, monkeypatch, question):
    stub(monkeypatch)
    assert client.post("/api/query", json={"question": question}).status_code == 400


def test_missing_question_field_is_a_422(client, session):
    assert client.post("/api/query", json={}).status_code == 422


def test_oversized_question_is_rejected(client, session, monkeypatch):
    stub(monkeypatch)
    assert client.post("/api/query", json={"question": "x" * 5000}).status_code == 400


# --- upstream failures map to meaningful status codes ---------------------------


def test_retrieval_failure_returns_503(client, session, monkeypatch):
    stub(monkeypatch, retrieve_error=OSError("embedding API unreachable"))

    r = client.post("/api/query", json={"question": "anything"})

    assert r.status_code == 503
    assert "unavailable" in r.json()["detail"].lower()


def test_missing_api_key_surfaces_as_a_readable_500(client, session, monkeypatch):
    stub(monkeypatch, llm_error=RuntimeError("ANTHROPIC_API_KEY is not set."))

    r = client.post("/api/query", json={"question": "anything"})

    assert r.status_code == 500
    assert "ANTHROPIC_API_KEY" in r.json()["detail"]


def test_rate_limit_maps_to_429(client, session, monkeypatch):
    stub(monkeypatch, llm_error=openai.RateLimitError(
        "rate limited", response=_fake_response(429), body=None
    ))

    assert client.post("/api/query", json={"question": "anything"}).status_code == 429


def test_unexpected_error_does_not_leak_a_traceback(client, session, monkeypatch):
    stub(monkeypatch, llm_error=ValueError("internal detail that must not leak"))

    r = client.post("/api/query", json={"question": "anything"})

    assert r.status_code == 500
    assert "internal detail" not in r.json()["detail"]


def test_offschema_model_output_is_rejected_without_logging(client, session, monkeypatch):
    """A malformed answer must fail the request, not commit a bad row to query_log."""
    stub(monkeypatch, content={"answer": "hi", "confidence": "high", "answerable": "not-a-bool"})

    r = client.post("/api/query", json={"question": "anything"})

    assert r.status_code == 500
    assert not session.committed
    assert session.added == []


# --- token + cost logging --------------------------------------------------------


def test_token_usage_and_cost_are_logged(client, session, monkeypatch):
    from app.config import settings

    stub(monkeypatch, usage=(1000, 200))

    client.post("/api/query", json={"question": "breach?"})

    log = session.added[0]
    assert log.input_tokens == 1000
    assert log.output_tokens == 200
    expected = (1000 * settings.price_per_mtok_input + 200 * settings.price_per_mtok_output) / 1e6
    assert log.cost_usd == pytest.approx(expected)
    assert log.grounded is True
    assert log.refusal_reason is None


def test_refusal_that_skips_the_model_logs_zero_cost(client, session, monkeypatch):
    monkeypatch.setattr(
        "app.query.pipeline.retrieve", lambda db, q, **kwargs: [retrieved(similarity=0.01)]
    )

    client.post("/api/query", json={"question": "unrelated"})

    log = session.added[0]
    assert log.cost_usd == 0.0
    assert log.grounded is False
    assert log.refusal_reason == "no_relevant_evidence"


# --- history ---------------------------------------------------------------------


def test_history_limit_is_capped(client, session):
    from app.config import settings

    assert client.get(f"/api/history?limit={settings.history_max_limit + 1}").status_code == 422
    assert client.get("/api/history?limit=0").status_code == 422


# --- ingestion -------------------------------------------------------------------


def test_ingest_returns_counts(client, session, monkeypatch):
    monkeypatch.setattr(
        "app.api.routes.ingestion.run_ingest",
        lambda db: {
            "documents": 26,
            "chunks": 983,
            "skipped_unchanged": 0,
            "superseded_documents": 1,
        },
    )

    r = client.post("/api/ingest")

    assert r.status_code == 200
    assert r.json()["documents"] == 26
    assert r.json()["superseded_documents"] == 1


def test_ingest_reports_skipped_unchanged_documents(client, session, monkeypatch):
    """Re-running ingestion with nothing changed should re-embed nothing."""
    monkeypatch.setattr(
        "app.api.routes.ingestion.run_ingest",
        lambda db: {"documents": 0, "chunks": 0, "skipped_unchanged": 26, "superseded_documents": 0},
    )

    body = client.post("/api/ingest").json()

    assert body["skipped_unchanged"] == 26
    assert body["chunks"] == 0


def test_ingest_failure_is_a_503_not_a_500(client, session, monkeypatch):
    def boom(db):
        raise OSError("/secret/internal/path is missing")

    monkeypatch.setattr("app.api.routes.ingestion.run_ingest", boom)

    r = client.post("/api/ingest")

    assert r.status_code == 503
    assert "/secret/internal/path" not in r.json()["detail"]


def _fake_response(status_code):
    import httpx

    return httpx.Response(status_code=status_code, request=httpx.Request("POST", "https://x"))
