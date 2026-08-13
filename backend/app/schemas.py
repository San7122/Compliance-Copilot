from typing import Literal

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str
    # Optional. When supplied it wins over anything inferred from the question text:
    # the caller knows which legal entity the user belongs to, which is better
    # information than phrasing. When absent, the entity is inferred from the question
    # and falls back to group scope.
    entity: str | None = None


class RetrievalMetadata(BaseModel):
    """How the evidence was selected. Exposed so an answer can be audited without
    reading server logs — particularly *why* a given document was or wasn't used."""

    entity_scope: str | None = None
    entity_source: str = "unspecified"  # explicit | inferred | unspecified
    intent: str = "current"  # current | historical
    superseded_included: bool = False
    candidates_considered: int = 0
    evidence_used: int = 0
    ranking: list[str] = Field(default_factory=list)


class ModelCitation(BaseModel):
    """What the model is allowed to say about a citation: which chunk, and what quote."""

    chunk_id: int
    excerpt: str


class ModelAnswer(BaseModel):
    """Schema for the model's structured output, validated before anything uses it.

    Forced tool-use already constrains the shape at the API boundary, but validating
    again here is cheap and closes a real gap: reading a field with `bool(...)` or
    `dict.get(...)` accepts whatever arrives, so an off-schema value becomes a
    plausible-looking answer instead of a loud failure. Parsing into a typed model means
    a malformed response fails as a malformed response.
    """

    answer: str
    citations: list[ModelCitation] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low"]
    answerable: bool


class Citation(BaseModel):
    """Every field except `excerpt` is copied from the stored chunk record.

    `chunk_id` is included so a citation can be traced back to the exact row it came
    from — which is what makes the "we never invent citations" claim checkable rather
    than merely asserted.
    """

    chunk_id: int
    document: str
    section: str
    excerpt: str
    document_id: str | None = None
    entity: str | None = None
    clause: str | None = None
    page: int | None = None
    version: str | None = None
    document_type: str | None = None
    status: str = "current"


class AnswerResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    # Application-level score, not a calibrated probability. See app/confidence.py.
    confidence: float = Field(ge=0.0, le=1.0)
    grounded: bool
    refusal_reason: str | None = None
    retrieval: RetrievalMetadata | None = None


class HistoryItem(BaseModel):
    id: int
    question: str
    answer: str
    citations: list[Citation]
    confidence: float
    grounded: bool
    refusal_reason: str | None = None
    latency_ms: int | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    created_at: str


class IngestResponse(BaseModel):
    documents: int
    chunks: int
    # Present but excluded from retrieval — surfaced rather than silently dropped.
    superseded_documents: int = 0
    # Unchanged files that were hash-matched and not re-embedded.
    skipped_unchanged: int = 0
    failed_documents: int = 0
    warning: str | None = None
