from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.embedder import embedding_service
from app.ingestion.extractor import CURRENT, SUPERSEDED
from app.models import Chunk, Document


@dataclass
class RetrievedChunk:
    """A chunk plus the stored metadata used for filtering, ranking and citation.

    This record -- not the model's output -- is the source of truth for citation
    fields. See app/citations/mapper.py.
    """

    chunk_id: int
    document: str
    section: str
    content: str
    similarity: float  # cosine similarity, 1.0 = identical, higher is better
    doc_id: str | None = None
    entity: str | None = None
    clause: str | None = None
    version: str | None = None
    page: int | None = None
    document_type: str | None = None
    status: str = CURRENT
    effective_date: str | None = None

    @property
    def is_superseded(self) -> bool:
        return self.status == SUPERSEDED


def retrieve(
    db: Session,
    question: str,
    limit: int | None = None,
    include_superseded: bool = False,
    doc_id: str | None = None,
) -> list[RetrievedChunk]:
    """Nearest chunks by cosine similarity.

    Two deliberate choices:

    - **Oversample.** `candidate_k` (default 20) candidates are fetched, not the five
      that reach the model. Filtering and reranking happen afterwards, so a chunk
      dropped for inapplicability is replaced by the next best applicable one instead
      of leaving a hole in the evidence.
    - **Status filtering is a parameter, not a constant.** Superseded documents are
      excluded by default because they are not a source of current requirements, but
      they remain reachable for explicitly historical questions. Deleting them, or
      filtering them unconditionally, would make "what did the policy previously
      require?" unanswerable -- and that is a legitimate question about a corpus that
      retains a superseded document precisely for audit reference.
    """
    query_vector = embedding_service.embed_query(question)
    k = limit or settings.candidate_k

    # pgvector's `<=>` operator is cosine *distance* (0 = identical, 2 = opposite).
    # We convert to similarity = 1 - distance for a more intuitive "higher is better" score.
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = select(Chunk, Document, distance.label("distance")).join(
        Document, Chunk.document_id == Document.id
    )
    if not include_superseded:
        stmt = stmt.where(Document.status != SUPERSEDED)
    if doc_id is not None:
        # Scope the same vector search to one document. Used only by the corpus-specific
        # governing-schedule rule (app/query/governing.py); leaving it None reproduces
        # the previous SQL exactly, so the normal path is unaffected.
        stmt = stmt.where(Document.doc_id == doc_id)
    stmt = stmt.order_by(distance).limit(k)

    return [
        RetrievedChunk(
            chunk_id=chunk.id,
            document=document.title,
            section=chunk.heading_path,
            content=chunk.content,
            similarity=1 - float(dist),
            doc_id=document.doc_id,
            entity=document.entity,
            clause=chunk.clause,
            version=document.version,
            page=chunk.page,
            document_type=document.document_type,
            status=document.status or CURRENT,
            effective_date=document.effective_date,
        )
        for chunk, document, dist in db.execute(stmt).all()
    ]


def filter_by_relevance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks below the similarity floor — these are noise, not weak evidence."""
    return [c for c in chunks if c.similarity >= settings.min_similarity]
