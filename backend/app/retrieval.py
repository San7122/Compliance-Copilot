from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import embed_query
from app.models import Chunk, Document


@dataclass
class RetrievedChunk:
    chunk_id: int
    document: str
    section: str
    content: str
    similarity: float  # cosine similarity, 1.0 = identical, higher is better


def retrieve(db: Session, question: str, top_k: int | None = None) -> list[RetrievedChunk]:
    query_vector = embed_query(question)
    k = top_k or settings.top_k

    # pgvector's `<=>` operator is cosine *distance* (0 = identical, 2 = opposite).
    # We convert to similarity = 1 - distance for a more intuitive "higher is better" score.
    distance = Chunk.embedding.cosine_distance(query_vector)
    stmt = (
        select(Chunk, Document.title, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(k)
    )
    rows = db.execute(stmt).all()

    results = []
    for chunk, doc_title, dist in rows:
        similarity = 1 - float(dist)
        results.append(
            RetrievedChunk(
                chunk_id=chunk.id,
                document=doc_title,
                section=chunk.heading_path,
                content=chunk.content,
                similarity=similarity,
            )
        )
    return results


def filter_by_relevance(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Drop chunks below the similarity floor — these are noise, not weak evidence."""
    return [c for c in chunks if c.similarity >= settings.min_similarity]
