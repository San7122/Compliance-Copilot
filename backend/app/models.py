from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    ARRAY,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.db import Base

EMBEDDING_DIM = 384


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True)
    filename = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    # Identity recovered from the document's front matter. `status` and `entity` are
    # the load-bearing ones: they decide whether a document may be used to answer at
    # all, and which legal entity its numbers bind. See app/pdf_extract.py.
    doc_id = Column(String)
    # policy / entity_policy / procedure / guidance / regulatory. Descriptive metadata
    # used as a reranking tiebreaker, never as a universal precedence order.
    document_type = Column(String)
    entity = Column(String)
    version = Column(String)
    status = Column(String, nullable=False, server_default="current")
    status_note = Column(Text)
    effective_date = Column(String)
    # SHA-256 of the source file. Document identity is (filename, content_hash): an
    # unchanged file is skipped entirely on re-ingest rather than re-embedded.
    content_hash = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    chunks = relationship("Chunk", back_populates="document", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(Integer, primary_key=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), nullable=False)
    heading_path = Column(String, nullable=False)
    # "7.1" / "E.1" -- the citable reference within the document. Null for unnumbered
    # blocks such as Definitions and Annexures.
    clause = Column(String)
    # Page the clause starts on, carried through to the citation.
    page = Column(Integer)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    embedding = Column(Vector(EMBEDDING_DIM))
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    document = relationship("Document", back_populates="chunks")


class QueryLog(Base):
    __tablename__ = "query_log"

    id = Column(Integer, primary_key=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    citations = Column(JSONB, nullable=False)
    # Application-level score in [0,1] combining retrieval strength, grounding, and
    # citation availability. Not a calibrated probability -- see app/confidence.py.
    confidence = Column(Float, nullable=False)
    grounded = Column(Boolean, nullable=False)
    refusal_reason = Column(Text)
    retrieved_chunk_ids = Column(ARRAY(Integer), nullable=False, default=list)
    latency_ms = Column(Integer)
    # Zero on refusals that never reached the API -- see llm.Generation.
    input_tokens = Column(Integer, nullable=False, default=0)
    output_tokens = Column(Integer, nullable=False, default=0)
    cost_usd = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
