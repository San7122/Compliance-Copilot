"""Ingestion must be safely re-runnable.

The operational scenario this protects: 26 documents are indexed, five more arrive, and
ingestion is run again. What must happen is that the five are added and the 26 are left
alone — not re-embedded (which costs money and time for no change) and above all not
deleted and rebuilt, which would leave the corpus empty or partial if the run failed
halfway.

Identity is (filename, content_hash). Filename alone cannot distinguish "same document"
from "edited document", so a filename-only design has to re-embed everything on every
run to stay correct.
"""

import shutil
from pathlib import Path

import pytest

from app.ingestion.pipeline import IngestionPipeline, content_hash
from app.models import Chunk, Document


def corpus_dir() -> Path:
    if Path("/app/docs").exists():
        return Path("/app/docs")
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs").is_dir():
            return parent / "docs"
    pytest.skip("policy corpus not available")


def a_policy_pdf() -> Path:
    matches = sorted(corpus_dir().glob("NFS-POL-001_v2.1*.pdf"))
    if not matches:
        pytest.skip("corpus document not available")
    return matches[0]


class FakeQuery:
    def __init__(self, db, model):
        self.db, self.model, self.filters = db, model, {}

    def filter_by(self, **kwargs):
        self.filters = kwargs
        return self

    def first(self):
        if self.model is Document:
            return next(
                (d for d in self.db.documents if d.filename == self.filters.get("filename")),
                None,
            )
        return None

    def delete(self):
        document_id = self.filters.get("document_id")
        before = len(self.db.chunks)
        self.db.chunks = [c for c in self.db.chunks if c.document_id != document_id]
        return before - len(self.db.chunks)


class FakeDB:
    """Enough SQLAlchemy Session surface for the pipeline, with no real database."""

    def __init__(self):
        self.documents: list[Document] = []
        self.chunks: list[Chunk] = []
        self._next_id = 1

    def query(self, model):
        return FakeQuery(self, model)

    def add(self, obj):
        (self.documents if isinstance(obj, Document) else self.chunks).append(obj)

    def flush(self):
        for document in self.documents:
            if document.id is None:
                document.id = self._next_id
                self._next_id += 1

    def commit(self):
        pass

    def rollback(self):
        pass


class CountingEmbedder:
    def __init__(self):
        self.calls = 0
        self.texts_embedded = 0

    def embed_documents(self, texts):
        self.calls += 1
        self.texts_embedded += len(texts)
        return [[0.0] * 384 for _ in texts]


@pytest.fixture
def docs_dir(tmp_path):
    shutil.copy(a_policy_pdf(), tmp_path / "policy_one.pdf")
    return tmp_path


@pytest.fixture
def embedder(monkeypatch):
    fake = CountingEmbedder()
    monkeypatch.setattr("app.ingestion.pipeline.embedding_service", fake)
    return fake


# --- content hash ---------------------------------------------------------------


def test_content_hash_is_stable_for_identical_bytes(tmp_path):
    a, b = tmp_path / "a.txt", tmp_path / "b.txt"
    a.write_bytes(b"same"), b.write_bytes(b"same")

    assert content_hash(a) == content_hash(b)


def test_content_hash_changes_when_content_changes(tmp_path):
    path = tmp_path / "a.txt"
    path.write_bytes(b"before")
    first = content_hash(path)
    path.write_bytes(b"after")

    assert content_hash(path) != first


# --- first run ------------------------------------------------------------------


def test_first_run_ingests_and_records_the_hash(docs_dir, embedder):
    db = FakeDB()

    result = IngestionPipeline(db, str(docs_dir)).run()

    assert result["documents"] == 1
    assert result["chunks"] > 0
    assert result["skipped_unchanged"] == 0
    assert db.documents[0].content_hash == content_hash(docs_dir / "policy_one.pdf")
    assert embedder.calls == 1


def test_chunks_carry_page_numbers(docs_dir, embedder):
    """Page must survive extraction -> chunk -> storage, or citations can't show it."""
    db = FakeDB()
    IngestionPipeline(db, str(docs_dir)).run()

    pages = [c.page for c in db.chunks]
    assert all(p is not None for p in pages)
    assert min(pages) == 1
    assert max(pages) > 1  # a multi-page policy should span pages


def test_chunks_carry_clause_references(docs_dir, embedder):
    db = FakeDB()
    IngestionPipeline(db, str(docs_dir)).run()

    assert any(c.clause == "7.1" for c in db.chunks)


# --- idempotency ----------------------------------------------------------------


def test_rerun_with_no_changes_skips_and_does_not_re_embed(docs_dir, embedder):
    db = FakeDB()
    IngestionPipeline(db, str(docs_dir)).run()
    chunks_after_first = len(db.chunks)
    calls_after_first = embedder.calls

    result = IngestionPipeline(db, str(docs_dir)).run()

    assert result["skipped_unchanged"] == 1
    assert result["documents"] == 0
    assert embedder.calls == calls_after_first  # nothing re-embedded
    assert len(db.chunks) == chunks_after_first  # nothing duplicated or dropped


def test_adding_a_document_leaves_existing_documents_untouched(docs_dir, embedder):
    db = FakeDB()
    IngestionPipeline(db, str(docs_dir)).run()
    original_chunk_ids = [id(c) for c in db.chunks]

    shutil.copy(a_policy_pdf(), docs_dir / "policy_two.pdf")
    result = IngestionPipeline(db, str(docs_dir)).run()

    assert result["documents"] == 1  # only the new one
    assert result["skipped_unchanged"] == 1
    assert len(db.documents) == 2
    # The first document's chunk objects are the same objects, never rebuilt.
    assert original_chunk_ids == [id(c) for c in db.chunks[: len(original_chunk_ids)]]


def test_changed_document_is_reprocessed_and_replaces_only_its_own_chunks(docs_dir, embedder):
    db = FakeDB()
    shutil.copy(a_policy_pdf(), docs_dir / "policy_two.pdf")
    IngestionPipeline(db, str(docs_dir)).run()
    two_chunks = [c for c in db.chunks if c.document_id == db.documents[1].id]

    # Replace one document's bytes with a different policy.
    other = sorted(corpus_dir().glob("NFS-POL-007*.pdf"))
    if not other:
        pytest.skip("second corpus document not available")
    shutil.copy(other[0], docs_dir / "policy_one.pdf")

    result = IngestionPipeline(db, str(docs_dir)).run()

    assert result["documents"] == 1
    assert result["skipped_unchanged"] == 1
    assert len(db.documents) == 2  # updated in place, not duplicated
    # The untouched document keeps exactly its original chunks.
    still_two = [c for c in db.chunks if c.document_id == db.documents[1].id]
    assert [id(c) for c in still_two] == [id(c) for c in two_chunks]


# --- failure isolation ------------------------------------------------------------


def test_one_bad_document_does_not_abort_the_run(docs_dir, embedder):
    """A scanned or corrupt PDF must not leave the rest of the corpus unindexed."""
    (docs_dir / "broken.pdf").write_bytes(b"not a pdf at all")
    db = FakeDB()

    result = IngestionPipeline(db, str(docs_dir)).run()

    assert result["documents"] == 1  # the good one still landed
    assert result["failed_documents"] == 1
    assert "broken.pdf" in result["warning"]


def test_empty_directory_reports_a_warning(tmp_path):
    result = IngestionPipeline(FakeDB(), str(tmp_path)).run()

    assert result["documents"] == 0
    assert "No .pdf or .md files found" in result["warning"]
