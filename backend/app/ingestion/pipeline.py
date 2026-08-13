"""Offline ingestion pipeline.

    document -> extract -> clean -> section/clause detection -> chunk -> embed -> store

This module owns orchestration and storage only. The stages themselves live next to it:
`extractor.py` (PDF text + front-matter identity + furniture removal), `chunker.py`
(section- and clause-aware splitting), `embedder.py` (hosted embedding API). Nothing
here answers questions -- that is the query pipeline's job, and the two are kept apart
deliberately.

Two chunking strategies, chosen by file type:

- **PDF (the real corpus): clause-aware.** Policy documents number their clauses, and
  those numbers are the natural citation target and the natural retrieval unit. Clauses
  here are short and self-contained, so they are not windowed further -- splitting one
  would sever an obligation from its own conditions.
- **Markdown: heading-aware with token windows.** Retained for non-PDF sources. Prose
  sections are of arbitrary length, so long ones are split into ~500-token windows with
  ~50-token overlap so a fact near a boundary keeps its context. Token count is
  approximated as words / 0.75 rather than by a real tokenizer -- a deliberate
  simplification at this corpus size.
"""

import hashlib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.ingestion.chunker import split_into_clauses
from app.ingestion.embedder import embedding_service
from app.ingestion.extractor import DocumentMeta, extract_document
from app.models import Chunk, Document

logger = logging.getLogger(__name__)

WORDS_PER_TOKEN = 0.75  # rough approximation: 1 token ~= 0.75 words


@dataclass
class Section:
    heading_path: str
    body: str
    clause: str | None = None
    page: int | None = None


@dataclass
class ParsedDocument:
    """A document reduced to what ingestion needs: identity plus citable chunks."""

    title: str
    sections: list[Section]
    meta: DocumentMeta | None = None  # PDFs carry front-matter identity; markdown doesn't


def _split_into_sections(markdown: str, doc_title: str) -> list[Section]:
    """Split markdown into sections using # / ## / ### headings, tracking heading path."""
    lines = markdown.splitlines()
    sections: list[Section] = []

    heading_stack: list[tuple[int, str]] = []  # (level, text)
    current_body: list[str] = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            path = " > ".join([doc_title] + [h for _, h in heading_stack])
            sections.append(Section(heading_path=path, body=body))

    for line in lines:
        match = re.match(r"^(#{1,3})\s+(.*)", line)
        if match:
            flush()
            current_body = []
            level = len(match.group(1))
            text = match.group(2).strip()
            # pop any headings at same or deeper level, then push this one
            heading_stack = [h for h in heading_stack if h[0] < level]
            # Skip the top-level (#) heading: it's already the document title, which is
            # prepended to every heading_path separately, so pushing it here would just
            # duplicate it (e.g. "Acceptable Use Policy > Acceptable Use Policy").
            if level > 1:
                heading_stack.append((level, text))
        else:
            current_body.append(line)

    flush()
    return sections


def _word_windows(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    words = text.split()
    # 1 token ~= 0.75 words, so a budget of N tokens holds N * 0.75 words.
    max_words = max(1, int(max_tokens * WORDS_PER_TOKEN))
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN)

    # Overlap must be strictly smaller than the window, or `start` below stops advancing
    # and the loop never terminates. Clamp rather than raise: a misconfigured overlap
    # should degrade the chunking, not take ingestion down.
    overlap_words = min(overlap_words, max_words - 1) if max_words > 1 else 0

    if len(words) <= max_words:
        return [text]

    windows = []
    start = 0
    while start < len(words):
        end = min(start + max_words, len(words))
        windows.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start = end - overlap_words
    return windows


def _extract_title(markdown: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.*)", markdown, re.MULTILINE)
    return match.group(1).strip() if match else fallback


def chunk_markdown(path: Path) -> ParsedDocument:
    markdown = path.read_text(encoding="utf-8")
    title = _extract_title(markdown, fallback=path.stem)
    sections = _split_into_sections(markdown, title)

    chunks: list[Section] = []
    for section in sections:
        # Markdown sections are prose of arbitrary length, so they still need windowing.
        windows = _word_windows(section.body, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        for w in windows:
            chunks.append(Section(heading_path=section.heading_path, body=w))
    return ParsedDocument(title=title, sections=chunks)


def chunk_pdf(path: Path) -> ParsedDocument:
    """Chunk a policy PDF at clause boundaries, carrying its front-matter identity.

    No token windowing here: clauses in this corpus are self-contained and short (the
    longest is ~170 words, well inside the chunk budget), so splitting them further
    would only sever an obligation from its own conditions.
    """
    meta, lines = extract_document(path)
    sections = [
        Section(
            heading_path=clause.heading_path(meta.title),
            body=clause.text,
            clause=clause.clause,
            page=clause.page,
        )
        for clause in split_into_clauses(lines)
    ]
    return ParsedDocument(title=meta.title, sections=sections, meta=meta)


def chunk_document(path: Path) -> ParsedDocument:
    """Parse one source document, dispatching on file type."""
    if path.suffix.lower() == ".pdf":
        return chunk_pdf(path)
    return chunk_markdown(path)


def content_hash(path: Path) -> str:
    """SHA-256 of the file bytes.

    Identity is (filename, content_hash) rather than filename alone. Filename alone
    can't tell an unchanged re-run from an edited document, so every ingest would have
    to re-embed everything to stay correct. Hashing makes the common case -- "nothing
    changed" -- free, which is what makes re-running ingestion after adding five new
    PDFs cheap instead of a full re-embed of the corpus.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _apply_metadata(document: Document, parsed: ParsedDocument) -> None:
    """Copy front-matter identity onto the document row.

    Markdown sources have no front matter, so they default to `current` with no entity.
    That default is deliberately permissive: an unknown-provenance document should still
    be searchable. It's only safe because the corpus that actually matters here does
    carry status, and because superseded documents announce themselves explicitly.
    """
    document.title = parsed.title
    if not parsed.meta:
        return
    document.doc_id = parsed.meta.doc_id
    document.document_type = parsed.meta.document_type
    document.entity = parsed.meta.entity
    document.version = parsed.meta.version
    document.status = parsed.meta.status
    document.status_note = parsed.meta.status_note
    document.effective_date = parsed.meta.effective_date


class IngestionPipeline:
    """Offline pipeline: discover -> extract -> clean -> chunk -> embed -> store.

    Thin on purpose. Each stage already lives in its own module (`extractor`,
    `chunker`, `embedder`); this class exists to make the order of those stages
    readable in one place, not to hold logic of its own.
    """

    def __init__(self, db: Session, docs_dir: str | None = None):
        self.db = db
        self.docs_path = Path(docs_dir or settings.docs_dir)

    def discover(self) -> list[Path]:
        return sorted(list(self.docs_path.glob("*.pdf")) + list(self.docs_path.glob("*.md")))

    def run(self) -> dict:
        source_files = self.discover()
        if not source_files:
            return {
                "documents": 0,
                "chunks": 0,
                "warning": f"No .pdf or .md files found in {self.docs_path}",
            }

        ingested = skipped = failed = superseded = total_chunks = 0
        errors: list[str] = []

        for path in source_files:
            existing = self.db.query(Document).filter_by(filename=path.name).first()
            digest = content_hash(path)

            # The whole point of hashing: an unchanged document costs one hash, not one
            # embedding call per chunk.
            if existing and existing.content_hash == digest:
                skipped += 1
                continue

            try:
                result = self._ingest_one(path, existing, digest)
            except Exception as exc:
                # One malformed or scanned PDF must not abort the run and leave the
                # rest of the corpus unindexed. Roll back just this document.
                self.db.rollback()
                logger.exception("ingestion failed for %s", path.name)
                errors.append(f"{path.name}: {exc}")
                failed += 1
                continue

            ingested += 1
            total_chunks += result["chunks"]
            superseded += result["superseded"]

        self.db.commit()

        summary = {
            "documents": ingested,
            "chunks": total_chunks,
            "skipped_unchanged": skipped,
            "superseded_documents": superseded,
        }
        if errors:
            summary["failed_documents"] = failed
            summary["warning"] = "; ".join(errors)
        return summary

    def _ingest_one(self, path: Path, existing: Document | None, digest: str) -> dict:
        parsed = chunk_document(path)

        if existing:
            # Replace only THIS document's chunks. Other documents are untouched, so
            # adding new files never disturbs what is already indexed.
            self.db.query(Chunk).filter_by(document_id=existing.id).delete()
            document = existing
        else:
            document = Document(filename=path.name, title=parsed.title)
            self.db.add(document)

        _apply_metadata(document, parsed)
        document.content_hash = digest
        self.db.flush()  # assigns document.id

        if not parsed.sections:
            raise ValueError("produced no chunks")

        vectors = embedding_service.embed_documents([s.body for s in parsed.sections])

        for idx, (section, vector) in enumerate(zip(parsed.sections, vectors)):
            self.db.add(
                Chunk(
                    document_id=document.id,
                    heading_path=section.heading_path,
                    clause=section.clause,
                    page=section.page,
                    chunk_index=idx,
                    content=section.body,
                    embedding=vector,
                )
            )

        return {
            "chunks": len(parsed.sections),
            "superseded": 1 if parsed.meta and parsed.meta.is_superseded else 0,
        }


def run_ingest(db: Session, docs_dir: str | None = None) -> dict:
    """Functional entry point kept for the API route, CLI, and tests."""
    return IngestionPipeline(db, docs_dir).run()
