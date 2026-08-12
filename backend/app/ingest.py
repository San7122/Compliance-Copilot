"""Ingestion pipeline: docs/*.md -> heading-based chunks -> embeddings -> Postgres.

Chunking strategy
------------------
1. Split each markdown file on headings (#, ##, ###). Each resulting section keeps a
   "heading path" like "Data Retention Policy > 2. Retention Periods > 2.1 Customer
   Account Data" — this is what we cite back to the user, since compliance readers care
   about *which section* an answer came from, not just the document name.
2. If a section's body is longer than ~500 tokens (approximated as words / 0.75), split
   it further into ~500-token windows with ~50-token overlap, so a single fact near a
   chunk boundary isn't cut off from its supporting context.
3. Very short sections (e.g. a lone heading with a one-line body) are kept as single
   chunks rather than merged with neighbours — simpler, and our corpus is small enough
   that we don't need to worry about chunk-count bloat.

This is a naive token approximation (word count), not a real tokenizer, which is a
deliberate simplification for a ~5-6 hour build — see README tradeoffs.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy.orm import Session

from app.config import settings
from app.embeddings import embed_texts
from app.models import Chunk, Document

WORDS_PER_TOKEN = 0.75  # rough approximation: 1 token ~= 0.75 words


@dataclass
class Section:
    heading_path: str
    body: str


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


def chunk_document(path: Path) -> tuple[str, list[Section]]:
    markdown = path.read_text(encoding="utf-8")
    title = _extract_title(markdown, fallback=path.stem)
    sections = _split_into_sections(markdown, title)

    chunks: list[Section] = []
    for section in sections:
        # Skip the H1 title-only section (front matter like owner/effective date is short,
        # but still useful context, so keep it rather than dropping it).
        windows = _word_windows(section.body, settings.chunk_max_tokens, settings.chunk_overlap_tokens)
        for w in windows:
            chunks.append(Section(heading_path=section.heading_path, body=w))
    return title, chunks


def run_ingest(db: Session, docs_dir: str | None = None) -> dict:
    docs_path = Path(docs_dir or settings.docs_dir)
    md_files = sorted(docs_path.glob("*.md"))

    if not md_files:
        return {"documents": 0, "chunks": 0, "warning": f"No .md files found in {docs_path}"}

    total_docs = 0
    total_chunks = 0

    for path in md_files:
        # Parse once -- both branches below need the title, and re-reading + re-chunking
        # the same file twice per document is pure waste.
        title, sections = chunk_document(path)

        existing = db.query(Document).filter_by(filename=path.name).first()
        if existing:
            # Re-ingest: drop old chunks for this doc, keep the doc row.
            db.query(Chunk).filter_by(document_id=existing.id).delete()
            document = existing
            document.title = title
        else:
            document = Document(filename=path.name, title=title)
            db.add(document)
            db.flush()  # get document.id

        texts = [s.body for s in sections]
        if not texts:
            continue
        vectors = embed_texts(texts)

        for idx, (section, vector) in enumerate(zip(sections, vectors)):
            chunk = Chunk(
                document_id=document.id,
                heading_path=section.heading_path,
                chunk_index=idx,
                content=section.body,
                embedding=vector,
            )
            db.add(chunk)

        total_docs += 1
        total_chunks += len(sections)

    db.commit()
    return {"documents": total_docs, "chunks": total_chunks}
