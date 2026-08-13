"""Build citations from stored records, not from model output.

The model is asked for a `chunk_id` and a supporting `excerpt` -- nothing else. Document
title, section, clause, page and entity are looked up from the chunk that was actually
retrieved. This makes fabricated citation metadata structurally impossible rather than
something to detect after the fact: there is no field for the model to invent, because
it never supplies one.

Two checks remain, and both matter:

1. **The chunk_id must be one that was actually given to the model.** A model can emit a
   plausible integer for a chunk it never saw. Only IDs present in this turn's retrieved
   set are accepted, so an invented ID cannot address an arbitrary row in the database.
2. **The excerpt must appear in that specific chunk.** Verifying against the whole
   context would let an excerpt lifted from chunk A be attributed to chunk B -- a
   citation pointing at the wrong clause while quoting real corpus text, which is
   exactly the kind of error that survives a casual read.
"""

from app.config import settings
from app.retrieval import RetrievedChunk
from app.verification import excerpt_is_grounded, normalize


def map_citations(
    raw_citations: list[dict], chunks: list[RetrievedChunk]
) -> tuple[list[dict], list[dict]]:
    """Turn the model's (chunk_id, excerpt) claims into verified citations.

    Returns (citations, rejected). Each citation's metadata comes from the retrieved
    chunk; only `excerpt` originates from the model, and only after being verified
    against that chunk's stored text.
    """
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    citations: list[dict] = []
    rejected: list[dict] = []
    seen: set[tuple[int, str]] = set()

    for raw in raw_citations or []:
        chunk_id = _as_int(raw.get("chunk_id"))
        excerpt = (raw.get("excerpt") or "").strip()

        chunk = by_id.get(chunk_id) if chunk_id is not None else None
        if chunk is None:
            rejected.append({**raw, "reason": "chunk_id was not in the retrieved set"})
            continue

        if not excerpt_is_grounded(excerpt, normalize(chunk.content), settings.citation_min_match):
            rejected.append({**raw, "reason": "excerpt does not appear in the cited chunk"})
            continue

        # The same clause can legitimately be cited once; twice is noise.
        key = (chunk.chunk_id, normalize(excerpt))
        if key in seen:
            continue
        seen.add(key)

        citations.append(
            {
                "chunk_id": chunk.chunk_id,
                "document": chunk.document,
                "document_id": chunk.doc_id,
                "entity": chunk.entity,
                "section": chunk.section,
                "clause": chunk.clause,
                "page": chunk.page,
                "version": chunk.version,
                "document_type": chunk.document_type,
                # Carried so a historical answer visibly cites superseded text rather
                # than looking like a statement of current requirements.
                "status": chunk.status,
                "excerpt": excerpt,
            }
        )

    return citations, rejected


def _as_int(value) -> int | None:
    """Tool-use schemas usually deliver a real int, but a numeric string is cheap to accept."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None
