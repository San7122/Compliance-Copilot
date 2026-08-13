"""Turn a Northwind policy PDF into clean text plus the metadata that governs how it
may be used.

Three things in this corpus decide whether an answer is safe to give, and none of them
live in the prose -- they live in the document's front matter:

1. **Status.** `NFS-POL-001-A` is marked SUPERSEDED on its front page and says outright
   "Do not rely on this version for current obligations". It still contains fluent,
   confident, retrievable policy text (an 8-hour breach reporting window against the
   current policy's 4 hours), so similarity search alone will happily surface it.
2. **Entity.** Northwind is a group of three legal entities. Subsidiary documents look
   almost identical to the group ones and deliberately carry different numbers, so a
   citation is only meaningful once you know which company it binds.
3. **Document kind.** `NFS-GUID-001` is a plain-language handbook extract that
   paraphrases the real policies loosely. It reads like a policy and is not one.

So the parser's job is not only to get the words out -- it is to recover the metadata
that lets retrieval refuse to use the wrong document.
"""

import re
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

# Repeated on every page: "Northwind Financial Services Pvt. Ltd.  |  NFS-POL-001 v2.1  |  Internal"
_RUNNING_HEADER = re.compile(r"^(?P<entity>.+?)\s*\|\s*(?P<doc_id>[A-Z]+-[A-Z]+-\d+(?:-[A-Z])?)\s+v(?P<version>[\d.]+)\s*\|\s*(?P<classification>.+?)\s*$")
_DISCLAIMER = re.compile(r"^Fictional document created for a hiring exercise")
_PAGE_MARKER = re.compile(r"^Page\s+\d+\s*$")

_FIELD = re.compile(r"^(?P<key>Document ID|Version|Effective date|Document owner|Classification|Review cycle):\s*(?P<value>.+?)\s*$")
_STATUS_START = re.compile(r"^STATUS:\s*(?P<rest>.+?)\s*$")

# A status block can wrap onto following lines; these mark where it must stop.
_STATUS_TERMINATORS = re.compile(r"^(Revision History|Definitions|\d+\.\s)")

SUPERSEDED = "superseded"
CURRENT = "current"

# Document series encoded in the ID (NFS-POL-001, NFS-SUB-002, ...). These are read off
# the corpus, not invented: POL/SUB/SOP/GUID/REG are the series actually present.
#
# `document_type` is descriptive metadata, not a ranking. It exists so that the handbook
# can be told apart from an authoritative policy, and it is applied only as a tiebreaker
# *after* entity applicability -- never as a universal precedence order.
POLICY = "policy"
ENTITY_POLICY = "entity_policy"
PROCEDURE = "procedure"
GUIDANCE = "guidance"
REGULATORY = "regulatory"
UNKNOWN_TYPE = "unknown"

_SERIES_TO_TYPE = {
    "POL": POLICY,
    "SUB": ENTITY_POLICY,
    "SOP": PROCEDURE,
    "GUID": GUIDANCE,
    "REG": REGULATORY,
}

_SERIES = re.compile(r"^[A-Z]+-(?P<series>[A-Z]+)-\d+")


def classify_document(doc_id: str | None) -> str:
    """Map a document ID to its type. Unrecognised series stay `unknown` rather than
    being guessed at, so an unfamiliar document is never silently given authority."""
    if not doc_id:
        return UNKNOWN_TYPE
    match = _SERIES.match(doc_id.strip().upper())
    if not match:
        return UNKNOWN_TYPE
    return _SERIES_TO_TYPE.get(match.group("series"), UNKNOWN_TYPE)


@dataclass
class Line:
    """A line of body text plus the page it came from.

    Page is carried from extraction all the way to the citation. A compliance reader
    checking an answer opens the PDF and needs a page to turn to; a clause number alone
    means scanning the document for it.
    """

    page: int
    text: str


@dataclass
class DocumentMeta:
    doc_id: str
    title: str
    entity: str
    version: str
    status: str
    document_type: str = UNKNOWN_TYPE
    status_note: str | None = None
    effective_date: str | None = None
    owner: str | None = None
    classification: str | None = None

    @property
    def is_superseded(self) -> bool:
        return self.status == SUPERSEDED


def _read_pages(path: Path) -> list[str]:
    return [page.extract_text() or "" for page in PdfReader(str(path)).pages]


def is_furniture(line: str) -> bool:
    """Per-page header/footer, carrying no document content."""
    return bool(
        _RUNNING_HEADER.match(line) or _DISCLAIMER.match(line) or _PAGE_MARKER.match(line)
    )


def strip_running_furniture(lines: list[str]) -> list[str]:
    """Drop the per-page header block so it can't pollute chunks or citations.

    Left in, this text appears mid-sentence every time a clause spans a page break --
    it would be embedded as content, and could even be quoted back as a citation
    excerpt.
    """
    return [line for line in lines if not is_furniture(line)]


def parse_metadata(lines: list[str], fallback_title: str) -> DocumentMeta:
    """Recover document identity from the front matter.

    The running header is the most reliable source for entity/doc id/version -- it is
    printed on every page in a fixed format -- so it's preferred over the front-matter
    fields, which are only on page 1.
    """
    entity = doc_id = version = classification = None
    for line in lines:
        match = _RUNNING_HEADER.match(line)
        if match:
            entity = match.group("entity").strip()
            doc_id = match.group("doc_id").strip()
            version = match.group("version").strip()
            classification = match.group("classification").strip()
            break

    body = strip_running_furniture(lines)

    fields: dict[str, str] = {}
    for line in body:
        match = _FIELD.match(line)
        if match:
            fields.setdefault(match.group("key"), match.group("value"))

    title = _extract_title(body, fallback_title)

    status_note = _extract_status(body)
    status = SUPERSEDED if status_note and "SUPERSEDED" in status_note.upper() else CURRENT

    resolved_doc_id = doc_id or fields.get("Document ID", fallback_title)

    return DocumentMeta(
        doc_id=resolved_doc_id,
        document_type=classify_document(resolved_doc_id),
        title=title,
        # The line under the title repeats the entity; fall back to it if the running
        # header was unreadable.
        entity=entity or (body[1].strip() if len(body) > 1 else "unknown"),
        version=version or fields.get("Version", "unknown"),
        status=status,
        status_note=status_note,
        effective_date=fields.get("Effective date"),
        owner=fields.get("Document owner"),
        classification=classification or fields.get("Classification"),
    )


def _extract_title(body: list[str], fallback: str) -> str:
    """First non-furniture line, re-joined if the title wrapped.

    Subsidiary titles are long enough to wrap ("Know Your Customer and
    Anti-Money-Laundering Policy - Northwind Capital Markets Ltd"), and the break lands
    after the hyphen. Taking only the first line leaves a title ending in a dangling
    "-" and drops the entity qualifier -- which then shows up in every citation.
    """
    lines = [line.strip() for line in body if line.strip()]
    if not lines:
        return fallback

    title = lines[0]
    if title.endswith("-") and len(lines) > 1:
        title = f"{title} {lines[1]}".strip()
    return title


def _extract_status(body: list[str]) -> str | None:
    """Pull the STATUS block, which may wrap across several lines."""
    for i, line in enumerate(body):
        match = _STATUS_START.match(line)
        if not match:
            continue
        parts = [match.group("rest")]
        for following in body[i + 1 :]:
            if not following.strip() or _STATUS_TERMINATORS.match(following) or _FIELD.match(following):
                break
            parts.append(following.strip())
        return " ".join(parts)
    return None


def extract_document(path: Path) -> tuple[DocumentMeta, list[Line]]:
    """Return (metadata, cleaned body lines with page numbers) for one policy PDF.

    A PDF with no extractable text is reported rather than silently yielding zero
    chunks: a scanned document that ingests "successfully" with nothing in it is worse
    than a failure, because retrieval then quietly cannot answer anything from it.
    """
    pages = _read_pages(path)
    if not any(page.strip() for page in pages):
        raise ValueError(
            f"{path.name}: no extractable text (likely a scanned image PDF). "
            "OCR is required before this document can be ingested."
        )

    lines = [
        Line(page=page_number, text=text)
        for page_number, page_text in enumerate(pages, start=1)
        for text in page_text.splitlines()
    ]

    meta = parse_metadata([line.text for line in lines], fallback_title=path.stem)
    return meta, [line for line in lines if not is_furniture(line.text)]
