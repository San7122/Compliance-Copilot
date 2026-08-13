"""Clause-level chunking for the Northwind policy corpus.

Every document numbers its clauses (1.1, 4.3, 7.2), and the corpus notes say outright
that those numbers "make good citation targets". They are also the natural retrieval
unit: a clause is a single self-contained obligation, which is exactly the granularity a
compliance reader wants pointed at -- "NFS-POL-001 clause 7.1" is checkable in a way that
"somewhere in the Data Protection Policy" is not.

Two structures in this corpus are easy to get wrong, and both corrupt citations rather
than merely losing text:

- **Lettered clauses.** Alongside 1.1/4.3 numbering, every document ends with `E.n`
  (Exceptions and Waivers), `C.n` (Compliance and Enforcement), and `R.n` (Document
  Review and Control) clauses -- 325 of them across the corpus. Matching only `\\d+\\.\\d+`
  silently appends all of them to whichever numbered clause happened to come last,
  which both loses the clause reference and attaches text to a clause that does not
  contain it.
- **Unnumbered sections.** "Roles and Responsibilities", "Related Documents" and the
  like carry no clause number but do carry content, and they follow numbered clauses.
  Unless they're recognised as section boundaries, their tables get glued onto the
  preceding clause -- so a citation to clause 3.3 would quote a roles table that clause
  3.3 never mentioned.

Both were found by reading the extracted output rather than by reasoning about the
format, which is the general argument for looking at real data before trusting a parser.
"""

import re
from dataclasses import dataclass

# "7. Breach Notification" -- checked only after the clause patterns, so "7.1 ..." wins.
_SECTION = re.compile(r"^(?P<number>\d+)\.\s+(?P<title>\S.*)$")
# "7.1  Any suspected breach ..." and "E.1  Requests for exception ..."
_CLAUSE = re.compile(r"^(?P<number>(?:\d+\.\d+|[A-Z]\.\d+))\s+(?P<text>\S.*)$")
_ANNEXURE = re.compile(r"^(?P<title>Annexure\s+[A-Z0-9]+.*)$")

# Unnumbered headings that appear in essentially every document in the corpus.
_NAMED_SECTIONS = {
    "Definitions",
    "Roles and Responsibilities",
    "Related Documents",
    "Exceptions and Waivers",
    "Compliance and Enforcement",
    "Document Review and Control",
}

# The revision history is a grid of versions and dates describing the document's own
# history. It states no obligation, and extracts as meaningless one-token lines.
_REVISION_HISTORY = "Revision History"

# Column headers from the two-column tables, which extract as standalone lines.
_TABLE_HEADERS = {"Role", "Responsibility", "Version", "Date", "Approved by", "Summary of change"}


@dataclass
class Clause:
    """One citable unit of a policy document."""

    clause: str | None  # "7.1" / "E.1", or None for unnumbered blocks
    section: str | None  # "7. Breach Notification" / "Roles and Responsibilities"
    text: str
    # Page the clause starts on. A clause that wraps a page break is cited by its
    # opening page, which is where a reader looking it up would start.
    page: int | None = None

    def heading_path(self, doc_title: str) -> str:
        """What the user sees as the citation's `section` field."""
        parts = [doc_title]
        if self.section:
            parts.append(self.section)
        if self.clause:
            parts.append(f"Clause {self.clause}")
        return " > ".join(parts)


def _as_page_lines(lines) -> list[tuple[int | None, str]]:
    """Accept `Line` objects (with pages) or bare strings (without).

    Bare strings keep the unit tests readable -- most chunking behaviour has nothing to
    do with pagination, and forcing every fixture to carry page numbers would obscure
    what those tests are actually asserting.
    """
    return [(None, line) if isinstance(line, str) else (line.page, line.text) for line in lines]


def split_into_clauses(lines) -> list[Clause]:
    """Split cleaned document lines into clause-level chunks.

    Accepts `Line` objects carrying page numbers, or bare strings.
    """
    clauses: list[Clause] = []
    section: str | None = None
    current: Clause | None = None
    skipping = False  # inside Revision History

    def flush() -> None:
        nonlocal current
        if current and current.text.strip():
            current.text = re.sub(r"\s+", " ", current.text).strip()
            clauses.append(current)
        current = None

    def start_named(title: str, page: int | None) -> None:
        nonlocal current, section, skipping
        flush()
        skipping = False
        section = title
        current = Clause(clause=None, section=title, text="", page=page)

    for page, line in _as_page_lines(lines):
        stripped = line.strip()
        if not stripped or stripped in _TABLE_HEADERS:
            continue

        if stripped == _REVISION_HISTORY:
            flush()
            skipping = True
            continue

        if stripped in _NAMED_SECTIONS:
            start_named(stripped, page)
            continue

        annexure = _ANNEXURE.match(stripped)
        if annexure:
            start_named(annexure.group("title"), page)
            continue

        clause_match = _CLAUSE.match(stripped)
        if clause_match:
            flush()
            skipping = False
            current = Clause(
                clause=clause_match.group("number"),
                section=section,
                text=clause_match.group("text"),
                page=page,
            )
            continue

        section_match = _SECTION.match(stripped)
        if section_match:
            flush()
            skipping = False
            section = f"{section_match.group('number')}. {section_match.group('title')}"
            continue

        if skipping:
            continue

        if current is not None:
            # A wrapped continuation of the clause or block being built.
            current.text += " " + stripped

    flush()
    return clauses
