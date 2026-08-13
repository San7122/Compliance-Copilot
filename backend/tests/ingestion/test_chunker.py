"""Clause boundaries are citation boundaries.

A chunk that silently absorbs the text after it produces a citation pointing at clause
N while quoting words that clause N does not contain — which is worse than no citation,
because it looks checkable. Both regressions pinned here (lettered clauses, unnumbered
section headings) were real: they were losing or misattributing roughly 40% of the
corpus before being caught by reading the extracted output.
"""

from pathlib import Path

import pytest

from app.ingestion.chunker import Clause, split_into_clauses
from app.ingestion.extractor import extract_document


def corpus_dir() -> Path:
    if Path("/app/docs").exists():
        return Path("/app/docs")
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs").is_dir():
            return parent / "docs"
    pytest.skip("policy corpus not available")


def clauses_of(pattern: str) -> list[Clause]:
    matches = sorted(corpus_dir().glob(pattern))
    if not matches:
        pytest.skip(f"no corpus document matching {pattern}")
    _, lines = extract_document(matches[0])
    return split_into_clauses(lines)


def find(clauses: list[Clause], ref: str) -> Clause:
    for c in clauses:
        if c.clause == ref:
            return c
    raise AssertionError(f"clause {ref} not found")


# --- structure -----------------------------------------------------------------


def test_numbered_clause_is_its_own_chunk():
    clauses = split_into_clauses(
        [
            "7. Breach Notification",
            "7.1  Any suspected breach must be reported within four (4) hours",
            "of discovery.",
            "7.2  A different obligation entirely.",
        ]
    )

    assert [c.clause for c in clauses] == ["7.1", "7.2"]
    assert clauses[0].text.endswith("of discovery.")
    assert "different obligation" not in clauses[0].text


def test_wrapped_continuation_lines_are_folded_in():
    clauses = split_into_clauses(["1.1  A clause that", "wraps across", "three lines."])

    assert len(clauses) == 1
    assert clauses[0].text == "A clause that wraps across three lines."


def test_lettered_clauses_are_recognised():
    """E.n / C.n / R.n clauses close every document; there are 325 in the corpus."""
    clauses = split_into_clauses(
        [
            "Exceptions and Waivers",
            "E.1  Requests for exception must be raised through the GRC tool.",
            "E.2  Exceptions expire after twelve months.",
        ]
    )

    assert [c.clause for c in clauses] == ["E.1", "E.2"]


def test_section_heading_does_not_become_a_clause():
    clauses = split_into_clauses(["7. Breach Notification", "7.1  The obligation."])

    assert len(clauses) == 1
    assert clauses[0].section == "7. Breach Notification"


def test_unnumbered_section_does_not_absorb_into_previous_clause():
    """Regression: 'Roles and Responsibilities' used to be glued onto the clause above."""
    clauses = split_into_clauses(
        [
            "3.3  Evidence is retained per the Records Retention Schedule.",
            "Roles and Responsibilities",
            "Head of Compliance",
            "Owns this calendar.",
        ]
    )

    clause_33 = find(clauses, "3.3")
    assert "Head of Compliance" not in clause_33.text
    assert any(c.section == "Roles and Responsibilities" for c in clauses)


def test_revision_history_is_dropped():
    """It records the document's own history and states no obligation."""
    clauses = split_into_clauses(
        [
            "Revision History",
            "2.1",
            "01 Apr 2026",
            "DPO",
            "Breach notification window aligned to 72 hours.",
            "1. Purpose",
            "1.1  The real content.",
        ]
    )

    assert [c.clause for c in clauses] == ["1.1"]
    assert all("01 Apr 2026" not in c.text for c in clauses)


def test_definitions_block_is_kept():
    clauses = split_into_clauses(
        ["Definitions", "Personal Data. Any information relating to a person."]
    )

    assert len(clauses) == 1
    assert clauses[0].clause is None
    assert clauses[0].section == "Definitions"


def test_heading_path_is_a_citable_reference():
    clause = Clause(clause="7.1", section="7. Breach Notification", text="...")

    assert (
        clause.heading_path("Data Protection and Privacy Policy")
        == "Data Protection and Privacy Policy > 7. Breach Notification > Clause 7.1"
    )


# --- against the real corpus ---------------------------------------------------


def test_breach_clause_extracts_the_current_figure():
    """The exact clause a naive system gets wrong by quoting the superseded policy."""
    clause = find(clauses_of("NFS-POL-001_v2.1*.pdf"), "7.1")

    assert "four (4) hours" in clause.text
    assert clause.section.startswith("7.")


def test_precedence_clause_is_captured():
    """NFS-POL-011 governs retention conflicts; clause 1.3 is where that's stated."""
    clause = find(clauses_of("NFS-POL-001_v2.1*.pdf"), "1.3")

    assert "NFS-POL-011" in clause.text
    assert "governs" in clause.text


def test_no_chunk_contains_page_furniture():
    for path in sorted(corpus_dir().glob("*.pdf")):
        _, lines = extract_document(path)
        for clause in split_into_clauses(lines):
            assert "Fictional document created" not in clause.text, path.name
            assert not clause.text.startswith("Page "), path.name


def test_every_corpus_document_produces_clauses():
    for path in sorted(corpus_dir().glob("*.pdf")):
        _, lines = extract_document(path)
        clauses = split_into_clauses(lines)

        assert clauses, f"{path.name}: no clauses"
        assert all(c.text.strip() for c in clauses), f"{path.name}: empty chunk"
        # Every document should yield real numbered obligations, not just prose blocks.
        assert any(c.clause for c in clauses), f"{path.name}: no numbered clauses"
