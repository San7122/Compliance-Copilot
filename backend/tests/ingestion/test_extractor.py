"""Front-matter parsing decides whether a document may be used to answer at all.

If `status` is misread, the system will quote a withdrawn policy. If `entity` is misread,
it will apply one legal entity's obligations to another. Both failures produce confident,
well-cited, wrong answers — so these tests pin the parse against the real corpus rather
than against a synthetic fixture that could drift from it.
"""

from pathlib import Path

import pytest

from app.ingestion.extractor import (
    CURRENT,
    SUPERSEDED,
    extract_document,
    parse_metadata,
    strip_running_furniture,
)

GROUP = "Northwind Financial Services Pvt. Ltd."
CAPITAL_MARKETS = "Northwind Capital Markets Ltd"


def corpus_dir() -> Path:
    """Locate docs/ whether tests run from backend/ or inside the container."""
    if Path("/app/docs").exists():
        return Path("/app/docs")
    for parent in Path(__file__).resolve().parents:
        if (parent / "docs").is_dir():
            return parent / "docs"
    pytest.skip("policy corpus not available")


def doc(pattern: str) -> Path:
    matches = sorted(corpus_dir().glob(pattern))
    if not matches:
        pytest.skip(f"no corpus document matching {pattern}")
    return matches[0]


# --- running furniture ---------------------------------------------------------


def test_running_header_block_is_stripped():
    """Left in, this repeats mid-clause at every page break and can be quoted back."""
    lines = [
        "Northwind Financial Services Pvt. Ltd.  |  NFS-POL-001 v2.1  |  Internal",
        "Fictional document created for a hiring exercise. Not legal or regulatory advice.",
        "Page 2",
        "2.5  Personal data is retained no longer than necessary.",
    ]

    assert strip_running_furniture(lines) == [
        "2.5  Personal data is retained no longer than necessary."
    ]


def test_stripping_keeps_ordinary_content_containing_a_pipe():
    lines = ["A table row | with a pipe | in it"]

    assert strip_running_furniture(lines) == lines


# --- metadata ------------------------------------------------------------------


def test_current_policy_metadata():
    meta, _ = extract_document(doc("NFS-POL-001_v2.1*.pdf"))

    assert meta.doc_id == "NFS-POL-001"
    assert meta.version == "2.1"
    assert meta.status == CURRENT
    assert meta.is_superseded is False
    assert meta.entity == GROUP
    assert meta.title == "Data Protection and Privacy Policy"
    assert meta.effective_date == "01 April 2026"


def test_superseded_policy_is_detected():
    """NFS-POL-001-A says on its own front page not to rely on it."""
    meta, _ = extract_document(doc("NFS-POL-001-A*SUPERSEDED*.pdf"))

    assert meta.doc_id == "NFS-POL-001-A"
    assert meta.status == SUPERSEDED
    assert meta.is_superseded is True
    assert "SUPERSEDED" in meta.status_note.upper()
    # It points at its replacement; keep that, it's what makes the exclusion explicable.
    assert "NFS-POL-001" in meta.status_note


def test_subsidiary_entity_is_not_read_as_the_group():
    meta, _ = extract_document(doc("NFS-SUB-001*.pdf"))

    assert meta.entity == CAPITAL_MARKETS
    assert meta.entity != GROUP
    assert meta.status == CURRENT  # current, but binding only on the subsidiary


def test_every_corpus_document_parses_with_usable_identity():
    docs = sorted(corpus_dir().glob("*.pdf"))
    assert docs, "no corpus PDFs found"

    for path in docs:
        meta, lines = extract_document(path)
        assert meta.doc_id, f"{path.name}: no document ID"
        assert meta.entity, f"{path.name}: no entity"
        assert meta.version, f"{path.name}: no version"
        assert meta.status in (CURRENT, SUPERSEDED), f"{path.name}: bad status"
        assert lines, f"{path.name}: no body text"


def test_corpus_has_exactly_one_superseded_document():
    """The corpus notes state this explicitly; if it changes, the fixtures moved."""
    statuses = [extract_document(p)[0].status for p in sorted(corpus_dir().glob("*.pdf"))]

    assert statuses.count(SUPERSEDED) == 1


def test_corpus_covers_three_legal_entities():
    entities = {extract_document(p)[0].entity for p in sorted(corpus_dir().glob("*.pdf"))}

    assert len(entities) == 3
    assert GROUP in entities


def test_missing_status_line_defaults_to_current():
    lines = [
        "Northwind Financial Services Pvt. Ltd.  |  NFS-POL-999 v1.0  |  Internal",
        "Some Policy",
        "Northwind Financial Services Pvt. Ltd.",
        "Document ID: NFS-POL-999",
        "Version: 1.0",
    ]

    meta = parse_metadata(lines, fallback_title="fallback")

    assert meta.status == CURRENT
    assert meta.status_note is None


# --- document type ---------------------------------------------------------------


def test_document_type_is_classified_from_the_id_series():
    from app.ingestion.extractor import (
        ENTITY_POLICY,
        GUIDANCE,
        POLICY,
        PROCEDURE,
        REGULATORY,
        UNKNOWN_TYPE,
        classify_document,
    )

    assert classify_document("NFS-POL-001") == POLICY
    assert classify_document("NFS-SUB-001") == ENTITY_POLICY
    assert classify_document("NFS-SOP-001") == PROCEDURE
    assert classify_document("NFS-GUID-001") == GUIDANCE
    assert classify_document("NFS-REG-001") == REGULATORY
    # Unfamiliar series must not be guessed at -- an unknown document should never be
    # handed authority it hasn't earned.
    assert classify_document("NFS-XYZ-001") == UNKNOWN_TYPE
    assert classify_document(None) == UNKNOWN_TYPE
    assert classify_document("garbage") == UNKNOWN_TYPE


def test_handbook_is_classified_as_guidance_not_policy():
    """NFS-GUID-001 says it paraphrases policy loosely; it must be distinguishable."""
    from app.ingestion.extractor import GUIDANCE

    meta, _ = extract_document(doc("NFS-GUID-001*.pdf"))

    assert meta.document_type == GUIDANCE


def test_every_corpus_document_gets_a_known_type():
    from app.ingestion.extractor import UNKNOWN_TYPE

    for path in sorted(corpus_dir().glob("*.pdf")):
        meta, _ = extract_document(path)
        assert meta.document_type != UNKNOWN_TYPE, path.name
