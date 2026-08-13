"""Citations must come from stored records, never from the model.

These are the tests that make the claim "we never invent citations" checkable. The
model supplies only a chunk_id and a quote; everything a reader would use to verify the
answer — document, section, clause, page — is looked up. So the failure modes worth
pinning are: an invented chunk_id, a quote lifted from a different chunk, and metadata
the model tried to supply itself.
"""

from app.citations.mapper import map_citations
from app.retrieval import RetrievedChunk

BREACH_TEXT = (
    "Any suspected personal data breach must be reported to the Data Protection "
    "Officer within four (4) hours of discovery, whether or not the breach has been "
    "confirmed."
)
RETENTION_TEXT = (
    "Financial transaction records are retained for a minimum of seven (7) years from "
    "the date of the transaction."
)


def chunk(chunk_id=7, content=BREACH_TEXT, clause="7.1", page=3) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        document="Data Protection and Privacy Policy",
        section="Data Protection and Privacy Policy > 7. Breach Notification > Clause 7.1",
        content=content,
        similarity=0.8,
        doc_id="NFS-POL-001",
        entity="Northwind Financial Services Pvt. Ltd.",
        clause=clause,
        version="2.1",
        page=page,
    )


# --- metadata comes from the record --------------------------------------------


def test_citation_metadata_is_taken_from_the_retrieved_chunk():
    citations, rejected = map_citations(
        [{"chunk_id": 7, "excerpt": "within four (4) hours of discovery"}], [chunk()]
    )

    assert rejected == []
    assert citations == [
        {
            "chunk_id": 7,
            "document": "Data Protection and Privacy Policy",
            "document_id": "NFS-POL-001",
            "entity": "Northwind Financial Services Pvt. Ltd.",
            "section": "Data Protection and Privacy Policy > 7. Breach Notification > Clause 7.1",
            "clause": "7.1",
            "page": 3,
            "version": "2.1",
            "document_type": None,
            "status": "current",
            "excerpt": "within four (4) hours of discovery",
        }
    ]


def test_model_supplied_metadata_is_ignored_entirely():
    """Even if the model volunteers a document or page, the record wins."""
    citations, _ = map_citations(
        [
            {
                "chunk_id": 7,
                "excerpt": "within four (4) hours of discovery",
                "document": "Some Other Policy",
                "page": 99,
                "clause": "1.1",
            }
        ],
        [chunk()],
    )

    assert citations[0]["document"] == "Data Protection and Privacy Policy"
    assert citations[0]["page"] == 3
    assert citations[0]["clause"] == "7.1"


def test_page_number_reaches_the_citation():
    citations, _ = map_citations(
        [{"chunk_id": 7, "excerpt": "within four (4) hours"}], [chunk(page=12)]
    )

    assert citations[0]["page"] == 12


# --- rejection paths ------------------------------------------------------------


def test_unknown_chunk_id_is_rejected():
    """A plausible integer for a chunk the model never saw must not address a row."""
    citations, rejected = map_citations(
        [{"chunk_id": 999, "excerpt": "within four (4) hours of discovery"}], [chunk()]
    )

    assert citations == []
    assert len(rejected) == 1
    assert "not in the retrieved set" in rejected[0]["reason"]


def test_excerpt_from_a_different_chunk_is_rejected():
    """The precise reason grounding is checked per-chunk, not against the whole context.

    The quote is real corpus text, but it is not in the chunk being cited — accepting it
    would produce a citation pointing at the wrong clause while quoting genuine text.
    """
    chunks = [chunk(chunk_id=7), chunk(chunk_id=8, content=RETENTION_TEXT, clause="5.1")]

    citations, rejected = map_citations(
        [{"chunk_id": 7, "excerpt": "retained for a minimum of seven (7) years"}], chunks
    )

    assert citations == []
    assert "does not appear in the cited chunk" in rejected[0]["reason"]


def test_same_excerpt_against_its_own_chunk_is_accepted():
    """Control for the test above: the identical quote, cited correctly, passes."""
    chunks = [chunk(chunk_id=7), chunk(chunk_id=8, content=RETENTION_TEXT, clause="5.1")]

    citations, rejected = map_citations(
        [{"chunk_id": 8, "excerpt": "retained for a minimum of seven (7) years"}], chunks
    )

    assert rejected == []
    assert citations[0]["chunk_id"] == 8
    assert citations[0]["clause"] == "5.1"


def test_invented_excerpt_is_rejected():
    citations, rejected = map_citations(
        [{"chunk_id": 7, "excerpt": "breaches must be reported within twenty-four hours"}],
        [chunk()],
    )

    assert citations == []
    assert rejected


def test_missing_or_malformed_chunk_id_is_rejected():
    for bad in (None, "not-a-number", True, {}):
        citations, rejected = map_citations(
            [{"chunk_id": bad, "excerpt": "within four (4) hours"}], [chunk()]
        )
        assert citations == [], bad
        assert rejected, bad


def test_numeric_string_chunk_id_is_accepted():
    citations, _ = map_citations(
        [{"chunk_id": "7", "excerpt": "within four (4) hours"}], [chunk()]
    )

    assert citations[0]["chunk_id"] == 7


# --- shape ----------------------------------------------------------------------


def test_duplicate_citations_are_collapsed():
    raw = [
        {"chunk_id": 7, "excerpt": "within four (4) hours of discovery"},
        {"chunk_id": 7, "excerpt": "within four (4) hours of discovery"},
    ]

    citations, _ = map_citations(raw, [chunk()])

    assert len(citations) == 1


def test_no_citations_returns_empty_lists():
    assert map_citations([], [chunk()]) == ([], [])
    assert map_citations(None, [chunk()]) == ([], [])


def test_partial_verification_keeps_the_good_citation():
    chunks = [chunk(chunk_id=7)]
    raw = [
        {"chunk_id": 7, "excerpt": "within four (4) hours of discovery"},
        {"chunk_id": 7, "excerpt": "a completely invented obligation"},
    ]

    citations, rejected = map_citations(raw, chunks)

    assert len(citations) == 1
    assert len(rejected) == 1
