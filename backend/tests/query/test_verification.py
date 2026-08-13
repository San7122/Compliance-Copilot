"""Grounding is the primitive citation mapping is built on.

If a fabricated quote passes this check, a fabricated citation reaches the user with a
real document, section and page attached to it — which reads as more trustworthy than
an uncited answer, not less. The tests therefore push from both directions: honest
quotes that were lightly reformatted must pass, and invented ones must not.
"""

from app.verification import excerpt_is_grounded, normalize

CHUNK_TEXT = (
    "Financial and billing records are retained for seven (7) years from the end of "
    "the fiscal year in which they were created, in line with statutory tax and audit "
    "requirements."
)

SOURCE = normalize(CHUNK_TEXT)


def test_exact_quote_is_grounded():
    assert excerpt_is_grounded(
        "Financial and billing records are retained for seven (7) years", SOURCE, 0.7
    )


def test_quote_with_reformatted_whitespace_is_grounded():
    """Models normalise line breaks when quoting; that isn't fabrication."""
    assert excerpt_is_grounded(
        "Financial and billing records\n   are retained for   seven (7) years", SOURCE, 0.7
    )


def test_quote_with_different_casing_is_grounded():
    assert excerpt_is_grounded(
        "FINANCIAL AND BILLING RECORDS ARE RETAINED FOR SEVEN (7) YEARS", SOURCE, 0.7
    )


def test_invented_quote_is_not_grounded():
    assert not excerpt_is_grounded(
        "Employees may carry over up to five unused vacation days.", SOURCE, 0.7
    )


def test_plausible_but_wrong_number_is_not_grounded():
    """The dangerous case: right topic, wrong figure. Must still be rejected."""
    assert not excerpt_is_grounded(
        "Financial and billing records are retained for three (3) months only.", SOURCE, 0.99
    )


def test_empty_excerpt_is_not_grounded():
    assert not excerpt_is_grounded("", SOURCE, 0.7)
    assert not excerpt_is_grounded("   ", SOURCE, 0.7)


def test_normalize_collapses_whitespace_and_case():
    assert normalize("  Seven   (7)\n YEARS ") == "seven (7) years"
