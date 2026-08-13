"""Chunking is the one place where a silent bug corrupts every downstream answer.

A wrong heading path doesn't crash anything — it just means every citation for that
document points at the wrong section, which in a compliance tool is worse than an
error. These tests pin the structural invariants.
"""

from pathlib import Path

from app.ingestion.pipeline import _extract_title, _split_into_sections, _word_windows, chunk_document

SAMPLE = """# Data Retention Policy

**Document owner:** Legal & Compliance

## 1. Purpose

Defines how long data is kept.

### 1.1 Scope

Applies to all employees.

## 2. Retention Periods

Financial records are kept for 7 years.
"""


def test_extract_title_uses_h1():
    assert _extract_title(SAMPLE, fallback="nope") == "Data Retention Policy"


def test_extract_title_falls_back_when_no_h1():
    assert _extract_title("no heading here", fallback="my-file") == "my-file"


def test_h1_is_not_duplicated_in_heading_path():
    """Regression: heading paths used to render as "X > X > 1. Purpose"."""
    sections = _split_into_sections(SAMPLE, "Data Retention Policy")
    for s in sections:
        assert not s.heading_path.startswith("Data Retention Policy > Data Retention Policy")


def test_heading_path_tracks_nesting():
    sections = _split_into_sections(SAMPLE, "Data Retention Policy")
    paths = [s.heading_path for s in sections]

    assert "Data Retention Policy > 1. Purpose > 1.1 Scope" in paths
    assert "Data Retention Policy > 2. Retention Periods" in paths


def test_deeper_heading_does_not_leak_into_later_sibling():
    """After 1.1 Scope closes, section 2 must not inherit it."""
    sections = _split_into_sections(SAMPLE, "Data Retention Policy")
    section_2 = next(s for s in sections if "Retention Periods" in s.heading_path)
    assert "1.1 Scope" not in section_2.heading_path


def test_front_matter_before_first_subheading_is_kept():
    """The owner/effective-date block sits under the H1 with no H2 — don't drop it."""
    sections = _split_into_sections(SAMPLE, "Data Retention Policy")
    bodies = " ".join(s.body for s in sections)
    assert "Legal & Compliance" in bodies


def test_short_text_is_a_single_window():
    assert _word_windows("a b c", max_tokens=500, overlap_tokens=50) == ["a b c"]


def test_long_text_is_split_with_overlap():
    text = " ".join(f"w{i}" for i in range(100))
    windows = _word_windows(text, max_tokens=20, overlap_tokens=8)

    assert len(windows) > 1
    # Consecutive windows must share their overlap region, or a fact sitting on the
    # boundary loses its context.
    first_tail = windows[0].split()[-6:]
    assert all(w in windows[1].split() for w in first_tail)


def test_windows_cover_the_whole_text():
    text = " ".join(f"w{i}" for i in range(100))
    windows = _word_windows(text, max_tokens=20, overlap_tokens=8)
    covered = {w for window in windows for w in window.split()}
    assert covered == set(text.split())


def test_overlap_larger_than_window_terminates():
    """Misconfigured overlap must degrade chunking, not hang ingestion in a loop."""
    text = " ".join(f"w{i}" for i in range(50))
    windows = _word_windows(text, max_tokens=10, overlap_tokens=999)
    assert len(windows) > 1
    assert windows[-1].split()[-1] == "w49"


def test_markdown_document_round_trips_through_chunk_document(tmp_path):
    """The markdown path is retained for non-PDF sources; keep it exercised.

    The real corpus is PDFs, so this writes its own fixture rather than depending on
    sample markdown existing in docs/.
    """
    path = tmp_path / "sample-policy.md"
    path.write_text(SAMPLE, encoding="utf-8")

    parsed = chunk_document(path)

    assert parsed.title == "Data Retention Policy"
    assert parsed.meta is None  # markdown carries no front-matter identity
    assert parsed.sections
    for section in parsed.sections:
        assert section.body.strip()
        assert section.heading_path.startswith(parsed.title)
        assert section.clause is None
