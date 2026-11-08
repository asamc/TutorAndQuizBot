"""Attack the parser. TUT-15. See docs/ingest-pipeline.md section 7.

Fixtures: synthetic PDFs from pypdf's own writer (no fixture binaries, no new
dependency), corrupted slices of a real deck, plain strings for .txt, and the
real decks behind a skipif -- data/ is gitignored, so those skip in CI.

Because they skip in CI, no security property may rest on a real deck alone.
The leak guards in particular are synthetic and run everywhere, with the real
decks asserting the same property as well rather than instead. Real-deck tests
are parametrised over every deck via each_real_deck; see the note on it for what
indexing REAL_DECKS cost us.
"""

import io
import logging
import pathlib

import pytest
from pypdf import PdfReader, PdfWriter

from ingest.model import Chunk, PageRecord
from ingest.parser import (
    MAX_OUTLINE_DEPTH,
    MESSAGE_ENCRYPTED,
    MESSAGE_NO_TEXT,
    MESSAGE_TOO_MANY_PAGES,
    MESSAGE_TOO_MUCH_TEXT,
    MESSAGE_UNREADABLE,
    TXT_PAGE_TARGET_CHARS,
    ParseFailed,
    _extract_headings,
    parse,
)

DOCS = pathlib.Path(__file__).resolve().parents[1] / "data" / "documents"
REAL_DECKS = sorted(DOCS.glob("*.pdf")) if DOCS.is_dir() else []

needs_real_decks = pytest.mark.skipif(
    not REAL_DECKS, reason="data/documents is gitignored and empty on this clone"
)

# Real-deck assertions run against every deck, never REAL_DECKS[0].
#
# Indexing this list is a trap. It is a sorted glob, so which deck is under test
# depends on whatever happens to be sitting in data/documents, and the decks are
# not interchangeable: Lecture_1 to 5 carry a classic xref table while Lecture 6
# to 27 carry PDF 1.5 cross-reference streams. " " sorts before "_", so adding
# the later lectures moved REAL_DECKS[0] from the first generation to the second
# and broke a leak test -- silently, because data/ is gitignored and the test
# was guarded out of CI. Parametrising means a new deck widens the coverage
# instead of changing the subject.
each_real_deck = pytest.mark.parametrize("deck", REAL_DECKS, ids=lambda path: path.name)


def blank_pdf(page_count: int = 1, outline: list[tuple[str, int]] | None = None) -> bytes:
    """A structurally valid PDF whose pages carry no text."""
    writer = PdfWriter()
    for _ in range(page_count):
        writer.add_blank_page(width=200, height=200)
    for title, page_index in outline or []:
        writer.add_outline_item(title, page_index)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# The silent failure: no text, no exception
# ---------------------------------------------------------------------------


def test_all_blank_pdf_fails_rather_than_reporting_success():
    """extract_text() returns "" for an image-only page and does not raise, so
    an all-scanned PDF would otherwise parse "successfully" and produce zero
    chunks. docs/ingest-pipeline.md section 5."""
    with pytest.raises(ParseFailed) as excinfo:
        parse(blank_pdf(5), "scanned.pdf")
    assert str(excinfo.value) == MESSAGE_NO_TEXT


def test_empty_txt_fails_as_no_text():
    with pytest.raises(ParseFailed) as excinfo:
        parse(b"", "empty.txt")
    assert str(excinfo.value) == MESSAGE_NO_TEXT


def test_whitespace_only_txt_fails_as_no_text():
    with pytest.raises(ParseFailed):
        parse(b"\n\n   \n\t\n", "blank.txt")


# ---------------------------------------------------------------------------
# Malformed input. Note most of these raise from PdfReader(), not extract_text()
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name,data",
    [
        ("empty", b""),
        ("header only", b"%PDF-1.7\n"),
        ("header plus garbage", b"%PDF-1.7\n" + b"\xde\xad\xbe\xef" * 256),
        ("no startxref", b"%PDF-1.7\n" + b"[" * 2000 + b"]" * 2000 + b"\n%%EOF\n"),
    ],
)
def test_malformed_pdf_raises_parse_failed(name, data):
    with pytest.raises(ParseFailed) as excinfo:
        parse(data, "broken.pdf")
    assert str(excinfo.value) == MESSAGE_UNREADABLE, name


@each_real_deck
@pytest.mark.parametrize("fraction", [0.1, 0.5, 0.99])
def test_truncated_real_pdf_raises_parse_failed(deck, fraction):
    data = deck.read_bytes()
    with pytest.raises(ParseFailed):
        parse(data[: int(len(data) * fraction)], "truncated.pdf")


@each_real_deck
def test_damaged_trailer_raises_parse_failed(deck):
    # Equal length and every occurrence, for the reasons PLANTED_PHRASE records.
    data = deck.read_bytes().replace(b"trailer", b"xxxxxxx")
    with pytest.raises(ParseFailed):
        parse(data, "damaged.pdf")


def test_unsupported_suffix_fails_closed():
    """Unreachable via the pipeline, since validate() gates the suffix. Asserted
    so a direct or CLI call cannot fall through the dispatch."""
    with pytest.raises(ParseFailed):
        parse(blank_pdf(1), "notes.docx")


def test_encrypted_pdf_reports_encryption_not_missing_text():
    """'No extractable text; may be scanned images' would be actively
    misleading for a password-protected file."""
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.encrypt("hunter2")
    buf = io.BytesIO()
    writer.write(buf)
    with pytest.raises(ParseFailed) as excinfo:
        parse(buf.getvalue(), "locked.pdf")
    assert str(excinfo.value) == MESSAGE_ENCRYPTED


# ---------------------------------------------------------------------------
# Leak channels. docs/ingest-pipeline.md 4.5
# ---------------------------------------------------------------------------


# Seven bytes, so it can overwrite b"trailer" without moving anything after it.
#
# How the document gets damaged is load-bearing, and the obvious way is wrong.
# A replacement of a different length shifts every byte offset that follows it,
# and replacing only the first occurrence leaves any later one intact -- a PDF
# carrying cross-reference streams then recovers, parses cleanly, and the test
# passes while proving nothing. Equal length and every occurrence instead, which
# also puts the phrase exactly where pypdf chokes. That is what gets it into
# pypdf's message, which is the whole thing being guarded against.
PLANTED_PHRASE = b"SECRET!"
assert len(PLANTED_PHRASE) == len(b"trailer")


def plant_and_damage(pdf: bytes) -> bytes:
    """A PDF that cannot be parsed, with PLANTED_PHRASE at the failure site."""
    return pdf.replace(b"trailer", PLANTED_PHRASE)


def test_parse_failed_message_never_quotes_the_document():
    """The highest-value test in this file, and the reason it is synthetic.

    pypdf's own error quotes ~100 bytes of the file it choked on. ParseFailed
    must carry a fixed string instead, and the original must not survive on
    __context__ or __cause__ either, where a traceback-logging handler would
    print it.

    This used to run only against the real decks, which meant it never ran in
    CI at all: data/ is gitignored, so the skipif fired on every clone and every
    runner. A synthetic PDF from pypdf's own writer reproduces the property
    exactly, so the guard now runs on every push on all three operating systems.
    The real decks are covered by the test below as well, not instead.
    """
    planted = plant_and_damage(blank_pdf(1))

    # Guard the premise. If pypdf stops quoting file bytes, every assertion
    # below would still pass while testing nothing at all.
    with pytest.raises(Exception) as leaked:
        PdfReader(io.BytesIO(planted))
    assert PLANTED_PHRASE.decode() in str(leaked.value), (
        "pypdf no longer leaks document bytes into its error; this test has "
        "become vacuous and the fixture needs rebuilding"
    )

    with pytest.raises(ParseFailed) as excinfo:
        parse(planted, "planted.pdf")

    error = excinfo.value
    assert PLANTED_PHRASE.decode() not in str(error)
    assert str(error) == MESSAGE_UNREADABLE
    assert error.__cause__ is None
    assert error.__context__ is None, "chained pypdf error still carries file bytes"


@each_real_deck
def test_parse_failed_never_quotes_a_real_deck(deck):
    """The same property on real course material.

    Not redundant with the synthetic test: these decks are a different PDF
    generation. pypdf's writer emits a classic xref table, while most of the
    lectures carry PDF 1.5 cross-reference streams, and that difference is
    exactly what broke the original version of this test.
    """
    with pytest.raises(ParseFailed) as excinfo:
        parse(plant_and_damage(deck.read_bytes()), "planted.pdf")

    error = excinfo.value
    assert PLANTED_PHRASE.decode() not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None, "chained pypdf error still carries file bytes"


def test_pypdf_logger_does_not_reach_application_logs(caplog):
    """pypdf logs warnings of its own, independent of this module: parsing a
    real deck emits "EOF marker not found". Those must not land in the
    application's log, where 4.5's metadata-only rule applies."""
    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ParseFailed):
            parse(b"%PDF-1.7\n", "broken.pdf")
    assert [r for r in caplog.records if r.name.startswith("pypdf")] == []


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_page_count_bound(monkeypatch):
    monkeypatch.setattr("ingest.parser.MAX_PAGES", 3)
    with pytest.raises(ParseFailed) as excinfo:
        parse(blank_pdf(4), "long.pdf")
    # Refused on the page count, before any text is pulled, so the all-blank
    # pages never get the chance to report MESSAGE_NO_TEXT instead.
    assert str(excinfo.value) == MESSAGE_TOO_MANY_PAGES


@each_real_deck
def test_extracted_char_bound(deck, monkeypatch):
    monkeypatch.setattr("ingest.parser.MAX_EXTRACTED_CHARS", 100)
    with pytest.raises(ParseFailed) as excinfo:
        parse(deck.read_bytes(), deck.name)
    assert str(excinfo.value) == MESSAGE_TOO_MUCH_TEXT


def test_txt_char_bound(monkeypatch):
    monkeypatch.setattr("ingest.parser.MAX_EXTRACTED_CHARS", 100)
    with pytest.raises(ParseFailed) as excinfo:
        parse(b"word " * 100, "big.txt")
    assert str(excinfo.value) == MESSAGE_TOO_MUCH_TEXT


def test_txt_page_count_bound(monkeypatch):
    monkeypatch.setattr("ingest.parser.MAX_PAGES", 2)
    body = "\n\n".join("x" * TXT_PAGE_TARGET_CHARS for _ in range(5))
    with pytest.raises(ParseFailed) as excinfo:
        parse(body.encode(), "long.txt")
    assert str(excinfo.value) == MESSAGE_TOO_MANY_PAGES


def test_outline_nesting_is_depth_bounded():
    """Nesting is attacker-controlled. Deeper than the limit is ignored, not a
    RecursionError out of a module whose job is surviving hostile input."""

    class FakeReader:
        outline = None

        def get_destination_page_number(self, item):
            return 0

    nested: object = [type("D", (), {"title": "too deep"})()]
    for _ in range(MAX_OUTLINE_DEPTH + 2):
        nested = [nested]
    reader = FakeReader()
    reader.outline = nested
    assert _extract_headings(reader) == {}


# ---------------------------------------------------------------------------
# .txt pagination
# ---------------------------------------------------------------------------


def test_short_txt_is_one_page():
    pages = parse(b"Confidentiality, integrity, availability.\n", "notes.txt")
    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].heading is None


def test_txt_splits_on_blank_lines_into_several_pages():
    body = "\n\n".join(f"Paragraph {i}. " + "word " * 200 for i in range(6))
    pages = parse(body.encode(), "notes.txt")
    assert len(pages) > 1
    assert [p.page_number for p in pages] == list(range(1, len(pages) + 1))
    assert all(len(p.text) <= TXT_PAGE_TARGET_CHARS * 2 for p in pages)


def test_long_txt_without_blank_lines_is_still_split():
    """The case the blank-line rule alone would silently fail on: one long
    wrapped block would otherwise become a single unbounded page, which is one
    unbounded chunk."""
    pages = parse(("word " * 4000).encode(), "wrapped.txt")
    assert len(pages) > 1
    assert all(len(p.text) <= TXT_PAGE_TARGET_CHARS for p in pages)


def test_txt_single_token_longer_than_a_page_is_split():
    """A long line with no whitespace must not grow a page until it finds one."""
    pages = parse(("x" * (TXT_PAGE_TARGET_CHARS * 3)).encode(), "nospaces.txt")
    assert len(pages) >= 3
    assert all(len(p.text) <= TXT_PAGE_TARGET_CHARS for p in pages)


def test_txt_pagination_loses_no_words():
    """Splitting is structural, not editorial: pages restructure whitespace, but
    every word survives in order."""
    body = "\n\n".join(f"Section {i} discusses availability." for i in range(50))
    pages = parse(body.encode(), "notes.txt")
    assert " ".join(" ".join(p.text.split()) for p in pages) == " ".join(body.split())


def test_txt_bom_is_consumed():
    pages = parse(b"\xef\xbb\xbfIntegrity assures data is unaltered.", "notes.txt")
    assert not pages[0].text.startswith("﻿")


def test_txt_invalid_encoding_fails_closed():
    """validate() screens this first, but parse() is also called directly."""
    with pytest.raises(ParseFailed) as excinfo:
        parse(b"\xff\xfe\x00\x41", "notes.txt")
    assert str(excinfo.value) == MESSAGE_UNREADABLE


# ---------------------------------------------------------------------------
# Headings: verbatim, and tolerant
# ---------------------------------------------------------------------------


def test_headings_absent_when_pdf_has_no_outline():
    pdf = blank_pdf(2)
    from pypdf import PdfReader

    assert _extract_headings(PdfReader(io.BytesIO(pdf))) == {}


def test_heading_is_verbatim_including_its_slide_prefix():
    """The decks' slide labels disagree with their PDF page numbers, so the
    outline title is the only record of the number the student sees. It is
    stored as-is: not reconstructed from the page index, not split on the
    colon."""
    from pypdf import PdfReader

    pdf = blank_pdf(3, outline=[("Slide 17: Example: two points", 2)])
    headings = _extract_headings(PdfReader(io.BytesIO(pdf)))
    assert headings == {3: "Slide 17: Example: two points"}


def test_first_bookmark_wins_for_a_repeated_page():
    from pypdf import PdfReader

    pdf = blank_pdf(2, outline=[("Section", 0), ("Subsection", 0)])
    assert _extract_headings(PdfReader(io.BytesIO(pdf))) == {1: "Section"}


def test_citation_prefers_heading_over_page_number():
    chunk = Chunk(
        document_name="Lecture_2_slides.pdf",
        page_number=4,
        heading="Slide 11: OSI Security Architecture",
        text="...",
    )
    assert chunk.citation() == (
        "Lecture_2_slides.pdf, Slide 11: OSI Security Architecture"
    )


def test_citation_falls_back_to_page_number_without_a_heading():
    chunk = Chunk(document_name="notes.txt", page_number=3, heading=None, text="...")
    assert chunk.citation() == "notes.txt, Page 3"


# ---------------------------------------------------------------------------
# The real decks
# ---------------------------------------------------------------------------


@needs_real_decks
def test_real_decks_parse_to_the_measured_shape():
    expected = {"Lecture_1_slides.pdf": 11, "Lecture_2_slides.pdf": 10}
    for deck in REAL_DECKS:
        if deck.name not in expected:
            continue
        pages = parse(deck.read_bytes(), deck.name)
        assert len(pages) == expected[deck.name], deck.name
        assert [p.page_number for p in pages] == list(range(1, len(pages) + 1))
        # One near-empty page in each: the shared 'A Hyperledger' image slide.
        assert sum(p.is_empty for p in pages) == 1, deck.name


@needs_real_decks
def test_real_deck_heading_records_the_slide_number_the_student_sees():
    deck = DOCS / "Lecture_2_slides.pdf"
    if not deck.exists():
        pytest.skip("Lecture_2_slides.pdf not present")
    pages = parse(deck.read_bytes(), deck.name)
    # PDF page 4 is the deck's slide 11 -- the mismatch citations depend on.
    page_four = pages[3]
    assert page_four.heading == "Slide 11: OSI Security Architecture"
    assert page_four.page_number == 4


@each_real_deck
def test_real_deck_text_is_not_normalised(deck):
    """The screener's non-mutation rule starts here: whatever pypdf returns for
    a page is what gets stored."""
    data = deck.read_bytes()
    pages = parse(data, deck.name)
    raw = PdfReader(io.BytesIO(data)).pages[1].extract_text()
    assert pages[1].text == raw


def test_pagerecord_is_the_returned_type():
    pages = parse(b"Availability assures timely access.", "notes.txt")
    assert all(isinstance(p, PageRecord) for p in pages)
