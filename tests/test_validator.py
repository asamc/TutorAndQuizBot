"""Attack the validator gate. TUT-14. See docs/ingest-pipeline.md section 7.

Mostly bypass attempts: extension spoofing, double extensions, binaries renamed
to .txt, and the precedence between the three checks. Every case is a byte
literal, so these run in CI where data/ is empty.
"""

import pytest

from ingest.model import Outcome
from ingest.validator import (
    ACCEPTED_SUFFIXES,
    MAX_UPLOAD_BYTES,
    validate,
)

# Smallest byte strings that satisfy each content rule. Not valid documents --
# the validator checks the signature, and parsing is TUT-15's problem.
PDF_BYTES = b"%PDF-1.7\n1 0 obj\n<< >>\nendobj\ntrailer\n%%EOF\n"
TXT_BYTES = b"Confidentiality, integrity, availability.\n"


# ---------------------------------------------------------------------------
# Accepted
# ---------------------------------------------------------------------------


def test_pdf_accepted():
    assert validate(PDF_BYTES, "Lecture_1_slides.pdf") is None


def test_txt_accepted():
    assert validate(TXT_BYTES, "notes.txt") is None


@pytest.mark.parametrize("filename", ["NOTES.TXT", "Lecture_1.PDF", "notes.Pdf"])
def test_suffix_case_is_not_a_signal(filename):
    """An uppercase extension is the same type, not an unknown one."""
    data = PDF_BYTES if filename.lower().endswith(".pdf") else TXT_BYTES
    assert validate(data, filename) is None


def test_utf8_bom_accepted():
    """Notepad and Excel both write a BOM; it must not be a rejection."""
    assert validate(b"\xef\xbb\xbf" + TXT_BYTES, "notes.txt") is None


def test_non_ascii_utf8_accepted():
    assert validate("Diffie–Hellman key exchange\n".encode(), "notes.txt") is None


def test_empty_txt_accepted_and_left_to_the_parser():
    """An empty file is valid text. It fails later as PARSE_FAILED, which
    reports "no extractable text" rather than an invented type error."""
    assert validate(b"", "empty.txt") is None


def test_windows_trailing_dot_and_space_accepted():
    """Characters the Windows filesystem itself discards must not reject a real
    upload: 'notes.pdf.' and 'notes.pdf ' name the same file as 'notes.pdf'."""
    assert validate(PDF_BYTES, "notes.pdf.") is None
    assert validate(PDF_BYTES, "notes.pdf ") is None


def test_traversal_filename_is_a_valid_type():
    """The validator gates *type*, not path safety. A traversal filename with
    real PDF bytes is an acceptable PDF; refusing to write under that name is
    manifest.storage_name's job -- see docs/ingest-pipeline.md 4.1. Asserted so
    that nobody 'fixes' this by adding a path check in the wrong module."""
    assert validate(PDF_BYTES, "../../.ssh/authorized_keys.pdf") is None


# ---------------------------------------------------------------------------
# REJECTED_SIZE
# ---------------------------------------------------------------------------


def test_oversized_rejected():
    assert validate(bytes(MAX_UPLOAD_BYTES + 1), "big.pdf") is Outcome.REJECTED_SIZE


def test_exactly_at_cap_is_not_oversized():
    """50 MB is the cap, not the first rejected size. These bytes are not a PDF,
    so reaching REJECTED_CONTENT_MISMATCH proves the size check let them past."""
    data = bytes(MAX_UPLOAD_BYTES)
    assert validate(data, "big.pdf") is Outcome.REJECTED_CONTENT_MISMATCH


def test_size_is_checked_before_type():
    """Ordering is user-visible: an oversized upload reports the size cap even
    when its extension is also wrong."""
    assert validate(bytes(MAX_UPLOAD_BYTES + 1), "huge.exe") is Outcome.REJECTED_SIZE


# ---------------------------------------------------------------------------
# REJECTED_TYPE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename",
    [
        "report.docx",  # listed in TUT-14, not implemented; see pipeline doc 8
        "slides.pptx",
        "payload.exe",
        "script.py",
        "archive.zip",
    ],
)
def test_unsupported_extensions_rejected(filename):
    assert validate(PDF_BYTES, filename) is Outcome.REJECTED_TYPE


def test_docx_rejects_as_type_not_as_a_parse_failure():
    """DOCX is refused at the gate on purpose. Accepting it would surface later
    as PARSE_FAILED and read like a corrupt document rather than a missing
    feature. Delete this test when python-docx lands."""
    assert ".docx" not in ACCEPTED_SUFFIXES
    assert validate(b"PK\x03\x04", "report.docx") is Outcome.REJECTED_TYPE


@pytest.mark.parametrize(
    "filename",
    [
        "notes.pdf.exe",  # only the final suffix counts
        "notes.pdf\x00.exe",  # no truncation at the null byte
        "notes",  # no suffix at all: undeterminable means reject
        ".pdf",  # a dotfile, not a PDF
        "",  # empty filename
        "notes.",  # bare trailing dot
    ],
)
def test_extension_bypass_attempts_rejected(filename):
    assert validate(PDF_BYTES, filename) is Outcome.REJECTED_TYPE


# ---------------------------------------------------------------------------
# REJECTED_CONTENT_MISMATCH
# ---------------------------------------------------------------------------


def test_pdf_extension_with_non_pdf_content_rejected():
    """The case the gate exists for: these bytes never reach pypdf.
    docs/ingest-pipeline.md section 6, first row."""
    assert validate(b"MZ\x90\x00" * 64, "malware.pdf") is Outcome.REJECTED_CONTENT_MISMATCH


def test_pdf_header_not_at_offset_zero_rejected():
    """Readers tolerate leading junk; this gate does not. Widening it is a
    documented one-line change in _looks_like_pdf, not a silent one."""
    assert validate(b"\n\n" + PDF_BYTES, "notes.pdf") is Outcome.REJECTED_CONTENT_MISMATCH


def test_empty_pdf_rejected():
    assert validate(b"", "empty.pdf") is Outcome.REJECTED_CONTENT_MISMATCH


def test_pdf_renamed_to_txt_rejected():
    """A PDF carries binary stream data, so it does not decode as text."""
    body = b"%PDF-1.7\nstream\n\x80\x81\x82\xfe\xff\nendstream\n"
    assert validate(body, "notes.txt") is Outcome.REJECTED_CONTENT_MISMATCH


def test_txt_with_invalid_utf8_rejected():
    assert validate(b"\xff\xfe\x00\x41", "notes.txt") is Outcome.REJECTED_CONTENT_MISMATCH


def test_txt_with_null_byte_rejected():
    """U+0000 is valid UTF-8, so the decode alone would pass this. No real text
    document contains a null byte."""
    assert validate(b"hello\x00world", "notes.txt") is Outcome.REJECTED_CONTENT_MISMATCH


# ---------------------------------------------------------------------------
# Fail-closed and non-mutation
# ---------------------------------------------------------------------------


def test_every_accepted_suffix_has_a_content_rule():
    """Adding a suffix to ACCEPTED_SUFFIXES without adding its content check
    must not let the file through on the extension alone. Each accepted suffix
    is asserted to reject bytes that are not of its type."""
    for suffix in ACCEPTED_SUFFIXES:
        outcome = validate(b"\x7fELF\x02\x01\x01\x00", f"probe{suffix}")
        assert outcome is Outcome.REJECTED_CONTENT_MISMATCH, suffix


def test_validate_does_not_modify_the_bytes():
    """The screener's non-mutation rule starts here: nothing in ingest edits
    document content. docs/ingest-pipeline.md 4.4."""
    data = bytearray(PDF_BYTES)
    before = bytes(data)
    validate(bytes(data), "notes.pdf")
    assert bytes(data) == before
