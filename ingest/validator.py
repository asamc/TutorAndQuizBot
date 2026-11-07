"""Accept or reject an uploaded file. TUT-14.

First gate in the pipeline, ahead of the parser: pypdf consumes hostile input
and should not see unchecked bytes. Runs in memory, before anything is written
to data/. Fails closed -- if the type cannot be determined, reject.
"""

from pathlib import PurePosixPath

from ingest.model import Outcome

# Enforced here even though Streamlit also caps uploads: the UI is not a trust
# boundary, and CLI ingestion bypasses it entirely.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB, matching docs/ingest-interface.md

ACCEPTED_SUFFIXES = (".pdf", ".txt")

# Checked against the actual bytes rather than trusting the extension. A .txt
# file has no signature, so it is validated by decoding.
PDF_MAGIC = b"%PDF-"

# Encoding a .txt upload must decode under. Shared with parser.py so a file
# cannot pass this gate and then fail extraction. utf-8-sig eats the BOM that
# Notepad writes. Rejects cp1252 -- the fail-closed direction, and the one false
# rejection to expect; a fallback would make the check vacuous.
TEXT_ENCODING = "utf-8-sig"


def validate(data: bytes, filename: str) -> Outcome | None:
    """Return the rejection Outcome, or None if the file is acceptable.

    Checks, in order:
      - size against MAX_UPLOAD_BYTES              -> REJECTED_SIZE
      - suffix against ACCEPTED_SUFFIXES           -> REJECTED_TYPE
      - content signature against the suffix       -> REJECTED_CONTENT_MISMATCH

    The content check is the one that matters: a .pdf whose bytes are not a PDF
    must be rejected, not handed to pypdf.

    Never log `data` or any slice of it. Rejections are logged as metadata only:
    document id, declared suffix, reason. Pipeline doc 4.5.
    """
    # Size first, so an oversized file is rejected on a len() rather than on a
    # 50 MB decode -- and reports the cap rather than its extension.
    if len(data) > MAX_UPLOAD_BYTES:
        return Outcome.REJECTED_SIZE

    suffix = declared_suffix(filename)
    if suffix not in ACCEPTED_SUFFIXES:
        return Outcome.REJECTED_TYPE

    if not _content_matches(data, suffix):
        return Outcome.REJECTED_CONTENT_MISMATCH

    return None


def declared_suffix(filename: str) -> str:
    """The lowercased extension the upload claims, or '' if it has none.

    Public because parser.py dispatches on the same answer. `filename` is user
    input and is never used to build a path here -- path safety belongs to
    manifest.storage_name (pipeline doc 4.1), so '../../etc/passwd.pdf' is a
    valid *type*.

    PurePosixPath, not Path, so a Windows-style path gives the same answer on
    every teammate's OS. Deliberately strict, since each of these is how an
    extension check gets bypassed:

      - 'notes.PDF'          -> '.pdf', accepted; case is not a signal
      - 'notes.pdf.exe'      -> '.exe', rejected; only the final suffix counts
      - 'notes.pdf\\x00.exe' -> '.exe', rejected; no truncation at the null
      - 'notes', '.pdf'      -> '',     rejected; undeterminable means reject
    """
    # Windows discards trailing dots and spaces, so strip them rather than
    # rejecting a real upload over a character the OS ignores.
    return PurePosixPath(filename.strip().rstrip(". ")).suffix.lower()


def _content_matches(data: bytes, suffix: str) -> bool:
    """True when the bytes are consistent with the accepted `suffix`.

    No third branch: the unreachable default returns False, so adding a suffix
    without adding its check fails closed.
    """
    if suffix == ".pdf":
        # Header required at offset 0. Readers tolerate it further in, but
        # scanning for it would accept a file whose first bytes are something
        # else -- the case this gate exists to catch.
        return data.startswith(PDF_MAGIC)
    if suffix == ".txt":
        return _looks_like_text(data)
    return False


def _looks_like_text(data: bytes) -> bool:
    """True when the bytes are plain text under TEXT_ENCODING.

    A .txt file has no magic number, so the decode is the signature. An empty
    file passes -- it is valid text -- and fails later as PARSE_FAILED.

    The null-byte check is not redundant: U+0000 is valid UTF-8, so a binary
    that happens to decode cleanly would otherwise pass.
    """
    if b"\x00" in data:
        return False
    try:
        data.decode(TEXT_ENCODING)
    except UnicodeDecodeError:
        # Swallowed, not logged: the message quotes the offending bytes.
        return False
    return True
