"""Extract text from an accepted file. TUT-15.

One PageRecord per page: a flat string loses the page boundaries that citations
and page-located flags need. Parses untrusted input, so it is bounded and never
lets a library exception reach the user. Design: docs/ingest-pipeline.md 4.2.
"""

import io
import logging

from pypdf import PdfReader

from ingest.model import PageRecord
from ingest.validator import TEXT_ENCODING, declared_suffix

# Bounds against malformed and hostile files. docs/ingest-pipeline.md 4.2.
# The char bound is the one that matters: a size cap does not bound memory.
# Real decks are 11 pages / 2,217 chars, so only synthetic fixtures reach these.
MAX_PAGES = 500
MAX_EXTRACTED_CHARS = 2_000_000

# NOT ENFORCED -- no portable in-process implementation exists, and a control in
# name only is worse than a recorded gap. Accepted risk; pipeline doc 4.2 and 8.
PARSE_TIMEOUT_SECONDS = 30  # unenforced; see above

# .txt has no pages, so we make them -- one unbounded chunk would be useless for
# retrieval. 1500 is a guess pending the embedding model. Pipeline doc 5.2.
TXT_PAGE_TARGET_CHARS = 1500

# Outline nesting is attacker-controlled; bound it. Real decks are flat.
MAX_OUTLINE_DEPTH = 8

# Fixed strings, never built from a caught exception: a pypdf error quotes the
# bytes it choked on. Pipeline doc 4.5.
MESSAGE_UNREADABLE = "The document could not be read. It may be corrupt or incomplete."
MESSAGE_ENCRYPTED = (
    "The document is password-protected. Remove the password and upload it again."
)
MESSAGE_NO_TEXT = "No extractable text was found. The document may be scanned images."
MESSAGE_TOO_MANY_PAGES = f"The document has more than {MAX_PAGES} pages or sections."
MESSAGE_TOO_MUCH_TEXT = "The document contains more text than can be ingested."
MESSAGE_UNSUPPORTED = "This file type cannot be read."


def _silence_pypdf_logger() -> None:
    """Stop pypdf writing to the application's logs. Pipeline doc 4.5.

    The level is the load-bearing line, not propagate: anything may attach a
    handler to the pypdf logger directly (pytest's plugin does). Children
    inherit the level, so the record is never constructed. Stopgap -- belongs in
    a central logging config, pipeline doc 8.
    """
    pypdf_logger = logging.getLogger("pypdf")
    pypdf_logger.setLevel(logging.CRITICAL + 1)
    pypdf_logger.addHandler(logging.NullHandler())
    pypdf_logger.propagate = False


_silence_pypdf_logger()


class ParseFailed(Exception):
    """Extraction failed. The message must be display-safe: no stack traces, no
    library internals, no document content. Shown to the user as Result.message."""


def parse(data: bytes, filename: str) -> list[PageRecord]:
    """Extract text as a list of PageRecord, one per page, 1-based.

    Dispatches on the suffix the validator accepted: PDFs through pypdf, .txt
    split into ~TXT_PAGE_TARGET_CHARS pages.

    Raises ParseFailed when a bound is exceeded, the file cannot be parsed, or
    *every* page is empty. That last case is the silent failure worth guarding:
    extract_text() returns "" for an image-only page and does not raise, so an
    all-scanned PDF would otherwise report success with zero chunks. Per-page
    emptiness is normal -- one near-empty page in each real deck. Pipeline doc 5.
    """
    suffix = declared_suffix(filename)

    if suffix == ".pdf":
        pages = _parse_pdf(data)
    elif suffix == ".txt":
        pages = _parse_txt(data)
    else:
        # Unreachable via the pipeline (validate() gates the suffix), but parse()
        # is also called directly, and an unknown type must fail closed.
        raise ParseFailed(MESSAGE_UNSUPPORTED)

    # all() of an empty list is True, so this covers "no pages at all" too.
    if all(page.is_empty for page in pages):
        raise ParseFailed(MESSAGE_NO_TEXT)

    return pages


def _guarded(call, message: str):
    """Run `call`, converting any exception out of it into ParseFailed.

    The raise happens *after* the except block exits. `raise X from None` only
    suppresses the display of the chained exception; it stays on __context__,
    where an error reporter that walks the chain prints the document bytes pypdf
    quoted. Pipeline doc 4.5.

    Catches Exception, not PyPdfError: DependencyError derives straight from
    Exception, and malformed input also surfaces as RecursionError, struct.error
    or zlib.error from inside the library. Never binds or logs the exception.
    """
    failed = False
    try:
        return call()
    except Exception:
        failed = True
    if failed:
        raise ParseFailed(message)


def _parse_pdf(data: bytes) -> list[PageRecord]:
    """Extract one PageRecord per PDF page.

    Every pypdf call goes through _guarded, including the constructor -- that is
    where truncated, empty and trailer-damaged files actually raise, not in
    extract_text(). Pipeline doc 4.2.
    """
    reader = _guarded(lambda: PdfReader(io.BytesIO(data)), MESSAGE_UNREADABLE)

    # Before the page count, not alongside it: reading .pages on an encrypted
    # file raises, which would report "corrupt" for a merely locked file.
    if _guarded(lambda: reader.is_encrypted, MESSAGE_UNREADABLE):
        raise ParseFailed(MESSAGE_ENCRYPTED)

    page_count = _guarded(lambda: len(reader.pages), MESSAGE_UNREADABLE)

    # Known before any text is pulled, so this refuses an oversized document
    # before it allocates anything.
    if page_count > MAX_PAGES:
        raise ParseFailed(MESSAGE_TOO_MANY_PAGES)

    headings = _extract_headings(reader)

    pages: list[PageRecord] = []
    total_chars = 0

    for index in range(page_count):
        try:
            text = reader.pages[index].extract_text() or ""
        except Exception:
            # One bad page degrades to empty rather than failing the document;
            # if every page raises, parse()'s all-empty rule still rejects it.
            # Swallowed, not logged: the message is document content.
            text = ""

        # Accumulated inside the loop, so a bomb is stopped partway rather than
        # after the whole document is in memory. One page is the granularity.
        total_chars += len(text)
        if total_chars > MAX_EXTRACTED_CHARS:
            raise ParseFailed(MESSAGE_TOO_MUCH_TEXT)

        pages.append(
            PageRecord(
                page_number=index + 1,
                heading=headings.get(index + 1),
                # Verbatim: nothing in ingest edits document text (4.4).
                text=text,
            )
        )

    return pages


def _extract_headings(reader: PdfReader) -> dict[int, str]:
    """Map 1-based page number to its outline title, where the PDF has one.

    The decks carry one bookmark per slide holding its title, which is where
    citations like 'Slide 6: Integrity' come from.

    The title is stored verbatim, including its own "Slide N:" prefix: the decks'
    slide labels do not match their PDF page numbers (hidden slides), so
    rebuilding the prefix from the page index would cite the wrong slide, and
    splitting on the colon breaks on titles holding a second one. Pipeline 5.1.

    Headings are a nicety, so any failure returns what was collected. Tolerates
    no outline, nested entries, and untitled ones. get_destination_page_number()
    is 0-based; add 1.
    """
    try:
        outline = reader.outline
    except Exception:
        return {}

    headings: dict[int, str] = {}

    for item in _flatten_outline(outline):
        try:
            page_index = reader.get_destination_page_number(item)
            title = (getattr(item, "title", None) or "").strip()
        except Exception:
            # A broken destination costs one heading, not the document.
            continue

        # Untitled bookmarks exist in the real decks and add no location.
        if not title:
            continue

        # First bookmark naming a page wins -- with nesting, the outermost entry.
        headings.setdefault(page_index + 1, title)

    return headings


def _flatten_outline(items: object, depth: int = 0) -> list[object]:
    """Flatten pypdf's nested outline into a list of destinations.

    Depth-bounded: nesting comes from the file, so a deeply nested outline would
    otherwise exhaust the recursion limit.
    """
    if depth > MAX_OUTLINE_DEPTH or not isinstance(items, list):
        return []

    flat: list[object] = []
    for item in items:
        if isinstance(item, list):
            flat += _flatten_outline(item, depth + 1)
        else:
            flat.append(item)
    return flat


def _parse_txt(data: bytes) -> list[PageRecord]:
    """Split a text file into page-sized PageRecords. Pipeline doc 5.2.

    Decoded under validator.TEXT_ENCODING, the rule that accepted the file.
    Re-checked rather than assumed, since parse() is also called directly.

    Splitting is structural, not editorial: no word is dropped or reordered.
    Headings are None, so these cite as 'Page N'.
    """
    # _guarded, so UnicodeDecodeError -- which quotes the offending bytes -- is
    # neither logged nor left on the raised exception.
    text = _guarded(lambda: data.decode(TEXT_ENCODING), MESSAGE_UNREADABLE)

    # On the decoded length, before splitting allocates.
    if len(text) > MAX_EXTRACTED_CHARS:
        raise ParseFailed(MESSAGE_TOO_MUCH_TEXT)

    pages = [
        PageRecord(page_number=number, heading=None, text=body)
        for number, body in enumerate(_paginate_text(text), start=1)
    ]

    if len(pages) > MAX_PAGES:
        raise ParseFailed(MESSAGE_TOO_MANY_PAGES)

    return pages


def _paginate_text(text: str) -> list[str]:
    """Pack paragraphs into ~TXT_PAGE_TARGET_CHARS blocks on blank-line bounds.

    An over-long paragraph is hard-split, because one long wrapped block is
    common and is exactly the input a blank-line-only rule silently fails on.
    """
    blocks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in text.split("\n\n"):
        if not paragraph.strip():
            continue

        for piece in _hard_split(paragraph, TXT_PAGE_TARGET_CHARS):
            if current and current_len + len(piece) > TXT_PAGE_TARGET_CHARS:
                blocks.append("\n\n".join(current))
                current, current_len = [], 0
            current.append(piece)
            current_len += len(piece)

    if current:
        blocks.append("\n\n".join(current))

    return blocks


def _hard_split(paragraph: str, limit: int) -> list[str]:
    """Break one over-long paragraph at whitespace, or mid-token if it has none.

    The mid-token fallback matters: a 2 MB line with no spaces is the shape of
    input that makes a "split on whitespace" implementation return one huge piece.
    """
    if len(paragraph) <= limit:
        return [paragraph]

    pieces: list[str] = []
    remaining = paragraph

    while len(remaining) > limit:
        window = remaining[:limit]
        cut = window.rfind(" ")
        # No space in the window: cut at the limit rather than growing the piece.
        if cut <= 0:
            cut = limit
        pieces.append(remaining[:cut].strip())
        remaining = remaining[cut:].lstrip()

    if remaining:
        pieces.append(remaining)

    return [piece for piece in pieces if piece]
