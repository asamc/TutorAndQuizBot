"""Extract text from an accepted file. TUT-15.

Returns one PageRecord per page or slide. Page structure must survive: a flat
string loses the page boundaries that citations and page-located flags depend on.

This module is attack surface. It parses untrusted input, so it is bounded on
three axes (see BOUNDS below) and never lets a library exception reach the user.
"""

from ingest.model import PageRecord

# Bounds against malformed and hostile files. docs/ingest-pipeline.md 4.2.
#
# MAX_EXTRACTED_CHARS is the one usually missed and the one that matters most:
# the upload size cap does not bound memory, because a few megabytes of PDF can
# expand to gigabytes of extracted text. Whether this becomes a fourth risk is
# still open (TUT-63); these bounds are cheap either way.
MAX_PAGES = 500
MAX_EXTRACTED_CHARS = 2_000_000
PARSE_TIMEOUT_SECONDS = 30


class ParseFailed(Exception):
    """Extraction failed. The message must be display-safe: no stack traces, no
    library internals, no document content. It is shown to the user as
    Result.message."""


def parse(data: bytes, filename: str) -> list[PageRecord]:
    """Extract text as a list of PageRecord, one per page, 1-based.

    Dispatches on the suffix the validator already accepted. PDFs go through
    pypdf; .txt becomes a single PageRecord with page_number=1.

    Raises ParseFailed when:
      - a bound in BOUNDS above is exceeded
      - the file cannot be parsed at all
      - *every* page is empty

    That last case is the silent failure worth guarding. page.extract_text()
    returns "" for an image-only page and does **not** raise, so an all-scanned
    PDF would otherwise parse "successfully", produce zero chunks, and report
    success. Per-page emptiness is normal -- measured one near-empty page in each
    real lecture deck -- so only an all-empty document fails.

    Wrap every pypdf call: a library exception must not propagate, because its
    message can carry fragments of the document.
    """
    raise NotImplementedError


def _extract_headings(reader: object) -> dict[int, str]:
    """Map 1-based page number to its outline title, where the PDF has one.

    The real lecture decks carry one bookmark per slide with its title, which is
    where citations like 'Slide 6: Integrity' come from -- no font-size
    heuristics needed. Three things to tolerate:

      - no outline at all
      - nested outline entries (multi-level bookmarks come back as nested lists)
      - untitled entries; 'Slide 4' in Lecture 1 has no title text

    Note reader.get_destination_page_number() is 0-based; add 1.
    """
    raise NotImplementedError
