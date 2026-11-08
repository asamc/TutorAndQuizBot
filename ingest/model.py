"""Domain model for the ingest pipeline.

The types under "published contract" are transcribed from
docs/ingest-interface.md section 3 and the UI is built against them, so their
names, string values and order are a contract rather than an implementation
detail. tests/test_contract.py compares the two.

PageRecord and Chunk are internal. PageRecord is the keystone: the screener and
the chunker both read it, so its shape decides what they can do.
"""

from dataclasses import dataclass
from enum import Enum

# --------------------------------------------------------------------------
# Published contract - see docs/ingest-interface.md section 3. Do not rename.
# --------------------------------------------------------------------------


class Stage(str, Enum):
    VALIDATING = "validating"
    PARSING = "parsing"
    SCREENING = "screening"
    CHUNKING = "chunking"
    EMBEDDING = "embedding"
    INDEXING = "indexing"


class Outcome(str, Enum):
    INDEXED = "indexed"
    ALREADY_INDEXED = "already_indexed"
    REJECTED_TYPE = "rejected_type"
    REJECTED_SIZE = "rejected_size"
    REJECTED_CONTENT_MISMATCH = "rejected_content_mismatch"
    PARSE_FAILED = "parse_failed"
    # The only outcome that is not a verdict on the document: the file was fine
    # and embedding or the index write could not complete. Retryable, and
    # nothing is stored or recorded. See docs/ingest-interface.md section 5.
    INDEXING_FAILED = "indexing_failed"
    FLAGGED_INJECTION = "flagged_injection"


@dataclass
class Progress:
    stage: Stage
    current: int | None = None  # e.g. chunks embedded so far
    total: int | None = None  # e.g. total chunks; None when not yet known


@dataclass
class Flag:
    page: int | None  # page or slide number where the match was found
    pattern: str  # which screening rule matched, not the document text
    excerpt: str | None  # bounded snippet; max length still an open decision


@dataclass
class Result:
    outcome: Outcome
    document_name: str
    chunk_count: int | None = None  # set when outcome is INDEXED or ALREADY_INDEXED
    message: str | None = None  # display-safe reason; never internals
    flagged: list[Flag] | None = None  # set only when outcome is FLAGGED_INJECTION


# --------------------------------------------------------------------------
# Internal to ingest
# --------------------------------------------------------------------------

# A page holding less than this many characters is treated as empty. Measured on
# the real lecture decks: one near-empty page in each (title and image slides),
# so per-page emptiness is normal and must not fail a document. Only an
# all-empty document fails. See docs/ingest-pipeline.md section 5.
EMPTY_PAGE_CHAR_THRESHOLD = 20


@dataclass
class PageRecord:
    """One page or slide, as extracted. Pages are retained even when empty so
    that the chunker can skip them and the screener can still report a page
    number for a flag."""

    page_number: int  # 1-based, as a human would cite it
    heading: str | None  # slide or section title where the document provides one
    text: str

    @property
    def is_empty(self) -> bool:
        """True for pages with no usable text, typically image-only slides."""
        return len(self.text.strip()) < EMPTY_PAGE_CHAR_THRESHOLD


@dataclass
class Chunk:
    """One retrievable unit. One chunk per page or slide, not word-budgeted."""

    document_name: str  # original filename, for display and citations
    page_number: int
    heading: str | None
    text: str

    def citation(self) -> str:
        """Source reference, e.g. 'Lecture_1_slides.pdf, Slide 6: Integrity'.

        The heading is preferred because it carries the slide number the
        student sees, which disagrees with the PDF page number: the decks have
        hidden slides, so Lecture 2's page 4 is "Slide 11". page_number stays
        the PDF index and is cited only for documents with no outline, which is
        every .txt file. See docs/ingest-pipeline.md 5.1.
        """
        if self.heading:
            return f"{self.document_name}, {self.heading}"
        return f"{self.document_name}, Page {self.page_number}"
