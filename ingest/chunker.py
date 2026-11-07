"""Turn page records into retrievable chunks. TUT-16.

One chunk per page or slide, not packed to a word budget. The original "~500
words per chunk" criterion was amended after measuring the real course material:
an entire 11-page lecture deck holds roughly 441 words, so a word budget would
collapse a whole lecture into a single chunk and destroy page-level citations.

Page alignment also means a chunk can never span a page boundary, so there is no
ambiguity about which page a chunk cites, and no sentence is ever split.
"""

from ingest.model import Chunk, PageRecord


def chunk(pages: list[PageRecord], document_name: str) -> list[Chunk]:
    """Build one Chunk per non-empty page.

    Pages where PageRecord.is_empty is True are skipped rather than emitted as
    empty chunks. Every chunk carries document_name, page_number and heading, so
    the Tutor Agent can render a citation naming the document and the location
    within it.

    Expected on the real material: 10 chunks from Lecture_1_slides.pdf (11 pages,
    one empty) and 9 from Lecture_2_slides.pdf (10 pages, one empty).
    """
    return [
        Chunk(
            document_name=document_name,
            page_number=page.page_number,
            heading=page.heading,
            text=page.text,
        )
        for page in pages
        if not page.is_empty
    ]
