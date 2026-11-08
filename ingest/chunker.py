"""Turn page records into chunks, one per page or slide.

Not packed to a word budget: an entire 11-page lecture holds about 441 words, so
a budget would collapse a whole lecture into one chunk and lose page-level
citations. Page alignment also means a chunk never spans a page boundary.
"""

from ingest.model import Chunk, PageRecord


def chunk(pages: list[PageRecord], document_name: str) -> list[Chunk]:
    """One Chunk per non-empty page; empty pages are skipped.

    Real material: 10 chunks from Lecture_1_slides.pdf (11 pages, one empty),
    9 from Lecture_2_slides.pdf (10 pages, one empty).
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
