"""Ingest pipeline.

Design and security requirements: docs/ingest-pipeline.md
Contract with the UI: docs/ingest-interface.md

ingest_document() -- the generator described in the interface doc -- is wired:
every stage from validating to indexing exists, so INDEXED is a claim the
pipeline can honestly make. It lives in ingest/pipeline.py and is re-exported
here, which is the import the interface doc implies:

    from ingest import ingest_document

ingest/cli.py drives it for the command line until a UI exists.

Until then the stages are used directly, which is also how they are tested and
how ingest/cli.py drives them:

    uv run python -m ingest.cli data/documents/Lecture_1_slides.pdf
"""

from ingest.model import (
    Chunk,
    Flag,
    Outcome,
    PageRecord,
    Progress,
    Result,
    Stage,
)
from ingest.pipeline import ingest_document, reindex_pending

__all__ = [
    "Chunk",
    "Flag",
    "Outcome",
    "PageRecord",
    "Progress",
    "Result",
    "Stage",
    "ingest_document",
    "reindex_pending",
]
