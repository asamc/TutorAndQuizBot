"""Ingest pipeline.

Design and security requirements: docs/ingest-pipeline.md
Contract with the UI: docs/ingest-interface.md

The orchestrator -- ingest_document(), the generator described in the interface
doc -- is deliberately not implemented yet. No Outcome in the published contract
can honestly describe a pipeline that stops after chunking: INDEXED would be
false, and adding a new Outcome would break the contract the UI is built
against. It gets wired once embedding and indexing exist, which are blocked on
TUT-58.

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

__all__ = [
    "Chunk",
    "Flag",
    "Outcome",
    "PageRecord",
    "Progress",
    "Result",
    "Stage",
]
