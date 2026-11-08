"""Ingest pipeline.

Design and security requirements: docs/ingest-pipeline.md
Contract with the UI: docs/ingest-interface.md

The orchestrator -- ingest_document(), the generator described in the interface
doc -- is deliberately not implemented yet, but the reason has narrowed. The
original blocker was that no Outcome could honestly describe a pipeline stopping
after chunking. Two of the three pieces are now in place: INDEXING_FAILED exists
for the case where embedding or the index write cannot complete, and
ingest/embedder.py implements the EMBEDDING stage. What remains is the index
itself (TUT-58), without which INDEXED is still a false claim.

The embedder is written to be driven by that orchestrator: embed_chunks() is a
generator yielding Progress and returning the vectors, so the orchestrator can
forward progress straight to the UI with

    vectors = yield from embedder.embed_chunks(chunks)

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
