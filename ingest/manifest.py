"""Document identity and integrity. TUT-17, read back by TUT-23.

Holds the SHA-256 of every ingested document, so that later tampering with a
document or the index is detectable rather than silently used.

Two design points worth not undoing:

1. **The manifest lives outside data/.** A manifest stored beside the documents
   can be rewritten by anyone able to tamper with those documents, which makes
   the integrity check theatre. The detector has to sit outside the blast radius
   of the thing it detects. See docs/threat-model.md section 4.3.

2. **The hash is the document identity.** Storage names are derived from it
   rather than from the user-supplied filename, which gives path safety and a
   correct duplicate check from the same decision. See docs/ingest-pipeline.md
   section 4.1.

Hash the bytes *as received*, before parsing. Hashing anything derived means the
integrity check compares against an artifact that was never on disk.
"""

from pathlib import Path

# Outside data/ by design -- see point 1 above. Created with owner-only
# permissions; it is a small target that invalidates a whole control.
MANIFEST_DIR = Path.home() / ".tutorandquizbot"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"


def document_id(data: bytes) -> str:
    """SHA-256 hex digest of the bytes as received. The document's identity."""
    raise NotImplementedError


def storage_name(data: bytes, filename: str) -> str:
    """Path-safe name for storing this document under data/documents/.

    Derived from document_id, never from `filename`. The supplied filename is
    user input: '../../.ssh/authorized_keys', absolute paths, Windows reserved
    names like CON and NUL, embedded null bytes and 4000-character names all
    arrive here. The original filename is kept as display metadata only.

    The suffix may be carried over for readability, but only after being checked
    against the validator's accepted list.
    """
    raise NotImplementedError


def record(doc_id: str, document_name: str, page_count: int) -> None:
    """Add or update this document's manifest entry.

    Stores document identity, original name, page count and timestamp. Never
    stores chunk text -- the manifest is metadata, and it sits outside the
    gitignored tree.
    """
    raise NotImplementedError


def lookup(doc_id: str) -> dict | None:
    """Return the manifest entry, or None if this document is not known.

    This is what makes re-ingest return ALREADY_INDEXED instead of duplicating.
    Note the asymmetry: same bytes under a different filename is the same
    document, while the same filename with different bytes is a *new* document
    and must not silently overwrite the old entry.
    """
    raise NotImplementedError


def verify(doc_id: str, data: bytes) -> bool:
    """True when `data` still hashes to its recorded digest. TUT-23.

    Called on index load. A mismatch means the document changed since ingest:
    surface it and do not serve the affected material.
    """
    raise NotImplementedError
