"""Document identity, integrity and storage. TUT-17, read back by TUT-23.

Holds the SHA-256 of every ingested document, so later tampering with a document
or the index is detectable rather than silently used.

The manifest lives outside data/ by design: one stored beside the documents can
be rewritten by anyone able to tamper with them, which makes the check theatre.
Pipeline doc 4.3.

Storage names come from the hash, not the user-supplied filename. That gives
path safety and the duplicate check from one decision. Pipeline doc 4.1.

Hash the bytes as received, before parsing -- hashing anything derived compares
the integrity check against an artifact that was never on disk.
"""

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from ingest.validator import ACCEPTED_SUFFIXES, declared_suffix

# Outside data/ by design. Owner-only: it is a small target that invalidates a
# whole control.
MANIFEST_DIR = Path.home() / ".tutorandquizbot"
MANIFEST_PATH = MANIFEST_DIR / "manifest.json"

# Every write under data/ happens in this module, so the R2 permission and
# gitignore story can be audited in one place. Interface doc 2.
DOCUMENT_DIR = Path("data/documents")


def document_id(data: bytes) -> str:
    """SHA-256 hex digest of the bytes as received. The document's identity."""
    return hashlib.sha256(data).hexdigest()


def storage_name(data: bytes, filename: str) -> str:
    """Path-safe name for storing this document, derived from document_id.

    `filename` is user input -- '../../.ssh/authorized_keys', absolute paths,
    Windows reserved names, embedded nulls -- and contributes only its suffix,
    and only if the validator accepts it. The original name is display metadata.
    """
    suffix = declared_suffix(filename)
    return document_id(data) + (suffix if suffix in ACCEPTED_SUFFIXES else "")


def store(data: bytes, filename: str) -> Path:
    """Write the document under DOCUMENT_DIR and return its path."""
    DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(DOCUMENT_DIR, 0o700)
    path = DOCUMENT_DIR / storage_name(data, filename)
    path.write_bytes(data)
    os.chmod(path, 0o600)
    return path


def stored_path(doc_id: str) -> Path | None:
    """The stored document for this id, or None if its bytes are not on disk.

    Matched by prefix: the stored name is the digest plus the accepted suffix,
    which the id alone does not carry.

    An empty id would glob every document, so it returns None rather than an
    arbitrary one -- this is a lookup used by deletion.
    """
    if not doc_id:
        return None
    found = sorted(DOCUMENT_DIR.glob(doc_id + "*")) if DOCUMENT_DIR.is_dir() else []
    return found[0] if found else None


def entries() -> dict:
    """Every manifest entry, keyed by document id.

    A corrupt manifest raises rather than returning {}: treating it as empty
    would re-ingest everything and overwrite the records that detect tampering.
    """
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


def lookup(doc_id: str) -> dict | None:
    """The manifest entry, or None if this document is not known.

    This is what makes re-ingest ALREADY_INDEXED instead of a duplicate. Note
    the asymmetry: the same bytes under a different filename are the same
    document, while the same filename with different bytes is a new one.
    """
    return entries().get(doc_id)


def _write_manifest(known: dict) -> None:
    """The only writer for MANIFEST_PATH. Atomic, owner-only.

    Written to a temp file and moved into place so a crash mid-write cannot
    leave a half-written manifest, which entries() would refuse to parse and
    which no longer detects tampering.
    """
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    os.chmod(MANIFEST_DIR, 0o700)
    tmp = MANIFEST_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(known, indent=2, sort_keys=True))
    os.chmod(tmp, 0o600)
    os.replace(tmp, MANIFEST_PATH)


def record(doc_id: str, document_name: str, page_count: int, chunk_count: int) -> None:
    """Add or update this document's entry. Metadata only, never chunk text."""
    known = entries()
    known[doc_id] = {
        "document_name": document_name,
        "page_count": page_count,
        "chunk_count": chunk_count,
        "ingested_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    _write_manifest(known)


def delete(doc_id: str) -> tuple[bool, Path | None]:
    """Forget a document: remove its stored bytes and its manifest entry.

    Returns (entry_removed, path_of_removed_file). An orphan -- bytes with no
    entry, or an entry with no bytes -- deletes the half that exists and
    reports the other as absent, so this doubles as drift repair.

    The file is unlinked *before* the entry is dropped, deliberately. If this
    dies halfway the surviving state should be the detectable one: verify() and
    cli.verify_stored() already report an entry whose file is missing, whereas
    bytes with no entry are an orphan nothing audits.

    Deleting needs no elevation. unlink(2) is authorised by write+execute on
    DOCUMENT_DIR, which store() leaves at 0o700 owned by the ingesting user;
    the 0o600 on the file governs open(), not removal. A PermissionError here
    means the artifacts are owned by someone else -- in practice that the
    pipeline was once run under sudo -- and is left to the caller to report.
    """
    path = stored_path(doc_id)
    if path is not None:
        path.unlink()

    known = entries()
    existed = known.pop(doc_id, None) is not None
    if existed:
        _write_manifest(known)
    return existed, path


def verify(doc_id: str, data: bytes) -> bool:
    """True when `data` still hashes to its recorded digest. TUT-23.

    Called on index load. A mismatch means the document changed since ingest:
    surface it and do not serve the affected material.
    """
    return doc_id in entries() and document_id(data) == doc_id
