"""The orchestrator: `ingest_document()` from docs/ingest-interface.md section 3.

Bytes and a filename in, `Progress` events out, exactly one `Result` as the
final item. This is the whole published contract with the UI, and the stage
order it runs is a security property in its own right -- ingest-pipeline 3.

    validating -> parsing -> screening -> chunking -> embedding -> indexing

Four orderings were already load-bearing before this module existed: validate
before parsing, hash the bytes as received, screen before indexing, validate in
memory before writing to disk. Wiring the last two stages adds a fifth.

**The manifest entry is the commit point, so it is written last.** Nothing is
recorded until the vectors are on disk. The alternative was measured in this
repo rather than imagined: ten documents sat in the manifest claiming a chunk
count while no index existed at all, because `record()` ran at the end of
chunking. A re-upload of any of them would have reported `already_indexed` with
a chunk count the system could not serve. So `already_indexed` now means
something stronger than it used to -- the manifest knows these bytes *and* the
index holds their vectors -- and a manifest entry without vectors is treated as
a document that still needs indexing. Those ten entries therefore heal on
re-upload instead of needing a migration.

The reverse drift is handled too. If a run dies between `index.save()` and
`manifest.record()`, the index holds vectors for a document the manifest has
never heard of. The next attempt at those bytes finds the manifest silent, the
index occupied, and clears the orphan before re-indexing rather than failing on
a duplicate id.

**Failure messages come from the stage that failed**, and every stage was built
to produce display-safe text: `ParseFailed`, `EmbeddingFailed` and `IndexFailed`
all carry fixed strings written by us, never a library message and never
document text (ingest-pipeline 4.5). That is what makes `str(exc)` safe to put
in `Result.message` here, and it is the only reason it is safe.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from ingest import chunker, manifest, screener
from ingest.embedder import Embedder, EmbeddingFailed
from ingest.index import INDEX_DIR, MESSAGE_MISMATCH, ChunkIndex, IndexFailed
from ingest.model import Flag, Outcome, PageRecord, Progress, Result, Stage
from ingest.parser import ParseFailed, parse
from ingest.validator import ACCEPTED_SUFFIXES, MAX_UPLOAD_BYTES, validate

# Display text for the three rejections. The validator returns an Outcome and
# no words, deliberately -- what to say about a refusal is a UI concern, and
# interface doc section 5 specifies it per outcome.
_MEGABYTE = 1024 * 1024
MESSAGE_REJECTED_TYPE = "Only {types} files can be uploaded.".format(
    types=" and ".join(suffix.upper().lstrip(".") for suffix in ACCEPTED_SUFFIXES)
)
MESSAGE_REJECTED_SIZE = f"Files must be smaller than {MAX_UPLOAD_BYTES // _MEGABYTE} MB."
MESSAGE_REJECTED_CONTENT = (
    "This file's contents do not match its extension, so it was not accepted."
)
MESSAGE_FLAGGED = (
    "This document contains instruction-like phrasing and is being held for "
    "review. Nothing has been stored or indexed."
)

# Failures that are the machine's fault, not the document's. All reach the user
# as Outcome.INDEXING_FAILED.
MESSAGE_MANIFEST_UNREADABLE = (
    "The record of ingested documents could not be read. It may be damaged."
)
MESSAGE_RECORD_FAILED = (
    "The document could not be saved. Check free space and permissions, then "
    "try again."
)
MESSAGE_ROLLBACK_FAILED = (
    "The document could not be saved, and the search index could not be "
    "returned to its previous state. Run --verify before continuing."
)

_REJECTION_MESSAGES = {
    Outcome.REJECTED_TYPE: MESSAGE_REJECTED_TYPE,
    Outcome.REJECTED_SIZE: MESSAGE_REJECTED_SIZE,
    Outcome.REJECTED_CONTENT_MISMATCH: MESSAGE_REJECTED_CONTENT,
}


@dataclass(frozen=True)
class Deletion:
    """What deleting one document actually removed."""

    doc_id: str
    document_name: str
    vectors_removed: int
    entry_removed: bool
    file_removed: Path | None


class FindingKind(str, Enum):
    """What verify() found about one document, or about the store itself."""

    OK = "ok"
    MISSING_BYTES = "missing_bytes"
    HASH_MISMATCH = "hash_mismatch"
    UNINDEXED = "unindexed"
    COUNT_MISMATCH = "count_mismatch"
    ORPHAN = "orphan"
    INDEX_UNREADABLE = "index_unreadable"
    MANIFEST_UNREADABLE = "manifest_unreadable"


@dataclass(frozen=True)
class Finding:
    """One verify result. Facts only -- the wording belongs to the caller."""

    kind: FindingKind
    doc_id: str | None = None
    document_name: str | None = None
    recorded_chunks: int | None = None
    indexed_chunks: int | None = None
    detail: str | None = None  # only for the two unreadable kinds

    @property
    def ok(self) -> bool:
        return self.kind is FindingKind.OK


def delete_document(doc_id: str, *, index_dir: Path = INDEX_DIR) -> Deletion:
    """Forget one document: its vectors, its stored bytes, its manifest entry.

    **The order is the point, and it is not symmetric.** Vectors go first and
    are made durable before the manifest entry is dropped. A crash the other
    way round leaves vectors with no entry -- content the user asked to delete
    that retrieval still answers from, and that verify() can report but not
    name. A crash this way round leaves an entry with no vectors, which
    verify() reports and reindex_pending() repairs.

    That rule lived in the CLI until now, which meant the first UI to grow a
    delete button would have had to rediscover it. Raises IndexFailed if the
    index cannot be read or updated, rather than deleting half the state; lets
    PermissionError through, because explaining it is the caller's job.
    """
    name = (manifest.lookup(doc_id) or {}).get("document_name") or doc_id[:16]

    vectors_removed = 0
    if ChunkIndex.exists(index_dir):
        index = ChunkIndex.load(index_dir)
        vectors_removed = index.remove(doc_id)
        if vectors_removed:
            index.save(index_dir)

    entry_removed, file_removed = manifest.delete(doc_id)
    return Deletion(
        doc_id=doc_id,
        document_name=name,
        vectors_removed=vectors_removed,
        entry_removed=entry_removed,
        file_removed=file_removed,
    )


def verify(*, index_dir: Path = INDEX_DIR) -> list[Finding]:
    """Re-hash every stored document and cross-check it against the index.

    TUT-23 is the hash half. The index half is the drift this project actually
    produced: a manifest entry with no vectors claims a chunk count nothing can
    serve, and vectors with no entry are an orphan retrieval would still answer
    from. Both are reported, never repaired here -- reindex_pending() fixes the
    first, delete_document() the second.
    """
    try:
        known = manifest.entries()
    except Exception:
        # Unbound deliberately: the error carries the manifest path.
        return [
            Finding(
                kind=FindingKind.MANIFEST_UNREADABLE,
                detail=MESSAGE_MANIFEST_UNREADABLE,
            )
        ]

    indexed, index_error = _indexed_documents(index_dir)
    findings: list[Finding] = []
    if index_error is not None:
        findings.append(
            Finding(kind=FindingKind.INDEX_UNREADABLE, detail=index_error)
        )

    for doc_id, entry in sorted(known.items()):
        name = entry["document_name"]
        recorded = entry.get("chunk_count")
        found = manifest.stored_path(doc_id)
        if found is None:
            kind = FindingKind.MISSING_BYTES
        elif not manifest.verify(doc_id, found.read_bytes()):
            kind = FindingKind.HASH_MISMATCH
        elif doc_id not in indexed:
            kind = FindingKind.UNINDEXED
        elif indexed[doc_id] != recorded:
            kind = FindingKind.COUNT_MISMATCH
        else:
            kind = FindingKind.OK
        findings.append(
            Finding(
                kind=kind,
                doc_id=doc_id,
                document_name=name,
                recorded_chunks=recorded,
                indexed_chunks=indexed.get(doc_id),
            )
        )

    for doc_id, count in sorted(indexed.items()):
        if doc_id not in known:
            findings.append(
                Finding(
                    kind=FindingKind.ORPHAN, doc_id=doc_id, indexed_chunks=count
                )
            )
    return findings


def _indexed_documents(index_dir: Path) -> tuple[dict[str, int], str | None]:
    """Chunk count per indexed document, plus why it could not be read.

    An unreadable index is reported and treated as empty rather than aborting:
    the manifest half of verify is still worth running, and "the index will not
    load" is itself the finding.
    """
    if not ChunkIndex.exists(index_dir):
        return {}, None
    try:
        return ChunkIndex.load(index_dir).documents(), None
    except IndexFailed as exc:
        # Ours, fixed message, safe to pass on.
        return {}, str(exc)


def ingest_document(
    data: bytes,
    filename: str,
    *,
    embedder: Embedder | None = None,
    index_dir: Path = INDEX_DIR,
) -> Iterator[Progress | Result]:
    """Yields Progress events, then exactly one Result as the final item.

    The two keyword arguments are not part of the published contract: the UI
    calls this with bytes and a filename only. They exist so tests can run the
    whole pipeline without a local Ollama and without touching `data/index/`,
    which is the same seam `Embedder(backend=...)` provides one layer down.
    """
    embedder = embedder if embedder is not None else Embedder()

    # --- validating ---------------------------------------------------
    yield Progress(stage=Stage.VALIDATING)
    rejection = validate(data, filename)
    if rejection is not None:
        yield Result(
            outcome=rejection,
            document_name=filename,
            message=_REJECTION_MESSAGES.get(rejection),
        )
        return

    # Hashed before parsing, so the digest covers the bytes as received. An
    # integrity check against anything derived would compare against an
    # artifact that was never on disk. Pipeline doc 3.
    doc_id = manifest.document_id(data)

    # Neither of the next two reads may raise past this point. The contract
    # promises a Result as the final item, so an exception here would hand the
    # UI a failure it has no channel for -- and both of these genuinely fail:
    # the index is refused when corrupt or torn, and manifest.entries() raises
    # on a damaged manifest *by design*, rather than reading as empty and
    # re-ingesting everything.
    #
    # The index is loaded without the embedder's spec on purpose: answering
    # "have we seen this file?" must not require a running model. The spec is
    # compared later, before anything is added.
    index: ChunkIndex | None = None
    known: dict | None = None
    failure: str | None = None
    try:
        index = _open_index(index_dir)
        known = manifest.lookup(doc_id)
    except IndexFailed as exc:
        # Ours, and its message is a fixed display-safe string.
        failure = str(exc)
    except Exception:
        # Deliberately unbound: a json or OS error message carries paths, and
        # 4.5 forbids rendering a broadly caught exception.
        failure = MESSAGE_MANIFEST_UNREADABLE

    if failure is not None:
        yield Result(
            outcome=Outcome.INDEXING_FAILED,
            document_name=filename,
            message=failure,
        )
        return

    if known is not None and index is not None and index.has_document(doc_id):
        yield Result(
            outcome=Outcome.ALREADY_INDEXED,
            document_name=filename,
            chunk_count=known.get("chunk_count"),
        )
        return

    # --- parsing ------------------------------------------------------
    yield Progress(stage=Stage.PARSING)
    try:
        pages = parse(data, filename)
    except ParseFailed as exc:
        yield Result(
            outcome=Outcome.PARSE_FAILED, document_name=filename, message=str(exc)
        )
        return

    # --- screening ----------------------------------------------------
    yield Progress(stage=Stage.SCREENING)
    flags = _screen(pages)
    if flags:
        # Fail closed and stop here: once a chunk is in the index retrieval can
        # surface it, and the gate has already failed. Threat model 3.3.
        yield Result(
            outcome=Outcome.FLAGGED_INJECTION,
            document_name=filename,
            message=MESSAGE_FLAGGED,
            flagged=flags,
        )
        return

    # --- chunking -----------------------------------------------------
    chunks = chunker.chunk(pages, filename)
    yield Progress(stage=Stage.CHUNKING, current=len(chunks), total=len(chunks))

    # --- embedding and indexing --------------------------------------
    # One try around both, because they are one unit of work: embedding exists
    # only to produce the index, and neither leaves anything behind on failure.
    try:
        vectors = yield from embedder.embed_chunks(chunks)

        yield Progress(stage=Stage.INDEXING)
        index = _prepare_index(index, embedder, doc_id)
        index.add(doc_id, chunks, vectors)
        index.save(index_dir)
    except (EmbeddingFailed, IndexFailed) as exc:
        yield Result(
            outcome=Outcome.INDEXING_FAILED, document_name=filename, message=str(exc)
        )
        return

    # --- commit -------------------------------------------------------
    # Only now, with the vectors durable, does this document become known. See
    # the module docstring: the manifest entry is the commit point.
    #
    # Which is exactly why this needs a rollback. The vectors are already on
    # disk, so a failure here without one would leave the document searchable
    # while no record of it exists -- an orphan that verify() can report but
    # cannot name, and that retrieval would happily answer from.
    commit_failure = None
    try:
        manifest.store(data, filename)
        manifest.record(doc_id, filename, len(pages), len(chunks))
    except Exception:
        # Unbound for the same reason as above: OSError carries a path.
        commit_failure = MESSAGE_RECORD_FAILED

    if commit_failure is not None:
        if not _rollback(index, index_dir, doc_id):
            commit_failure = MESSAGE_ROLLBACK_FAILED
        yield Result(
            outcome=Outcome.INDEXING_FAILED,
            document_name=filename,
            message=commit_failure,
        )
        return

    yield Result(
        outcome=Outcome.INDEXED, document_name=filename, chunk_count=len(chunks)
    )


def _screen(pages: list[PageRecord]) -> list[Flag]:
    """Screen, treating a screener error as flagged rather than clean.

    A gate that fails open is worse than no gate, because it produces the
    belief that screening ran when it silently did not. Pipeline doc 4.6.
    """
    failed = False
    try:
        return screener.screen(pages)
    except Exception:
        failed = True
    if failed:
        return [Flag(page=None, pattern="screener-error", excerpt=None)]
    return []


def _open_index(index_dir: Path) -> ChunkIndex | None:
    """The stored index, or None when there is not one yet.

    Deliberately loads without an expected spec. A spec mismatch is a refusal,
    but refusing here would turn "this index was built with another model" into
    a failure to answer whether a file is a duplicate, which is a worse message
    for the same problem. `_prepare_index` compares the spec before adding.
    """
    if not ChunkIndex.exists(index_dir):
        return None
    return ChunkIndex.load(index_dir)


def _rollback(index: ChunkIndex, index_dir: Path, doc_id: str) -> bool:
    """Undo this document's index entry. False if that could not be done.

    Best effort by necessity: if the write that failed was caused by a full
    disk, the write that undoes it may fail too. Reporting that honestly is
    better than a message implying the index is clean.
    """
    ok = True
    try:
        index.remove(doc_id)
        index.save(index_dir)
    except Exception:
        ok = False
    return ok


def _prepare_index(
    index: ChunkIndex | None, embedder: Embedder, doc_id: str
) -> ChunkIndex:
    """The index to add to, created, spec-checked, or cleared of an orphan."""
    spec = embedder.spec
    if index is None:
        return ChunkIndex.create(spec)
    if index.spec != spec:
        # Stored vectors are not comparable with the ones just computed, and
        # the symptom would be quietly worse ranking rather than an error. The
        # message is index.py's, not a second copy of it.
        raise IndexFailed(MESSAGE_MISMATCH)
    if index.has_document(doc_id):
        # The manifest said unknown and the index disagrees, so these vectors
        # are the residue of a run that died before recording. Clear them
        # rather than failing on a duplicate.
        index.remove(doc_id)
    return index


def reindex_pending(
    *, embedder: Embedder | None = None, index_dir: Path = INDEX_DIR
) -> Iterator[Progress | Result]:
    """Re-run ingest for every manifest entry whose vectors are missing.

    The repair path for documents recorded before indexing existed. It reads
    the stored bytes back rather than trusting the manifest's counts, so a
    document whose file is gone is reported rather than silently skipped, and
    one whose bytes no longer hash to their recorded id never gets indexed.
    """
    # Which documents need work is decided once, up front. Re-reading the index
    # after every document meant deserialising the whole thing per file -- 773
    # KB at semester size -- to answer a question ingest_document() already
    # handles itself through its orphan-vector path.
    index = _open_index(index_dir)
    already = set(index.documents()) if index is not None else set()
    pending = [
        (doc_id, entry)
        for doc_id, entry in sorted(
            manifest.entries().items(), key=lambda item: item[1]["ingested_at"]
        )
        if doc_id not in already
    ]

    for doc_id, entry in pending:
        name = entry["document_name"]
        path = manifest.stored_path(doc_id)
        if path is None:
            yield Result(
                outcome=Outcome.INDEXING_FAILED,
                document_name=name,
                message="The stored copy of this document is missing.",
            )
            continue

        data = path.read_bytes()
        if not manifest.verify(doc_id, data):
            yield Result(
                outcome=Outcome.INDEXING_FAILED,
                document_name=name,
                message="This document has changed since it was ingested.",
            )
            continue

        yield from ingest_document(
            data, name, embedder=embedder, index_dir=index_dir
        )
