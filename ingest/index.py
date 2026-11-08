"""The vector index and its chunk sidecar. TUT-58, the INDEXING stage.

One flat faiss index over unit-norm 768-dimension vectors, decided in
architecture doc 5.1: the whole semester is ~250 chunks, orders of magnitude
below where an approximate index earns its complexity, so `IndexFlatIP` gives
exact nearest-neighbour search with no training step and no recall to tune.

Four structural decisions, each with a failure mode behind it:

**Vectors carry explicit ids, through `IndexIDMap2`.** A bare `IndexFlatIP`
addresses vectors by position and `remove_ids` compacts by swapping the last
vector into the hole -- so deleting one document silently renumbers every vector
after it, and a sidecar keyed by position then describes the wrong chunks. The
CLI already exposes `--delete`, so that is reachable today, not hypothetical.
Ids are allocated from a monotonic counter and **never reused**, so a stale
reference resolves to nothing rather than to a different document's text.

**Chunk text and citations live in the sidecar, not in faiss**, which stores
vectors only. The sidecar is keyed by vector id and holds everything
`Chunk.citation()` needs, so retrieval can render a citation without re-reading
the document.

**The sidecar owns the document-to-vector mapping, not the manifest.** The
manifest is the integrity artifact that deliberately lives outside `data/`
(ingest-pipeline 4.3), and growing index internals into it would put the
detector inside the blast radius of the thing it detects. The cost is that
deletion has to load the index, which at this corpus size is nothing.

**Reads and writes funnel through `_read_blob`/`_write_blob`.** TUT-49's
encryption at rest is the bonus build: `faiss.serialize_index()` hands back one
buffer, which is exactly the shape AES-256-GCM wants, and the sidecar is already
a single JSON document. Keeping all persistence in two functions means that
becomes a wrapper rather than a rewrite.

The two files have to agree, and nothing in faiss or JSON enforces that. Each
write stamps a fresh random `generation` token into both, and load refuses a
pair that disagrees -- so a torn write or a swapped file is a refusal instead of
an index that describes chunks it does not contain.

Comparing vector counts is not enough, which is worth stating because it was the
first attempt. Two indexes holding the same number of chunks -- unremarkable
across a set of uniform lecture decks -- can have their vector files exchanged,
and a count check passes: the index then answers with another document's vector
while citing this one's filename and page, at a similarity of 1.0. The token is
what closes that, and it is why the vectors file carries a small header of its
own rather than being a bare `faiss.serialize_index()` buffer. Architecture 5.1
raises exactly this worry about a store owning its on-disk layout; this is the
base-build answer, and TUT-49 can additionally bind each file to its name and
generation as AES-GCM associated data.

Note for R2: the sidecar holds document text at rest. It lives under the
gitignored, permission-restricted `data/` tree, is written 0600 in a 0700
directory, and is what TUT-49 encrypts.
"""

from __future__ import annotations

import json
import os
import re
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np

from ingest.embedder import EmbeddingSpec
from ingest.model import Chunk

# Every write under data/ is accounted for in one place per interface doc 2;
# this is the index half, manifest.py is the documents half.
INDEX_DIR = Path("data/index")
VECTORS_NAME = "vectors.faiss"
SIDECAR_NAME = "sidecar.json"

# Bumped when the sidecar layout changes in a way an older reader would
# misinterpret. An unknown version is refused rather than guessed at.
FORMAT_VERSION = 1

# A unit vector's norm after float32 rounding. Looser than the 1e-6 measured
# off the embedder, tight enough that raw unnormalised vectors are caught.
NORM_TOLERANCE = 1e-3

# secrets.token_hex(16). Validated on both read and write so a hand-edited or
# truncated token is a refusal rather than a token that happens to compare equal.
_GENERATION = re.compile(r"[0-9a-f]{32}")

MESSAGE_FAILED = "The document could not be indexed."
MESSAGE_UNREADABLE = (
    "The search index could not be read. It may be incomplete or from a "
    "different version."
)
MESSAGE_MISMATCH = (
    "The search index was built with different embedding settings and cannot "
    "be used. Rebuild it to continue."
)


class IndexFailed(Exception):
    """Index read, write or update could not complete.

    Maps to Outcome.INDEXING_FAILED. Carries a fixed, display-safe message
    written here: never a library error, never document text.
    """


@dataclass(frozen=True)
class Hit:
    """One search result, with everything a citation needs."""

    score: float
    doc_id: str
    vector_id: int
    chunk: Chunk

    def citation(self) -> str:
        return self.chunk.citation()


class ChunkIndex:
    """Vectors plus their chunk sidecar, held together by one generation token."""

    def __init__(
        self,
        spec: EmbeddingSpec,
        index: faiss.Index | None = None,
        records: dict[int, dict] | None = None,
        next_id: int = 0,
        generation: str | None = None,
    ):
        self.spec = spec
        self.index = index if index is not None else _new_index(spec.dim)
        self.records: dict[int, dict] = records if records is not None else {}
        self.next_id = next_id
        self.generation = generation or secrets.token_hex(16)

    # -- construction ----------------------------------------------------

    @classmethod
    def create(cls, spec: EmbeddingSpec) -> ChunkIndex:
        """An empty index bound to one embedder's settings."""
        return cls(spec=spec)

    @classmethod
    def load(
        cls, directory: Path = INDEX_DIR, expected_spec: EmbeddingSpec | None = None
    ) -> ChunkIndex:
        """Read the index from disk, refusing anything inconsistent.

        `expected_spec` is the live embedder's settings. A stored index built
        with a different model, digest, dimension or prefix holds vectors that
        are not comparable with freshly computed ones, and the symptom is
        plausible-looking wrong rankings rather than an error -- so a mismatch
        is refused here, which is the only place it can be caught.
        """
        sidecar = _read_sidecar(directory)
        spec = _spec_from(sidecar.get("spec"))

        if expected_spec is not None and spec != expected_spec:
            raise IndexFailed(MESSAGE_MISMATCH)

        records = _records_from(sidecar.get("records"))
        generation = _generation_from(sidecar.get("generation"))

        # The two files must come from the same write. Compared before the
        # vectors are even deserialised, because a mismatch means this pair
        # should not be assembled into an index at all.
        stamped, blob = _unpack_vectors(_read_blob(directory / VECTORS_NAME))
        if stamped != generation:
            raise IndexFailed(MESSAGE_UNREADABLE)
        index = _deserialize(blob)

        # Cheap internal consistency on top: the token catches a swap or a torn
        # write, this catches a pair that was written inconsistently to begin
        # with.
        if index.ntotal != len(records):
            raise IndexFailed(MESSAGE_UNREADABLE)
        if index.d != spec.dim:
            raise IndexFailed(MESSAGE_MISMATCH)

        next_id = int(sidecar.get("next_id") or 0)
        if records and next_id <= max(records):
            # Ids must never be reused; a counter at or below a live id would
            # hand the next document an id that already means something else.
            raise IndexFailed(MESSAGE_UNREADABLE)

        return cls(
            spec=spec,
            index=index,
            records=records,
            next_id=next_id,
            generation=generation,
        )

    @staticmethod
    def exists(directory: Path = INDEX_DIR) -> bool:
        return (directory / SIDECAR_NAME).is_file() and (
            directory / VECTORS_NAME
        ).is_file()

    # -- mutation --------------------------------------------------------

    def add(self, doc_id: str, chunks: list[Chunk], vectors: np.ndarray) -> list[int]:
        """Index one document's chunks. Returns the ids assigned, in order.

        Row i of `vectors` must be the vector for `chunks[i]`; that pairing is
        what holds a citation to a vector and nothing downstream could detect it
        being wrong, so it is checked rather than trusted.
        """
        if not doc_id:
            raise IndexFailed(MESSAGE_FAILED)
        if self.has_document(doc_id):
            # The manifest decides what counts as already ingested; reaching
            # here means the two disagree, which is not something to paper over
            # by indexing the same document twice.
            raise IndexFailed(MESSAGE_FAILED)
        if len(chunks) != len(vectors):
            raise IndexFailed(MESSAGE_FAILED)
        if not chunks:
            return []

        matrix = _validated(vectors, self.spec.dim)
        ids = list(range(self.next_id, self.next_id + len(chunks)))
        id_array = np.asarray(ids, dtype=np.int64)

        _guarded(
            lambda: self.index.add_with_ids(matrix, id_array), MESSAGE_FAILED
        )

        for vector_id, chunk in zip(ids, chunks):
            self.records[vector_id] = {
                "doc_id": doc_id,
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "heading": chunk.heading,
                "text": chunk.text,
            }
        self.next_id += len(chunks)
        self.generation = secrets.token_hex(16)
        return ids

    def remove(self, doc_id: str) -> int:
        """Forget one document's vectors and sidecar rows. Returns the count.

        The ids are not recycled. `IndexIDMap2` keeps every surviving vector's
        id across a removal, which is the reason it is used at all.
        """
        ids = [
            vector_id
            for vector_id, record in self.records.items()
            if record["doc_id"] == doc_id
        ]
        if not ids:
            return 0

        id_array = np.asarray(sorted(ids), dtype=np.int64)
        _guarded(lambda: self.index.remove_ids(id_array), MESSAGE_FAILED)
        for vector_id in ids:
            del self.records[vector_id]
        self.generation = secrets.token_hex(16)
        return len(ids)

    # -- query -----------------------------------------------------------

    def search(self, query: np.ndarray, top_k: int = 5) -> list[Hit]:
        """The `top_k` most similar chunks, best first.

        Scores are inner products of unit vectors, so they are cosine
        similarities in [-1, 1]. faiss pads a short result with id -1, which is
        dropped here rather than surfaced as a hit -- that happens whenever
        top_k exceeds the number of indexed chunks, which for a fresh index with
        one lecture in it is the normal case.
        """
        if top_k < 1:
            raise IndexFailed(MESSAGE_FAILED)
        if self.index.ntotal == 0:
            return []

        vector = _validated(np.asarray(query, dtype=np.float32).reshape(1, -1), self.spec.dim)
        scores, ids = _guarded(
            lambda: self.index.search(vector, min(top_k, self.index.ntotal)),
            MESSAGE_FAILED,
        )

        hits = []
        for score, vector_id in zip(scores[0].tolist(), ids[0].tolist()):
            if vector_id < 0:
                continue
            record = self.records.get(int(vector_id))
            if record is None:
                # A vector with no sidecar row cannot be cited, so it is not a
                # usable result. load() refuses this state; reaching it here
                # means it appeared after load, so skip rather than serve
                # uncitable text.
                continue
            hits.append(
                Hit(
                    score=float(score),
                    doc_id=record["doc_id"],
                    vector_id=int(vector_id),
                    chunk=Chunk(
                        document_name=record["document_name"],
                        page_number=record["page_number"],
                        heading=record["heading"],
                        text=record["text"],
                    ),
                )
            )
        return hits

    # -- inspection ------------------------------------------------------

    def has_document(self, doc_id: str) -> bool:
        return any(record["doc_id"] == doc_id for record in self.records.values())

    def documents(self) -> dict[str, int]:
        """Chunk count per indexed document id."""
        counts: dict[str, int] = {}
        for record in self.records.values():
            counts[record["doc_id"]] = counts.get(record["doc_id"], 0) + 1
        return counts

    def __len__(self) -> int:
        return len(self.records)

    # -- persistence -----------------------------------------------------

    def save(self, directory: Path = INDEX_DIR) -> None:
        """Write both files, owner-only, in a 0700 directory.

        Written to temporaries and moved into place, so a crash mid-write
        cannot leave a half-written file that load() would misread. The two
        renames are not one atomic step: if the process dies between them the
        vector count and the sidecar disagree, which load() refuses. Failing
        closed on a torn write is the point -- the alternative is an index that
        describes chunks it does not hold.
        """
        if self.index.ntotal != len(self.records):
            raise IndexFailed(MESSAGE_FAILED)

        directory.mkdir(parents=True, exist_ok=True)
        os.chmod(directory, 0o700)

        sidecar = {
            "format_version": FORMAT_VERSION,
            "generation": self.generation,
            "next_id": self.next_id,
            "spec": asdict(self.spec),
            "records": {str(key): value for key, value in self.records.items()},
        }
        _write_blob(
            directory / VECTORS_NAME,
            _pack_vectors(self.generation, _serialize(self.index)),
        )
        _write_blob(
            directory / SIDECAR_NAME,
            json.dumps(sidecar, indent=2, sort_keys=True).encode("utf-8"),
        )


# ---------------------------------------------------------------------------
# faiss and persistence seams
# ---------------------------------------------------------------------------


def _new_index(dim: int) -> faiss.Index:
    """IndexIDMap2 over IndexFlatIP: exact cosine search with stable ids."""
    if dim < 1:
        raise IndexFailed(MESSAGE_FAILED)
    return faiss.IndexIDMap2(faiss.IndexFlatIP(dim))


def _serialize(index: faiss.Index) -> bytes:
    """The index as one buffer, which is the shape TUT-49 encrypts.

    `faiss.serialize_index` returns a numpy uint8 array rather than `bytes`,
    which architecture doc 5.1 describes loosely; the conversion belongs here
    so callers and the encryption wrapper see plain bytes.
    """
    return bytes(_guarded(lambda: faiss.serialize_index(index), MESSAGE_FAILED))


def _deserialize(blob: bytes) -> faiss.Index:
    array = np.frombuffer(blob, dtype=np.uint8)
    return _guarded(lambda: faiss.deserialize_index(array), MESSAGE_UNREADABLE)


def _pack_vectors(generation: str, blob: bytes) -> bytes:
    """Stamp the vectors file with the generation it was written in.

    A bare serialized index carries nothing identifying, so a file from another
    index -- or from an earlier write of this one -- substitutes cleanly. The
    header is the fix: 32 hex characters and a newline, ahead of the faiss
    buffer.
    """
    if not _GENERATION.fullmatch(generation):
        raise IndexFailed(MESSAGE_FAILED)
    return generation.encode("ascii") + b"\n" + blob


def _unpack_vectors(payload: bytes) -> tuple[str, bytes]:
    """Split the stamped vectors file back into its token and the faiss buffer."""
    header, separator, blob = payload.partition(b"\n")
    if not separator:
        raise IndexFailed(MESSAGE_UNREADABLE)
    generation = _guarded(
        lambda: header.decode("ascii"), MESSAGE_UNREADABLE
    )
    if not _GENERATION.fullmatch(generation):
        raise IndexFailed(MESSAGE_UNREADABLE)
    return generation, blob


def _generation_from(raw: object) -> str:
    if not isinstance(raw, str) or not _GENERATION.fullmatch(raw):
        raise IndexFailed(MESSAGE_UNREADABLE)
    return raw


def _write_blob(path: Path, payload: bytes) -> None:
    """One of the two persistence seams. TUT-49 wraps this."""
    temp = path.with_suffix(path.suffix + ".tmp")
    _guarded(lambda: temp.write_bytes(payload), MESSAGE_FAILED)
    os.chmod(temp, 0o600)
    _guarded(lambda: os.replace(temp, path), MESSAGE_FAILED)


def _read_blob(path: Path) -> bytes:
    """The other persistence seam. TUT-49 wraps this."""
    return _guarded(lambda: path.read_bytes(), MESSAGE_UNREADABLE)


def _read_sidecar(directory: Path) -> dict:
    raw = _read_blob(directory / SIDECAR_NAME)
    sidecar = _guarded(lambda: json.loads(raw.decode("utf-8")), MESSAGE_UNREADABLE)
    if not isinstance(sidecar, dict):
        raise IndexFailed(MESSAGE_UNREADABLE)
    if sidecar.get("format_version") != FORMAT_VERSION:
        raise IndexFailed(MESSAGE_UNREADABLE)
    return sidecar


def _spec_from(raw: object) -> EmbeddingSpec:
    if not isinstance(raw, dict):
        raise IndexFailed(MESSAGE_UNREADABLE)
    fields = {field for field in EmbeddingSpec.__dataclass_fields__}
    if set(raw) != fields:
        raise IndexFailed(MESSAGE_UNREADABLE)
    return _guarded(lambda: EmbeddingSpec(**raw), MESSAGE_UNREADABLE)


def _records_from(raw: object) -> dict[int, dict]:
    if not isinstance(raw, dict):
        raise IndexFailed(MESSAGE_UNREADABLE)
    required = {"doc_id", "document_name", "page_number", "heading", "text"}
    records: dict[int, dict] = {}
    for key, value in raw.items():
        if not isinstance(value, dict) or not required <= set(value):
            raise IndexFailed(MESSAGE_UNREADABLE)
        vector_id = _guarded(lambda key=key: int(key), MESSAGE_UNREADABLE)
        records[vector_id] = value
    return records


def _validated(vectors: np.ndarray, dim: int) -> np.ndarray:
    """float32, right width, finite, and unit length.

    The norm check is not pedantry: IndexFlatIP computes inner products, which
    equal cosine similarity only for unit vectors. Unnormalised input would
    rank by magnitude as much as by direction and never raise, so this is the
    one place that can catch it.
    """
    matrix = np.asarray(vectors, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dim or matrix.shape[0] == 0:
        raise IndexFailed(MESSAGE_FAILED)
    if not np.isfinite(matrix).all():
        raise IndexFailed(MESSAGE_FAILED)
    norms = np.linalg.norm(matrix, axis=1)
    if not np.allclose(norms, 1.0, atol=NORM_TOLERANCE):
        raise IndexFailed(MESSAGE_FAILED)
    return np.ascontiguousarray(matrix, dtype=np.float32)


def _guarded(call, message: str):
    """Run `call`, converting any exception out of it into IndexFailed.

    Third copy of the pattern, after parser._guarded and embedder._guarded, and
    non-obvious for the same reason: the raise happens after the except block
    exits, because `raise X from None` leaves the original on __context__ where
    an error reporter walking the chain prints whatever it quoted. faiss and the
    filesystem both raise errors carrying paths. Pipeline doc 4.5, enforced by
    tests/test_leak_invariants.py.
    """
    failed = False
    try:
        return call()
    except Exception:
        failed = True
    if failed:
        raise IndexFailed(message)
