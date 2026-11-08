"""The vector index and its chunk sidecar under data/index/.

One flat faiss index over unit-norm 768-dimension vectors. The whole semester is
~250 chunks, far below where an approximate index is worth its complexity, so
IndexFlatIP gives exact search with no training step.

Four structural points:

- Vectors carry explicit ids, through IndexIDMap2. A bare IndexFlatIP addresses
  by position and compacts on removal, so deleting one document would renumber
  the rest and the sidecar would then describe the wrong chunks. Ids come from a
  counter and are never reused.
- Chunk text and citations live in the sidecar, not in faiss, keyed by vector
  id. The sidecar also owns the document-to-vector mapping; the manifest stays
  the integrity artifact outside data/ and holds no index internals.
- Each write stamps a random `generation` into both files and load refuses a
  pair that disagrees. Counting vectors is not enough: two indexes with the same
  chunk count can have their vector files swapped, and the result loads happily
  while citing the wrong document at similarity 1.0.
- All persistence goes through _read_blob/_write_blob, so encryption at rest
  becomes a wrapper rather than a rewrite.

The sidecar holds document text at rest: 0600 in a 0700 directory, gitignored.
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

# The index half of "every write under data/ lives in one place"; manifest.py
# is the documents half.
INDEX_DIR = Path("data/index")
VECTORS_NAME = "vectors.faiss"
SIDECAR_NAME = "sidecar.json"

# Bumped when the sidecar layout changes; an unknown version is refused.
FORMAT_VERSION = 1

# Loose enough for float32 rounding, tight enough to catch raw vectors.
NORM_TOLERANCE = 1e-3

# secrets.token_hex(16), validated on read and write.
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

    Becomes Outcome.INDEXING_FAILED. Carries a fixed message, never library
    text and never document text.
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

        `expected_spec` is the live embedder's settings. An index built with a
        different model, dimension or prefix holds vectors that are not
        comparable with fresh ones, and the symptom is plausible wrong rankings
        rather than an error, so this is the only place it can be caught.
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

        Row i of `vectors` must belong to chunks[i]. That pairing is what holds
        a citation to a vector, so it is checked rather than trusted.
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

        Ids are not recycled. IndexIDMap2 keeps every surviving vector's id
        across a removal, which is why it is used at all.
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

        Scores are inner products of unit vectors, so cosine similarities in
        [-1, 1]. faiss pads a short result with id -1; those are dropped, which
        is the normal case whenever top_k exceeds the number of chunks.
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

    def chunks_for(self, doc_id: str) -> list[Chunk]:
        """One document's indexed chunks, in vector id order.

        Returns Chunk objects so callers get Chunk.citation() rather than
        reimplementing it.
        """
        return [
            Chunk(
                document_name=record["document_name"],
                page_number=record["page_number"],
                heading=record["heading"],
                text=record["text"],
            )
            for _, record in sorted(self.records.items())
            if record["doc_id"] == doc_id
        ]

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

        Each goes to a temporary and is moved into place, so a crash cannot
        leave a half-written file. The two renames are not one atomic step: if
        the process dies between them the generation tokens disagree and load
        refuses the pair, which is the intended outcome.
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
    """The index as one buffer, ready to encrypt.

    faiss.serialize_index returns a numpy uint8 array rather than bytes, so the
    conversion belongs here.
    """
    return bytes(_guarded(lambda: faiss.serialize_index(index), MESSAGE_FAILED))


def _deserialize(blob: bytes) -> faiss.Index:
    array = np.frombuffer(blob, dtype=np.uint8)
    return _guarded(lambda: faiss.deserialize_index(array), MESSAGE_UNREADABLE)


def _pack_vectors(generation: str, blob: bytes) -> bytes:
    """Stamp the vectors file with the generation it was written in.

    A bare serialized index carries nothing identifying, so a file from another
    index substitutes cleanly. Header is 32 hex characters and a newline.
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
    """One of the two persistence seams; encryption would wrap this."""
    temp = path.with_suffix(path.suffix + ".tmp")
    _guarded(lambda: temp.write_bytes(payload), MESSAGE_FAILED)
    os.chmod(temp, 0o600)
    _guarded(lambda: os.replace(temp, path), MESSAGE_FAILED)


def _read_blob(path: Path) -> bytes:
    """The other persistence seam; encryption would wrap this."""
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

    The norm check matters: IndexFlatIP computes inner products, which equal
    cosine only for unit vectors. Unnormalised input would rank partly by
    magnitude and never raise.
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
    """Run `call`, turning any exception into IndexFailed.

    The raise happens after the except block exits: `raise X from None` only
    hides the chained exception, which stays on __context__. faiss and the
    filesystem both raise errors carrying paths.
    """
    failed = False
    try:
        return call()
    except Exception:
        failed = True
    if failed:
        raise IndexFailed(message)
