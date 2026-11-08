"""Turn chunks into vectors with nomic-embed-text, through Ollama.

Chunks in, one unit-norm 768-dimension float32 row each, in the same order.
embed_query does the same for a single question.

Four things fail silently if changed:

- truncate=False on every call. Ollama's default embeds only the leading tokens
  of an over-long chunk, so a page gets cited in full when part of it was read.
- A character cap cannot replace that. The limit is in tokens and the ratio
  depends on content: ~12,000 chars for prose, far fewer for symbols.
- The search_document:/search_query: prefixes are part of the index format. A
  mismatch between the two sides never errors, it only ranks worse.
- An empty string embeds to a confident unit vector rather than failing, so
  empty chunk text is rejected here.

Batch size only changes how often Progress is reported; batched output is
identical to embedding one at a time.

Errors carry fixed strings from this module, never a library message and never
chunk text. See docs/ingest-pipeline.md 4.5.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ingest.model import Chunk, Progress, Stage

DEFAULT_MODEL = "nomic-embed-text"

# Part of the index format. Changing either invalidates every stored vector.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# How often Progress is reported, not a speed setting. ~17 chunks/s either way.
BATCH_SIZE = 16

# Shown to the user as Result.message, so they name a remedy, not an internal.
MESSAGE_UNAVAILABLE = (
    "The local embedding service is not running. Start Ollama and try again."
)
MESSAGE_MODEL_MISSING = (
    "The embedding model is not installed. Run: ollama pull " + DEFAULT_MODEL
)
MESSAGE_PAGE_TOO_LONG = (
    "Page {page} of this document is too long to index. Split it and try again."
)
MESSAGE_TOO_LONG = "A page of this document is too long to index."
MESSAGE_FAILED = "The document could not be indexed."


class EmbeddingFailed(Exception):
    """Embedding could not complete. Becomes Outcome.INDEXING_FAILED."""


class InputTooLong(EmbeddingFailed):
    """One chunk is past the model's context window.

    Its own type because it is the only failure here caused by the document
    rather than the machine, and so the only one worth locating to a page.
    """


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything that invalidates stored vectors if it changes.

    Written into the index sidecar and compared on load. A re-pulled model or
    an edited prefix otherwise gives plausible, wrong rankings rather than an
    error. asdict() serialises it for the sidecar header.
    """

    model: str
    digest: str  # "" when the runtime reports none; see OllamaBackend._digest
    dim: int
    context_tokens: int
    doc_prefix: str
    query_prefix: str


class Backend(Protocol):
    """The seam tests replace, so they need no Ollama.

    Implementations turn their library's errors into this module's types, so no
    library message reaches a caller.
    """

    def spec(self) -> EmbeddingSpec: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _guarded(call: Callable[[], object], message: str) -> object:
    """Run `call`, turning any exception into EmbeddingFailed.

    The raise happens after the except block exits, deliberately. `raise X from
    None` only hides the chained exception from display -- it stays on
    __context__, where anything walking the chain prints what it quoted.
    """
    failed = False
    try:
        return call()
    except Exception:
        failed = True
    if failed:
        raise EmbeddingFailed(message)


class OllamaBackend:
    """The only part of this module that imports ollama."""

    def __init__(self, model: str = DEFAULT_MODEL, client: object | None = None):
        self.model = model
        if client is None:
            import ollama

            client = ollama
        self._client = client
        self._spec: EmbeddingSpec | None = None

    def spec(self) -> EmbeddingSpec:
        """The model's declared shape, read from the model and cached.

        Not hardcoded, so a model of a different shape is caught by the index's
        spec check rather than producing a wrongly shaped index.
        """
        if self._spec is not None:
            return self._spec

        info = _guarded(lambda: self._client.show(self.model), MESSAGE_MODEL_MISSING)
        model_info = _lookup(info, "modelinfo") or _lookup(info, "model_info") or {}

        # Keys are family-prefixed, e.g. "nomic-bert.embedding_length", so match
        # on the suffix rather than hardcoding the family.
        dim = _suffixed(model_info, ".embedding_length")
        context = _suffixed(model_info, ".context_length")
        if not dim:
            raise EmbeddingFailed(MESSAGE_FAILED)

        self._spec = EmbeddingSpec(
            model=self.model,
            digest=self._digest(),
            dim=int(dim),
            context_tokens=int(context or 0),
            doc_prefix=DOC_PREFIX,
            query_prefix=QUERY_PREFIX,
        )
        return self._spec

    def _digest(self) -> str:
        """The model's content digest, or "" if the runtime will not say.

        Detects an `ollama pull` silently replacing the model under a stored
        index. Best effort: losing that one check beats refusing to embed.
        """
        try:
            listed = self._client.list()
            for entry in _lookup(listed, "models") or []:
                name = _lookup(entry, "model") or _lookup(entry, "name") or ""
                if name in (self.model, f"{self.model}:latest"):
                    return str(_lookup(entry, "digest") or "")
        except Exception:
            return ""
        return ""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed texts in order. Raises only this module's exception types.

        truncate=False is the important part: without it an over-long chunk is
        silently embedded in part. The status code is read off the error to
        classify it; the error itself is never logged or chained.
        """
        status = None
        unavailable = False
        failed = False
        try:
            response = self._client.embed(
                model=self.model, input=texts, truncate=False
            )
            return list(_lookup(response, "embeddings") or [])
        except ConnectionError:
            # ollama raises the builtin for a refused loopback connection.
            unavailable = True
            failed = True
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            failed = True

        if failed:
            if unavailable:
                raise EmbeddingFailed(MESSAGE_UNAVAILABLE)
            if status == 400:
                # Only an over-long input can produce a 400 here.
                raise InputTooLong(MESSAGE_TOO_LONG)
            if status == 404:
                raise EmbeddingFailed(MESSAGE_MODEL_MISSING)
            raise EmbeddingFailed(MESSAGE_FAILED)


class Embedder:
    """Chunks in, unit-norm float32 vectors out, Progress along the way."""

    def __init__(self, backend: Backend | None = None, batch_size: int = BATCH_SIZE):
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.backend = backend if backend is not None else OllamaBackend()
        self.batch_size = batch_size

    @property
    def spec(self) -> EmbeddingSpec:
        return self.backend.spec()

    def embed_chunks(
        self, chunks: list[Chunk]
    ) -> Generator[Progress, None, np.ndarray]:
        """Yield Progress per batch; return the (n, dim) float32 matrix.

        A generator that returns a value, so the caller can write

            vectors = yield from embedder.embed_chunks(chunks)

        and forward every Progress onward while still getting the matrix.
        Nothing runs until the generator is advanced.

        Row i belongs to chunks[i]. The count is checked, not assumed: a short
        response would shift every later citation onto the wrong page.
        """
        total = len(chunks)
        dim = self.spec.dim
        if total == 0:
            return np.zeros((0, dim), dtype=np.float32)

        self._reject_unusable(chunks)
        yield Progress(stage=Stage.EMBEDDING, current=0, total=total)

        rows: list[list[float]] = []
        for start in range(0, total, self.batch_size):
            batch = chunks[start : start + self.batch_size]
            rows.extend(self._embed_batch(batch))
            yield Progress(stage=Stage.EMBEDDING, current=len(rows), total=total)

        if len(rows) != total:
            raise EmbeddingFailed(MESSAGE_FAILED)
        return _as_unit_matrix(rows, dim)

    def embed_query(self, text: str) -> np.ndarray:
        """One (dim,) unit-norm vector for a question.

        Here rather than in retrieval so the query prefix cannot drift from the
        document one.
        """
        if not text or not text.strip():
            raise EmbeddingFailed(MESSAGE_FAILED)
        rows = self.backend.embed([self.spec.query_prefix + text])
        if len(rows) != 1:
            raise EmbeddingFailed(MESSAGE_FAILED)
        return _as_unit_matrix(rows, self.spec.dim)[0]

    def _embed_batch(self, batch: list[Chunk]) -> list[list[float]]:
        """One backend call, with an over-long chunk located to its page.

        A batch is all or nothing and the error names no item, so on that one
        failure the batch is retried a chunk at a time just to find the page.
        """
        texts = [self.spec.doc_prefix + chunk.text for chunk in batch]
        too_long = False
        try:
            rows = self.backend.embed(texts)
        except InputTooLong:
            too_long = True
        if too_long:
            raise self._locate_too_long(batch)
        return rows

    def _locate_too_long(self, batch: list[Chunk]) -> InputTooLong:
        """The exception to raise, naming the page where it can.

        Returns rather than raises, so the caller raises outside an except
        block -- see _guarded.
        """
        if len(batch) == 1:
            return InputTooLong(MESSAGE_PAGE_TOO_LONG.format(page=batch[0].page_number))
        for chunk in batch:
            culprit = False
            try:
                self.backend.embed([self.spec.doc_prefix + chunk.text])
            except InputTooLong:
                culprit = True
            except EmbeddingFailed:
                # The runtime went away mid-search; do not guess at a page.
                return InputTooLong(MESSAGE_TOO_LONG)
            if culprit:
                return InputTooLong(
                    MESSAGE_PAGE_TOO_LONG.format(page=chunk.page_number)
                )
        # Every chunk passed alone, so the batch as a whole was too large.
        return InputTooLong(MESSAGE_TOO_LONG)

    @staticmethod
    def _reject_unusable(chunks: list[Chunk]) -> None:
        """Refuse empty chunk text before it reaches the model.

        The chunker skips empty pages, so this should never fire. It exists
        because an empty string embeds to a confident unit vector that
        retrieval would rank, rather than to an error.
        """
        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                raise EmbeddingFailed(MESSAGE_FAILED)


def _as_unit_matrix(rows: list[list[float]], dim: int) -> np.ndarray:
    """Rows to a validated, L2-normalised float32 matrix.

    IndexFlatIP is cosine only for unit vectors, so normalising here is what
    makes the index's scores mean what retrieval thinks they mean.
    """
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise EmbeddingFailed(MESSAGE_FAILED)
    if not np.isfinite(matrix).all():
        raise EmbeddingFailed(MESSAGE_FAILED)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not (norms > 0).all():
        # Normalising a zero vector is a divide by zero and a matrix of NaN.
        raise EmbeddingFailed(MESSAGE_FAILED)
    return (matrix / norms).astype(np.float32, copy=False)


def _lookup(obj: object, key: str) -> object:
    """Read `key` from a dict or as an attribute, whichever the client returns."""
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)


def _suffixed(model_info: object, suffix: str) -> int | None:
    """The value of the one key ending in `suffix`, e.g. ".embedding_length"."""
    if not isinstance(model_info, dict):
        return None
    for key, value in model_info.items():
        if isinstance(key, str) and key.endswith(suffix):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None
