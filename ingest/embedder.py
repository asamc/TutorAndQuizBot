"""Turn chunks into vectors. The EMBEDDING stage of docs/ingest-interface.md.

The stack was decided elsewhere: `nomic-embed-text` served by Ollama
(architecture doc 5.1, TUT-65) feeding one flat faiss index (TUT-58). This
module is only the step between `list[Chunk]` and the float32 matrix the index
takes, plus the single query vector retrieval needs.

Five things here look like preferences and are not. Each is a measured result on
the reference machine, and getting any of them wrong fails silently:

1. **`truncate=False` on every call.** Ollama truncates by default and reports
   nothing. Measured: with the default, a chunk past the 2048-token context is
   embedded on its leading tokens and the remainder is ignored *bitwise* -- the
   vectors for `text` and for `text + anything` are byte-identical. That is a
   page embedded in part while cited in whole, which is TUT-67, and it has no
   failure surface at all. With `truncate=False` the same input is a 400 and the
   document is refused instead of quietly misindexed.

2. **A character cap cannot replace rule 1.** The obvious fix is to cap chunk
   length, but the limit is in tokens and the ratio is content-dependent: the
   cut landed near 12,000 characters for repetitive English prose, while
   symbol-heavy material tokenises far worse. A cap in the chunker is a sensible
   belt; this is the braces.

3. **The task prefixes are part of the index format.** `nomic-embed-text` is
   trained with `search_document:` on stored text and `search_query:` on
   queries. Measured by self-retrieval over 84 headed slides: recall@1 0.643
   with both prefixes against 0.583 with neither, and 0.631 with only the
   document side prefixed. Those gaps sit inside the noise for that sample size,
   so the honest reason to use them is that the model was trained that way --
   but note the failure shape. A prefix mismatch between indexing and querying
   never raises; it just quietly ranks worse. So both prefixes live in one
   place, go into `EmbeddingSpec`, and are stored with the index.

4. **Batch size is a progress knob, not a speed knob.** Measured 16.5 chunks/s
   at batch 1 against 17.3 at batch 32, and batched output is byte-identical to
   embedding the same texts one at a time. Batching buys about 5% and cannot
   change a vector, so it is sized here for how often `Progress` is reported.

5. **An empty string embeds successfully**, to a confident unit-norm vector
   rather than an error or a zero vector. The chunker already skips empty pages,
   so this module refuses empty text instead of trusting that -- the fail-closed
   rule of ingest-pipeline 4.6, applied to the one input that fails open.

Leak discipline (ingest-pipeline 4.5): `EmbeddingFailed` carries a fixed string
written here, never a library message and never chunk text. Page numbers are
allowed and used, because a user whose document was refused needs to know which
page to shorten. See `_guarded` for why a raise cannot happen inside an `except`
block.
"""

from __future__ import annotations

from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ingest.model import Chunk, Progress, Stage

# Decided in architecture doc 5.1 (TUT-65). The dimension is deliberately not a
# constant here: it is read back from the model, so a swapped model becomes a
# spec mismatch rather than a silently wrong-shaped index.
DEFAULT_MODEL = "nomic-embed-text"

# Part of the index format, not tuning parameters. See note 3 above.
DOC_PREFIX = "search_document: "
QUERY_PREFIX = "search_query: "

# Progress granularity. At ~17 chunks/s a 10-page lecture is under a second, so
# this reports about once per lecture and the whole semester in ~16 events.
BATCH_SIZE = 16

# Display-safe messages. These reach the user as Result.message under
# Outcome.INDEXING_FAILED, so they name a remedy and never an internal.
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
    """Embedding could not complete. Maps to Outcome.INDEXING_FAILED.

    Always carries a fixed, display-safe message from this module. Never a
    library error, never document text.
    """


class InputTooLong(EmbeddingFailed):
    """One chunk exceeds the model's context window.

    Separate from its parent because it is the only failure here that is the
    document's fault rather than the machine's, and so the only one worth
    locating to a page.
    """


@dataclass(frozen=True)
class EmbeddingSpec:
    """Everything about this embedder that invalidates stored vectors.

    Written into the index sidecar and compared on load. Any difference means
    stored vectors are not comparable with freshly computed ones, and nothing
    else would notice: a re-pulled model or an edited prefix produces plausible,
    wrong rankings rather than an error.

    `dataclasses.asdict` serialises this as-is for the sidecar header.
    """

    model: str
    digest: str  # "" when the runtime reports none; see OllamaBackend._digest
    dim: int
    context_tokens: int
    doc_prefix: str
    query_prefix: str


class Backend(Protocol):
    """The seam. Tests replace this; nothing else in the module imports ollama.

    Implementations translate their library's errors into EmbeddingFailed or
    InputTooLong, so no library message can reach a caller.
    """

    def spec(self) -> EmbeddingSpec: ...

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def _guarded(call: Callable[[], object], message: str) -> object:
    """Run `call`, converting any exception out of it into EmbeddingFailed.

    The twin of parser._guarded, and non-obvious for the same reason: the raise
    happens *after* the except block exits. `raise X from None` only suppresses
    the display of the chained exception, which stays on __context__ where an
    error reporter walking the chain prints it. Pipeline doc 4.5.

    Kept local rather than shared with the parser so that neither module reaches
    into the other's privates for a security-critical eight lines.
    """
    failed = False
    try:
        return call()
    except Exception:
        failed = True
    if failed:
        raise EmbeddingFailed(message)


class OllamaBackend:
    """Talks to the local Ollama runtime, which binds to loopback only.

    The only part of this module that knows the embedding library exists.
    """

    def __init__(self, model: str = DEFAULT_MODEL, client: object | None = None):
        self.model = model
        if client is None:
            import ollama

            client = ollama
        self._client = client
        self._spec: EmbeddingSpec | None = None

    def spec(self) -> EmbeddingSpec:
        """The model's own declared shape, cached.

        Dimension and context length are read from the model rather than fixed
        here, so a model whose shape differs is caught by the index's spec
        comparison instead of producing a wrongly shaped index.
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

        Best effort on purpose. The digest is what detects a silent
        `ollama pull` replacing the model under a stored index, which is worth
        having -- but not worth refusing to embed over, so a runtime that does
        not report one degrades to losing that single check.
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

        `truncate=False` is the whole point of this method -- note 1 in the
        module docstring. Without it an over-long chunk is silently embedded in
        part. The status code is read off the error to classify it, but the
        error itself is never bound beyond this block, logged, or chained.
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
                # "the input length exceeds the context length" -- the only 400
                # this call can provoke, since the model and the input shape are
                # both built here.
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

        A generator that returns a value, so the orchestrator can write

            vectors = yield from embedder.embed_chunks(chunks)

        and have every Progress forwarded to the UI untouched while still
        getting the matrix back. Nothing runs until the generator is advanced,
        so validation errors surface on the first `next`, not at the call.

        Row i belongs to chunks[i]. That correspondence is the only thing
        holding citations to vectors, so the count is asserted rather than
        assumed: a short response would shift every citation after the gap onto
        the wrong page.
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

        Lives here rather than in retrieval so the query prefix cannot drift
        from the document prefix: they are two halves of one format, and a
        mismatch between them degrades ranking without ever raising.
        """
        if not text or not text.strip():
            raise EmbeddingFailed(MESSAGE_FAILED)
        rows = self.backend.embed([self.spec.query_prefix + text])
        if len(rows) != 1:
            raise EmbeddingFailed(MESSAGE_FAILED)
        return _as_unit_matrix(rows, self.spec.dim)[0]

    def _embed_batch(self, batch: list[Chunk]) -> list[list[float]]:
        """One backend call, with an over-long chunk located to its page.

        A batch is all or nothing: one over-long chunk fails the whole call and
        the runtime's error does not say which item caused it. So on that one
        failure the batch is retried a chunk at a time, purely to name the page.
        That costs an extra round trip per chunk, and only for a document that
        is already being refused.
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
        """The exception to raise, naming the offending page where it can.

        Returns rather than raises so the caller raises it outside an except
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
                # The runtime went away mid-search. Stop guessing at pages.
                return InputTooLong(MESSAGE_TOO_LONG)
            if culprit:
                return InputTooLong(
                    MESSAGE_PAGE_TOO_LONG.format(page=chunk.page_number)
                )
        # Every chunk passed alone, so the batch as a whole was too large rather
        # than any single page. Possible if a backend bounds the request, not
        # the item.
        return InputTooLong(MESSAGE_TOO_LONG)

    @staticmethod
    def _reject_unusable(chunks: list[Chunk]) -> None:
        """Refuse empty chunk text before it reaches the model.

        The chunker skips empty pages, so this should never fire. It is here
        because the failure is invisible if it ever does: an empty string embeds
        to a perfectly confident unit vector that retrieval will happily rank,
        rather than to an error or a zero vector. Fail closed.
        """
        for chunk in chunks:
            if not chunk.text or not chunk.text.strip():
                raise EmbeddingFailed(MESSAGE_FAILED)


def _as_unit_matrix(rows: list[list[float]], dim: int) -> np.ndarray:
    """Rows to a validated, L2-normalised float32 matrix.

    Ollama returns unit-norm vectors today, measured, but documents no such
    guarantee and another backend need not. IndexFlatIP is cosine only for
    normalised input, so normalising here is what makes the index's scores mean
    what retrieval thinks they mean.
    """
    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise EmbeddingFailed(MESSAGE_FAILED)
    if not np.isfinite(matrix).all():
        raise EmbeddingFailed(MESSAGE_FAILED)

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if not (norms > 0).all():
        # A zero vector has no direction, so every similarity against it is 0.
        # Normalising it would be a divide by zero and a matrix of NaN.
        raise EmbeddingFailed(MESSAGE_FAILED)
    return (matrix / norms).astype(np.float32, copy=False)


def _lookup(obj: object, key: str) -> object:
    """Read `key` from a mapping or as an attribute of a response object.

    The Ollama client has returned plain dicts and pydantic models at different
    versions, and both shapes turn up depending on what is installed.
    """
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
