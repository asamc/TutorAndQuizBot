"""Attack the embedder. See ingest/embedder.py and docs/ingest-interface.md.

Everything here runs in CI. The Ollama runtime is behind the Backend protocol,
so the fake below stands in for it -- which is the whole reason that seam
exists. The measured properties the module relies on (truncate=False refusing
instead of truncating, batched output matching single, bitwise determinism) are
asserted against the real runtime too, at the bottom, behind a skipif.

The fake deliberately returns vectors that are NOT unit length, so that any test
asserting normalisation is testing this module rather than Ollama's habits.
"""

import hashlib

import numpy as np
import pytest

from ingest.embedder import (
    BATCH_SIZE,
    DOC_PREFIX,
    MESSAGE_FAILED,
    MESSAGE_MODEL_MISSING,
    MESSAGE_TOO_LONG,
    MESSAGE_UNAVAILABLE,
    QUERY_PREFIX,
    Embedder,
    EmbeddingFailed,
    EmbeddingSpec,
    InputTooLong,
    OllamaBackend,
)
from ingest.model import Chunk, Progress, Stage

DIM = 8


def spec(dim: int = DIM) -> EmbeddingSpec:
    return EmbeddingSpec(
        model="fake-embed",
        digest="0" * 64,
        dim=dim,
        context_tokens=128,
        doc_prefix=DOC_PREFIX,
        query_prefix=QUERY_PREFIX,
    )


class FakeBackend:
    """Deterministic, order-preserving, and never unit length."""

    def __init__(self, dim: int = DIM, scale: float = 7.5):
        self.dim = dim
        self.scale = scale
        self.calls: list[list[str]] = []

    def spec(self) -> EmbeddingSpec:
        return spec(self.dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return [self.vector(text) for text in texts]

    def vector(self, text: str) -> list[float]:
        seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        return (rng.normal(size=self.dim) * self.scale).tolist()


class LengthLimited(FakeBackend):
    """Fails the whole batch when any item is too long, as Ollama does."""

    def __init__(self, limit: int, **kwargs):
        super().__init__(**kwargs)
        self.limit = limit

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        if any(len(text) > self.limit for text in texts):
            raise InputTooLong(MESSAGE_TOO_LONG)
        return [self.vector(text) for text in texts]


class Broken(FakeBackend):
    """Returns a deliberately wrong response shape."""

    def __init__(self, rows, **kwargs):
        super().__init__(**kwargs)
        self.rows = rows

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        return self.rows


def pages(*texts: str) -> list[Chunk]:
    return [
        Chunk(document_name="deck.pdf", page_number=i, heading=None, text=text)
        for i, text in enumerate(texts, start=1)
    ]


def run(embedder: Embedder, chunks: list[Chunk]):
    """Drive the generator the way the orchestrator will, keeping the events."""
    events: list[Progress] = []
    generator = embedder.embed_chunks(chunks)
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            return events, stop.value


# ---------------------------------------------------------------------------
# The matrix handed to the index
# ---------------------------------------------------------------------------


def test_shape_and_dtype_match_the_index_contract():
    """faiss takes float32 and nothing else; a float64 matrix raises there."""
    _, vectors = run(Embedder(FakeBackend()), pages("alpha", "beta", "gamma"))
    assert vectors.shape == (3, DIM)
    assert vectors.dtype == np.float32


def test_vectors_are_normalised_even_though_the_backend_returns_raw():
    """IndexFlatIP is cosine only for unit vectors, so this is what makes a
    stored score mean similarity rather than magnitude."""
    backend = FakeBackend(scale=7.5)
    _, vectors = run(Embedder(backend), pages("alpha", "beta"))
    raw = np.linalg.norm(backend.vector(DOC_PREFIX + "alpha"))
    assert raw > 2, "fixture is supposed to be far from unit length"
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-6)


def test_row_order_follows_chunk_order():
    """The only thing holding a citation to a vector. If this slips, every
    answer cites the wrong page and nothing raises."""
    backend = FakeBackend()
    chunks = pages("alpha", "beta", "gamma", "delta")
    _, vectors = run(Embedder(backend, batch_size=2), chunks)
    for i, chunk in enumerate(chunks):
        expected = np.asarray(backend.vector(DOC_PREFIX + chunk.text), dtype=np.float32)
        expected /= np.linalg.norm(expected)
        assert np.allclose(vectors[i], expected, atol=1e-6), chunk.text


def test_empty_chunk_list_returns_an_empty_matrix_and_no_events():
    """A document whose every page was empty. The index needs a well-shaped
    (0, dim) array, not an error and not a bare list."""
    events, vectors = run(Embedder(FakeBackend()), [])
    assert events == []
    assert vectors.shape == (0, DIM)
    assert vectors.dtype == np.float32


# ---------------------------------------------------------------------------
# Prefixes. Note 3 in the module docstring: a mismatch never raises.
# ---------------------------------------------------------------------------


def test_document_prefix_is_applied_to_every_chunk():
    backend = FakeBackend()
    run(Embedder(backend), pages("alpha", "beta"))
    sent = [text for call in backend.calls for text in call]
    assert sent == [DOC_PREFIX + "alpha", DOC_PREFIX + "beta"]


def test_query_uses_the_query_prefix_not_the_document_one():
    backend = FakeBackend()
    Embedder(backend).embed_query("what is integrity")
    assert backend.calls == [[QUERY_PREFIX + "what is integrity"]]
    assert QUERY_PREFIX != DOC_PREFIX


def test_query_vector_is_one_normalised_row():
    vector = Embedder(FakeBackend()).embed_query("what is integrity")
    assert vector.shape == (DIM,)
    assert vector.dtype == np.float32
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-6)


def test_the_same_text_embeds_differently_as_document_and_query():
    """Proof the two prefixes are actually reaching the model. If both sides
    used one prefix this passes trivially; if neither did, it fails."""
    backend = FakeBackend()
    _, document = run(Embedder(backend), pages("integrity"))
    query = Embedder(backend).embed_query("integrity")
    assert not np.allclose(document[0], query)


# ---------------------------------------------------------------------------
# Progress. docs/ingest-interface.md section 4 promises current and total here.
# ---------------------------------------------------------------------------


def test_progress_starts_at_zero_ends_at_total_and_never_goes_backwards():
    chunks = pages(*[f"chunk {i}" for i in range(7)])
    events, _ = run(Embedder(FakeBackend(), batch_size=3), chunks)

    assert all(event.stage is Stage.EMBEDDING for event in events)
    assert all(event.total == 7 for event in events)
    assert events[0].current == 0
    assert events[-1].current == 7
    currents = [event.current for event in events]
    assert currents == sorted(currents)
    assert currents == [0, 3, 6, 7]


def test_batch_size_decides_call_count_not_the_result():
    """Measured: batched output is byte-identical to single. So batch size may
    change how often progress is reported and nothing else."""
    chunks = pages(*[f"chunk {i}" for i in range(5)])
    _, one = run(Embedder(FakeBackend(), batch_size=1), chunks)
    _, many = run(Embedder(FakeBackend(), batch_size=5), chunks)
    assert np.array_equal(one, many)

    backend = FakeBackend()
    run(Embedder(backend, batch_size=2), chunks)
    assert [len(call) for call in backend.calls] == [2, 2, 1]


def test_default_batch_size_is_sane():
    assert BATCH_SIZE >= 1
    chunks = pages(*[f"chunk {i}" for i in range(BATCH_SIZE + 1)])
    backend = FakeBackend()
    run(Embedder(backend), chunks)
    assert [len(call) for call in backend.calls] == [BATCH_SIZE, 1]


def test_rejects_a_batch_size_below_one():
    with pytest.raises(ValueError):
        Embedder(FakeBackend(), batch_size=0)


def test_nothing_runs_until_the_generator_is_advanced():
    """It is a generator, so the backend must not be touched at call time."""
    backend = FakeBackend()
    generator = Embedder(backend).embed_chunks(pages("alpha"))
    assert backend.calls == []
    next(generator)
    assert backend.calls == []  # first event is the 0-of-total, before any work


# ---------------------------------------------------------------------------
# Fail closed. ingest-pipeline 4.6
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("text", ["", " ", "\n\n", "\t  \n"])
def test_empty_chunk_text_is_refused_before_the_model_sees_it(text):
    """An empty string embeds to a confident unit vector rather than an error,
    so retrieval would rank meaningless content. Measured on nomic-embed-text."""
    backend = FakeBackend()
    with pytest.raises(EmbeddingFailed):
        run(Embedder(backend), pages("fine", text))
    assert backend.calls == [], "refused after sending data to the model"


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_query_is_refused(text):
    backend = FakeBackend()
    with pytest.raises(EmbeddingFailed):
        Embedder(backend).embed_query(text)
    assert backend.calls == []


def test_short_response_is_refused_rather_than_misaligned():
    """Dropping a vector would silently shift every later citation onto the
    wrong page, which no downstream check could detect."""
    backend = Broken(rows=[[1.0] * DIM])  # one vector for two chunks
    with pytest.raises(EmbeddingFailed):
        run(Embedder(backend), pages("alpha", "beta"))


def test_wrong_dimension_is_refused():
    backend = Broken(rows=[[1.0] * (DIM + 1)])
    with pytest.raises(EmbeddingFailed):
        run(Embedder(backend), pages("alpha"))


def test_zero_vector_is_refused_instead_of_producing_nan():
    """Normalising a zero vector is a divide by zero, and a NaN row poisons
    every search result rather than just its own."""
    backend = Broken(rows=[[0.0] * DIM])
    with pytest.raises(EmbeddingFailed):
        run(Embedder(backend), pages("alpha"))


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_non_finite_values_are_refused(bad):
    backend = Broken(rows=[[bad] + [1.0] * (DIM - 1)])
    with pytest.raises(EmbeddingFailed):
        run(Embedder(backend), pages("alpha"))


# ---------------------------------------------------------------------------
# Over-long input. TUT-67, and the reason truncate=False exists.
# ---------------------------------------------------------------------------


def test_over_long_chunk_is_located_to_its_page():
    """The runtime fails the whole batch and does not say which item. A user
    told only 'a page is too long' cannot act; the page number is metadata and
    4.5 permits it."""
    backend = LengthLimited(limit=50)
    chunks = pages("short", "short", "x" * 500, "short")
    with pytest.raises(InputTooLong) as excinfo:
        run(Embedder(backend, batch_size=4), chunks)
    assert "3" in str(excinfo.value), str(excinfo.value)


def test_over_long_chunk_is_located_even_in_a_batch_of_one():
    backend = LengthLimited(limit=50)
    with pytest.raises(InputTooLong) as excinfo:
        run(Embedder(backend, batch_size=1), pages("short", "y" * 500))
    assert "2" in str(excinfo.value), str(excinfo.value)


def test_too_long_is_a_subclass_so_callers_can_treat_it_as_one_failure():
    assert issubclass(InputTooLong, EmbeddingFailed)


def test_locating_stops_guessing_if_the_runtime_disappears_mid_search():
    """Between failing the batch and locating the page, Ollama can go away.
    Reporting a page number then would be a guess."""

    class DiesWhileLocating(FakeBackend):
        def __init__(self):
            super().__init__()
            self.first = True

        def embed(self, texts):
            self.calls.append(list(texts))
            if self.first:
                self.first = False
                raise InputTooLong(MESSAGE_TOO_LONG)
            raise EmbeddingFailed(MESSAGE_UNAVAILABLE)

    with pytest.raises(InputTooLong) as excinfo:
        run(Embedder(DiesWhileLocating(), batch_size=3), pages("a", "b", "c"))
    assert str(excinfo.value) == MESSAGE_TOO_LONG


# ---------------------------------------------------------------------------
# Leak channels. ingest-pipeline 4.5
# ---------------------------------------------------------------------------


PLANTED_PHRASE = "ZEROCOOLCANARY"


@pytest.mark.parametrize(
    "backend_factory",
    [
        lambda: Broken(rows=[[0.0] * DIM]),
        lambda: Broken(rows=[[1.0] * (DIM + 1)]),
        lambda: Broken(rows=[]),
        lambda: LengthLimited(limit=5),
    ],
    ids=["zero-vector", "wrong-dim", "no-rows", "too-long"],
)
def test_no_failure_path_quotes_the_chunk_text(backend_factory):
    """Every raise in this module carries a fixed string. Document text reaching
    a message is an R2 leak, and the message is shown to the user."""
    chunks = pages(f"page one mentioning {PLANTED_PHRASE} in passing")
    with pytest.raises(EmbeddingFailed) as excinfo:
        run(Embedder(backend_factory()), chunks)

    error = excinfo.value
    assert PLANTED_PHRASE not in str(error)
    assert PLANTED_PHRASE not in repr(error.args)


def test_backend_errors_do_not_survive_on_the_exception_chain():
    """The parser's lesson applies here unchanged: a raise inside an except
    block leaves the original on __context__, where a traceback-logging handler
    prints whatever it quoted. parser._guarded and embedder._guarded both exist
    for this."""
    backend = OllamaBackend(
        model="fake-embed", client=_FakeClient(raise_exc=RuntimeError(PLANTED_PHRASE))
    )
    with pytest.raises(EmbeddingFailed) as excinfo:
        backend.embed(["anything"])

    error = excinfo.value
    assert PLANTED_PHRASE not in str(error)
    assert error.__cause__ is None
    assert error.__context__ is None, "library error still on the chain"


def test_spec_failure_does_not_leak_the_runtime_error():
    backend = OllamaBackend(client=_FakeClient(show_exc=RuntimeError(PLANTED_PHRASE)))
    with pytest.raises(EmbeddingFailed) as excinfo:
        backend.spec()
    assert PLANTED_PHRASE not in str(excinfo.value)
    assert excinfo.value.__context__ is None


# ---------------------------------------------------------------------------
# OllamaBackend: the only part that knows the library exists
# ---------------------------------------------------------------------------


class _FakeResponseError(Exception):
    """Shaped like ollama.ResponseError, which carries status_code."""

    def __init__(self, status_code: int, message: str = "upstream detail"):
        super().__init__(message)
        self.status_code = status_code


class _FakeClient:
    def __init__(
        self,
        embeddings=None,
        raise_exc=None,
        show_exc=None,
        model_info=None,
        models=None,
    ):
        self.embeddings = embeddings if embeddings is not None else [[1.0] * DIM]
        self.raise_exc = raise_exc
        self.show_exc = show_exc
        self.model_info = (
            model_info
            if model_info is not None
            else {
                "nomic-bert.embedding_length": DIM,
                "nomic-bert.context_length": 2048,
            }
        )
        self.models = models
        self.embed_kwargs: dict = {}

    def show(self, model):
        if self.show_exc is not None:
            raise self.show_exc
        return {"modelinfo": self.model_info}

    def list(self):
        if self.models is None:
            return {"models": [{"model": "fake-embed:latest", "digest": "abc123"}]}
        return {"models": self.models}

    def embed(self, **kwargs):
        self.embed_kwargs = kwargs
        if self.raise_exc is not None:
            raise self.raise_exc
        return {"embeddings": self.embeddings}


def test_truncate_is_always_disabled():
    """The single most important line in the module. With Ollama's default, an
    over-long chunk is embedded on its leading tokens and the rest is dropped
    bitwise -- a page embedded in part and cited in whole, with no error."""
    client = _FakeClient()
    OllamaBackend(model="fake-embed", client=client).embed(["text"])
    assert client.embed_kwargs.get("truncate") is False, (
        "truncate must be passed explicitly as False; Ollama's default silently "
        "embeds the leading tokens of an over-long chunk and drops the rest"
    )


def test_spec_is_read_from_the_model_not_hardcoded():
    client = _FakeClient(
        model_info={"someother.embedding_length": 1024, "someother.context_length": 512}
    )
    got = OllamaBackend(model="fake-embed", client=client).spec()
    assert got.dim == 1024
    assert got.context_tokens == 512
    assert got.doc_prefix == DOC_PREFIX
    assert got.query_prefix == QUERY_PREFIX


def test_spec_is_cached_so_it_costs_one_round_trip():
    class CountingClient(_FakeClient):
        shows = 0

        def show(self, model):
            CountingClient.shows += 1
            return super().show(model)

    backend = OllamaBackend(model="fake-embed", client=CountingClient())
    backend.spec()
    backend.spec()
    assert CountingClient.shows == 1


def test_spec_records_the_model_digest_to_catch_a_silent_repull():
    backend = OllamaBackend(model="fake-embed", client=_FakeClient())
    assert backend.spec().digest == "abc123"


def test_a_runtime_that_reports_no_digest_still_embeds():
    """Losing one check is better than refusing to work."""
    backend = OllamaBackend(model="fake-embed", client=_FakeClient(models=[]))
    assert backend.spec().digest == ""


def test_a_model_without_a_declared_dimension_is_refused():
    backend = OllamaBackend(model="fake-embed", client=_FakeClient(model_info={}))
    with pytest.raises(EmbeddingFailed):
        backend.spec()


@pytest.mark.parametrize(
    "exc, expected",
    [
        (_FakeResponseError(404), MESSAGE_MODEL_MISSING),
        (ConnectionError("refused"), MESSAGE_UNAVAILABLE),
        (RuntimeError("something else"), MESSAGE_FAILED),
    ],
)
def test_runtime_errors_become_our_own_messages(exc, expected):
    backend = OllamaBackend(model="fake-embed", client=_FakeClient(raise_exc=exc))
    with pytest.raises(EmbeddingFailed) as excinfo:
        backend.embed(["text"])
    assert str(excinfo.value) == expected


def test_a_400_becomes_input_too_long_specifically():
    """Measured: truncate=False turns an over-long input into exactly this."""
    backend = OllamaBackend(
        model="fake-embed", client=_FakeClient(raise_exc=_FakeResponseError(400))
    )
    with pytest.raises(InputTooLong):
        backend.embed(["text"])


# ---------------------------------------------------------------------------
# Against the real runtime. Skipped in CI, which has no Ollama.
# ---------------------------------------------------------------------------


def _ollama_ready() -> bool:
    try:
        import ollama

        names = {m["model"] for m in ollama.list()["models"]}
    except Exception:
        return False
    return any(name.startswith("nomic-embed-text") for name in names)


needs_ollama = pytest.mark.skipif(
    not _ollama_ready(), reason="no local Ollama with nomic-embed-text"
)


@needs_ollama
def test_real_runtime_reports_the_shape_the_architecture_doc_claims():
    got = OllamaBackend().spec()
    assert got.dim == 768
    assert got.context_tokens == 2048
    assert len(got.digest) == 64


@needs_ollama
def test_real_runtime_is_deterministic_and_normalised():
    """Determinism is what makes a stored index comparable with a later query,
    and it is the premise the fake backend imitates."""
    embedder = Embedder()
    chunks = pages("Integrity means data has not been modified in transit.")
    _, first = run(embedder, chunks)
    _, second = run(embedder, chunks)
    assert np.array_equal(first, second)
    assert np.allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-6)


@needs_ollama
def test_real_runtime_refuses_an_over_long_page_instead_of_truncating():
    """TUT-67 end to end. Without truncate=False this silently returns a vector
    for the first ~2048 tokens of a page that gets cited in full."""
    embedder = Embedder()
    long_page = pages("Integrity means data has not been modified. " * 900)
    with pytest.raises(InputTooLong) as excinfo:
        run(embedder, long_page)
    assert "1" in str(excinfo.value)


@needs_ollama
def test_real_runtime_batching_does_not_change_a_vector():
    """The measurement the batch-size comment rests on."""
    chunks = pages(*[f"Network security topic number {i}." for i in range(6)])
    _, single = run(Embedder(batch_size=1), chunks)
    _, batched = run(Embedder(batch_size=6), chunks)
    assert np.array_equal(single, batched)
