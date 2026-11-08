"""Attack the index. TUT-58. See ingest/index.py.

Everything here runs in CI: vectors are just arrays, so no Ollama is needed, and
every persistence test uses tmp_path rather than data/index.

The tests worth reading first are the id-stability ones. A bare IndexFlatIP
renumbers vectors when one is removed, which would make the sidecar describe the
wrong chunks, and the symptom is a confidently wrong citation rather than an
error. That is the failure this module's whole shape exists to prevent.
"""

import json

import numpy as np
import pytest

from ingest.embedder import DOC_PREFIX, QUERY_PREFIX, EmbeddingSpec
from ingest.index import (
    FORMAT_VERSION,
    SIDECAR_NAME,
    VECTORS_NAME,
    ChunkIndex,
    IndexFailed,
)
from ingest.model import Chunk

DIM = 4


def spec(**overrides) -> EmbeddingSpec:
    base = dict(
        model="fake-embed",
        digest="a" * 64,
        dim=DIM,
        context_tokens=2048,
        doc_prefix=DOC_PREFIX,
        query_prefix=QUERY_PREFIX,
    )
    base.update(overrides)
    return EmbeddingSpec(**base)


def unit(*values: float) -> np.ndarray:
    """One unit-length row, which is what the index requires."""
    row = np.asarray([values], dtype=np.float32)
    return row / np.linalg.norm(row)


def rows(*vectors: np.ndarray) -> np.ndarray:
    return np.vstack(vectors)


def chunk(name: str = "deck.pdf", page: int = 1, heading: str | None = None) -> Chunk:
    return Chunk(
        document_name=name,
        page_number=page,
        heading=heading,
        text=f"text of {name} page {page}",
    )


def populated() -> ChunkIndex:
    """Two documents: 'aaa' with two chunks, 'bbb' with one."""
    index = ChunkIndex.create(spec())
    index.add(
        "aaa",
        [chunk("a.pdf", 1), chunk("a.pdf", 2)],
        rows(unit(1, 0, 0, 0), unit(0, 1, 0, 0)),
    )
    index.add("bbb", [chunk("b.pdf", 1)], unit(0, 0, 1, 0))
    return index


# ---------------------------------------------------------------------------
# Adding
# ---------------------------------------------------------------------------


def test_ids_are_allocated_in_order_and_do_not_restart_per_document():
    index = ChunkIndex.create(spec())
    first = index.add("aaa", [chunk(), chunk(page=2)], rows(unit(1, 0, 0, 0), unit(0, 1, 0, 0)))
    second = index.add("bbb", [chunk("b.pdf")], unit(0, 0, 1, 0))
    assert first == [0, 1]
    assert second == [2]
    assert len(index) == 3
    assert index.documents() == {"aaa": 2, "bbb": 1}


def test_adding_no_chunks_is_a_no_op():
    """A document whose every page was empty still reaches the index."""
    index = ChunkIndex.create(spec())
    assert index.add("aaa", [], np.zeros((0, DIM), dtype=np.float32)) == []
    assert len(index) == 0


def test_a_vector_per_chunk_is_required():
    """Row i belongs to chunks[i]. A length mismatch means that pairing is
    already broken, and nothing downstream could detect it."""
    index = ChunkIndex.create(spec())
    with pytest.raises(IndexFailed):
        index.add("aaa", [chunk(), chunk(page=2)], unit(1, 0, 0, 0))


def test_the_same_document_cannot_be_indexed_twice():
    """Duplicate suppression is the manifest's job; reaching here means the
    manifest and the index disagree, which is not fixed by indexing twice."""
    index = ChunkIndex.create(spec())
    index.add("aaa", [chunk()], unit(1, 0, 0, 0))
    with pytest.raises(IndexFailed):
        index.add("aaa", [chunk(page=2)], unit(0, 1, 0, 0))
    assert len(index) == 1


def test_unnormalised_vectors_are_refused():
    """The highest-value check in this file after id stability.

    IndexFlatIP computes inner products, which are cosine similarity only for
    unit vectors. Raw vectors would rank partly by magnitude, score above 1.0,
    and never raise -- so this is the one place it can be caught.
    """
    index = ChunkIndex.create(spec())
    raw = np.asarray([[1.0, 1.0, 0.0, 0.0]], dtype=np.float32)  # norm sqrt(2)
    assert not np.isclose(np.linalg.norm(raw), 1.0)
    with pytest.raises(IndexFailed):
        index.add("aaa", [chunk()], raw)


def test_wrong_width_is_refused():
    index = ChunkIndex.create(spec())
    with pytest.raises(IndexFailed):
        index.add("aaa", [chunk()], unit(1, 0, 0, 0, 0))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_non_finite_vectors_are_refused(bad):
    index = ChunkIndex.create(spec())
    vector = np.asarray([[bad, 0.0, 0.0, 0.0]], dtype=np.float32)
    with pytest.raises(IndexFailed):
        index.add("aaa", [chunk()], vector)


def test_an_empty_document_id_is_refused():
    index = ChunkIndex.create(spec())
    with pytest.raises(IndexFailed):
        index.add("", [chunk()], unit(1, 0, 0, 0))


# ---------------------------------------------------------------------------
# Searching
# ---------------------------------------------------------------------------


def test_search_returns_the_nearest_chunk_with_a_cosine_score():
    index = populated()
    hits = index.search(unit(1, 0, 0, 0)[0], top_k=1)
    assert len(hits) == 1
    assert hits[0].doc_id == "aaa"
    assert hits[0].chunk.page_number == 1
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_scores_are_ordered_best_first():
    index = populated()
    hits = index.search(unit(1, 0.5, 0, 0)[0], top_k=3)
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)


def test_a_top_k_larger_than_the_index_drops_the_padding():
    """faiss pads a short result with id -1. That is the normal case for a
    fresh index holding one lecture, not an edge case."""
    index = ChunkIndex.create(spec())
    index.add("aaa", [chunk()], unit(1, 0, 0, 0))
    hits = index.search(unit(1, 0, 0, 0)[0], top_k=10)
    assert len(hits) == 1
    assert all(hit.vector_id >= 0 for hit in hits)


def test_searching_an_empty_index_returns_nothing():
    assert ChunkIndex.create(spec()).search(unit(1, 0, 0, 0)[0], top_k=5) == []


def test_top_k_below_one_is_refused():
    with pytest.raises(IndexFailed):
        populated().search(unit(1, 0, 0, 0)[0], top_k=0)


def test_a_query_of_the_wrong_width_is_refused():
    with pytest.raises(IndexFailed):
        populated().search(np.zeros(DIM + 1, dtype=np.float32), top_k=1)


def test_a_hit_carries_the_citation_the_chunker_built():
    """The heading is preferred over the page number because it holds the slide
    number the student sees -- ingest-pipeline 5.1."""
    index = ChunkIndex.create(spec())
    index.add(
        "aaa",
        [chunk("Lecture_2_slides.pdf", 4, "Slide 11: OSI Security Architecture")],
        unit(1, 0, 0, 0),
    )
    hit = index.search(unit(1, 0, 0, 0)[0], top_k=1)[0]
    assert hit.citation() == "Lecture_2_slides.pdf, Slide 11: OSI Security Architecture"


def test_a_chunk_without_a_heading_cites_its_page():
    index = ChunkIndex.create(spec())
    index.add("aaa", [chunk("notes.txt", 3, None)], unit(1, 0, 0, 0))
    hit = index.search(unit(1, 0, 0, 0)[0], top_k=1)[0]
    assert hit.citation() == "notes.txt, Page 3"


# ---------------------------------------------------------------------------
# Deletion and id stability. The reason IndexIDMap2 is used at all.
# ---------------------------------------------------------------------------


def test_removing_a_document_leaves_every_other_id_untouched():
    """A bare IndexFlatIP compacts by swapping the last vector into the hole,
    so ids after the removal shift and the sidecar then describes the wrong
    chunks. This asserts the property that makes the sidecar safe."""
    index = populated()
    before = {
        vector_id: record["document_name"] for vector_id, record in index.records.items()
    }
    index.remove("aaa")
    for vector_id, record in index.records.items():
        assert before[vector_id] == record["document_name"]
    assert sorted(index.records) == [2]


def test_removing_a_document_takes_exactly_its_vectors():
    index = populated()
    assert index.remove("aaa") == 2
    assert len(index) == 1
    assert index.documents() == {"bbb": 1}
    assert index.index.ntotal == 1


def test_removed_chunks_stop_being_searchable():
    index = populated()
    index.remove("aaa")
    hits = index.search(unit(1, 0, 0, 0)[0], top_k=5)
    assert all(hit.doc_id == "bbb" for hit in hits)


def test_ids_are_never_reused_after_a_removal():
    """A recycled id would make a stale reference resolve to a different
    document's text rather than to nothing."""
    index = populated()
    index.remove("aaa")
    fresh = index.add("ccc", [chunk("c.pdf")], unit(0, 0, 0, 1))
    assert fresh == [3], "id 0 or 1 was recycled"


def test_removing_an_unknown_document_changes_nothing():
    index = populated()
    assert index.remove("nope") == 0
    assert len(index) == 3


def test_a_document_can_be_removed_then_reindexed():
    """What a re-upload after --delete has to do."""
    index = populated()
    index.remove("aaa")
    assert not index.has_document("aaa")
    index.add("aaa", [chunk("a.pdf", 1)], unit(1, 0, 0, 0))
    assert index.has_document("aaa")
    assert index.documents()["aaa"] == 1


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_round_trip_preserves_everything_search_depends_on(tmp_path):
    index = populated()
    index.save(tmp_path)
    loaded = ChunkIndex.load(tmp_path, expected_spec=spec())

    assert len(loaded) == len(index)
    assert loaded.documents() == index.documents()
    assert loaded.next_id == index.next_id
    assert loaded.spec == index.spec
    assert loaded.records == index.records

    before = index.search(unit(1, 0.25, 0, 0)[0], top_k=3)
    after = loaded.search(unit(1, 0.25, 0, 0)[0], top_k=3)
    assert [h.vector_id for h in before] == [h.vector_id for h in after]
    assert [round(h.score, 6) for h in before] == [round(h.score, 6) for h in after]


def test_both_files_are_written_owner_only_in_a_private_directory(tmp_path):
    """R2: the sidecar holds document text at rest."""
    target = tmp_path / "index"
    populated().save(target)
    assert oct(target.stat().st_mode)[-3:] == "700"
    for name in (VECTORS_NAME, SIDECAR_NAME):
        assert oct((target / name).stat().st_mode)[-3:] == "600"


def test_no_temporary_files_are_left_behind(tmp_path):
    populated().save(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(
        [SIDECAR_NAME, VECTORS_NAME]
    )


def test_exists_reports_both_halves(tmp_path):
    assert not ChunkIndex.exists(tmp_path)
    populated().save(tmp_path)
    assert ChunkIndex.exists(tmp_path)
    (tmp_path / VECTORS_NAME).unlink()
    assert not ChunkIndex.exists(tmp_path)


def test_saving_again_overwrites_rather_than_accumulating(tmp_path):
    index = populated()
    index.save(tmp_path)
    index.add("ccc", [chunk("c.pdf")], unit(0, 0, 0, 1))
    index.save(tmp_path)
    assert len(ChunkIndex.load(tmp_path)) == 4


# ---------------------------------------------------------------------------
# Refusing an index that cannot be trusted
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"model": "other-model"},
        {"digest": "b" * 64},
        {"doc_prefix": "passage: "},
        {"query_prefix": "question: "},
    ],
    ids=["model", "digest", "doc-prefix", "query-prefix"],
)
def test_a_spec_mismatch_is_refused(tmp_path, overrides):
    """Every one of these makes stored vectors incomparable with fresh ones,
    and every one of them is silent: the symptom is worse ranking, not an
    error. load() is the only place it can be caught."""
    populated().save(tmp_path)
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path, expected_spec=spec(**overrides))


def test_a_dimension_change_is_refused(tmp_path):
    populated().save(tmp_path)
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path, expected_spec=spec(dim=768))


def test_loading_without_an_expected_spec_still_reads_the_stored_one(tmp_path):
    """Used by --list and --delete, which need the mapping but do not embed."""
    populated().save(tmp_path)
    assert ChunkIndex.load(tmp_path).spec == spec()


def test_an_unknown_format_version_is_refused(tmp_path):
    populated().save(tmp_path)
    path = tmp_path / SIDECAR_NAME
    sidecar = json.loads(path.read_text())
    sidecar["format_version"] = FORMAT_VERSION + 1
    path.write_text(json.dumps(sidecar))
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_torn_write_is_refused_rather_than_half_served(tmp_path):
    """The two renames in save() are not one atomic step. If the process dies
    between them the vector count and the sidecar disagree, and serving that
    would mean citing chunks the index does not hold."""
    small = ChunkIndex.create(spec())
    small.add("aaa", [chunk()], unit(1, 0, 0, 0))
    small.save(tmp_path)

    other = tmp_path / "other"
    populated().save(other)
    # Sidecar from the three-chunk index, vectors from the one-chunk index.
    (tmp_path / SIDECAR_NAME).write_bytes((other / SIDECAR_NAME).read_bytes())

    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_swapped_vector_file_is_refused_even_at_the_same_chunk_count(tmp_path):
    """Comparing vector counts does not close this, which is why the generation
    token exists. Measured before it did: two one-chunk indexes, vectors file
    exchanged, loads happily and answers with the other document's vector while
    citing this one's filename and page, at similarity 1.0.
    """
    for name, vector in (("one", unit(1, 0, 0, 0)), ("two", unit(0, 1, 0, 0))):
        index = ChunkIndex.create(spec())
        index.add(f"doc-{name}", [chunk(f"{name}.pdf", 1)], vector)
        index.save(tmp_path / name)

    assert (tmp_path / "one" / SIDECAR_NAME).is_file()
    (tmp_path / "one" / VECTORS_NAME).write_bytes(
        (tmp_path / "two" / VECTORS_NAME).read_bytes()
    )

    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path / "one")


def test_an_older_write_of_the_same_index_is_refused(tmp_path):
    """The same hole in its subtler form: not another index's file, but a stale
    copy of this one's, which a backup or a half-finished sync can leave."""
    index = populated()
    index.save(tmp_path)
    stale = (tmp_path / VECTORS_NAME).read_bytes()

    index.add("ccc", [chunk("c.pdf")], unit(0, 0, 0, 1))
    index.save(tmp_path)
    (tmp_path / VECTORS_NAME).write_bytes(stale)

    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_every_write_stamps_a_new_generation(tmp_path):
    """Two saves of changed content must not share a token, or a stale file
    compares equal to a current one."""
    index = populated()
    index.save(tmp_path)
    first = json.loads((tmp_path / SIDECAR_NAME).read_text())["generation"]
    index.add("ccc", [chunk("c.pdf")], unit(0, 0, 0, 1))
    index.save(tmp_path)
    second = json.loads((tmp_path / SIDECAR_NAME).read_text())["generation"]
    assert first != second
    assert len(second) == 32


def test_the_vectors_file_carries_the_same_token_as_the_sidecar(tmp_path):
    populated().save(tmp_path)
    sidecar = json.loads((tmp_path / SIDECAR_NAME).read_text())
    header = (tmp_path / VECTORS_NAME).read_bytes().split(b"\n", 1)[0]
    assert header.decode() == sidecar["generation"]


@pytest.mark.parametrize(
    "token", ["", "not-hex", "A" * 32, "0" * 31], ids=["empty", "nonhex", "upper", "short"]
)
def test_a_malformed_generation_token_is_refused(tmp_path, token):
    populated().save(tmp_path)
    path = tmp_path / SIDECAR_NAME
    sidecar = json.loads(path.read_text())
    sidecar["generation"] = token
    path.write_text(json.dumps(sidecar))
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_vectors_file_with_no_header_is_refused(tmp_path):
    """What a bare faiss.serialize_index() buffer looks like -- the shape this
    used to write."""
    import faiss

    populated().save(tmp_path)
    bare = bytes(faiss.serialize_index(faiss.IndexIDMap2(faiss.IndexFlatIP(DIM))))
    (tmp_path / VECTORS_NAME).write_bytes(bare)
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_an_id_counter_that_would_reuse_a_live_id_is_refused(tmp_path):
    populated().save(tmp_path)
    path = tmp_path / SIDECAR_NAME
    sidecar = json.loads(path.read_text())
    sidecar["next_id"] = 1  # id 1 and 2 are in use
    path.write_text(json.dumps(sidecar))
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_sidecar_missing_a_citation_field_is_refused(tmp_path):
    """A record without a page number cannot be cited, so it is not loadable
    rather than loadable-and-uncitable."""
    populated().save(tmp_path)
    path = tmp_path / SIDECAR_NAME
    sidecar = json.loads(path.read_text())
    first = next(iter(sidecar["records"]))
    del sidecar["records"][first]["page_number"]
    path.write_text(json.dumps(sidecar))
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_corrupt_sidecar_is_refused(tmp_path):
    populated().save(tmp_path)
    (tmp_path / SIDECAR_NAME).write_bytes(b"{not json at all")
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_corrupt_vector_file_is_refused(tmp_path):
    populated().save(tmp_path)
    (tmp_path / VECTORS_NAME).write_bytes(b"\x00\x01\x02not a faiss index")
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_a_missing_index_is_refused_rather_than_treated_as_empty(tmp_path):
    """Treating absence as an empty index would answer every question with
    'nothing found' while the real index sat unreadable."""
    with pytest.raises(IndexFailed):
        ChunkIndex.load(tmp_path)


def test_save_refuses_when_vectors_and_records_disagree(tmp_path):
    index = populated()
    del index.records[0]  # desynchronise by hand
    with pytest.raises(IndexFailed):
        index.save(tmp_path)


# ---------------------------------------------------------------------------
# Leak discipline. ingest-pipeline 4.5
# ---------------------------------------------------------------------------


PLANTED_PHRASE = "ZEROCOOLCANARY"


def test_no_failure_message_quotes_the_chunk_text(tmp_path):
    text_chunk = Chunk(
        document_name="deck.pdf",
        page_number=1,
        heading=None,
        text=f"a page mentioning {PLANTED_PHRASE}",
    )
    index = ChunkIndex.create(spec())
    raw = np.asarray([[2.0, 0.0, 0.0, 0.0]], dtype=np.float32)  # unnormalised
    with pytest.raises(IndexFailed) as excinfo:
        index.add("aaa", [text_chunk], raw)
    assert PLANTED_PHRASE not in str(excinfo.value)


def test_a_library_error_does_not_survive_on_the_exception_chain(tmp_path):
    """faiss and the filesystem both raise errors carrying paths. The third
    copy of the _guarded pattern has to behave like the other two."""
    populated().save(tmp_path)
    (tmp_path / VECTORS_NAME).write_bytes(b"not an index")
    with pytest.raises(IndexFailed) as excinfo:
        ChunkIndex.load(tmp_path)
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_the_sidecar_is_what_holds_document_text(tmp_path):
    """Recorded as a test because it is what TUT-49 has to encrypt, and what
    makes data/index/ an R2 asset rather than just derived data."""
    index = ChunkIndex.create(spec())
    index.add(
        "aaa",
        [Chunk("deck.pdf", 1, None, f"slide text {PLANTED_PHRASE}")],
        unit(1, 0, 0, 0),
    )
    index.save(tmp_path)
    assert PLANTED_PHRASE in (tmp_path / SIDECAR_NAME).read_text()
    assert PLANTED_PHRASE.encode() not in (tmp_path / VECTORS_NAME).read_bytes()


# ---------------------------------------------------------------------------
# End to end with the embedder, still without Ollama
# ---------------------------------------------------------------------------


def test_embedder_output_feeds_the_index_unchanged(tmp_path):
    """The two modules' contracts meet here: the embedder promises unit-norm
    float32 in chunk order, and the index requires exactly that."""
    import hashlib

    from ingest.embedder import Embedder

    class FakeBackend:
        def spec(self):
            return spec(dim=8)

        def embed(self, texts):
            out = []
            for text in texts:
                seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")
                out.append((np.random.default_rng(seed).normal(size=8) * 5).tolist())
            return out

    chunks = [chunk("deck.pdf", page, f"Slide {page}: topic") for page in range(1, 6)]
    embedder = Embedder(FakeBackend(), batch_size=2)

    generator = embedder.embed_chunks(chunks)
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            vectors = stop.value
            break

    index = ChunkIndex.create(spec(dim=8))
    ids = index.add("aaa", chunks, vectors)
    assert ids == [0, 1, 2, 3, 4]

    # Each chunk's own vector retrieves that chunk first.
    for position, chunk_obj in enumerate(chunks):
        hit = index.search(vectors[position], top_k=1)[0]
        assert hit.chunk.page_number == chunk_obj.page_number
        assert hit.score == pytest.approx(1.0, abs=1e-5)

    index.save(tmp_path)
    assert len(ChunkIndex.load(tmp_path, expected_spec=spec(dim=8))) == 5
