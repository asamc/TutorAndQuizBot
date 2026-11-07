"""The document library: uploading, reading slides, and finding them by topic."""

import pytest

from agents.quiz_bot.documents import DocumentError, Library, topic_name

SLIDE_TEXT = (
    "A stateful firewall tracks connection state so that reply traffic for "
    "connections started from inside the network is allowed back through."
)


@pytest.fixture
def library(tmp_path):
    return Library(tmp_path / "documents")


@pytest.fixture
def notes(tmp_path):
    """A valid document waiting to be uploaded, outside the library folder."""
    path = tmp_path / "Firewalls.txt"
    path.write_text(f"Stateful Firewalls\n{SLIDE_TEXT}", encoding="utf-8")
    return path


def test_upload_stores_the_file_and_makes_it_quizzable(library, notes):
    assert library.upload(notes) == ("Firewalls.txt", 1)
    assert (library.folder / "Firewalls.txt").is_file()
    assert library.documents() == ["Firewalls.txt"]


def test_uploaded_documents_are_there_next_time(library, notes):
    library.upload(notes)
    assert Library(library.folder).documents() == ["Firewalls.txt"]


def test_upload_accepts_a_quoted_windows_style_name(library, notes):
    name, _ = library.upload(str(notes))
    assert name == "Firewalls.txt"


@pytest.mark.parametrize(
    "filename, content, reason",
    [
        ("slides.pptx", b"x", "only PDF and TXT"),
        ("broken.pdf", b"%PDF-1.4 not really a pdf", "could not be read as a PDF"),
        ("title.txt", b"Lecture 3: Firewalls", "no slides with enough text"),
        ("CON.txt", SLIDE_TEXT.encode(), "filename is not allowed"),
    ],
)
def test_bad_uploads_are_refused_and_nothing_is_written(
    library, tmp_path, filename, content, reason
):
    path = tmp_path / filename
    try:
        path.write_bytes(content)
    except OSError:
        pytest.skip("this OS will not create the file at all")
    with pytest.raises(DocumentError, match=reason):
        library.upload(path)
    assert not library.folder.exists() or not any(library.folder.iterdir())


def test_the_same_document_cannot_be_uploaded_twice(library, notes):
    library.upload(notes)
    with pytest.raises(DocumentError, match="already uploaded"):
        library.upload(notes)


def test_a_missing_file_is_reported(library, tmp_path):
    with pytest.raises(DocumentError, match="not found"):
        library.upload(tmp_path / "nope.pdf")


def test_unreadable_file_in_the_folder_is_skipped_not_fatal(tmp_path):
    (tmp_path / "broken.pdf").write_bytes(b"%PDF-1.4 not really a pdf")
    (tmp_path / "notes.txt").write_text(SLIDE_TEXT, encoding="utf-8")
    library = Library(tmp_path)
    assert library.documents() == ["notes.txt"]
    assert library.problems == ["broken.pdf: could not be read as a PDF"]


def test_first_line_is_the_title_when_there_is_no_outline(library, notes):
    library.upload(notes)
    assert library.chunks[0].heading == "Stateful Firewalls"


def test_a_long_first_line_is_not_taken_as_a_title(tmp_path):
    (tmp_path / "notes.txt").write_text(SLIDE_TEXT, encoding="utf-8")
    assert Library(tmp_path).chunks[0].heading is None


@pytest.mark.parametrize(
    "filename, topic",
    [
        ("1-2 Cryptograpy - Cryptographic Hash Function.pdf", "Cryptographic Hash Function"),
        ("02 Introduction - Computer Security Concept.pdf", "Computer Security Concept"),
        ("Lecture_1_slides.pdf", "Lecture 1 slides"),
        ("firewalls.txt", "firewalls"),
    ],
)
def test_topic_names_come_from_filenames(filename, topic):
    assert topic_name(filename) == topic


def test_choosing_a_lecture_topic_draws_from_that_document(tmp_path):
    (tmp_path / "03 Firewalls.txt").write_text(SLIDE_TEXT, encoding="utf-8")
    (tmp_path / "04 Hashing.txt").write_text(
        "A cryptographic hash maps any input to a fixed-length digest and should "
        "resist both preimage and collision attacks in practice.",
        encoding="utf-8",
    )
    library = Library(tmp_path)
    assert library.topics() == ["Firewalls", "Hashing"]
    assert library.search("Hashing", 1)[0].document_name == "04 Hashing.txt"
    assert library.search("zzz-no-such-topic", 3) == []
