"""The user's course documents: upload, read, and find slides to quiz on.

Uploaded files are stored in data/documents/. That folder is gitignored, so
course material never reaches the repository (threat model R2). Each slide or
page becomes one Chunk, the unit a quiz question is written from and cited to.

A stopgap until the ingest pipeline and vector store exist. It follows the rules
the pipeline is specified to follow (docs/ingest-pipeline.md): validate in
memory before writing to disk, never build a path from an untrusted filename,
bound parsing against hostile files, and one chunk per page. When ingest and
retrieval are implemented, Library should call them and its own parsing and
keyword search go away.

Not yet done here, and required by the threat model: SHA-256 hashing of
uploads and injection screening (ingest/manifest.py, ingest/screener.py).
"""

import io
import pathlib
import random
import re

from pypdf import PdfReader

from ingest.model import EMPTY_PAGE_CHAR_THRESHOLD, Chunk
from ingest.parser import MAX_EXTRACTED_CHARS, MAX_PAGES
from ingest.validator import MAX_UPLOAD_BYTES

DEFAULT_FOLDER = pathlib.Path(__file__).resolve().parents[2] / "data" / "documents"

SUPPORTED_SUFFIXES = {".pdf", ".txt"}

# A slide with fewer words than this is a title or image slide: too little to
# write a fair question from, so it is not offered to the quiz.
MIN_QUIZ_WORDS = 15

# Longer than this, a first line is body text rather than a slide title.
MAX_TITLE_CHARS = 80

_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}  # fmt: skip

# Filler words that would otherwise match every slide, so "what is a firewall"
# scores on "firewall" alone.
_STOPWORDS = {
    "a", "an", "and", "are", "about", "as", "at", "be", "by", "for", "from", "how",
    "in", "is", "it", "me", "no", "not", "of", "on", "or", "quiz", "such", "the",
    "to", "what", "with",
}  # fmt: skip


class DocumentError(Exception):
    """A file could not be uploaded or read. The message names the file but never
    carries library internals or document text, so it is safe to display."""


class Library:
    """The documents in one folder, read once and cached.

    Holds every slide with enough text to quiz on. Call reload() after the
    folder changes outside upload().
    """

    def __init__(self, folder: pathlib.Path | str = DEFAULT_FOLDER):
        self.folder = pathlib.Path(folder)
        self.chunks: list[Chunk] = []
        self.problems: list[str] = []  # files that could not be read, display-safe
        self.reload()

    def reload(self) -> None:
        self.chunks, self.problems = [], []
        if not self.folder.is_dir():
            return
        for path in sorted(self.folder.iterdir()):
            if path.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            try:
                self.chunks.extend(_quizzable(parse(path.read_bytes(), path.name)))
            except DocumentError as exc:
                self.problems.append(str(exc))

    def upload(self, source: pathlib.Path | str) -> tuple[str, int]:
        """Copy a document into the library and return (stored name, slides
        added).

        The file is fully checked in memory first -- type, size, and that it
        parses with quizzable text -- so nothing unreadable or hostile is ever
        written into the documents folder.
        """
        source = pathlib.Path(source)
        name = _safe_name(source.name)
        if not source.is_file():
            raise DocumentError(f"{source.name}: file not found")
        if source.stat().st_size > MAX_UPLOAD_BYTES:
            raise DocumentError(f"{name}: over the {MAX_UPLOAD_BYTES // 2**20} MB limit")
        if (self.folder / name).exists():
            raise DocumentError(f"{name}: already uploaded")

        data = source.read_bytes()
        chunks = _quizzable(parse(data, name))
        if not chunks:
            raise DocumentError(f"{name}: no slides with enough text to quiz on")

        self.folder.mkdir(parents=True, exist_ok=True)
        (self.folder / name).write_bytes(data)
        self.chunks.extend(chunks)
        return name, len(chunks)

    def documents(self) -> list[str]:
        return sorted({chunk.document_name for chunk in self.chunks})

    def topics(self) -> list[str]:
        """One topic per document, named from its filename, e.g.
        '1-2 Cryptograpy - Cryptographic Hash Function.pdf' becomes
        'Cryptographic Hash Function'. Passing a topic to search() draws
        questions from that document."""
        return [topic_name(name) for name in self.documents()]

    def search(self, topic: str, k: int) -> list[Chunk]:
        """Return up to k slides most relevant to topic, best first.

        Keyword scoring until embeddings exist: topic words found in the slide
        text, heading matches counting double and document-name matches once, so
        choosing a document's topic draws from that document. Ties are broken at
        random, so repeating a topic does not repeat the same quiz. Slides with
        no overlap are not returned.
        """
        topic_words = _words(topic)
        scored = []
        for chunk in self.chunks:
            score = len(topic_words & _words(chunk.text))
            score += 2 * len(topic_words & _words(chunk.heading or ""))
            score += len(topic_words & _words(chunk.document_name))
            if score:
                scored.append((score, random.random(), chunk))
        scored.sort(key=lambda entry: entry[:2], reverse=True)
        return [chunk for _, _, chunk in scored[:k]]

    def random_chunks(self, n: int) -> list[Chunk]:
        """Return n distinct slides chosen at random, or all of them if n is larger."""
        return random.sample(self.chunks, min(n, len(self.chunks)))


def parse(data: bytes, name: str) -> list[Chunk]:
    """Read a PDF or TXT file's bytes into one Chunk per non-empty page."""
    if name.lower().endswith(".txt"):
        text = data.decode("utf-8", errors="replace")
        if len(text) > MAX_EXTRACTED_CHARS:
            raise DocumentError(f"{name}: too much text")
        pages = [(1, None, text)]
    elif name.lower().endswith(".pdf"):
        pages = _read_pdf(data, name)
    else:
        raise DocumentError(f"{name}: only PDF and TXT files are supported")

    return [
        Chunk(name, number, heading or _first_line(text), text.strip())
        for number, heading, text in pages
        if len(text.strip()) >= EMPTY_PAGE_CHAR_THRESHOLD
    ]


def topic_name(document_name: str) -> str:
    stem = pathlib.Path(document_name).stem
    stem = stem.rsplit(" - ", 1)[-1]  # "Lecture 3 - Firewalls" -> "Firewalls"
    stem = re.sub(r"^[\d\s._-]+", "", stem)  # leading numbering like "02 " or "1-2 "
    return stem.replace("_", " ").strip() or pathlib.Path(document_name).stem


def _safe_name(filename: str) -> str:
    """A filename that is safe to create inside the documents folder.

    Only the final path component is used, and anything that could escape the
    folder or misbehave on Windows is refused. docs/ingest-pipeline.md 4.1.
    """
    name = pathlib.PurePath(filename.replace("\\", "/")).name
    stem, suffix = pathlib.Path(name).stem, pathlib.Path(name).suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise DocumentError(f"{name or filename}: only PDF and TXT files are supported")
    if (
        not stem.strip(" .")
        or len(name) > 150
        or "\0" in name
        or re.search(r'[<>:"|?*\x00-\x1f]', name)
        or stem.split(".")[0].lower() in _WINDOWS_RESERVED
    ):
        raise DocumentError("that filename is not allowed; rename the file and try again")
    return name


def _quizzable(chunks: list[Chunk]) -> list[Chunk]:
    return [c for c in chunks if len(c.text.split()) >= MIN_QUIZ_WORDS]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - _STOPWORDS


def _first_line(text: str) -> str | None:
    """A slide's title when the PDF has no outline. Measured on real lecture
    decks: pypdf extracts the slide title as the first line of the page."""
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line if len(line) <= MAX_TITLE_CHARS else None
    return None


def _read_pdf(data: bytes, name: str) -> list[tuple[int, str | None, str]]:
    # Every pypdf call is wrapped: its exception messages can carry fragments of
    # the document, so they are never passed on.
    try:
        reader = PdfReader(io.BytesIO(data))
        if len(reader.pages) > MAX_PAGES:
            raise DocumentError(f"{name}: more than {MAX_PAGES} pages")
        headings = _headings(reader)
        pages = []
        total = 0
        for index, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            total += len(text)
            if total > MAX_EXTRACTED_CHARS:
                raise DocumentError(f"{name}: too much text")
            pages.append((index + 1, headings.get(index + 1), text))
    except DocumentError:
        raise
    except Exception:
        raise DocumentError(f"{name}: could not be read as a PDF") from None

    if all(len(text.strip()) < EMPTY_PAGE_CHAR_THRESHOLD for _, _, text in pages):
        # Image-only PDFs extract to "" without raising. Say so rather than
        # loading nothing silently.
        raise DocumentError(f"{name}: no text found (scanned or image-only?)")
    return pages


def _headings(reader: PdfReader) -> dict[int, str]:
    """Map 1-based page number to its outline title. Lecture decks carry one
    bookmark per slide; PDFs without an outline just get no headings."""
    headings: dict[int, str] = {}

    def walk(items: list) -> None:
        for item in items:
            if isinstance(item, list):  # nested outline level
                walk(item)
                continue
            title = (getattr(item, "title", "") or "").strip()
            if not title:
                continue
            try:
                page = reader.get_destination_page_number(item) + 1
            except Exception:
                continue
            headings.setdefault(page, title)

    try:
        walk(reader.outline)
    except Exception:
        return {}
    return headings
