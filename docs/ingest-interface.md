# Ingest Interface

Status: Draft for UI integration. The contract in sections 2 to 5 is stable enough to build against. Section 6 lists what is still open.

## 1. Scope

How the UI hands an uploaded course file to the ingest pipeline, and what it gets back. Written so the upload flow can be built before `ingest/` exists.

**Provisional assumption:** ingest runs in the same process as the UI, so this is a Python function call. If TUT-55 decides on separate processes, section 3 becomes a wire format instead and both sides need serialization. Nothing else in this document changes.

## 2. Ownership Boundary

The UI passes **file bytes and the original filename**, and nothing else.

Ingest performs **every write under `data/`**. The UI never touches the filesystem.

This is worth stating because the obvious Streamlit idiom is for the UI to save the upload itself. Two reasons not to:

- TUT-51 requires the file to land only in `data/documents/`.
- Keeping all `data/` writes inside one module means the R2 filesystem-permission and gitignore story can be audited in one place rather than across two.

## 3. Call Signature

One entry point. It is a generator: it yields progress events as it works, then returns a terminal result.

```python
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

class Stage(str, Enum):
    VALIDATING = "validating"
    PARSING    = "parsing"
    SCREENING  = "screening"
    CHUNKING   = "chunking"
    EMBEDDING  = "embedding"
    INDEXING   = "indexing"

class Outcome(str, Enum):
    INDEXED                   = "indexed"
    ALREADY_INDEXED           = "already_indexed"
    REJECTED_TYPE             = "rejected_type"
    REJECTED_SIZE             = "rejected_size"
    REJECTED_CONTENT_MISMATCH = "rejected_content_mismatch"
    PARSE_FAILED              = "parse_failed"
    INDEXING_FAILED           = "indexing_failed"
    FLAGGED_INJECTION         = "flagged_injection"

@dataclass
class Progress:
    stage: Stage
    current: int | None = None   # e.g. chunks embedded so far
    total: int | None = None     # e.g. total chunks; None when not yet known

@dataclass
class Result:
    outcome: Outcome
    document_name: str
    chunk_count: int | None = None   # set when outcome is INDEXED or ALREADY_INDEXED
    message: str | None = None       # display-safe reason; never internals (see section 5)
    flagged: list["Flag"] | None = None   # set only when outcome is FLAGGED_INJECTION

@dataclass
class Flag:
    page: int | None        # page or slide number where the match was found
    pattern: str            # which screening rule matched, not the document text
    excerpt: str | None     # bounded snippet - length still open, see section 6

def ingest_document(data: bytes, filename: str) -> Iterator[Progress | Result]:
    """Yields Progress events, then exactly one Result as the final item."""
```

Progress is part of the contract, not an optional extra. TUT-51 requires progress shown during parsing, chunking and indexing, so a single blocking call cannot satisfy that story.

## 4. Progress Stages

Fixed and ordered. The UI can render all six before ingest is implemented.

```
validating -> parsing -> screening -> chunking -> embedding -> indexing
```

Not every stage reports `current`/`total`. Validation and screening are fast and report neither; chunking and embedding report both.

Chunking is **page-aligned: one chunk per slide or page**, not word-count based. This matters for the UI because it sets the magnitude of `chunk_count` — expect roughly one chunk per page, so a 10-page deck reports about 10 chunks, not 1. Measured on the first two lecture decks: an entire 11-page deck holds only ~440 words, so packing to a word budget would collapse a whole lecture into one chunk and destroy page-level citations.

## 5. Outcome States

Every run ends in exactly one of these. Each needs different UI treatment, so this is not a success/failure pair.

| Outcome | Meaning | UI treatment |
|---|---|---|
| `indexed` | Success | Confirm document name and chunk count |
| `already_indexed` | Same document ingested previously | **Not an error.** TUT-19 forbids duplicates on re-ingest |
| `rejected_type` | Not PDF, DOCX or TXT | Rejection, state the accepted types |
| `rejected_size` | Over the 50 MB cap | Rejection, state the limit |
| `rejected_content_mismatch` | Extension disagrees with actual content | Rejection, no file internals |
| `parse_failed` | Text extraction failed | Show `message` only |
| `indexing_failed` | Embedding or the index write could not complete | Show `message` only, and offer a retry. See below |
| `flagged_injection` | Instruction-like phrasing found | Review screen, see section 6 |

Two display rules apply to every `message`:

- Never show file internals, stack traces or library errors (TUT-15).
- Never show vectors, raw chunks, embeddings or internal document IDs (TUT-54, threat-model section 4.3).

### `indexing_failed` is the one outcome that is not about the document

Every other outcome is a verdict on the uploaded file: its type, its size, whether its bytes match its
extension, whether text could be extracted, how it is phrased. Those are deterministic — the same bytes
produce the same outcome every time, so "rejected" means *do not bother trying this file again*.

`indexing_failed` is a verdict on **the machine**. It means the document was accepted, parsed and screened
cleanly, and then embedding or the index write could not complete. The usual causes are the local Ollama
runtime not running, the `nomic-embed-text` model not pulled, no space or permission to write
`data/index/`, or no keychain available to unlock the encrypted index (threat-model 4.3 requires refusing
rather than falling back to a key file).

Three consequences for the UI:

- **It is retryable, and the remedy is on the machine, not in the file.** The same upload will succeed once
  the cause is fixed. Do not word it as a rejection, and do not tell the user their document is faulty.
- **Nothing was stored, recorded or indexed.** A failed run leaves no trace, so a retry is a plain re-upload
  rather than a resume, and the next attempt will not report `already_indexed`.
- **`message` carries the remedy**, as a fixed string written by us — "The local embedding model is not
  available." — never the underlying exception. `ollama`'s own errors are safe-looking but they are library
  errors, which the first display rule above already forbids, and the parser's are measurably worse
  (ingest-pipeline 4.5).

Before this existed there was no honest answer here: `parse_failed` was the closest fit and it is a lie,
because it would tell someone whose Ollama is simply not started that their PDF is corrupt.

Which stage failed is available from the last `Progress` event the UI received — `embedding` or `indexing` —
so the outcome deliberately does not encode it twice.

### Upload size cap

**50 MB.** Image-heavy slide decks commonly run 10 to 30 MB, so this accepts normal course material while preventing one upload from stalling CPU-only embedding.

Set `server.maxUploadSize = 50` in `.streamlit/config.toml` so the browser rejects oversized files early with a usable message. Ingest still enforces the cap itself — the caller is never trusted — so `rejected_size` remains reachable.

## 6. Open Items

- **Process layout** (TUT-55, Jacob). In-process is assumed above. A separate process turns section 3 into a wire format.
- **How much flagged text may be displayed** (TUT-18 against TUT-54). TUT-18 requires the user be told what was flagged and where; TUT-54 and threat-model section 4.3 say the UI never displays raw document content. A flagged excerpt *is* document text. Defensible, because whoever uploaded the file already has it, but it needs an explicit decision and a maximum snippet length. `Flag.excerpt` is typed optional until then; `Flag.pattern` and `Flag.page` are safe to show either way.
- **Expected ingest duration.** Partly measured, and smaller than first assumed. The real lecture decks hold ~440 to ~490 words each across 10 to 11 pages, which is roughly 10 chunks per lecture and a few hundred for a whole semester. Embedding that volume is quick, so **no stage is slow enough to need a progress bar for its own sake** — progress is still worth showing for the larger documents the pipeline should tolerate, but do not design the UI around a long wait. Now measured: `nomic-embed-text` on CPU runs **~13.2 chunks/s**, so a 10-chunk lecture is under a second and the whole ~250-chunk semester is about 20 seconds.

### The flagged-injection review

`flagged_injection` is neither success nor rejection. TUT-18 requires that a flagged document is not silently indexed and that the user is told what was flagged and where, which makes this a decision screen rather than a message: the user chooses to index anyway or discard. That choice then has to come back into ingest, so expect a second entry point once the display question above is settled.
