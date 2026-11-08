# Ingest Pipeline

Status: Design and security guidance. Stable enough to implement against. Section 8 lists what is still open.

Sources: docs/threat-model.md sections 3 and 4; docs/architecure.md section 3.1.

## 1. Scope

How the ingest pipeline is built internally, and the security properties it has to hold.

The boundary with the UI is **not** covered here. See [docs/ingest-interface.md](ingest-interface.md) for the call signature, the `Stage` / `Outcome` / `Progress` / `Result` / `Flag` types, the progress stages, the outcome states, and the 50 MB cap. Those definitions live there only — duplicating them in two documents is how contracts drift.

### Why this needs its own security section

Ingest is the only place untrusted data enters the system. Everything downstream — retrieval, both agents, the index — operates on whatever ingest decided to accept.

The documents also have a dual nature that most pipelines do not deal with:

- they are **the asset** being protected (R2)
- they are **the attack vector** for indirect injection (R1)

"Protect this" and "distrust this" apply to the same bytes at the same time.

## 2. Modules

| Module | Responsibility | Story |
|---|---|---|
| `ingest/model.py` | `PageRecord`, `Chunk`; re-exports the contract types from the interface doc | — |
| `ingest/validator.py` | Type, size and content-sniff gate | TUT-14 |
| `ingest/parser.py` | PDF and TXT to `list[PageRecord]` | TUT-15 |
| `ingest/manifest.py` | SHA-256 hashing; manifest stored outside `data/` | TUT-17 |
| `ingest/screener.py` | Injection screening to `list[Flag]` | TUT-18 |
| `ingest/chunker.py` | One chunk per page, with citation metadata | TUT-16 |
| `ingest/embedder.py` | Chunks to unit-norm vectors via Ollama | TUT-65 |
| `security/patterns.py` | The injection pattern list itself | TUT-18 |
| `ingest/cli.py` | Drives the stages from the command line | — |

Every module in the table is implemented. The pattern list belongs in `security/` rather than beside the screener because TUT-18 requires it there, so it can be extended during R1 testing without touching ingest code.

**`model.py` is the keystone.** The screener and the chunker both read `PageRecord`, so its shape decides what they can do. A `PageRecord` that loses the page number cannot support citations or page-located flags. Settle it first; it is the most expensive thing here to change later.

**There is deliberately no orchestrator yet.** No `Outcome` in the published contract can honestly describe a pipeline that stops after chunking — `indexed` would be false, and inventing a new outcome would break the contract the UI is being built against. Wire the generator when embedding and indexing exist.

Until then `ingest/cli.py` drives the stages in order and reports where it stopped, which is how the pipeline is exercised before the UI exists:

```
uv run python -m ingest.cli data/documents/Lecture_1_slides.pdf   # ingest
uv run python -m ingest.cli --chunks notes.txt                    # + citations
uv run python -m ingest.cli --list                                # the manifest
uv run python -m ingest.cli --verify                              # re-hash stored docs
```

It prints the real `Outcome` value for every state that is reachable today — `rejected_*`, `parse_failed`, `flagged_injection`, `already_indexed` — and for a clean document says `chunked, not indexed` rather than claiming an outcome the pipeline cannot deliver. When TUT-58 lands, `ingest_document()` wraps the same calls and that last line becomes `indexed`.

## 3. Stage Order Is a Security Property

The order in `ingest-interface.md` section 4 is not stylistic. Four orderings are load-bearing.

**Validate before parsing.** `pypdf` is a parser consuming hostile input. Do not hand it bytes that have not been checked.

**Hash the bytes as received, before parsing.** Hashing after any transformation hashes the wrong artifact, and the integrity check in TUT-23 would then compare against something that was never on disk.

**Screen before indexing, never after.** Once a chunk is in the index, retrieval can surface it, and the gate has already failed. Section 3.3 of the threat model requires flagged documents to be held for review before indexing.

**Validate in memory, then write to disk.** Persisting the upload before validating puts an unvalidated file inside the protected directory. At a 50 MB cap, holding the bytes in memory costs nothing.

## 4. Security Rules

### 4.1 Filenames are user input

The UI passes the original `filename`. Using it to build a path is a traversal bug: `../../.ssh/authorized_keys`, absolute paths, Windows reserved names (`CON`, `NUL`, `AUX`), embedded null bytes, 4000-character names.

**Storage identity is not the display name.** Derive the on-disk name from the content hash. Keep the original filename only as metadata, for display and citations.

This gives path safety and correct idempotency from the same decision, because the hash is also the document identity for the duplicate check.

### 4.2 The parser is attack surface

Untrusted PDFs expose decompression bombs, deeply nested structures that exhaust the recursion limit, and malformed files that throw from inside the library.

Bound three things:

- page count — **enforced**, before any text is extracted
- **total extracted characters** — **enforced**, accumulated inside the page loop
- wall-clock time — **declared but not enforced**, see below

The character bound matters most and is the one usually missed. A file-size cap does not bound memory: a few megabytes of PDF can expand to gigabytes of extracted text. It is accumulated *during* the page loop rather than checked afterwards, so a bomb is stopped partway instead of after the whole document is in memory. Its granularity is one page, because `extract_text()` returns a whole page or nothing — a single catastrophically expanding page is not covered.

For scale, measured on the real decks: 11 pages and 2,217 extracted characters, parsed in 33 ms. Both enforced bounds sit three orders of magnitude above real course material, so **nothing in `data/` exercises them** and they are driven by synthetic fixtures in `tests/test_parser.py`.

#### The wall-clock bound is not implemented, deliberately

`PARSE_TIMEOUT_SECONDS` remains in `parser.py` as the documented intent, and nothing enforces it. No portable in-process implementation exists: `signal.alarm` is Unix-only and `pyproject.toml` commits to Windows, and a watchdog thread cannot kill the work it watches — pypdf would keep consuming CPU and memory after the "timeout" returned. That is a control in name only, and the manifest-placement argument in 4.3 applies to it: a control that cannot do its job is worse than a recorded gap, because the Round 2 report would claim it.

The gap is real and the other two bounds do not cover it. A file that spins inside `PdfReader()` construction produces no pages and no characters, so neither bound ever trips. Closing it needs a subprocess, which is TUT-55's call. Until then it is an accepted risk in section 8, not a control.

Whether malicious file ingestion becomes a fourth risk is still open (TUT-63). The enforced bounds are cheap regardless, and "out of scope in the threat model" does not mean it will not happen during a demo.

#### What pypdf actually does with malformed input

Worth knowing before writing the error handling, because it is not where you would guess:

| Input | Result |
|---|---|
| Empty, header-only, truncated at 50% or 99% | raises from the **constructor**, not from `extract_text()` |
| Trailer overwritten | `PdfReadError` from the constructor |
| One byte flipped mid-stream | **Parses fine.** 11 pages, byte-identical extracted text |

So a `try`/`except` wrapped around extraction alone misses every real failure mode. Catch `Exception` rather than `PyPdfError`: `DependencyError` and `DeprecationError` derive straight from `Exception`, and malformed input also surfaces as `RecursionError`, `struct.error`, `zlib.error` and `KeyError` from inside the library.

The flipped byte is the row that matters for the threat model: corruption is silently survivable here, so **the parser is structurally incapable of detecting tampering**. That is `manifest.verify`'s job (TUT-23), and it is the argument for hashing the bytes as received rather than anything derived.

### 4.3 Keep the manifest away from what it protects

The manifest lives outside `data/`, at a path such as `~/.tutorandquizbot/manifest.json`, with restrictive file permissions.

The reason, not just the rule: **a manifest stored beside the documents can be rewritten by anyone who can tamper with those documents.** The integrity check would then always pass and the control becomes theatre. The detector has to sit outside the blast radius of the thing it detects.

### 4.4 Screening detects, it does not sanitise

The pattern list will catch naive injections and miss anything sophisticated. The real defence against indirect injection is **structural** — the labeled context blocks in TUT-26 that tell the model that retrieved text is reference material, not instructions. Screening is a speed bump that surfaces a suspicious document to a human.

So the screener must **never modify document text**. Stripping a matched phrase and indexing the result corrupts course material and creates false confidence at the same time. It flags; a person decides.

**The trap specific to this project:** the course material is *about security*. A lecture on prompt injection legitimately contains "ignore previous instructions". A slide on privilege escalation contains `system:`. The screener will fire on correct, benign content, repeatedly.

That is not a bug in the patterns — it is inherent to screening security material for security attack strings. Two consequences:

1. Expect a high false-positive rate, and do not weaken the patterns to suppress it.
2. The review flow in TUT-51 will be used constantly rather than exceptionally, so it has to be cheap to clear a document.

### 4.5 Logging is the leak channel

Section 4.1 of the threat model lists cached and temporary copies as an R2 attack path. Logs are the easiest one to create by accident. Log **document ID, page number, and pattern name**. Never text.

**Measured, not assumed:** planting the string `SECRETPHRASE` into a deck and damaging its trailer produces

```
PdfReadError: Invalid Elementary Object starting with b'S' @790092:
b'0000784380 00000 n\r\nSECRETPHRASE\n<<\n/Size 279\n/Root 1 0 R\n/Info 7 0 R\n/ID [<4CAD'
```

— roughly 100 bytes of file content inside the exception message. Two consequences that are easy to get wrong:

- `ParseFailed` carries a **fixed string written by us**, never the caught message, not even truncated.
- `raise ... from None` is **not sufficient**. It suppresses the *display* of the chained exception but leaves it on `__context__`, where an error reporter that walks the chain still prints those bytes. The raise has to happen after the `except` block has exited, which is why `parser._guarded` exists.

**pypdf writes to the logs on its own**, independent of our code: parsing a real deck emits `EOF marker not found` four times. 185 of its warnings go through the `pypdf` logger, six build their message dynamically, and one pipes an external subprocess's stderr straight through. Disabling it by **level** is the load-bearing part — `propagate = False` only blocks ancestor handlers, and anything may attach a handler to the `pypdf` logger directly. pytest's logging plugin does exactly that, which is how the first version of the muzzle was caught.

The four real leak paths in ingest:

- **Parser exception messages carry the data that caused them.** Never log a raw exception from `pypdf`.
- **A `Flag` contains document text.** Logging one wholesale writes protected content to disk.
- **Debug logging of extracted text**, added while developing and never removed. This is how it actually happens.
- **Intermediate files in `/tmp`**, which is outside the permission-restricted, gitignored `data/` tree.

### 4.6 Fail closed

- Type cannot be determined, reject.
- Hashing fails, abort.
- **The screener itself throws: treat the document as flagged, not clean.**

A gate that fails open is worse than no gate, because it produces the belief that screening ran when it silently did not.

Related: enforce the 50 MB cap in `validator.py` even though Streamlit also caps it. The UI is not a trust boundary, and CLI ingestion for testing bypasses it entirely.

## 5. Empty Pages and the Silent Failure

`page.extract_text()` returns an empty string for an image-only page. It does **not** raise.

So an all-scanned PDF parses "successfully", produces zero chunks, and reports success. TUT-15's acceptance criteria only describe extraction *failures*, which assumes something throws.

Measured on the two real lecture decks: **one near-empty page in each** (11 pages and 10 pages respectively), from title or image slides. Both are in fact the *same* slide — Lecture 2 opens by repeating Lecture 1's closing `A Hyperledger` image slide, so overlapping content across documents is already real in this corpus.

Per-page extracted characters, for anyone tempted to retune `EMPTY_PAGE_CHAR_THRESHOLD`:

```
Lecture 1:  39  362  154   29  425  381  209  212  313   79   14
Lecture 2:  14  191   25  560  318  653  177  271  108  127
```

The two empty pages are both 14 characters; the next-smallest pages are 25 and 29 — section dividers whose entire text is their own title. So any threshold from 15 to 24 gives identical results and 20 is a safe middle, but the margin is an 11-character gap: a divider slide titled `OSI Model` would be dropped.

That sets the rule:

- Per-page emptiness is **normal**. Retain the record, mark it, and let the chunker skip it.
- Only when **every** page is empty does the document fail, as `PARSE_FAILED`, with a message along the lines of "no extractable text; the document may be scanned images".

Deciding between rejecting such files and adding OCR is a scope question worth settling before it appears in a demo. OCR is a large piece of work nobody has budgeted.

A password-protected PDF is a separate case and must not land here. `is_encrypted` is checked before the page count — reading `.pages` on an encrypted file raises `FileNotDecryptedError`, which would otherwise be reported as "corrupt or incomplete" and tell the user the wrong thing about their file.

### 5.1 Headings carry the slide number, and it is not the page number

Settled decision, because it is invisible until a citation is wrong in front of a marker.

The decks carry one outline bookmark per slide holding its title, which is where citations like `Slide 6: Integrity` come from — no font-size heuristics needed. **But the slide labels do not match the PDF page numbers**, because the original decks had hidden slides:

| | PDF page | Outline title |
|---|---|---|
| Lecture 1 | 10 | `Slide 11: Question` |
| Lecture 2 | 3 | `Slide 10` |
| Lecture 2 | 4 | `Slide 11: OSI Security Architecture` |

Lecture 1 skips slide 10; Lecture 2 jumps from 2 to 10. Citing "page 4" sends a student to the wrong slide by seven.

The rules that follow:

- The outline title is stored **verbatim, including its own `Slide N:` prefix**. It is the only record of the number the student sees.
- The prefix is never regenerated from the page index, and never parsed off — `Slide 17: Example: two points communication` has a second colon, and a trailing space.
- `PageRecord.page_number` stays the **PDF page index**. It orders chunks and locates flags; it is not the citation.
- `Chunk.citation()` prefers the heading and falls back to `Page N` only when a document has no outline — which is every `.txt` file.
- Untitled bookmarks (`Slide 4`, `Slide 10`) are skipped; they carry no location the page number does not already give.

### 5.2 A `.txt` upload has no pages, so the parser makes them

The chunker is page-aligned, one chunk per page. A single `PageRecord` for a whole text file would therefore become one unbounded chunk — past any embedding model's context window and useless for retrieval. Nobody hit this because the course material is PDFs, but TUT-14 accepts `.txt` and the validator passes it: it is a hole *between* two stories, which is why it was unowned.

So `parser` splits text into ~1500-character pages on blank lines, packing paragraphs in order. A paragraph longer than the target is hard-split at whitespace, because a text file that is one long wrapped block is common and is exactly the input a blank-line-only rule silently fails on.

Splitting is structural, not editorial: no word is dropped or reordered, which `test_txt_pagination_loses_no_words` asserts. Two consequences worth recording: the effective ceiling for `.txt` is `MAX_PAGES × 1500` = 750k characters, below `MAX_EXTRACTED_CHARS`; and 1500 characters, originally a guess pending the embedding model, is now confirmed safe — `nomic-embed-text` has a 2048-token context (architecture doc 5.1) and 1500 characters is roughly 375 tokens, so the target stands with ~5.5× headroom.

## 6. Risk Traceability

For the Round 2 report: each control, the risk it serves, where it comes from, and what proves it.

| Control | Risk | Source | Evidence |
|---|---|---|---|
| Content sniffing, not extension trust | R1, R4? | TUT-14 | A `.pdf` that is not a PDF returns `rejected_content_mismatch` |
| Page and extracted-character bounds | R4? | TUT-63 open | Oversized and bomb-shaped synthetic fixtures are rejected, memory stays bounded |
| Wall-clock parse bound | R4? | 4.2 | **None — not enforced.** Recorded as an accepted risk in section 8 |
| `ParseFailed` carries a fixed message | R2 | 4.5 | A planted phrase appears in pypdf's error, and in neither `ParseFailed` nor its `__context__` |
| pypdf's own logger disabled by level | R2 | 4.5 | No `pypdf.*` record reaches the application log |
| Encrypted PDFs reported as locked | — | 5 | A password-protected upload says so instead of "corrupt" |
| Storage name derived from content hash | R2 | section 4.1 | A `../../etc/passwd` filename never writes outside `data/documents/` |
| Manifest stored outside `data/` | R2 | section 4.3 | One flipped byte makes the integrity check fail |
| Screening runs before indexing | R1 | section 3.3 | A planted-instruction fixture is flagged and not indexed |
| Screener never modifies text | R1 | section 3.3 | Flagged document text is byte-identical to input |
| Metadata-only logging | R2 | section 4.1 | Log grep for a known document phrase returns nothing |
| Fail-closed on screener error | R1 | section 3.4 | An induced screener exception yields `flagged_injection`, not success |
| `data/` permissions and gitignore | R2 | section 4.3 | `git check-ignore` covers `data/`; no commit contains it |

Section 3.4 of the threat model already concedes that model-level defences are probabilistic and that output filtering can miss paraphrase. Nothing in this pipeline changes that; these controls reduce the attack surface around it.

## 7. Verification

Security properties need tests that attack them, not tests that confirm the happy path.

| Test | Expected |
|---|---|
| Filename `../../etc/passwd` | Written inside `data/documents/`, or rejected. Never outside |
| `.pdf` extension, non-PDF content | `rejected_content_mismatch` |
| All-image PDF | `parse_failed`, not success with zero chunks |
| Truncated, header-only or trailer-damaged PDF | `parse_failed` with our fixed message, nothing from pypdf |
| Password-protected PDF | `parse_failed` saying it is locked, not "no extractable text" |
| `.txt` with no blank lines in it | Split into bounded pages, every word preserved |
| Outline nested past the depth limit | Headings ignored, no `RecursionError` |
| Same bytes ingested twice | `already_indexed`, no duplicate |
| Same filename, different bytes | Treated as a new document, no silent overwrite |
| One byte of a stored document flipped | Integrity check fails |
| Planted-instruction fixture | One `Flag`, correct page, bounded excerpt |
| Clean document | Zero flags |
| Log grep after a full run | No document phrases present |

The **log grep is the highest-value single test here**, because it catches the entire class of accidental-leak bugs in one line rather than one at a time.

Note that `data/` is gitignored, so tests depending on the real lectures cannot run in CI or on a teammate's clone. Keep committed synthetic fixtures for the properties above, and guard real-material assertions so they skip when `data/documents/` is empty.

### No security property may rest on a real deck alone

Learned the hard way. The test asserting `ParseFailed` never quotes the document — the guard on the leak channel 4.5 is about — was written against the real decks only, so the skipif above fired on every clone and every CI runner. It had never once run outside the one machine holding the lectures, and it then broke on that machine too, from *adding data*: no code changed.

Two things caused it, both worth knowing before writing another parser test:

- **The decks are two different PDF generations.** `Lecture_1` to `Lecture 5` carry a classic xref table and one `trailer` keyword. `Lecture 6` to `Lecture 27` carry PDF 1.5 cross-reference streams and two. They are not interchangeable fixtures.
- **Damaging a PDF is not as easy as it looks.** Replacing `trailer` with a *longer* string shifts every byte offset after it, and replacing only the first occurrence leaves the second intact — a cross-reference-stream PDF then recovers and parses cleanly, so the test passes while proving nothing. Replace with equal length, every occurrence; the phrase then also lands exactly where pypdf chokes, which is what gets it into pypdf's message in the first place.

And because a sorted glob decides which deck `REAL_DECKS[0]` is, ` ` sorting before `_` moved the test from the first generation to the second. Real-deck assertions are parametrised over every deck now, so a new lecture widens the coverage instead of silently changing the subject.

The resulting rule: **the synthetic fixture is the guard, and the real decks are the corroboration.** A synthetic PDF from pypdf's own writer reproduces the leak property exactly, so that test runs on every push on all three operating systems. It also asserts that pypdf *still* leaks the planted phrase into its own error, because the day that stops being true the test would keep passing while testing nothing.

The planted-instruction PDF needed for the screener test is also the fixture TUT-43 needs for adversarial testing. Build it once.

## 8. Open Items

- **Flagged excerpt length** (TUT-18 against TUT-54). TUT-18 wants the user told what was flagged and where; TUT-54 and threat-model section 4.3 forbid the UI showing raw document content. A flagged excerpt *is* document text. Needs an explicit decision and a maximum length.
- **The wall-clock parse bound is unenforced** (4.2). Accepted risk, not a control: no portable in-process implementation exists, and a file that spins inside `PdfReader()` construction is bounded by nothing. Needs either a subprocess under TUT-55 or an explicit decision to live with it. Do not let it be reported as a control while this item is open.
- **No central logging configuration.** `parser._silence_pypdf_logger` disables pypdf's logger at import time, which is a module doing something global. It belongs in one application-level logging setup that also enforces the metadata-only rule in 4.5 for our own log calls.
- **A PDF page has no size cap** (TUT-67). Only `.txt` is bounded, by `TXT_PAGE_TARGET_CHARS` in the parser; a PDF page becomes one chunk at whatever length it extracts to. Not reachable on the course decks, whose worst page is 1122 characters, but reachable the moment a textbook PDF is uploaded. The cap still belongs in the chunker, where it applies to every source and not only `.txt`.

  **No longer silent, though, which was the dangerous half.** `ingest/embedder.py` passes `truncate=False` on every call, so an over-long page is a refusal naming the page rather than a vector built from its leading tokens and cited as the whole page. Measured while building the embedder: under Ollama's default the vectors for `text` and for `text + anything` are *byte-identical* once the context fills, so the old behaviour had no failure surface whatsoever. Also measured: the cut is in tokens and the ratio is content-dependent — it landed near 12,000 characters for repetitive English prose, not the ~8000 this item previously assumed — so a character cap in the chunker is a belt that keeps users out of the error, and cannot be the control itself.
- **Malicious file ingestion as R4** (TUT-63). Still undecided. Section 4.2 above holds either way.
- **Process layout** (TUT-55). If ingest moves to its own process, section 3's ordering still holds but the call boundary becomes a wire format.
- **DOCX is not implemented.** TUT-14 lists PDF, DOCX and TXT, but `python-docx` is not installed and no course material is DOCX. Accepting a file that cannot be parsed would surface as `parse_failed` and read like a corrupt document rather than a missing feature, so DOCX is rejected explicitly until the dependency lands.
- ~~**Embedding throughput is unmeasured.**~~ Measured 2026-11-07: `nomic-embed-text` on CPU runs ~13.2 chunks/s, steady from 10 to 100 chunks per call, so the ~250-chunk semester embeds in about 20 seconds. See architecture doc 5.1.
- **The flagged-review flow does not exist** (TUT-51). The CLI refuses a flagged document outright: nothing is stored, recorded or chunked. That is the fail-closed half of section 3.3; the "index anyway" decision needs the second entry point the interface doc's section 6 describes.
- **`record()` takes a chunk count**, which the original TUT-17 signature did not. `Result.chunk_count` has to be reportable for `already_indexed`, and it is not derivable from the page count alone because empty pages are skipped.
