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
| `security/patterns.py` | The injection pattern list itself | TUT-18 |

`ingest/manifest.py` does not exist yet; everything else in the table is scaffolded. The pattern list belongs in `security/` rather than beside the screener because TUT-18 requires it there, so it can be extended during R1 testing without touching ingest code.

**`model.py` is the keystone.** The screener and the chunker both read `PageRecord`, so its shape decides what they can do. A `PageRecord` that loses the page number cannot support citations or page-located flags. Settle it first; it is the most expensive thing here to change later.

**There is deliberately no orchestrator yet.** No `Outcome` in the published contract can honestly describe a pipeline that stops after chunking — `indexed` would be false, and inventing a new outcome would break the contract the UI is being built against. Wire the generator when embedding and indexing exist.

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

- page count
- wall-clock time
- **total extracted characters**

The third matters most and is the one usually missed. A file-size cap does not bound memory: a few megabytes of PDF can expand to gigabytes of extracted text.

Whether this becomes a fourth risk is still open (TUT-63). These bounds are cheap regardless, and "out of scope in the threat model" does not mean it will not happen during a demo.

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

Measured on the two real lecture decks: **one near-empty page in each** (11 pages and 10 pages respectively), from title or image slides.

That sets the rule:

- Per-page emptiness is **normal**. Retain the record, mark it, and let the chunker skip it.
- Only when **every** page is empty does the document fail, as `PARSE_FAILED`, with a message along the lines of "no extractable text; the document may be scanned images".

Deciding between rejecting such files and adding OCR is a scope question worth settling before it appears in a demo. OCR is a large piece of work nobody has budgeted.

## 6. Risk Traceability

For the Round 2 report: each control, the risk it serves, where it comes from, and what proves it.

| Control | Risk | Source | Evidence |
|---|---|---|---|
| Content sniffing, not extension trust | R1, R4? | TUT-14 | A `.pdf` that is not a PDF returns `rejected_content_mismatch` |
| Page, time and extracted-character bounds | R4? | TUT-63 open | Oversized and bomb-shaped fixtures are rejected, memory stays bounded |
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
| Same bytes ingested twice | `already_indexed`, no duplicate |
| Same filename, different bytes | Treated as a new document, no silent overwrite |
| One byte of a stored document flipped | Integrity check fails |
| Planted-instruction fixture | One `Flag`, correct page, bounded excerpt |
| Clean document | Zero flags |
| Log grep after a full run | No document phrases present |

The **log grep is the highest-value single test here**, because it catches the entire class of accidental-leak bugs in one line rather than one at a time.

Note that `data/` is gitignored, so tests depending on the real lectures cannot run in CI or on a teammate's clone. Keep committed synthetic fixtures for the properties above, and guard real-material assertions so they skip when `data/documents/` is empty.

The planted-instruction PDF needed for the screener test is also the fixture TUT-43 needs for adversarial testing. Build it once.

## 8. Open Items

- **Flagged excerpt length** (TUT-18 against TUT-54). TUT-18 wants the user told what was flagged and where; TUT-54 and threat-model section 4.3 forbid the UI showing raw document content. A flagged excerpt *is* document text. Needs an explicit decision and a maximum length.
- **Malicious file ingestion as R4** (TUT-63). Still undecided. Section 4.2 above holds either way.
- **Process layout** (TUT-55). If ingest moves to its own process, section 3's ordering still holds but the call boundary becomes a wire format.
- **DOCX is not implemented.** TUT-14 lists PDF, DOCX and TXT, but `python-docx` is not installed and no course material is DOCX. Accepting a file that cannot be parsed would surface as `parse_failed` and read like a corrupt document rather than a missing feature, so DOCX is rejected explicitly until the dependency lands.
- **Embedding throughput is unmeasured.** The `ml` extra is not installed. Relevant once stages 5 and 6 exist.
