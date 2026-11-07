# Setup and Usage

Every command for this project, in the order you need them. For what the project *is*, the
architecture, and the threat-model rationale behind the secret scanning, see
[README.md](README.md).

Prefix everything with `uv run`. **There is no activation step** — this is identical on Windows,
Linux and macOS, which avoids the `source .venv/bin/activate` versus `.venv\Scripts\activate`
split.

---

## 1. Environment

### Install uv

| OS | Command |
|---|---|
| Linux / macOS | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `irm https://astral.sh/uv/install.ps1 \| iex` |

### Clone and create the environment

```bash
git clone https://github.com/asamc/TutorAndQuizBot.git
cd TutorAndQuizBot
uv sync
uv run pre-commit install    # required, see section 2
```

`uv sync` reads `.python-version` and `pyproject.toml`, fetches Python 3.12 if you do not have it,
creates `.venv`, and installs the locked dependencies including the hook tooling. No flags are
needed for ingest work.

### Optional heavy extras — not needed for ingest

```bash
uv sync --extra ml
```

`--extra ml` adds `faiss-cpu`, `sentence-transformers`, `torch` and `streamlit`: roughly 2–3 GB.
Pull it in only when retrieval (TUT-4) or the UI (TUT-10) lands. On Linux, `pyproject.toml` pins
`torch` to the CPU-only index so it does not drag in ~18 CUDA packages.

### Without uv

```bash
pip install -r requirements.txt
pip install pre-commit detect-secrets pytest
```

`requirements.txt` is runtime-only, exported from `uv.lock`, so the hook and test tooling has to
be installed separately.

---

## 2. Secret scanning

Required once per clone. Without it the hooks do not fire for you.

```bash
uv run pre-commit install
```

Writes the hook into `.git/hooks/pre-commit`. Required defence for R3 — see
[README.md](README.md#secret-scanning) for why, and `docs/threat-model.md` section 5.3.

### What runs on every `git commit`

| Hook | Effect |
|---|---|
| `detect-secrets` | Scans staged changes against `.secrets.baseline` (`--baseline .secrets.baseline`) |
| `detect-private-key` | Catches SSH/PEM keys regardless of filename |
| `check-added-large-files` | `--maxkb=5000`; a 5 MB cap backstopping R2 against model weights or course PDFs that slip past `.gitignore` |
| `check-merge-conflict` | Blocks committed conflict markers |

A match exits non-zero and no commit is created.

### Run the hooks without committing

```bash
uv run pre-commit run --all-files
```

`--all-files` checks the whole tree instead of just staged changes. Use it after cloning, or
before a large push.

### Update the baseline

```bash
uv run detect-secrets scan --baseline .secrets.baseline
uv run detect-secrets audit .secrets.baseline
```

`scan --baseline` rewrites `.secrets.baseline` in place, keeping existing audit decisions.
`audit` steps through each finding interactively so you mark it real or false.

### Mark a genuine false positive inline

Preferred over disabling the hook:

```python
SOME_VALUE = "looks-random-but-is-not"  # pragma: allowlist secret
```

### Bypassing

```bash
git commit --no-verify              # skips all local hooks
SKIP=detect-secrets git commit      # skips one hook
```

Both exist; neither is a way out. CI re-runs a secret scan over the full history on every push,
and that is what gates a merge.

---

## 3. Ollama and the model

Needed for the agents, **not** for ingest.

| OS | Command |
|---|---|
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| macOS | `brew install ollama` |
| Windows | Installer from [ollama.com/download](https://ollama.com/download) |

```bash
ollama pull llama3.2:3b
```

One model serves both agents via separate prompt profiles (TUT-57).

---

## 4. Verify your environment

Run each of these and check the result. No application code is needed.

| # | Command | Expected |
|---|---|---|
| 1 | `uv --version` | 0.13 or newer |
| 2 | `uv run python -V` | `Python 3.12.x` |
| 3 | `ollama --version` | 0.40 or newer |
| 4 | `ollama list` | `llama3.2:3b` is listed |
| 5 | `ollama run llama3.2:3b "In one sentence: what does TLS protect?"` | A coherent sentence about TLS |

**6. Confirm Ollama is bound to loopback only.** This is the one check whose command differs per
OS. The result must show `127.0.0.1`, never `0.0.0.0` — otherwise the model is reachable from your
network, which breaks the trust boundary in the threat model.

| OS | Command |
|---|---|
| Linux | `ss -tln \| grep 11434` |
| macOS | `lsof -iTCP:11434 -sTCP:LISTEN` |
| Windows | `netstat -an \| findstr 11434` |

---

## 5. Running the ingest pipeline

Stages run in order: validate → hash → parse → screen → chunk → store → record. A flagged
document stops at screening and nothing is stored or indexed. Embedding and indexing are not
wired yet (TUT-58). The pipeline never calls the LLM, so section 3 is not a prerequisite.

### Ingest one or more files

```bash
uv run python -m ingest.cli data/documents/Lecture_1_slides.pdf
uv run python -m ingest.cli notes0.txt notes1.txt
```

`paths` is positional and variadic: one or more PDF or TXT files. DOCX is **rejected** despite
appearing in the README's component table — `python-docx` is not installed, and accepting a file
we cannot parse would surface later as `PARSE_FAILED` and read like a corrupt document.

### Flags

| Flag | Effect |
|---|---|
| `--chunks` | Print one line per chunk: document, page/slide, heading, character count. Metadata only; chunk text is never printed. |
| `--list` | Show the manifest: 16-char document id, filename, page count, chunk count, ingest timestamp. The id prefix shown here is what `--delete` takes. |
| `--verify` | Re-hash every stored document against its recorded SHA-256 (TUT-23). Prints `ok` / `MISMATCH` / `missing` per document, and exits 1 if anything fails. |
| `--delete ID [ID ...]` | Remove the manifest entry **and** the stored file. Takes the id prefixes `--list` prints, or full 64-char ids. An ambiguous prefix is refused and names the candidates rather than guessing. |
| `--all` | With `--delete` only: every recorded document. This is "purge everything recorded", not "purge test files" — run `--list` first. |
| `--yes` | Skip the confirmation prompt. Required when stdin is not a terminal, so CI cannot silently wipe state. |

```bash
uv run python -m ingest.cli --chunks notes0.txt
uv run python -m ingest.cli --list
uv run python -m ingest.cli --verify
uv run python -m ingest.cli --delete 83e080edc119614d a641aaf772d53290
uv run python -m ingest.cli --yes --delete --all
```

`--delete` only touches files the manifest points at, so documents sitting in `data/documents`
under their original names are unreachable from it.

**Do not run any of this under `sudo`.** Nothing needs root; it writes only to `data/documents`
and `~/.tutorandquizbot`. One elevated run leaves root-owned artifacts that a later `--delete`
cannot remove.

The manifest lives at `~/.tutorandquizbot/manifest.json`, outside `data/` by design so tampering
with documents cannot rewrite their hashes. `data/` is entirely gitignored (`**/data`), so nothing
you ingest is ever committed, and a fresh clone starts with an empty manifest even if
`data/documents` already has files in it.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Every file got through |
| 1 | A rejection, parse failure, injection flag, verify mismatch, or a failed delete |

---

## 6. Tests

```bash
uv run pytest                            # whole suite
uv run pytest -q                         # quiet, one line per file
uv run pytest tests/test_patterns.py     # pattern hygiene only
uv run pytest -k screener                # match test names
```

`pyproject.toml` sets `pythonpath = ["."]` and `testpaths = ["tests"]`, so `import ingest` works
without installing the package — there is no `[build-system]`, this is a uv virtual project.

Expect one pre-existing failure,
`test_parser.py::test_parse_failed_message_never_quotes_the_document`. It depends on a real deck
in `data/`, which is gitignored, so it skips on a fresh clone and in CI.
