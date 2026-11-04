Fall 2026

Edit this to add your name to the list when you join, so you show up as contributor

- Asa Mcdaniel
- Reid Layne
- Kelly Nunez
- Jacob Wallace
- Alejandro Rubio

# TutorAndQuizBot

A privacy-preserving, local network-security tutor and quiz generator. Two local Llama-based agents answer course questions with citations and generate graded quizzes, while all course documents stay on the local machine.

Course: CS5342 Network Security (Fall 2026)
Team: Asa McDaniel, Alejandro Rubio, Elena MacCormack, Jacob Wallace, Kelly Nunez, Reid Layne, Michelle Nguyen

## Goals

- **Tutor Agent:** answers user questions from ingested documents, with citations to the source document and, where possible, the section within it.
- **Quiz Agent:** generates randomly chosen or topic-specific quizzes (multiple-choice, true/false, open-ended), grades answers, and gives cited feedback.
- **Privacy:** document content never leaves the machine. Internet lookups go through a separate module that never receives ingested documents.
- **Security:** resistance to prompt injection, protection of the vector database and embeddings, and safe handling of API keys.
- **Single UI:** one central interface for chatting, quizzing, and uploading documents.

## Architecture

```
User -> UI -> Tutor Agent / Quiz Agent -> Retrieval (vector DB) -> Local Llama
                                                                       |
                                  Internet Lookup Module (no doc access) -> external API
```

See [docs/architecure.md](docs/architecure.md) for the full design and [docs/threat-model.md](docs/threat-model.md) for the threat model.

Work is tracked in Jira: [TutorAndQuizBot board](https://alejoserubio.atlassian.net/jira/software/projects/TUT/boards).

## Repository Structure

```
TutorAndQuizBot/
├── agents/
│   ├── quiz-bot/      # Quiz Agent: question generation, grading, feedback
│   └── tutor/         # Tutor Agent: Q&A with citations
├── data/
│   ├── documents/     # Ingested course files (gitignored, never committed)
│   └── index/         # Vector database and embedding cache (encrypted at rest)
├── docs/
│   ├── architecure.md     # System architecture and data flow
│   └── threat-model.md    # Attack surface, risks, and mitigations
├── ingest/            # Document parsing, chunking (~500 words), hashing, indexing
├── internet/          # Internet lookup module (only component with outbound access)
├── llm/               # Local Llama model loading and inference wrapper
├── models/            # Model weights and configs (gitignored)
├── security/          # Prompt-injection filters, output redaction, encryption, key handling
├── tests/             # Unit tests and adversarial test cases
├── ui/                # Central web interface (chat, quiz, upload)
└── README.md
```

### Component Responsibilities

| Directory | Responsibility |
|---|---|
| `agents/tutor` | Retrieves context, answers questions, attaches citations |
| `agents/quiz-bot` | Builds quizzes, checks answers, returns feedback with citations |
| `data/documents` | Source files as ingested; access restricted to the service account |
| `data/index` | Vector store and embeddings; encrypted when not in use |
| `ingest` | Converts PDF/DOCX/TXT to text, chunks it, computes SHA-256 hashes, writes to the index |
| `internet` | Sends search queries only, after scrubbing document content; holds the API key |
| `llm` | Wraps the local Llama runtime; no outbound network calls |
| `models` | Local model files; not committed to the repository |
| `security` | Injection screening, output filtering for verbatim quotes and internal IDs, key loading |
| `tests` | Adversarial prompts, traffic-verification scripts, integrity checks |
| `ui` | Single interface for queries, quiz sessions, and document upload |

## Prerequisites

- **Python 3.12** — pinned in `.python-version`; uv installs it for you, so no manual install needed
- **[uv](https://docs.astral.sh/uv/)** for environment management
- **[Ollama](https://ollama.com)** as the local model runtime
- **`llama3.2:3b`** — one model serves both agents via separate prompt profiles
- Optional: an API key for the internet lookup provider, read from the OS keychain

Versions are pinned in `pyproject.toml` and `uv.lock`. A pip-compatible `requirements.txt` is
exported from the lockfile for anyone not using uv.

The sentence-embedding model, vector store, and Streamlit are declared as the `ml` extra but are
**not installed by default** — they add roughly 2–3 GB. Install them when you start work that needs
them: `uv sync --extra ml`.

## Setup

### 1. Install uv

| OS | Command |
|---|---|
| Linux / macOS | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Windows (PowerShell) | `irm https://astral.sh/uv/install.ps1 \| iex` |

### 2. Clone and create the environment

```bash
git clone https://github.com/asamc/TutorAndQuizBot.git
cd TutorAndQuizBot
uv sync
uv run pre-commit install    # required - see Secret scanning below
```

`uv sync` reads `.python-version` and `pyproject.toml`, fetches Python 3.12 if you do not have it,
creates `.venv`, and installs the locked dependencies including the hook tooling.

**There is no activation step.** Prefix commands with `uv run` instead — `uv run python -V`,
`uv run pytest`. This is identical on Windows, Linux, and macOS, which avoids the
`source .venv/bin/activate` versus `.venv\Scripts\activate` split.

Not using uv? `pip install -r requirements.txt` into your own Python 3.12 environment works.
Note that `requirements.txt` holds runtime dependencies only — install `pre-commit` and
`detect-secrets` yourself if you go that route.

### Secret scanning

A secret scanner runs on every commit. This is a required defence for R3
(see [docs/threat-model.md](docs/threat-model.md) section 5.3), so **everyone must run
`uv run pre-commit install` once after cloning.** Without it the hooks do not fire for you.

On each commit, `detect-secrets` checks staged changes against `.secrets.baseline` and blocks the
commit on a match. Three backstops run alongside it: private-key detection, a merge-conflict check,
and a 5 MB file-size cap that catches model weights or course PDFs slipping past `.gitignore`.

Verified behaviour: a planted fake Brave key and AWS secret were both caught, and `git commit`
exited non-zero without creating the commit.

If the scanner flags something that is genuinely not a secret, mark it inline rather than disabling
the hook:

```python
SOME_VALUE = "looks-random-but-is-not"  # pragma: allowlist secret
```

The local hook is opt-in by nature, so it is a convenience, not the enforcement point — CI re-runs a
secret scan over the full history on every push, and that is what gates a merge.

### 3. Install Ollama and pull the model

| OS | Command |
|---|---|
| Linux | `curl -fsSL https://ollama.com/install.sh \| sh` |
| macOS | `brew install ollama` |
| Windows | Installer from [ollama.com/download](https://ollama.com/download) |

```bash
ollama pull llama3.2:3b
```

### 4. Verify your environment

Run each of these and check the result. No application code is needed.

| # | Command | Expected |
|---|---|---|
| 1 | `uv --version` | 0.13 or newer |
| 2 | `uv run python -V` | `Python 3.12.x` |
| 3 | `ollama --version` | 0.40 or newer |
| 4 | `ollama list` | `llama3.2:3b` is listed |
| 5 | `ollama run llama3.2:3b "In one sentence: what does TLS protect?"` | A coherent sentence about TLS |

**6. Confirm Ollama is bound to loopback only.** This is the one check whose command differs per OS.
The result must show `127.0.0.1`, never `0.0.0.0` — otherwise the model is reachable from your
network, which breaks the trust boundary in the threat model.

| OS | Command |
|---|---|
| Linux | `ss -tln \| grep 11434` |
| macOS | `lsof -iTCP:11434 -sTCP:LISTEN` |
| Windows | `netstat -an \| findstr 11434` |

### Measured baseline

On the reference machine (15 GiB RAM, Intel Arc integrated graphics, no CUDA, CPU inference):

| Metric | Value |
|---|---|
| Generation speed | ~17.7 tok/s |
| Resident memory, model loaded | ~2.4 GB (Ollama reports 2.6 GB at 4096 context) |
| Short answer, end to end | ~5 s |

Expect slower results on a machine with fewer cores, and faster on Apple Silicon.

## Usage (Planned)

1. Start the UI and upload course PDFs or slides; the ingest step hashes, chunks, and indexes them.
2. Ask a question in the Tutor tab; the answer includes citations to document and section.
3. In the Quiz tab, choose random or topic-specific mode, answer the questions, and receive graded feedback with citations.

## Security Notes

- Documents and the index never leave the machine. The only outbound path is the internet lookup module, which receives search queries with document content removed.
- The index is encrypted at rest and unlocked only while the application runs.
- Document SHA-256 hashes are recorded at ingest and checked on load to detect tampering.
- API keys are read from the environment or OS keychain and are never committed.

## Status

Early scaffolding. Directory layout and documentation are in place; agent, ingestion, and UI code are not yet implemented.

## Team Contributions

| Member | Role |
|---|---|
| Jacob Wallace | Tutor Bot assistance; UI / central interface lead |
| Alejandro Rubio | Tutor Bot: subject-material correctness training |
| Reid Layne | Quiz Bot: subject-material correctness training |
| Asa McDaniel | Tutor Bot: citation lookup training |
| Elena MacCormack | Quiz Bot: citation lookup training |
| Michelle Nguyen | Quiz Bot: attack-resistance testing |
| Kelly Nunez | Tutor Bot: attack-resistance testing |

