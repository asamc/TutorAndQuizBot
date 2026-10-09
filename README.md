Fall 2026

Edit this to add your name to the list when you join, so you show up as contributor

- Asa Mcdaniel
- Reid Layne
- Kelly Nunez
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

See [docs/architecture.md](docs/architecture.md) for the full design and [docs/threat-model.md](docs/threat-model.md) for the threat model.

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
│   ├── architecture.md    # System architecture and data flow
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

- Python 3.10+ (planned)
- A local Llama-family model in GGUF format (planned runtime: llama.cpp or Ollama)
- A sentence-embedding model from the Sentence Transformers library
- A vector store (Faiss planned; Qdrant, Weaviate, or Milvus are alternatives)
- Streamlit for the UI (planned)
- Optional: an API key for the internet lookup provider, stored in an environment variable

Exact versions will be listed in `requirements.txt` once the stack is confirmed.

## Setup (Planned)

```bash
git clone https://github.com/asamc/TutorAndQuizBot.git
cd TutorAndQuizBot
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the model file under models/
export LOOKUP_API_KEY=...   # only if internet lookup is enabled
streamlit run ui/app.py
```

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

