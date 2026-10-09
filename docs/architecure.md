# Architecture

Status: High-level draft. Component roles and communication are set. Process layout, model runtime, and storage details are still open (see section 5).

## 1. Overview

TutorAndQuizBot is a local question-answering and quiz system for network security course material. A user uploads course files, asks questions, and takes quizzes. Answers and feedback cite the source documents. Course content stays on the local machine. The only outbound connection is the Internet Lookup module, which sends scrubbed search queries and never receives document content.

## 2. Components and Communication

```mermaid
flowchart LR
    User([User])

    subgraph LOCAL["Local machine (trust boundary)"]
        UI[UI<br/>chat, quiz, upload]
        Tutor[Tutor Agent]
        Quiz[Quiz Agent]
        Retrieval[Retrieval]
        Index[(Vector index<br/>+ chunk store)]
        Ingest[Ingest pipeline]
        LLM[Local LLM<br/>loopback only]
        Lookup[Internet Lookup<br/>scrub + allowlist]
    end

    Brave[(Brave Search API)]

    User --> UI
    UI -->|questions| Tutor
    UI -->|quiz requests and answers| Quiz
    UI -->|uploaded files| Ingest
    Ingest -->|chunks + embeddings| Index
    Tutor --> Retrieval
    Quiz --> Retrieval
    Retrieval <--> Index
    Tutor <--> LLM
    Quiz <--> LLM
    Tutor -.->|structured search request| Lookup
    Quiz -.->|structured search request| Lookup
    Lookup -->|HTTPS, scrubbed query only| Brave
    Brave -->|results with URLs| Lookup
```

Solid arrows show the normal data path. Dashed arrows show a request that goes through the Internet Lookup module, which is the only component allowed to reach the network.

| Component | Role |
|---|---|
| UI | Single interface for chat, quiz sessions, and document upload |
| Tutor Agent | Answers questions from retrieved material, with citations |
| Quiz Agent | Generates quizzes, holds reference answers, grades responses, gives cited feedback |
| Retrieval | Finds the most relevant chunks for a question or quiz topic |
| Vector index and chunk store | Holds embeddings and chunk text with source metadata |
| Ingest pipeline | Validates, parses, screens, hashes, chunks, embeds, and indexes files |
| Local LLM | One local Llama model serving both agents |
| Internet Lookup | Scrubs outbound queries, enforces the host allowlist, returns cited results |

## 3. Main Flows

### 3.1 Ingestion

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant Ingest
    participant Index

    User->>UI: upload course file
    UI->>Ingest: file
    Ingest->>Ingest: type/size check, parse, hash, screen
    Ingest->>Ingest: chunk (~500 words), embed
    Ingest->>Index: chunks + embeddings + metadata
    Index-->>UI: indexed (document name, count)
    UI-->>User: confirmation
```

Each chunk keeps its document name, section heading, and page or slide number, so citations can point to a specific location.

### 3.2 Tutor Question

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant Tutor
    participant Retrieval
    participant LLM
    participant Lookup

    User->>UI: question
    UI->>Tutor: question
    Tutor->>Retrieval: find relevant chunks
    Retrieval-->>Tutor: top chunks with citation metadata
    Tutor->>LLM: question + chunks in labeled context blocks
    LLM-->>Tutor: draft answer
    opt local material is insufficient
        Tutor->>Lookup: structured search request
        Lookup-->>Tutor: web results with URLs
    end
    Tutor->>Tutor: output check (quote length, internal IDs)
    Tutor-->>UI: answer with citations
    UI-->>User: answer
```

The model treats retrieved text as reference material, not as instructions.

### 3.3 Quiz

```mermaid
sequenceDiagram
    actor User
    participant UI
    participant Quiz
    participant Retrieval
    participant LLM
    participant Store as Session store

    User->>UI: request quiz (random or topic)
    UI->>Quiz: request
    Quiz->>Retrieval: chunks for topic or random sample
    Quiz->>LLM: generate questions
    LLM-->>Quiz: questions + reference answers
    Quiz->>Store: save reference answers
    Quiz-->>UI: question text only
    UI-->>User: questions
    User->>UI: answers
    UI->>Quiz: answers
    Quiz->>Store: read reference answers
    Quiz->>LLM: grade against reference
    Quiz-->>UI: score + cited feedback
    UI-->>User: feedback
```

Reference answers never leave the server side before the user submits.

### 3.4 Internet Lookup

```mermaid
sequenceDiagram
    participant Agent as Tutor or Quiz Agent
    participant Lookup as Internet Lookup
    participant Brave as Brave Search API

    Agent->>Lookup: structured request {need_web, query}
    Lookup->>Lookup: scrub query of document text
    Lookup->>Lookup: check host allowlist
    Lookup->>Brave: HTTPS request
    Brave-->>Lookup: results
    Lookup-->>Agent: results with URLs
```

Agents never make network calls themselves. They only send structured requests to the lookup module.

## 4. Trust Boundary

```mermaid
flowchart LR
    subgraph Inside["Inside trust boundary"]
        UI2[UI]
        Agents[Agents]
        Ret[Retrieval]
        Model[Local LLM]
        Store[(Index)]
        Ing[Ingest]
    end

    Gate{{Internet Lookup<br/>only outbound path}}
    Ext[(Brave Search)]

    Inside --> Gate
    Gate -->|scrubbed queries only| Ext
```

Everything except the Internet Lookup module stays inside the boundary. The threat model describes the risks at this boundary.

## 5. Open Items

These are deliberately left open until the team agrees on them.

- **Process layout.** The system will run as several processes, but the split and the communication method depend on the team's operating systems (Windows, Linux, macOS).
- **Model runtime.** Candidates include the llama.cpp server, Ollama, and llama-cpp-python. Cross-platform setup effort will drive the choice.
- **One model or two.** Current plan is one base model with two prompt profiles. Separate adapters remain possible.
- **Vector store.** Faiss is the planned option. The encrypted-at-rest approach for the bonus build depends on this choice.
- **Citation granularity.** Page or slide numbers are likely sufficient, but PDF section detection may need work.
- **Ingest isolation.** Whether parsing runs in a separate process depends on the process layout decision and on whether malicious file ingestion becomes R4.
