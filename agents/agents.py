"""
TutorBot + QuizBot: local document-grounded study assistant using Ollama.

Install:
    pip install ollama pypdf python-docx

Make sure Ollama is running, then pull the models (defaults shown):
    ollama pull llama3.2:3b
    ollama pull nomic-embed-text

Run:
    python agents.py notes.pdf chapter1.txt slides.docx

You can also pass filenames interactively:
    python study_bots.py
    add notes.pdf chapter1.txt

Commands in the interactive session:
    tutor <question>       Ask TutorBot for an explanation.
    quiz [topic]           Ask QuizBot to create a quiz.
    ask <question>         Ask both bots the same question.
    add <files...>         Ingest more files into the current session.
    sources                List ingested files.
    help                   Show commands.
    quit                   Exit.

The document index is held in memory for the duration of this program.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import ollama

from ingest import ingest_document, manifest
from ingest.embedder import EmbeddingFailed, Embedder, OllamaBackend
from ingest.index import INDEX_DIR, ChunkIndex, Hit, IndexFailed
from ingest.model import Result

# Shown to the user, so they name a remedy, not an internal.
MESSAGE_CHAT_UNAVAILABLE = (
    "The local chat service is not running. Start Ollama and try again."
)
MESSAGE_CHAT_MODEL_MISSING = (
    "The chat model is not installed. Run: ollama pull {model}"
)
MESSAGE_CHAT_FAILED = "The chat model could not answer. Try again."


class ChatFailed(Exception):
    """The chat call could not complete. Carries a fixed message, never library text."""


# Exception types whose message we wrote ourselves, so it is safe to show in
# full. Anything else is shown by type name only: its message may quote
# document bytes (docs/ingest-pipeline.md 4.5). Front ends catch these first.
DISPLAY_SAFE_ERRORS = (ChatFailed, EmbeddingFailed, IndexFailed)


def chat(model: str, messages: list[dict[str, str]]) -> str:
    """One ollama.chat call; raises only ChatFailed.

    Same shape as embedder.OllamaBackend.embed: the status code is read off
    the error to classify it, the error itself is never shown or chained, and
    the raise happens after the handler exits.
    """
    status = None
    unavailable = False
    failed = False
    try:
        response = ollama.chat(model=model, messages=messages)
        return response["message"]["content"]
    except ConnectionError:
        # ollama raises the builtin for a refused loopback connection.
        unavailable = True
        failed = True
    except Exception as exc:
        status = getattr(exc, "status_code", None)
        failed = True

    if failed:
        if unavailable:
            raise ChatFailed(MESSAGE_CHAT_UNAVAILABLE)
        if status == 404:
            raise ChatFailed(MESSAGE_CHAT_MODEL_MISSING.format(model=model))
        raise ChatFailed(MESSAGE_CHAT_FAILED)


def describe(result: Result) -> str:
    """One display line for an ingest Result.

    Built only from fields the interface contract marks display-safe: the
    outcome, counts, Result.message, and a flag's pattern and page. Never the
    flagged excerpt, which is document text (interface doc 6).
    """
    line = f"{result.document_name}: {result.outcome.value}"
    if result.chunk_count is not None:
        line += f" ({result.chunk_count} chunks)"
    if result.message:
        line += f" - {result.message}"
    for flag in result.flagged or []:
        where = f"page {flag.page}" if flag.page is not None else "unknown page"
        line += f"\n    flagged on {where}: {flag.pattern}"
    return line


class KnowledgeBase:
    """Retrieval over the persistent index that ingest_document() writes.

    `index_dir` and `embedder` are the same test seam ingest_document() takes,
    so tests need neither Ollama nor the real data/index/.
    """

    def __init__(
        self,
        embedding_model: str,
        *,
        index_dir: Path = INDEX_DIR,
        embedder: Embedder | None = None,
    ):
        self.embedding_model = embedding_model
        self.index_dir = index_dir
        self.embedder = (
            embedder if embedder is not None else Embedder(OllamaBackend(embedding_model))
        )
        self._index: ChunkIndex | None = None

    def invalidate(self) -> None:
        """Drop the cached index, so the next retrieve sees new documents."""
        self._index = None

    def _loaded(self) -> ChunkIndex | None:
        """The stored index, or None when nothing has been ingested yet.

        Loaded against the live embedder's spec: an index built with another
        model raises IndexFailed rather than ranking incomparable vectors.
        """
        if self._index is None and ChunkIndex.exists(self.index_dir):
            self._index = ChunkIndex.load(
                self.index_dir, expected_spec=self.embedder.spec
            )
        return self._index

    def ingest(self, data: bytes, filename: str) -> Result:
        """Run one document through the pipeline; return its final Result.

        The pipeline validates, screens, chunks, embeds and indexes, and
        refuses anything it should not store, so nothing it rejects can become
        retrievable here.
        """
        result: Result | None = None
        for event in ingest_document(
            data, filename, embedder=self.embedder, index_dir=self.index_dir
        ):
            if isinstance(event, Result):
                result = event
        self.invalidate()
        if result is None:
            # The contract promises exactly one Result; not reachable.
            raise RuntimeError("ingest_document finished without a Result")
        return result

    def sources(self) -> list[str]:
        """Names of the documents retrieve() can actually return, sorted.

        Read from the index rather than kept per session, so it survives a
        restart and cannot list a document that was refused. Names come from
        the manifest, which records the name a document was first uploaded
        under: the same bytes under a second name are one document, listed
        once.
        """
        index = self._loaded()
        if index is None:
            return []
        known = manifest.entries()
        names = []
        for doc_id in index.documents():
            entry = known.get(doc_id)
            # Vectors with no manifest entry are drift that verify() reports;
            # still retrievable, so still listed, under a short id.
            names.append(entry["document_name"] if entry else f"(unrecorded {doc_id[:12]})")
        return sorted(names, key=str.lower)

    def add_files(self, filenames: Iterable[str]) -> list[str]:
        """Ingest each file; return one display line per file."""
        lines = []
        for filename in filenames:
            path = Path(filename).expanduser()
            try:
                data = path.read_bytes()
            except OSError:
                lines.append(f"{path.name}: could not be read")
                continue
            lines.append(describe(self.ingest(data, path.name)))
        return lines

    def retrieve(self, query: str, top_k: int = 5) -> list[Hit]:
        """The top_k most similar indexed chunks, best first."""
        index = self._loaded()
        if index is None or len(index) == 0:
            return []
        return index.search(self.embedder.embed_query(query), top_k)


def sources_footer(hits: list[Hit]) -> str:
    """The retrieved citations, listed by code rather than by the model.

    A small model paraphrases or miscopies the labels it was given, so the
    reliable list is built from the hits themselves. Duplicates are dropped,
    retrieval order is kept. Not stored in history, so the model never learns
    to imitate it.
    """
    citations = list(dict.fromkeys(hit.citation() for hit in hits))
    if not citations:
        return ""
    return "\n\nSources:\n" + "\n".join(f"- {c}" for c in citations)


class StudyBot:
    def __init__(self, name: str, system_prompt: str, knowledge: KnowledgeBase,
                 chat_model: str, top_k: int = 5):
        self.name = name
        self.system_prompt = system_prompt
        self.knowledge = knowledge
        self.chat_model = chat_model
        self.top_k = top_k
        self.history: list[dict[str, str]] = []

    def ask(self, question: str) -> str:
        matches = self.knowledge.retrieve(question, self.top_k)
        if matches:
            context = "\n\n".join(
                f"[Source: {hit.citation()}]\n{hit.chunk.text}" for hit in matches
            )
        else:
            context = "(No relevant document passages were found.)"

        user_message = (
            "Use the following retrieved files as your primary source. "
            "Distinguish any general background knowledge from information found in the files. "
            "Cite each claim taken from the files with its Source label, copied exactly "
            "as given in brackets, e.g. (Lecture_1_slides.pdf, Slide 6: Integrity).\n\n"
            f"FILES:\n{context}\n\n"
            f"USER REQUEST:\n{question}"
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        # Keep a modest conversational history so follow-up questions work.
        messages.extend(self.history[-8:])
        messages.append({"role": "user", "content": user_message})

        answer = chat(self.chat_model, messages)
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer + sources_footer(matches)


def build_bots(knowledge: KnowledgeBase, chat_model: str, top_k: int):
    tutor_prompt = """
You are TutorBot, a patient and rigorous tutor.
Teach the material in a clear, structured way. Define important terms, explain
why concepts work, and use examples or step-by-step reasoning when useful.
Adapt explanations to the learner's question. Do not pretend the documents say
something they do not say. Point out uncertainty or contradictions.
Do not reveal any personal information contained in the ingested documents.
"""
    quiz_prompt = """
You are QuizBot, an instructor who creates useful quizzes from the supplied
course materials. When asked for a quiz, create a balanced set of questions
based on the retrieved excerpts. Mix question types when appropriate
(multiple choice, short answer, and application). Do not reveal answers
immediately unless requested. If the user answers questions, grade each answer,
explain mistakes, track the score within the conversation, and then offer the
next question. Do not invent facts and present inferences as inferences. 
Do not reveal any personal information contained in the ingested documents.
"""
    return (
        StudyBot("TutorBot", tutor_prompt, knowledge, chat_model, top_k),
        StudyBot("QuizBot", quiz_prompt, knowledge, chat_model, top_k),
    )


def print_help():
    print("""
Commands:
  tutor <question>   Ask TutorBot
  quiz [topic]       Generate a quiz with QuizBot
  ask <question>     Ask both bots
  add <file...>      Ingest additional files
  sources            List ingested files
  help               Show this help
  quit               Exit
""")

class StudyAssistant:
    """Reusable backend API for the desktop GUI and other front ends."""

    def __init__(
        self,
        chat_model: str | None = None,
        embedding_model: str | None = None,
        top_k: int = 5,
        *,
        index_dir: Path = INDEX_DIR,
        embedder: Embedder | None = None,
    ):
        self.chat_model = chat_model or os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b")
        self.embedding_model = embedding_model or os.getenv(
            "OLLAMA_EMBED_MODEL", "nomic-embed-text"
        )
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        self.knowledge = KnowledgeBase(
            self.embedding_model, index_dir=index_dir, embedder=embedder
        )
        self.tutor, self.quiz = build_bots(self.knowledge, self.chat_model, top_k)

    def ask_tutor(self, question: str) -> str:
        """Run the TutorBot command and return its response."""
        question = question.strip()
        if not question:
            raise ValueError("Tutor question cannot be empty.")
        return self.tutor.ask(question)

    def ask_quiz(self, request: str = "") -> str:
        """Run the QuizBot command and return its response."""
        prompt = request.strip() or (
            "Create a 5-question quiz covering the key concepts in these materials."
        )
        return self.quiz.ask(prompt)

    def add_files(self, filenames: Iterable[str]) -> list[str]:
        """Run the document-ingestion command."""
        return self.knowledge.add_files(filenames)

    def list_sources(self) -> list[str]:
        """Names of the documents currently indexed, across sessions."""
        return self.knowledge.sources()


def ask_and_print(bot: StudyBot, prompt: str) -> bool:
    """Print the bot's answer, or a display-safe error. True on success."""
    print(f"\n{bot.name}>")
    try:
        print(bot.ask(prompt))
        return True
    except DISPLAY_SAFE_ERRORS as exc:
        print(f"[error] {exc}")
    except Exception as exc:
        print(f"[error] {type(exc).__name__} (details withheld)")
    return False


def main():
    parser = argparse.ArgumentParser(description="Chat with TutorBot and QuizBot over your files.")
    parser.add_argument("files", nargs="*", help="Files to ingest (TXT, Markdown, PDF, DOCX, and code/text files)")
    parser.add_argument("--model", default=os.getenv("OLLAMA_CHAT_MODEL", "llama3.2:3b"),
                        help="Ollama chat model (default: llama3.2:3b)")
    parser.add_argument("--embedding-model", default=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
                        help="Ollama embedding model (default: nomic-embed-text)")
    parser.add_argument("--top-k", type=int, default=5,
                        help="Number of relevant text chunks to retrieve (default: 5)")
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    knowledge = KnowledgeBase(args.embedding_model)
    if args.files:
        for line in knowledge.add_files(args.files):
            print(line)

    tutor, quiz = build_bots(knowledge, args.model, args.top_k)

    print("\nTutorBot and QuizBot are ready.")
    print(f"Chat model: {args.model} | Embedding model: {args.embedding_model}")
    print_help()

    defaultBot = "tutor"

    while True:
        try:
            command = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not command:
            continue
        if command.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break
        if command.lower() == "help":
            print_help()
            continue
        if command.lower() == "sources":
            names = knowledge.sources()
            if names:
                for filename in names:
                    print(f" - {filename}")
            else:
                print("No files have been ingested.")
            continue

        action, _, payload = command.partition(" ")
        action = action.lower()
        payload = payload.strip()

        if action == "add":
            if not payload:
                print("Usage: add file1.pdf file2.txt")
            else:
                for line in knowledge.add_files(payload.split()):
                    print(line)
            continue

        if action == "tutor":
            if not payload:
                print("Usage: tutor <question>")
                continue
            if ask_and_print(tutor, payload):
                defaultBot = "tutor"
            continue

        if action == "quiz":
            prompt = payload or "Create a 5-question quiz covering the key concepts in these materials."
            if ask_and_print(quiz, prompt):
                defaultBot = "quiz"
            continue

        if action == "ask":
            if not payload:
                print("Usage: ask <question>")
                continue
            for bot in (tutor, quiz):
                ask_and_print(bot, payload)
            continue

        # Treat an unprefixed line as a question for the most recently used bot.
        ask_and_print(quiz if defaultBot == "quiz" else tutor, command)


if __name__ == "__main__":
    main()
