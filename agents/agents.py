#!/usr/bin/env python3
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
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import ollama


SUPPORTED_TEXT = {".txt", ".md", ".rst", ".csv", ".json", ".py", ".java",
                  ".c", ".cpp", ".h", ".hpp", ".html", ".xml", ".yaml", ".yml"}
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


@dataclass
class Chunk:
    text: str
    source: str
    embedding: list[float]


def read_file(path: Path) -> str:
    """Extract text from a supported file."""
    suffix = path.suffix.lower()

    if suffix in SUPPORTED_TEXT:
        return path.read_text(encoding="utf-8", errors="replace")

    if suffix == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise RuntimeError("PDF support requires: pip install pypdf") from exc
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        try:
            from docx import Document
        except ImportError as exc:
            raise RuntimeError("DOCX support requires: pip install python-docx") from exc
        document = Document(str(path))
        return "\n".join(p.text for p in document.paragraphs)

    raise ValueError(f"Unsupported file type: {suffix or '(no extension)'}")


def split_into_chunks(text: str, size: int = CHUNK_SIZE,
                      overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, preferring paragraph boundaries."""
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n", start + size // 2, end)
            if boundary > start:
                end = boundary

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)

    return chunks


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


class KnowledgeBase:
    def __init__(self, embedding_model: str):
        self.embedding_model = embedding_model
        self.chunks: list[Chunk] = []
        self.files: list[str] = []

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # The Ollama Python package returns one embedding per input string.
        response = ollama.embed(model=self.embedding_model, input=texts)
        return response["embeddings"]

    def add_files(self, filenames: Iterable[str]) -> None:
        added_any = False
        for filename in filenames:
            path = Path(filename).expanduser()
            if not path.is_file():
                print(f"[skip] File not found: {path}")
                continue

            resolved = str(path.resolve())
            if resolved in self.files:
                print(f"[skip] Already ingested: {path}")
                continue

            try:
                text = read_file(path)
                pieces = split_into_chunks(text)
                if not pieces:
                    print(f"[skip] No extractable text: {path}")
                    continue

                print(f"[index] Embedding {len(pieces)} chunk(s) from {path.name}...")
                vectors = self._embed(pieces)
                if len(vectors) != len(pieces):
                    raise RuntimeError("Ollama returned an unexpected number of embeddings.")

                self.chunks.extend(
                    Chunk(piece, path.name, vector)
                    for piece, vector in zip(pieces, vectors)
                )
                self.files.append(resolved)
                print(f"[ok] Added {path.name}")
                added_any = True
            except Exception as exc:
                print(f"[error] Could not ingest {path}: {exc}")

        if added_any:
            print(f"[index] Knowledge base now contains {len(self.chunks)} chunks.")

    def retrieve(self, query: str, top_k: int = 5) -> list[Chunk]:
        if not self.chunks:
            return []
        query_embedding = self._embed([query])[0]
        ranked = sorted(
            self.chunks,
            key=lambda chunk: cosine_similarity(query_embedding, chunk.embedding),
            reverse=True,
        )
        return ranked[:min(top_k, len(ranked))]


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
                f"[Source: {chunk.source} | Retrieved passage {i}]\n{chunk.text}"
                for i, chunk in enumerate(matches, start=1)
            )
        else:
            context = "(No relevant document passages were found.)"

        user_message = (
            "Use the following retrieved files as your primary source. "
            "Distinguish any general background knowledge from information found in the files."
            "Give a citation(s) for your answer(s) of the form (filename): Pg (page or slide number).\n\n"
            f"FILES:\n{context}\n\n"
            f"USER REQUEST:\n{question}"
        )

        messages = [{"role": "system", "content": self.system_prompt}]
        # Keep a modest conversational history so follow-up questions work.
        messages.extend(self.history[-8:])
        messages.append({"role": "user", "content": user_message})

        response = ollama.chat(model=self.chat_model, messages=messages)
        answer = response["message"]["content"]
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        return answer


def build_bots(knowledge: KnowledgeBase, chat_model: str, top_k: int):
    tutor_prompt = """
You are TutorBot, a patient and rigorous tutor.
Teach the material in a clear, structured way. Define important terms, explain
why concepts work, and use examples or step-by-step reasoning when useful.
Adapt explanations to the learner's question. Do not pretend the documents say
something they do not say. Point out uncertainty or contradictions.
"""
    quiz_prompt = """
You are QuizBot, an instructor who creates useful quizzes from the supplied
course materials. When asked for a quiz, create a balanced set of questions
based on the retrieved excerpts. Mix question types when appropriate
(multiple choice, short answer, and application). Do not reveal answers
immediately unless requested. If the user answers questions, grade each answer,
explain mistakes, track the score within the conversation, and then offer the
next question. Do not invent facts and present inferences as inferences.
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
        knowledge.add_files(args.files)

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
            if knowledge.files:
                for filename in knowledge.files:
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
                knowledge.add_files(payload.split())
            continue

        if action == "tutor":
            if not payload:
                print("Usage: tutor <question>")
                continue
            bot = tutor
            prompt = payload
            print(f"\n{bot.name}>")
            try:
                print(bot.ask(prompt))
                defaultBot = "tutor"
            except Exception as exc:
                print(f"[error] {exc}")
            continue

        if action == "quiz":
            prompt = payload or "Create a 5-question quiz covering the key concepts in these materials."
            print("\nQuizBot>")
            try:
                print(quiz.ask(prompt))
                defaultBot = "quiz"
            except Exception as exc:
                print(f"[error] {exc}")
            continue

        if action == "ask":
            if not payload:
                print("Usage: ask <question>")
                continue
            for bot in (tutor, quiz):
                print(f"\n{bot.name}>")
                try:
                    print(bot.ask(payload))
                except Exception as exc:
                    print(f"[error] {exc}")
            continue

        # Treat an unprefixed line as a TutorBot question for convenience.
        if defaultBot == "tutor":
            print("\nTutorBot>")
            try:
                print(tutor.ask(command))
            except Exception as exc:
                print(f"[error] {exc}")
            continue

        if defaultBot == "quiz":
            print("\nQuizBot>")
            try:
                print(quiz.ask(command))
            except Exception as exc:
                print(f"[error] {exc}")
            continue

        #failcase default to tutor
        print("\nTutorBot>")
        try:
            print(tutor.ask(command))
        except Exception as exc:
            print(f"[error] {exc}")

        


if __name__ == "__main__":
    main()
