"""Terminal interface: a stopgap until the web UI exists.

The assignment rules out a command-line interface as the product UI (section
5.2), so this is a prototype harness for development and the Round 2 demo, not
the final interface. It calls the same functions the web UI will --
Library.upload(), start_quiz() and submit() -- so nothing here needs porting
later.

    uv run python -m ui.terminal

Uploaded documents are stored in data/documents/, which is gitignored.
"""

import string
import time

from agents.quiz_bot.documents import DocumentError, Library
from agents.quiz_bot.model import Mode
from agents.quiz_bot.quiz import QuizError, start_quiz, submit
from llm import LLMError

DEFAULT_QUESTIONS = 3
MAX_QUESTIONS = 10


class Quit(Exception):
    """The user typed q, or closed input, at a prompt."""


def ask(prompt: str) -> str:
    """Read one line. 'q' at any prompt leaves the current activity."""
    try:
        # PowerShell prefixes piped input with a byte-order mark; drop it.
        answer = input(prompt).replace("﻿", "").strip()
    except EOFError:
        raise Quit from None
    if answer.lower() == "q":
        raise Quit
    return answer


def choose(prompt: str, options: list[str]) -> int:
    """Show numbered options and return the 0-based index of the user's pick."""
    for number, option in enumerate(options, 1):
        print(f"  {number}) {option}")
    while True:
        answer = ask(prompt)
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer) - 1
        print(f"  Enter a number from 1 to {len(options)}, or q to go back.")


def upload(library: Library) -> None:
    print("\nEnter the path to a lecture PDF or TXT file.")
    print("Tip: drag the file into this window to paste its path. q to go back.")
    while True:
        # Dragging a file in, or "Copy as path", wraps it in quotes.
        path = ask("File: ").strip("\"'")
        if not path:
            continue
        try:
            name, slides = library.upload(path)
        except DocumentError as exc:
            print(f"  Not uploaded: {exc}")
            continue
        print(f"  Uploaded {name}: {slides} slides ready to quiz on.")
        return


def list_documents(library: Library) -> None:
    if not library.chunks:
        print("\nNo documents yet. Choose 'Upload a document' first.")
        return
    print(f"\nYour documents ({len(library.chunks)} slides in total):")
    for name in library.documents():
        count = sum(1 for c in library.chunks if c.document_name == name)
        print(f"  - {name} ({count} slides)")


def pick_topic(library: Library) -> tuple[Mode, str | None]:
    print("\nWhat should the quiz cover?")
    kind = choose("Choice: ", ["Random mix of everything", "A lecture topic", "Type my own topic"])
    if kind == 0:
        return Mode.RANDOM, None
    if kind == 1:
        topics = library.topics()
        print()
        return Mode.TOPIC, topics[choose("Topic number: ", topics)]
    return Mode.TOPIC, ask("Topic (e.g. firewalls, hash functions): ")


def pick_count() -> int:
    while True:
        answer = ask(f"How many questions? (1-{MAX_QUESTIONS}, Enter for {DEFAULT_QUESTIONS}): ")
        if not answer:
            return DEFAULT_QUESTIONS
        if answer.isdigit() and 1 <= int(answer) <= MAX_QUESTIONS:
            return int(answer)


def run_quiz(library: Library) -> None:
    if not library.chunks:
        print("\nNo documents yet. Choose 'Upload a document' first.")
        return
    mode, topic = pick_topic(library)
    count = pick_count()

    print(f"\nWriting {count} question(s)" + (f" on '{topic}'" if topic else "") + ".")
    print("This takes about 15 seconds per question on a CPU...")
    started = time.monotonic()
    try:
        session_id, views = start_quiz(library, mode, topic, count)
    except (QuizError, LLMError) as exc:
        print(exc)
        return
    print(f"Ready in {time.monotonic() - started:.0f}s. Type q at any answer to stop.\n")

    score = 0.0
    answered = 0
    try:
        for number, view in enumerate(views, 1):
            letters = string.ascii_uppercase[: len(view.options)]
            print(f"Q{number}/{len(views)}. {view.stem}")
            for letter, option in zip(letters, view.options):
                print(f"   {letter}) {option}")
            choice = ""
            while len(choice) != 1 or choice not in letters:
                choice = ask(f"Your answer ({'/'.join(letters)}): ").upper()
            result = submit(session_id, view.id, view.options[letters.index(choice)])
            score += result.score
            answered += 1
            print(f"   {result.feedback}")
            print(f"   Source: {result.citation}\n")
    except Quit:
        print()

    if answered:
        print(f"Score: {score:g}/{answered}" + ("" if answered == len(views) else " (stopped early)"))


MENU = {
    "1": ("Upload a document", upload),
    "2": ("Take a quiz", run_quiz),
    "3": ("List my documents", list_documents),
}


def main() -> None:
    print("=== Network Security Tutor & Quiz Bot (terminal prototype) ===")
    library = Library()
    for problem in library.problems:
        print(f"Could not read {problem}")
    if library.chunks:
        print(f"{len(library.documents())} document(s) loaded, {len(library.chunks)} slides.")
    else:
        print("No documents yet. Upload your lecture slides to get started.")

    while True:
        print("\nWhat would you like to do?")
        for key, (label, _) in MENU.items():
            print(f"  {key}) {label}")
        print("  q) Quit")
        try:
            choice = ask("Choice: ")
        except Quit:
            break
        if choice not in MENU:
            print(f"  Enter {', '.join(MENU)}, or q to quit.")
            continue
        try:
            MENU[choice][1](library)
        except Quit:
            continue
        except KeyboardInterrupt:
            print("\nCancelled.")
    print("Goodbye.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nGoodbye.")
