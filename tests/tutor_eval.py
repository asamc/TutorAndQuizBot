#!/usr/bin/env python3
"""
Repeatable evaluation harness for TutorBot (agents/agents.py).

Drives StudyBot directly via KnowledgeBase + build_bots instead of the
interactive loop. agents.py is NOT modified: KnowledgeBase.retrieve and
ollama.chat are wrapped at runtime so each turn's retrieval and token usage
can be logged.

Usage (from the repo root):
    uv run python tests/tutor_eval.py
    uv run python tests/tutor_eval.py --runs 3 --file ..\\studymat\\Network_Security_Midterm_Slides.pdf

Output: tests/results/tutor_eval_<timestamp>.json
Scoring is done afterwards by a human/LLM reviewer against tests/tutor_gold.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "agents"))

import ollama  # noqa: E402
from pypdf import PdfReader  # noqa: E402

import agents  # noqa: E402  (agents/agents.py)
from agents import KnowledgeBase, build_bots  # noqa: E402

DEFAULT_FILE = REPO_ROOT.parent / "studymat" / "Network_Security_Midterm_Slides.pdf"
DEFAULT_GOLD = REPO_ROOT / "tests" / "tutor_gold.json"
RESULTS_DIR = REPO_ROOT / "tests" / "results"
TRUNCATION_WARN = 4000  # prompt_eval_count at/above this may mean context truncation
PREVIEW_CHARS = 150

# Windows consoles default to cp1252, which crashes on slide bullets/arrows.
for stream in (sys.stdout, sys.stderr):
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass


# --------------------------------------------------------------------------
# Chunk -> true page mapping
# --------------------------------------------------------------------------
class PageMapper:
    """Map a chunk produced by agents.split_into_chunks back to PDF page(s).

    agents.read_file joins page texts with "\\n" and split_into_chunks then
    strips every line and drops blank ones before slicing. We rebuild that same
    normalized text, remembering which page each character came from, and
    locate each chunk in it by exact text match.
    """

    def __init__(self, pdf_path: Path):
        reader = PdfReader(str(pdf_path))
        self.num_pages = len(reader.pages)
        lines: list[str] = []
        line_pages: list[int] = []
        for page_no, page in enumerate(reader.pages, start=1):
            for line in (page.extract_text() or "").splitlines():
                if line.strip():
                    lines.append(line.strip())
                    line_pages.append(page_no)

        self.text = "\n".join(lines)
        # page number for every character (newline separators inherit the
        # page of the line they follow)
        self.char_pages: list[int] = []
        for line, page_no in zip(lines, line_pages):
            self.char_pages.extend([page_no] * (len(line) + 1))
        self.char_pages = self.char_pages[: len(self.text)]

    def pages_for(self, chunk_text: str) -> dict:
        start = self.text.find(chunk_text)
        if start < 0:
            return {"pages": [], "match": "not_found"}
        second = self.text.find(chunk_text, start + 1)
        end = start + len(chunk_text)
        pages = sorted(set(self.char_pages[start:end]))
        return {
            "pages": pages,
            "match": "ambiguous" if second >= 0 else "exact",
        }


# --------------------------------------------------------------------------
# Instrumentation (monkeypatch; agents.py is untouched)
# --------------------------------------------------------------------------
class Recorder:
    def __init__(self, mapper: PageMapper | None):
        self.mapper = mapper
        self.turn: dict | None = None

    def start_turn(self) -> None:
        self.turn = {"retrieved": [], "chat_calls": []}

    def end_turn(self) -> dict:
        turn, self.turn = self.turn, None
        return turn or {}


def install_patches(recorder: Recorder) -> None:
    original_retrieve = agents.KnowledgeBase.retrieve
    original_chat = ollama.chat

    def retrieve_wrapper(self, query, top_k=5):
        chunks = original_retrieve(self, query, top_k)
        if recorder.turn is not None:
            for rank, chunk in enumerate(chunks, start=1):
                mapping = (recorder.mapper.pages_for(chunk.text)
                           if recorder.mapper else {"pages": [], "match": "n/a"})
                recorder.turn["retrieved"].append({
                    "rank": rank,
                    "source": chunk.source,
                    "true_pages": mapping["pages"],
                    "page_match": mapping["match"],
                    "preview": chunk.text[:PREVIEW_CHARS],
                    # full text kept so citation grounding can be scored
                    "text": chunk.text,
                })
        return chunks

    def chat_wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        response = original_chat(*args, **kwargs)
        elapsed = time.perf_counter() - t0
        if recorder.turn is not None:
            messages = kwargs.get("messages") or (args[1] if len(args) > 1 else [])
            recorder.turn["chat_calls"].append({
                "model": kwargs.get("model") or (args[0] if args else None),
                "num_messages": len(messages),
                "prompt_chars": sum(len(m.get("content", "")) for m in messages),
                "prompt_eval_count": _field(response, "prompt_eval_count"),
                "eval_count": _field(response, "eval_count"),
                "ollama_total_duration_s": _ns_to_s(_field(response, "total_duration")),
                "seconds": round(elapsed, 3),
            })
        return response

    agents.KnowledgeBase.retrieve = retrieve_wrapper
    # agents.py calls `ollama.chat(...)` through the module, so patching the
    # module attribute is enough.
    ollama.chat = chat_wrapper


def _field(response, name):
    value = getattr(response, name, None)
    if value is None:
        try:
            value = response[name]
        except (KeyError, TypeError):
            value = None
    return value


def _ns_to_s(ns):
    return round(ns / 1e9, 3) if isinstance(ns, (int, float)) else None


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------
def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return None


def ask(bot, question: dict, recorder: Recorder, run: int, label: str) -> dict:
    print(f"  [run {run}] {question['id']} ({label}) asking... ", end="", flush=True)
    recorder.start_turn()
    t0 = time.perf_counter()
    error = None
    answer = None
    try:
        answer = bot.ask(question["question"])
    except Exception as exc:  # keep going; record the failure
        error = f"{type(exc).__name__}: {exc}"
    seconds = round(time.perf_counter() - t0, 3)
    turn = recorder.end_turn()

    chat = turn["chat_calls"][-1] if turn.get("chat_calls") else {}
    pec = chat.get("prompt_eval_count")
    flag = isinstance(pec, int) and pec >= TRUNCATION_WARN
    pages = sorted({p for c in turn.get("retrieved", []) for p in c["true_pages"]})

    # Retrieval hit: was any gold slide inside the retrieved chunks' page ranges?
    gold_pages = set(question.get("source_pages") or [])
    hit_pages = sorted(gold_pages & set(pages))
    retrieval_hit = bool(hit_pages) if gold_pages else None  # None = n/a (out of scope)
    hit_txt = "n/a" if retrieval_hit is None else ("yes " + str(hit_pages) if hit_pages else "NO")

    status = "ERROR " + error if error else "done"
    print(f"{status} in {seconds:.1f}s | prompt_eval={pec} eval={chat.get('eval_count')} "
          f"| gold hit: {hit_txt}" + ("  !! prompt_eval>=4000" if flag else ""))

    return {
        "run": run,
        "question_id": question["id"],
        "question_type": question.get("type"),
        "session": label,
        "gold_pages": sorted(gold_pages),
        "retrieval_hit": retrieval_hit,
        "retrieval_hit_pages": hit_pages,
        "retrieved_pages": pages,
        "question": question["question"],
        "history_len_before": len(bot.history) - (0 if error else 2),
        "retrieved": turn.get("retrieved", []),
        "answer": answer,
        "error": error,
        "prompt_eval_count": pec,
        "eval_count": chat.get("eval_count"),
        "seconds": seconds,
        "chat_call": chat,
        "flag_possible_truncation": flag,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--file", type=Path, default=DEFAULT_FILE)
    parser.add_argument("--gold", type=Path, default=DEFAULT_GOLD)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--embedding-model", default="nomic-embed-text")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    if not args.file.is_file():
        print(f"Test file not found: {args.file}")
        return 2
    gold = json.loads(args.gold.read_text(encoding="utf-8"))
    questions = {q["id"]: q for q in gold["questions"]}

    print(f"File:   {args.file}")
    print(f"Models: chat={args.model} embed={args.embedding_model} top_k={args.top_k}")
    print(f"Commit: {git_commit()}")

    mapper = PageMapper(args.file)
    recorder = Recorder(mapper)
    install_patches(recorder)

    print("Indexing file (once; shared by every fresh TutorBot)...")
    knowledge = KnowledgeBase(args.embedding_model)
    knowledge.add_files([str(args.file)])
    if not knowledge.chunks:
        print("No chunks indexed; aborting.")
        return 2
    unmapped = sum(1 for c in knowledge.chunks if not mapper.pages_for(c.text)["pages"])
    print(f"{len(knowledge.chunks)} chunks, {mapper.num_pages} pages, "
          f"{unmapped} chunk(s) not mappable to a page")

    results = []
    started = time.perf_counter()
    for run in range(1, args.runs + 1):
        print(f"\n=== Run {run}/{args.runs} ===")
        sessions: dict[str, object] = {}
        for qid, q in questions.items():
            follows = q.get("follows")
            if follows:
                bot = sessions[follows]  # same TutorBot, history intact
                label = f"continues {follows}"
            else:
                bot, _quiz = build_bots(knowledge, args.model, args.top_k)
                label = "fresh"
            sessions[qid] = bot
            results.append(ask(bot, q, recorder, run, label))

    out = {
        "meta": {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "commit": git_commit(),
            "chat_model": args.model,
            "embedding_model": args.embedding_model,
            "top_k": args.top_k,
            "runs": args.runs,
            "file": str(args.file.resolve()),
            "file_pages": mapper.num_pages,
            "num_chunks": len(knowledge.chunks),
            "unmapped_chunks": unmapped,
            "gold": str(args.gold.resolve()),
            "truncation_warn_threshold": TRUNCATION_WARN,
            "total_seconds": round(time.perf_counter() - started, 1),
        },
        "results": results,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"tutor_eval_{datetime.now():%Y%m%d_%H%M%S}.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    flagged = [f"run{r['run']}/{r['question_id']}" for r in results if r["flag_possible_truncation"]]
    print(f"\nSaved {out_path}")
    print(f"Flagged (prompt_eval_count >= {TRUNCATION_WARN}): {flagged or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
