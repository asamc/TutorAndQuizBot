"""Drive the ingest pipeline from the command line.

The only front end there is, so it is what exercises the published contract. It
calls ingest_document() and renders the Progress events and the single Result,
which is what a UI would do.

    uv run python -m ingest.cli data/documents/Lecture_1_slides.pdf
    uv run python -m ingest.cli --chunks notes.txt
    uv run python -m ingest.cli --list
    uv run python -m ingest.cli --verify
    uv run python -m ingest.cli --reindex
    uv run python -m ingest.cli --delete 83e080edc119614d
    uv run python -m ingest.cli --delete --all

Ingesting needs Ollama on loopback. --list, --verify and --delete deliberately
do not: "what do I have" and "get rid of this" must not depend on a model.

Never run this under sudo. Nothing here needs root, and one elevated run leaves
root-owned artifacts that a later --delete cannot remove.
"""

import argparse
import sys
from pathlib import Path

from ingest import manifest
from ingest.embedder import Embedder
from ingest.index import INDEX_DIR, ChunkIndex, IndexFailed
from ingest.model import Outcome, Progress, Result
from ingest.pipeline import (
    Finding,
    FindingKind,
    delete_document,
    ingest_document,
    reindex_pending,
    verify,
)


SUCCESSFUL = (Outcome.INDEXED, Outcome.ALREADY_INDEXED)


def render(event: Progress | Result) -> None:
    """Print one item from the generator, as a UI would render it."""
    if isinstance(event, Progress):
        counts = ""
        if event.total is not None:
            counts = f" {event.current}/{event.total}"
        print(f"  {event.stage.value:11}{counts}")
        return

    print(f"  -> {event.outcome.value}", end="")
    if event.chunk_count is not None:
        print(f": {event.chunk_count} chunks", end="")
    print()
    if event.message:
        print(f"     {event.message}")
    for flag in event.flagged or []:
        where = f"page {flag.page}" if flag.page else "document"
        # pattern and page are safe to show; the excerpt is document text and
        # its maximum length is still an open decision -- interface doc 6.
        print(f"     {where}: {flag.pattern}  {flag.excerpt!r}")


def ingest_path(
    path: Path,
    show_chunks: bool = False,
    *,
    embedder: Embedder | None = None,
    index_dir: Path | None = None,
) -> bool:
    """Read one file, run the pipeline, print what happened.

    Reads the bytes rather than passing a path: ingest owns every write under
    data/ and takes bytes and a filename only, so the CLI stands where the UI
    will.
    """
    index_dir = index_dir if index_dir is not None else INDEX_DIR
    data = path.read_bytes()
    print(path.name)

    outcome = None
    for event in ingest_document(
        data, path.name, embedder=embedder, index_dir=index_dir
    ):
        render(event)
        if isinstance(event, Result):
            outcome = event.outcome

    if show_chunks and outcome in SUCCESSFUL:
        # Read back what was actually indexed rather than reprinting what was
        # about to be: these citations are the ones retrieval will produce.
        for chunk in indexed_chunks(manifest.document_id(data), index_dir):
            print(f"     {chunk.citation()}  ({len(chunk.text)} chars)")

    return outcome in SUCCESSFUL


def indexed_chunks(doc_id: str, index_dir: Path | None = None) -> list:
    """This document's indexed chunks, or [] when there is no index."""
    index_dir = index_dir if index_dir is not None else INDEX_DIR
    if not ChunkIndex.exists(index_dir):
        return []
    return ChunkIndex.load(index_dir).chunks_for(doc_id)


def reindex(
    *, embedder: Embedder | None = None, index_dir: Path | None = None
) -> bool:
    """Index every recorded document whose vectors are missing."""
    index_dir = index_dir if index_dir is not None else INDEX_DIR
    saw_failure = False
    saw_work = False
    for event in reindex_pending(embedder=embedder, index_dir=index_dir):
        if isinstance(event, Result):
            saw_work = True
            print(event.document_name)
            render(event)
            if event.outcome not in SUCCESSFUL:
                saw_failure = True
    if not saw_work:
        print("every recorded document already has vectors in the index")
    return not saw_failure


def show_manifest() -> None:
    known = manifest.entries()
    if not known:
        print(f"no documents recorded in {manifest.MANIFEST_PATH}")
        return
    print(f"{len(known)} document(s) in {manifest.MANIFEST_PATH}")
    for doc_id, entry in sorted(known.items(), key=lambda kv: kv[1]["ingested_at"]):
        print(
            f"  {doc_id[:16]}  {entry['document_name']:30} "
            f"{entry['page_count']:>3} pages  {entry['chunk_count']:>3} chunks  "
            f"{entry['ingested_at']}"
        )


def render_finding(finding: Finding) -> None:
    """One verify finding as a line of terminal output."""
    name = finding.document_name or (finding.doc_id or "")[:16]
    labels = {
        FindingKind.OK: f"  ok        {name}  {finding.indexed_chunks} chunks indexed",
        FindingKind.MISSING_BYTES: (
            f"  missing   {name} ({(finding.doc_id or '')[:16]}) -- no stored bytes"
        ),
        FindingKind.HASH_MISMATCH: (
            f"  MISMATCH  {name} ({(finding.doc_id or '')[:16]}) -- bytes changed "
            "since ingest"
        ),
        FindingKind.UNINDEXED: (
            f"  unindexed {name} ({(finding.doc_id or '')[:16]}) -- run --reindex"
        ),
        FindingKind.COUNT_MISMATCH: (
            f"  COUNT     {name} ({(finding.doc_id or '')[:16]}) -- manifest says "
            f"{finding.recorded_chunks} chunks, index holds {finding.indexed_chunks}"
        ),
        FindingKind.ORPHAN: (
            f"  ORPHAN    {(finding.doc_id or '')[:16]} -- {finding.indexed_chunks} "
            "vectors with no manifest entry; retrieval can still answer from them"
        ),
        FindingKind.INDEX_UNREADABLE: f"  INDEX     unreadable -- {finding.detail}",
        FindingKind.MANIFEST_UNREADABLE: f"  MANIFEST  unreadable -- {finding.detail}",
    }
    print(labels[finding.kind])


def verify_stored(*, index_dir: Path | None = None) -> bool:
    """Print every verify finding. True when nothing is wrong."""
    index_dir = index_dir if index_dir is not None else INDEX_DIR
    findings = verify(index_dir=index_dir)
    if not findings:
        print("nothing to verify")
        return True
    for finding in findings:
        render_finding(finding)
    return all(finding.ok for finding in findings)


def resolve(prefix: str, known: dict) -> str:
    """The single document id `prefix` identifies. Raises LookupError otherwise.

    --list prints doc_id[:16], so a prefix is what people have to paste. An
    ambiguous prefix names its candidates rather than picking one, and an empty
    prefix is refused instead of expanding to every document.
    """
    prefix = prefix.strip().lower()
    if not prefix:
        raise LookupError("empty document id")
    matches = [doc_id for doc_id in known if doc_id.startswith(prefix)]
    if not matches:
        raise LookupError(f"{prefix}: no such document")
    if len(matches) > 1:
        listed = ", ".join(doc_id[:16] for doc_id in sorted(matches))
        raise LookupError(f"{prefix}: ambiguous, matches {len(matches)}: {listed}")
    return matches[0]


def delete_documents(
    prefixes: list[str],
    every: bool,
    assume_yes: bool,
    *,
    index_dir: Path | None = None,
) -> bool:
    """Delete documents by id prefix, or every recorded one. True on success.

    Only ever touches files the manifest points at, so documents sitting in
    data/documents under their original names -- the lecture decks -- are not
    reachable from here even with --all.
    """
    index_dir = index_dir if index_dir is not None else INDEX_DIR
    known = manifest.entries()
    if not known:
        print(f"no documents recorded in {manifest.MANIFEST_PATH}")
        return True

    if every:
        targets = sorted(known, key=lambda doc_id: known[doc_id]["ingested_at"])
    else:
        targets = []
        for prefix in prefixes:
            try:
                targets.append(resolve(prefix, known))
            except LookupError as exc:
                print(f"  {exc}")
                return False
        # Two prefixes can name one document; delete it once.
        targets = list(dict.fromkeys(targets))

    print(f"{len(targets)} document(s) to delete:")
    for doc_id in targets:
        print(f"  {doc_id[:16]}  {known[doc_id]['document_name']}")

    if not assume_yes:
        if not sys.stdin.isatty():
            print("refusing to delete without --yes when stdin is not a terminal")
            return False
        if input("delete these? [y/N] ").strip().lower() not in ("y", "yes"):
            print("nothing deleted")
            return True

    deleted = 0
    for doc_id in targets:
        name = known[doc_id]["document_name"]
        try:
            removal = delete_document(doc_id, index_dir=index_dir)
        except PermissionError as exc:
            print(f"  {'denied':11} {exc.filename or name}")
            print(
                "  nothing in this CLI needs root, so these artifacts belong to\n"
                "  another user -- most likely from one elevated run. To reclaim:\n"
                '    chown -R "$(id -un)" data/documents ~/.tutorandquizbot'
            )
            return False
        except IndexFailed as exc:
            # Refusing beats deleting half the state: dropping the entry while
            # the vectors stay would leave an orphan nothing can name.
            print(f"  {'index':11} {exc}")
            print("  refusing to delete while the index cannot be updated to match")
            return False

        print(
            f"  {'index':11} {removal.vectors_removed} vector(s) removed  {name}"
        )
        print(
            f"  {'manifest':11} {'removed' if removal.entry_removed else 'absent'}"
            f"  {name}"
        )
        print(
            f"  {'file':11} "
            f"{('removed  ' + str(removal.file_removed)) if removal.file_removed else 'absent'}"
        )
        deleted += 1

    print(f"  -> {deleted} document(s) deleted")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ingest pipeline on a file.")
    ap.add_argument("paths", nargs="*", type=Path, help="PDF or TXT files to ingest")
    ap.add_argument("--chunks", action="store_true", help="print each chunk's citation")
    ap.add_argument("--list", action="store_true", help="show the manifest")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="re-hash stored documents and cross-check them against the index",
    )
    ap.add_argument(
        "--reindex",
        action="store_true",
        help="index any recorded document whose vectors are missing",
    )
    ap.add_argument(
        "--delete",
        nargs="*",
        metavar="ID",
        help="delete documents by id prefix, as --list prints them",
    )
    ap.add_argument(
        "--all", action="store_true", help="with --delete: every recorded document"
    )
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args(argv)

    if args.list:
        show_manifest()
        return 0
    if args.verify:
        return 0 if verify_stored() else 1
    if args.reindex:
        return 0 if reindex() else 1
    if args.delete is not None:
        if args.all and args.delete:
            ap.error("--delete takes ids or --all, not both")
        if not args.all and not args.delete:
            ap.error("--delete needs at least one id, or --all")
        return 0 if delete_documents(args.delete, args.all, args.yes) else 1
    if args.all:
        ap.error("--all only makes sense with --delete")
    if not args.paths:
        ap.error(
            "give at least one file, or --list / --verify / --reindex / --delete"
        )

    ok = True
    for path in args.paths:
        if not path.is_file():
            print(f"{path}: not a file")
            ok = False
            continue
        ok = ingest_path(path, args.chunks) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
