"""Run the ingest stages from the command line.

ingest_document() in the interface doc is still unwired: no Outcome can honestly
report INDEXED while embedding and indexing are blocked on TUT-58. So this
drives the stages directly and says where it stopped.

    uv run python -m ingest.cli data/documents/Lecture_1_slides.pdf
    uv run python -m ingest.cli --chunks notes.txt
    uv run python -m ingest.cli --list
    uv run python -m ingest.cli --verify
    uv run python -m ingest.cli --delete 83e080edc119614d a641aaf772d53290
    uv run python -m ingest.cli --delete --all

Never run this under sudo. Nothing here needs root -- it writes only to
data/documents and ~/.tutorandquizbot, both owned by the invoking user. Running
it elevated once leaves root-owned artifacts that a later --delete cannot
remove, which is the only way deletion ever hits a permission error.
"""

import argparse
import sys
from pathlib import Path

from ingest import chunker, manifest, screener
from ingest.model import Flag, Outcome, Stage
from ingest.parser import ParseFailed, parse
from ingest.validator import validate


def ingest(path: Path, show_chunks: bool = False) -> bool:
    """Run every implemented stage against one file. True if it got through."""
    data = path.read_bytes()
    print(path.name)

    rejection = validate(data, path.name)
    if rejection:
        print(f"  {Stage.VALIDATING.value:11} {rejection.value}")
        return False
    print(f"  {Stage.VALIDATING.value:11} ok, {len(data):,} bytes")

    # Hashed before parsing, so the digest covers the bytes as received.
    doc_id = manifest.document_id(data)
    known = manifest.lookup(doc_id)
    print(f"  {'hashing':11} {doc_id[:16]}")
    if known:
        print(
            f"  -> {Outcome.ALREADY_INDEXED.value}: {known['chunk_count']} chunks, "
            f"first ingested {known['ingested_at']}"
        )
        return True

    try:
        pages = parse(data, path.name)
    except ParseFailed as exc:
        print(f"  {Stage.PARSING.value:11} {Outcome.PARSE_FAILED.value}: {exc}")
        return False
    empty = sum(page.is_empty for page in pages)
    print(f"  {Stage.PARSING.value:11} {len(pages)} pages, {empty} empty")

    # Fails closed: a screener that raises means flagged, not clean.
    try:
        flags = screener.screen(pages)
    except Exception:
        flags = [Flag(page=None, pattern="screener-error", excerpt=None)]

    if flags:
        print(f"  {Stage.SCREENING.value:11} {len(flags)} flagged")
        for flag in flags:
            where = f"page {flag.page}" if flag.page else "document"
            print(f"              {where}: {flag.pattern}  {flag.excerpt!r}")
        print(
            f"  -> {Outcome.FLAGGED_INJECTION.value}: held for review, "
            "nothing stored or indexed"
        )
        return False
    print(f"  {Stage.SCREENING.value:11} clean")

    chunks = chunker.chunk(pages, path.name)
    print(f"  {Stage.CHUNKING.value:11} {len(chunks)} chunks")
    if show_chunks:
        for chunk in chunks:
            print(f"              {chunk.citation()}  ({len(chunk.text)} chars)")

    stored = manifest.store(data, path.name)
    manifest.record(doc_id, path.name, len(pages), len(chunks))
    print(f"  {'stored':11} {stored}")
    print("  -> chunked, not indexed: embedding and indexing needed")
    return True


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


def verify_stored() -> bool:
    """Re-hash every stored document against its manifest entry. TUT-23."""
    known = manifest.entries()
    if not known:
        print("nothing to verify")
        return True

    all_ok = True
    for doc_id, entry in sorted(known.items()):
        found = manifest.stored_path(doc_id)
        if found is None:
            print(f"  missing   {entry['document_name']} ({doc_id[:16]})")
            all_ok = False
            continue
        if manifest.verify(doc_id, found.read_bytes()):
            print(f"  ok        {entry['document_name']}")
        else:
            print(f"  MISMATCH  {entry['document_name']} ({doc_id[:16]})")
            all_ok = False
    return all_ok


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


def delete_documents(prefixes: list[str], every: bool, assume_yes: bool) -> bool:
    """Delete documents by id prefix, or every recorded one. True on success.

    Only ever touches files the manifest points at, so documents sitting in
    data/documents under their original names -- the lecture decks -- are not
    reachable from here even with --all.
    """
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
            entry_removed, removed = manifest.delete(doc_id)
        except PermissionError as exc:
            print(f"  {'denied':11} {exc.filename or name}")
            print(
                "  nothing in this CLI needs root, so these artifacts belong to\n"
                "  another user -- most likely from one elevated run. To reclaim:\n"
                '    chown -R "$(id -un)" data/documents ~/.tutorandquizbot'
            )
            return False
        print(f"  {'manifest':11} {'removed' if entry_removed else 'absent'}  {name}")
        print(f"  {'file':11} {('removed  ' + str(removed)) if removed else 'absent'}")
        deleted += 1

    print(f"  -> {deleted} document(s) deleted")
    return True


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the ingest pipeline on a file.")
    ap.add_argument("paths", nargs="*", type=Path, help="PDF or TXT files to ingest")
    ap.add_argument("--chunks", action="store_true", help="print each chunk's citation")
    ap.add_argument("--list", action="store_true", help="show the manifest")
    ap.add_argument("--verify", action="store_true", help="re-hash stored documents")
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
    if args.delete is not None:
        if args.all and args.delete:
            ap.error("--delete takes ids or --all, not both")
        if not args.all and not args.delete:
            ap.error("--delete needs at least one id, or --all")
        return 0 if delete_documents(args.delete, args.all, args.yes) else 1
    if args.all:
        ap.error("--all only makes sense with --delete")
    if not args.paths:
        ap.error("give at least one file, or --list / --verify / --delete")

    ok = True
    for path in args.paths:
        if not path.is_file():
            print(f"{path}: not a file")
            ok = False
            continue
        ok = ingest(path, args.chunks) and ok
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
