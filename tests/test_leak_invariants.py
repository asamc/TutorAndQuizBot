"""Static guards on the leak rules in docs/ingest-pipeline.md 4.5.

Two rules there are followed by convention and, until this file, enforced by
nothing. Both protect the same R2 channel -- document text reaching a log or a
message -- and both are counterintuitive enough that a careful reviewer would
approve a violation:

1. **No `raise` inside an `except` handler.** `raise X from None` reads as "do
   not chain this" and does not do that: it suppresses the *display* of the
   original while leaving it on `__context__`, where an error reporter walking
   the chain prints whatever the library quoted. Measured in 4.5: pypdf's error
   carries ~100 bytes of the file it choked on. The raise has to happen after
   the handler exits, which is what parser._guarded and embedder._guarded are
   for -- and what the five hand-rolled copies of that pattern in embedder.py
   are also doing.

2. **A broad `except` must never render the exception it caught.** If you catch
   `Exception` you do not know what you caught, so `f"...{exc}"` or `str(exc)`
   may be interpolating a library message built from document bytes. Catching a
   specific project type is different: ParseFailed and EmbeddingFailed carry
   fixed strings we wrote, so printing those is safe and is not flagged.

Why static analysis rather than more unit tests: these are properties of every
raise and every handler, present and future. A unit test covers the sites
someone thought to cover -- embedder.py has five deferred-raise handlers and
tests pin two of them -- while this covers all of them, plus whatever index.py
adds next. It needs no Ollama, no data/, and runs in milliseconds.

The checks are lexical and therefore blunt. A deliberate, safe chain would also
trip rule 1. That is intentional: the exemption list below is the only way out,
so an exemption is a visible edit to a security test rather than an inline
comment that rides along in an unrelated diff.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]

# Shipped packages. tests/ is excluded deliberately: test code constructs leaky
# exceptions on purpose, which is how rule 1 and rule 2 are proven to fire.
SOURCE_PACKAGES = (
    "ingest",
    "security",
    "llm",
    "internet",
    "ui",
    "models",
    "agents",
)

BROAD_CATCHES = {"Exception", "BaseException"}

# Files allowed to violate these rules, with the reason. Strictly temporary:
# every entry is a known leak, not an accepted one.
EXEMPT = {
    "agents/agents.py": (
        "Standalone prototype that predates the pipeline and bypasses it "
        "entirely -- its own reader, chunker and in-memory search, no "
        "validator, screener or manifest. It breaks both rules (raise inside "
        "except at the import guards, and five `print(f\"[error] {exc}\")` "
        "sites under `except Exception`). Not fixed here because the file is "
        "being rewired onto ingest/ rather than patched; this exemption goes "
        "away with that work, and until then the leak is recorded rather than "
        "claimed as absent."
    ),
}


def source_files() -> list[pathlib.Path]:
    found: list[pathlib.Path] = []
    for package in SOURCE_PACKAGES:
        directory = ROOT / package
        if directory.is_dir():
            found.extend(sorted(directory.rglob("*.py")))
    return [p for p in found if "__pycache__" not in p.parts]


def relative(path: pathlib.Path) -> str:
    return path.relative_to(ROOT).as_posix()


def caught_types(handler: ast.ExceptHandler) -> set[str]:
    """The exception names this handler catches. A bare `except:` is broadest."""
    if handler.type is None:
        return {"BaseException"}
    node = handler.type
    candidates = node.elts if isinstance(node, ast.Tuple) else [node]
    names = set()
    for candidate in candidates:
        if isinstance(candidate, ast.Name):
            names.add(candidate.id)
        elif isinstance(candidate, ast.Attribute):
            names.add(candidate.attr)
    return names


def raises_inside_handlers(tree: ast.AST) -> list[int]:
    """Line numbers of `raise` statements lexically inside an except handler."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler):
            for inner in ast.walk(node):
                if isinstance(inner, ast.Raise):
                    found.append(inner.lineno)
    return sorted(set(found))


def broad_catches_rendering_exception(tree: ast.AST) -> list[int]:
    """Line numbers where a broad handler turns its exception into text.

    Only the exception rendered *as a whole* counts. Reading one attribute --
    `exc.filename`, `exc.status_code` -- is a deliberate narrow access rather
    than a library message, and embedder.OllamaBackend.embed depends on it to
    classify a status code without ever printing one.
    """
    found = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler) or not node.name:
            continue
        if not (caught_types(node) & BROAD_CATCHES):
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.FormattedValue):
                value = inner.value
                if isinstance(value, ast.Name) and value.id == node.name:
                    found.append(inner.lineno)
            elif isinstance(inner, ast.Call):
                function = inner.func
                named = isinstance(function, ast.Name) and function.id in {
                    "str",
                    "repr",
                }
                if named and any(
                    isinstance(arg, ast.Name) and arg.id == node.name
                    for arg in inner.args
                ):
                    found.append(inner.lineno)
    return sorted(set(found))


CHECKED = [path for path in source_files() if relative(path) not in EXEMPT]


@pytest.mark.parametrize("path", CHECKED, ids=relative)
def test_no_raise_inside_an_except_handler(path):
    """Rule 1. The raise must happen after the handler exits.

    `raise X from None` is not sufficient -- it clears __cause__ and leaves
    __context__. See parser._guarded for the shape that works.
    """
    offenders = raises_inside_handlers(ast.parse(path.read_text()))
    assert not offenders, (
        f"{relative(path)}: raise inside an except handler at "
        f"line(s) {offenders}. The caught exception stays on __context__ even "
        "with `from None`, where a traceback-logging handler prints the bytes "
        "it quoted. Set a flag in the handler and raise after it, as "
        "parser._guarded and embedder._guarded do."
    )


@pytest.mark.parametrize("path", CHECKED, ids=relative)
def test_a_broad_catch_never_renders_the_caught_exception(path):
    """Rule 2. If you caught everything, you do not know what you are printing."""
    offenders = broad_catches_rendering_exception(ast.parse(path.read_text()))
    assert not offenders, (
        f"{relative(path)}: a handler catching Exception renders that "
        f"exception as text at line(s) {offenders}. Its message may be built "
        "from document bytes (ingest-pipeline 4.5). Catch the specific type "
        "whose message you wrote, or report a fixed string instead."
    )


# ---------------------------------------------------------------------------
# Guard the guards. A detector that silently matches nothing passes everything.
# ---------------------------------------------------------------------------


def test_rule_one_detects_each_way_of_writing_the_violation():
    sources = {
        "plain": "try:\n    f()\nexcept ValueError:\n    raise Boom('x')\n",
        "from None": "try:\n    f()\nexcept ValueError:\n    raise Boom('x') from None\n",
        "from exc": "try:\n    f()\nexcept ValueError as e:\n    raise Boom('x') from e\n",
        "bare except": "try:\n    f()\nexcept:\n    raise Boom('x')\n",
        "nested in if": "try:\n    f()\nexcept ValueError:\n    if g():\n        raise Boom('x')\n",
    }
    for label, source in sources.items():
        assert raises_inside_handlers(ast.parse(source)), label


def test_rule_one_does_not_flag_the_correct_shape():
    """The parser's actual pattern must pass, or the rule is unusable."""
    source = (
        "failed = False\n"
        "try:\n"
        "    return call()\n"
        "except Exception:\n"
        "    failed = True\n"
        "if failed:\n"
        "    raise Boom('fixed message')\n"
    )
    assert raises_inside_handlers(ast.parse(source)) == []


def test_rule_two_detects_a_broad_catch_printing_its_exception():
    for source in (
        "try:\n    f()\nexcept Exception as e:\n    print(f'oops {e}')\n",
        "try:\n    f()\nexcept Exception as e:\n    log(str(e))\n",
        "try:\n    f()\nexcept (Exception,) as e:\n    print(f'{e}')\n",
    ):
        assert broad_catches_rendering_exception(ast.parse(source)), source


def test_rule_two_allows_a_narrow_catch_and_an_attribute_read():
    """Both shapes are load-bearing elsewhere: ingest/cli.py prints its own
    ParseFailed, and embedder reads status_code off a response error."""
    narrow = "try:\n    f()\nexcept ParseFailed as e:\n    print(f'{e}')\n"
    attribute = "try:\n    f()\nexcept Exception as e:\n    print(e.status_code)\n"
    assert broad_catches_rendering_exception(ast.parse(narrow)) == []
    assert broad_catches_rendering_exception(ast.parse(attribute)) == []


def test_the_checks_are_actually_looking_at_something():
    """If the package globs break, both rules above pass by scanning nothing."""
    names = {relative(path) for path in CHECKED}
    assert "ingest/parser.py" in names
    assert "ingest/embedder.py" in names
    assert len(names) >= 8, names


def test_every_exemption_still_names_a_file_that_exists():
    """An exemption outliving its file is an exemption nobody will revisit."""
    for name, reason in EXEMPT.items():
        assert (ROOT / name).is_file(), f"stale exemption: {name}"
        assert len(reason) > 40, f"exemption needs a real reason: {name}"


def test_exempt_files_are_genuinely_still_violating():
    """The other half: once a file is clean, its exemption has to go, or the
    list rots into a permanent blind spot."""
    for name in EXEMPT:
        tree = ast.parse((ROOT / name).read_text())
        still_bad = raises_inside_handlers(tree) or broad_catches_rendering_exception(
            tree
        )
        assert still_bad, (
            f"{name} no longer violates these rules -- remove it from EXEMPT "
            "so it is checked from now on."
        )
