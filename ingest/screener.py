r"""Screen documents for instruction-like phrasing. TUT-18.

Scans PageRecords rather than flattened text, so every Flag carries a real page
number and the UI can say *where* something was found.

Three properties this module must hold:

1. It **detects, it does not sanitise.** Document text is never modified.
   Stripping a matched phrase and indexing the result corrupts course material
   and creates false confidence at the same time. This flags; a person decides.

   Normalization below does not break this. Each normalized form is a throwaway
   copy built for matching; PageRecord.text is never touched, and nothing
   downstream -- chunking, embedding, citation -- ever sees a normalized form.

2. It **fails closed.** If screening raises, the caller treats the document as
   FLAGGED_INJECTION, not as clean. A gate that fails open is worse than no
   gate, because it produces the belief that screening ran when it did not.

3. It screens **several views of the same page.** A regex matches surface forms
   of what is really a semantic property, so any transformation that preserves
   meaning for a reader while changing the bytes defeats every pattern at once.
   Measured against data/test_files: letter spacing ("I g n o r e  a l l"), a
   U+200D inside a keyword, fullwidth characters and a base64 payload each
   evaded all 27 patterns in their raw form. Widening the patterns cannot fix
   that -- it is the wrong layer -- so the text is also screened normalized and
   base64-decoded.

Indirect injection is the more serious R1 path: the course material is the asset
being protected, and anyone who can upload a file can attempt it. See
docs/threat-model.md section 3.1.
"""

import base64
import re
import unicodedata

from security.patterns import INJECTION_PATTERNS

from ingest.model import Flag, PageRecord

# Maximum characters of document text carried in Flag.excerpt.
#
# Open decision: TUT-18 requires the user be told what was flagged and where,
# while TUT-54 and threat-model 4.3 say the UI never displays raw document
# content -- and an excerpt *is* document content. Bounded at 80 for now and
# kept as a module constant so changing it later is one line. Flag.pattern and
# Flag.page are safe to display regardless.
EXCERPT_MAX_CHARS = 80

# Reported when a rule matched only after normalization or decoding -- i.e. the
# phrasing was hidden. That is a stronger signal than the rule alone: course
# material has no reason to letter-space an instruction or base64-encode it, so
# unlike the patterns themselves these two carry no false-positive budget.
# Both are rule names, safe to display and log, like any other Flag.pattern.
NORMALIZED_EVASION = "normalized-evasion"
ENCODED_PAYLOAD = "encoded-payload"

# Zero-width and joiner characters, stripped before matching. Placed inside a
# keyword they defeat every literal in the pattern list while leaving the text
# visually identical.
_ZERO_WIDTH = dict.fromkeys(
    map(ord, "​‌‍⁠﻿­"),
    None,
)

# Letter-spacing is only recovered on lines that are mostly single *letters*.
#
# Two earlier attempts were wrong in instructive ways. A token-level rule glued
# "a resistor" into "aresistor" on 139 of 275 real slide pages, because English
# prose is full of one-letter words. Gating on single *characters* then caught
# the maths instead -- "a + b = c + d (mod m)" is 80% single-character tokens --
# and fired on 180 of 275 pages. Counting only single-letter tokens separates
# the two: letter-spacing is all letters, while notation carries operators,
# digits and parentheses. The absolute floor matters as much as the ratio, since
# a short formula can be all letters.
_DESPACE_MIN_TOKENS = 6
_DESPACE_MIN_LETTER_TOKENS = 6
_DESPACE_MIN_LETTER_RATIO = 0.6

# A standalone operator or digit means notation, not letter-spacing: spaced-out
# prose is letters and sentence punctuation only. Vetoing on these separates
# "x y z = p q r + s" from "I g n o r e  a l l" where counting cannot.
_NOTATION_TOKENS = frozenset(
    "=+-*/^<>≡≤≥±×÷∈∧∨⊕|~"
)

# Base64 runs to decode. The floor keeps ordinary words and short hex strings
# out; the caps keep a page of padding from turning into real work.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{24,}={0,2}")
_B64_MAX_RUNS = 20
_B64_MAX_DECODED_CHARS = 20_000


def screen(pages: list[PageRecord]) -> list[Flag]:
    """Return one Flag per (page, rule). An empty list means clean.

    Patterns come from security.patterns.INJECTION_PATTERNS and are matched
    case-insensitively. Each Flag records the rule *name*, not the document
    text, so a Flag is safe to log once the excerpt is dropped.

    One Flag per rule per page, not per match: a rule that fires in the raw text
    and again in a normalized view of the same page is one finding, and the raw
    view wins so the excerpt stays faithful to the document. A rule that fires
    *only* in a derived view also yields a NORMALIZED_EVASION or ENCODED_PAYLOAD
    flag, because the hiding is itself the finding.

    Expect frequent false positives: this course's material is about security,
    so a lecture on prompt injection legitimately contains "ignore previous
    instructions". Do not weaken the patterns to suppress that -- the review
    flow is meant to absorb it.

    Never log a Flag wholesale; the excerpt is document text.
    """
    flags: list[Flag] = []
    for page in pages:
        seen: set[str] = set()
        hidden: set[str] = set()
        for view, text in _views(page.text):
            for name, pattern in INJECTION_PATTERNS:
                if name in seen:
                    continue
                match = re.search(pattern, text, re.IGNORECASE)
                if match is None:
                    continue
                seen.add(name)
                if view != "raw":
                    hidden.add(view)
                flags.append(
                    Flag(
                        page=page.page_number,
                        pattern=name,
                        excerpt=_excerpt(text, match.start()),
                    )
                )
        for view in sorted(hidden):
            flags.append(
                Flag(
                    page=page.page_number,
                    pattern=(
                        NORMALIZED_EVASION if view == "normalized" else ENCODED_PAYLOAD
                    ),
                    excerpt=None,
                )
            )
    return flags


def _excerpt(text: str, at: int) -> str:
    """A bounded window of `text` around `at`, whitespace collapsed."""
    start = max(at - EXCERPT_MAX_CHARS // 2, 0)
    return " ".join(text[start : start + EXCERPT_MAX_CHARS].split())


def _views(text: str) -> list[tuple[str, str]]:
    """The forms of one page to screen, raw first so it wins on a tie.

    A derived form is dropped when it comes back identical to the raw text,
    which is the common case and keeps clean documents at one pass.
    """
    views = [("raw", text)]
    normalized = _normalize(text)
    if normalized != text:
        views.append(("normalized", normalized))
    decoded = _decode_base64(text)
    if decoded:
        views.append(("decoded", decoded))
    return views


def _normalize(text: str) -> str:
    """A matching-only copy with meaning-preserving obfuscation undone.

    NFKC folds fullwidth and other compatibility forms onto their ASCII
    equivalents; zero-width characters are dropped; letter-spaced lines are
    re-joined. Offsets do not survive this, which is why the excerpt is taken
    from the view that matched rather than from the original.
    """
    folded = unicodedata.normalize("NFKC", text).translate(_ZERO_WIDTH)
    # Rewrite each line in place rather than splitlines()/join: rejoining drops
    # the trailing newline, which made every ordinary page look transformed and
    # bought a second, pointless screening pass over it.
    return re.sub(r"[^\n]+", lambda m: _despace_line(m.group(0)), folded)


def _despace_line(line: str) -> str:
    """Re-join "I g n o r e   a l l" into "Ignore  all" on letter-spaced lines.

    Word gaps are assumed to be two or more spaces, which is what makes the
    text readable in the first place. A payload spaced with a single space
    throughout collapses into one run and is not recovered -- restoring word
    boundaries there needs a dictionary, and a wrong guess is worse than a miss.
    """
    tokens = line.split()
    if len(tokens) < _DESPACE_MIN_TOKENS:
        return line
    if any(token in _NOTATION_TOKENS or token.isdigit() for token in tokens):
        return line
    letters = sum(1 for token in tokens if len(token) == 1 and token.isalpha())
    if letters < _DESPACE_MIN_LETTER_TOKENS:
        return line
    if letters / len(tokens) < _DESPACE_MIN_LETTER_RATIO:
        return line
    groups = re.split(r"[ \t]{2,}", line.strip())
    return "  ".join(re.sub(r"[ \t]+", "", group) for group in groups)


def _decode_base64(text: str) -> str:
    """Plausible base64 payloads in `text`, decoded and concatenated.

    Only decodings that look like text are kept. A crypto deck is full of hex
    and base64 blobs that decode to bytes, and feeding that noise to the
    pattern list would trade one problem for another.
    """
    out: list[str] = []
    size = 0
    for run in _B64_RUN.findall(text)[:_B64_MAX_RUNS]:
        try:
            raw = base64.b64decode(run + "=" * (-len(run) % 4), validate=False)
        except Exception:
            continue
        try:
            candidate = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if not _looks_like_text(candidate):
            continue
        out.append(candidate)
        size += len(candidate)
        if size >= _B64_MAX_DECODED_CHARS:
            break
    return "\n".join(out)


def _looks_like_text(candidate: str) -> bool:
    """True for a decoding worth screening: prose, not accidental bytes."""
    if len(candidate) < 12 or " " not in candidate:
        return False
    printable = sum(1 for char in candidate if char.isprintable() or char in "\n\t")
    return printable / len(candidate) > 0.9
