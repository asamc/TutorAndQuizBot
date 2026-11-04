"""Screen documents for instruction-like phrasing. TUT-18.

Scans PageRecords rather than flattened text, so every Flag carries a real page
number and the UI can say *where* something was found.

Two properties this module must hold:

1. It **detects, it does not sanitise.** Document text is never modified.
   Stripping a matched phrase and indexing the result corrupts course material
   and creates false confidence at the same time. This flags; a person decides.

2. It **fails closed.** If screening raises, the caller treats the document as
   FLAGGED_INJECTION, not as clean. A gate that fails open is worse than no
   gate, because it produces the belief that screening ran when it did not.

Indirect injection is the more serious R1 path: the course material is the asset
being protected, and anyone who can upload a file can attempt it. See
docs/threat-model.md section 3.1.
"""

from ingest.model import Flag, PageRecord

# Maximum characters of document text carried in Flag.excerpt.
#
# Open decision: TUT-18 requires the user be told what was flagged and where,
# while TUT-54 and threat-model 4.3 say the UI never displays raw document
# content -- and an excerpt *is* document content. Bounded at 80 for now and
# kept as a module constant so changing it later is one line. Flag.pattern and
# Flag.page are safe to display regardless.
EXCERPT_MAX_CHARS = 80


def screen(pages: list[PageRecord]) -> list[Flag]:
    """Return one Flag per pattern match. An empty list means clean.

    Patterns come from security.patterns.INJECTION_PATTERNS and are matched
    case-insensitively. Each Flag records the pattern *name*, not the matched
    text, so a Flag is safe to log once the excerpt is dropped.

    Expect frequent false positives: this course's material is about security,
    so a lecture on prompt injection legitimately contains "ignore previous
    instructions". Do not weaken the patterns to suppress that -- the review
    flow is meant to absorb it.

    Never log a Flag wholesale; the excerpt is document text.
    """
    raise NotImplementedError
