"""Injection screening patterns.

This list lives in security/ rather than beside the screener because TUT-18
requires it here, so it can be extended during R1 testing without touching
ingest code.

The three patterns below are the ones named in docs/threat-model.md section 3.3.
Add to this list as adversarial testing finds new phrasings, and record each
addition against the attempt in the log (tests/adversarial-log-template.md).

Two things to keep in mind before extending it:

1. This is a speed bump, not a guarantee. The real defence against indirect
   injection is structural: the labeled context blocks in TUT-26 that tell the
   model retrieved text is reference material, not instructions.

2. This course's material is *about* security. A lecture on prompt injection
   legitimately contains "ignore previous instructions"; a slide on privilege
   escalation contains "system:". Expect frequent false positives on correct
   content, and do not weaken these patterns to suppress them. See
   docs/ingest-pipeline.md section 4.4.
"""

# (name, regex) pairs. The name is what gets reported in a Flag and written to
# logs; it is safe to display. The regex is matched case-insensitively.
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ignore-previous", r"ignore\s+(?:all\s+)?previous"),
    ("you-must-now", r"you\s+must\s+now"),
    ("system-prefix", r"system\s*:"),
)
