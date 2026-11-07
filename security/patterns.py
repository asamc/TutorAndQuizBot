r"""Injection screening patterns.

This list lives in security/ rather than beside the screener because TUT-18
requires it here, so it can be extended during R1 testing without touching
ingest code.

Seeded from the three patterns named in docs/threat-model.md section 3.3 and
extended during R1 adversarial testing. Record each addition against the
attempt in the log (tests/adversarial-log-template.md).

Two things to keep in mind before extending it:

1. This is a speed bump, not a guarantee. The real defence against indirect
   injection is structural: the labeled context blocks tell the
   model retrieved text is reference material, not instructions.

   Measured, against the 14-file corpus in data/test_files: widening these
   patterns took detection from 1/8 to 4/8 injected files. The other four carry
   no jailbreak vocabulary at all -- an answer-key request, an API-key
   exfiltration phrased as verification, an instruction buried mid-paragraph,
   and a forged instructor exemption. No pattern list of any length reaches
   those. Do not read a low number here as a tuning problem.

2. This course's material is *about* security. A lecture on prompt injection
   legitimately contains "ignore previous instructions"; a slide on privilege
   escalation contains "system:". Expect frequent false positives on correct
   content, and do not weaken these patterns to suppress them. See
   docs/ingest-pipeline.md section 4.4.

   As of this writing the 27 real decks contain the word "ignore" zero times,
   so the current false-positive count of zero is thin evidence rather than a
   clean bill of health. Add a prompt-injection lecture and expect that to
   change.

Authoring constraints -- screener.py calls re.finditer(pattern, text, re.I):

- IGNORECASE is applied to every pattern. An acronym that doubles as a word
  needs (?-i:...) or it fires on prose; `DAN` matching the name "Dan" is how
  that was found.
- MULTILINE is NOT applied. A pattern anchoring with ^ or $ must carry its own
  (?m) or it silently anchors to the whole page and matches almost nothing.
  `assistant-prefix` shipped broken for exactly this reason.
- Patterns match the extracted text verbatim: no Unicode normalization, no
  zero-width stripping, no base64 decoding happens first. Surface-form evasion
  (letter spacing, U+200D inside a keyword, fullwidth characters) therefore
  defeats any regex here, and belongs in a normalization pass in screener.py
  rather than in a cleverer pattern.
- A bounded filler gap -- (?:\w+\s+){0,3}? -- is preferred over a single
  optional qualifier. "override your previous instructions" slipped past
  override-instructions because only one qualifier slot existed.
"""

# (name, regex) pairs. The name is what gets reported in a Flag and written to
# logs; it is safe to display. The regex is matched case-insensitively.
INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    # Negating prior instructions. One verb alternation per object family; the
    # bounded gap absorbs qualifiers ("all of the", "your previous").
    ("ignore-previous", r"ignore\s+(?:\w+\s+){0,3}?(?:previous|prior|above|earlier|preceding|foregoing)"),
    ("disregard-instructions", r"disregard\s+(?:\w+\s+){0,3}?(?:instructions|rules|guidelines|directions|material|content|previous|prior|above|everything)"),
    ("forget-everything", r"forget\s+(?:everything|all|what|the|your)\b(?:\s+\w+){0,2}?\s*(?:you\s+)?(?:were\s+told|above|before|previous(?:ly)?|prior|instructions|rules)"),
    ("override-instructions", r"override\s+(?:\w+\s+){0,3}?(?:instructions|rules|settings|programming|policy|guidelines)"),
    ("new-instructions", r"(?:new|updated|revised|real|actual)\s+instructions\b"),
    ("instead-do", r"instead\s*,?\s+(?:you\s+(?:should|will|must)|do\s+the\s+following)"),

    # Persona and role reassignment.
    ("you-must-now", r"you\s+must\s+now"),
    ("from-now-on", r"from\s+now\s+on\b"),
    ("you-are-now", r"you\s+are\s+now\b"),
    ("act-as", r"\b(?:act|behave|respond)\s+as\s+(?:if\s+you\s+(?:are|were)|an?\b|[\"'A-Z])"),
    ("pretend-to-be", r"pretend\s+(?:to\s+be|you\s+are|that\s+you)"),

    # Forged turn boundaries and chat scaffolding.
    ("system-prefix", r"system\s*:"),
    ("assistant-prefix", r"(?m)^\s*assistant\s*:"),
    ("role-banner", r"(?m)^\s*[#*=_|-]{2,}\s*(?:system|user|assistant|human|ai)\b[\s#*=_|-]*$"),
    ("chat-template-tokens", r"<\|(?:im_start|im_end|system|user|assistant|endoftext|begin_of_text|start_header_id|end_header_id|eot_id)\|>"),
    ("inst-tags", r"\[/?(?:INST|SYS)\]|<</?SYS>>"),
    ("fake-xml-system", r"</?\s*(?:system|instructions?|admin)(?:[_-]\w+)?\s*/?>"),

    # Jailbreak idiom.
    ("end-of-prompt", r"(?:end|close)\s+of\s+(?:system\s+)?(?:prompt|instructions|document|context|input|transcript)"),
    ("developer-mode", r"(?:developer|dev|debug|admin|god|sudo|jailbroken)[\s_-]*mode"),
    # DAN is case-sensitive: under the screener's global IGNORECASE a bare
    # \bDAN\b matches the name "Dan" in a worked example.
    ("dan-jailbreak", r"(?-i:\bDAN\b)|do\s+anything\s+now"),
    ("no-restrictions", r"(?:without|no|free\s+of)\s+(?:any\s+)?(?:restrictions|limitations|filters|guidelines|rules|policy|policies|safeguards|constraints|refusal)|\b(?:unrestricted|unfiltered|uncensored)\b"),
    ("jailbreak-word", r"\bjail[\s_-]*break"),

    # Prompt extraction. The bounded gaps matter: "print the full text of your
    # system prompt" puts three words between the verb and its object.
    ("reveal-prompt", r"(?:reveal|show|print|repeat|output|display|dump|leak|give|tell)\s+(?:\w+\s+){0,4}?(?:your|the)\s+(?:\w+\s+){0,3}?(?:system\s+)?(?:prompt|instructions)"),
    ("what-are-instructions", r"what\s+(?:are|were|is|was)\s+your\s+(?:original\s+|initial\s+|system\s+)?(?:instructions|rules|prompt)"),
    ("repeat-above", r"repeat\s+(?:\w+\s+){0,3}?(?:above|before|verbatim)"),

    # Exfiltration. Low value on its own -- the outbound host allowlist (TUT-38)
    # is the real control, and agents never call the network (TUT-36) -- but
    # cheap, and it catches a document that states its intent plainly.
    ("send-to-url", r"(?:send|post|upload|forward)\s+(?:this|it|the\s+\w+|all\s+\w+)\s+to\s+https?://"),
    ("markdown-image-exfil", r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*="),
)
