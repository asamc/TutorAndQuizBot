r"""Injection screening patterns, as (name, regex) pairs.

Lives in security/ rather than beside the screener so it can be extended during
adversarial testing without touching ingest code. Record each addition against
the attempt in tests/adversarial-log-template.md.

A speed bump, not a guarantee. Widening these patterns took detection from 1/8
to 4/8 injected files in data/test_files; the other four carry no jailbreak
vocabulary at all (an answer-key request, an exfiltration phrased as
verification, an instruction buried mid-paragraph, a forged exemption). No
pattern list reaches those. The real defence is structural: labeled context
blocks telling the model retrieved text is reference material.

This course's material is *about* security, so expect false positives on correct
content and do not weaken patterns to suppress them. The 27 real decks contain
"ignore" zero times, so today's count of zero is thin evidence, not a clean bill
of health. See docs/ingest-pipeline.md 4.4.

Authoring constraints -- screener.py calls re.finditer(pattern, text, re.I):

- IGNORECASE applies to every pattern. An acronym that doubles as a word needs
  (?-i:...), or it fires on prose; `DAN` matched the name "Dan".
- MULTILINE does NOT apply. A pattern using ^ or $ must carry its own (?m) or it
  anchors to the whole page; `assistant-prefix` shipped broken this way.
- Text is matched verbatim: no normalization, zero-width stripping or base64
  decoding happens first. Surface-form evasion defeats any regex here and
  belongs in screener.py's normalization pass instead.
- Prefer a bounded filler gap, (?:\w+\s+){0,3}?, over one optional qualifier.
  "override your previous instructions" slipped past with only one slot.
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

    # Exfiltration. Low value alone -- the outbound host allowlist is the real
    # control, and agents never call the network -- but cheap, and it catches a
    # document that states its intent plainly.
    ("send-to-url", r"(?:send|post|upload|forward)\s+(?:this|it|the\s+\w+|all\s+\w+)\s+to\s+https?://"),
    ("markdown-image-exfil", r"!\[[^\]]*\]\(https?://[^)]*\?[^)]*="),
)
