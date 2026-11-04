"""Accept or reject an uploaded file. TUT-14.

First gate in the pipeline. Runs before the parser, because pypdf is a parser
consuming hostile input and should not see unchecked bytes. Runs in memory,
before anything is written to data/, so an unvalidated file never lands inside
the protected directory.

Fails closed: if the type cannot be determined, reject.
"""

from ingest.model import Outcome

# Enforced here even though Streamlit also caps uploads. The UI is not a trust
# boundary, and CLI ingestion bypasses it entirely.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB, matching docs/ingest-interface.md

# TUT-14 lists PDF, DOCX and TXT. DOCX is deliberately absent: python-docx is
# not installed and no course material is DOCX. Accepting a file we cannot parse
# would surface later as PARSE_FAILED and read like a corrupt document rather
# than a missing feature. See docs/ingest-pipeline.md section 8.
ACCEPTED_SUFFIXES = (".pdf", ".txt")

# Content signatures, checked against the actual bytes rather than trusting the
# extension. A .txt file has no signature, so it is validated by decoding.
PDF_MAGIC = b"%PDF-"


def validate(data: bytes, filename: str) -> Outcome | None:
    """Return the rejection Outcome, or None if the file is acceptable.

    Checks, in order:
      - size against MAX_UPLOAD_BYTES              -> REJECTED_SIZE
      - suffix against ACCEPTED_SUFFIXES           -> REJECTED_TYPE
      - content signature against the suffix       -> REJECTED_CONTENT_MISMATCH

    The content check is the one that matters: a .pdf whose bytes are not a PDF
    must be rejected, not handed to pypdf.

    Never log `data` or any slice of it. Rejections are logged as metadata only:
    document id, declared suffix, reason. See docs/ingest-pipeline.md 4.5.
    """
    raise NotImplementedError
