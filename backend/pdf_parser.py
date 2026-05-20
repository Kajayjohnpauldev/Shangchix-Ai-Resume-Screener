"""PDF → text extraction using pypdf."""
from __future__ import annotations

import io
import re
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


_WHITESPACE_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def extract_text(pdf: str | Path | bytes) -> str:
    """Extract plain text from a PDF.

    Accepts a filesystem path or raw bytes. Returns an empty string for
    unreadable or empty PDFs rather than raising — callers treat empty
    text as a zero-content resume.
    """
    try:
        if isinstance(pdf, (bytes, bytearray)):
            reader = PdfReader(io.BytesIO(pdf))
        else:
            reader = PdfReader(str(pdf))
    except (PdfReadError, FileNotFoundError, OSError):
        return ""

    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            continue

    return _clean("\n".join(pages))


def candidate_name_from_filename(filename: str) -> str:
    """Turn 'Test_Candidate_One.pdf' into 'Test Candidate One'."""
    # Don't use Path.stem — it treats ".pdf" as a dotfile and returns
    # ".pdf" as the stem, which would render as ".Pdf".
    name = Path(filename).name
    if name.lower().endswith(".pdf"):
        name = name[:-4]
    cleaned = re.sub(r"[_\-]+", " ", name).strip(" .")
    if not cleaned:
        return "Unknown Candidate"
    return cleaned.title()
