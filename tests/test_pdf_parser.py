"""Tests for backend.pdf_parser."""
from __future__ import annotations

from pathlib import Path

import pytest

from backend.pdf_parser import candidate_name_from_filename, extract_text


SAMPLES = Path(__file__).parent / "sample_resumes"


def test_extract_text_from_real_pdf() -> None:
    pdf = SAMPLES / "Test_Candidate_One.pdf"
    assert pdf.exists(), "run scripts/generate_sample_resumes.py first"
    text = extract_text(pdf)
    assert "Test Candidate One" in text
    assert "FastAPI" in text
    assert len(text) > 100


def test_extract_text_handles_missing_file() -> None:
    # Should not raise — returns empty string for any unreadable input.
    text = extract_text(SAMPLES / "does_not_exist.pdf")
    assert text == ""


def test_extract_text_handles_garbage_bytes() -> None:
    text = extract_text(b"this is not a pdf at all")
    assert text == ""


def test_candidate_name_from_filename() -> None:
    assert (
        candidate_name_from_filename("Test_Candidate_One.pdf")
        == "Test Candidate One"
    )
    assert candidate_name_from_filename("jane-doe.pdf") == "Jane Doe"
    assert candidate_name_from_filename("resume.pdf") == "Resume"


def test_candidate_name_handles_empty() -> None:
    assert candidate_name_from_filename(".pdf") == "Unknown Candidate"


@pytest.mark.parametrize(
    "filename",
    [
        "Test_Candidate_One.pdf",
        "Test_Candidate_Two.pdf",
        "Test_Candidate_Three.pdf",
    ],
)
def test_all_sample_pdfs_parse_non_empty(filename: str) -> None:
    text = extract_text(SAMPLES / filename)
    assert len(text) > 200, f"{filename} produced only {len(text)} chars"
