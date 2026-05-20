"""Tests for backend.scorer.

These exercise the real embedding model — they're slow on first run
(the model downloads ~80MB) but quick afterwards. We assert relative
orderings, not absolute scores.
"""
from __future__ import annotations

import pytest

from backend.scorer import ScoredResume, rank


JD_PYTHON_BACKEND = (
    "We are looking for a Python backend engineer with strong experience "
    "in FastAPI, Docker, PostgreSQL, and AWS. Familiarity with REST APIs "
    "and CI/CD pipelines is required."
)

RESUME_STRONG = (
    "Senior Python backend engineer. Built REST APIs in FastAPI on AWS "
    "ECS. PostgreSQL, Docker, GitHub Actions CI/CD. 6 years of experience."
)
RESUME_WEAK = (
    "Frontend developer specializing in React and TypeScript. Built "
    "design systems and accessibility-first components. No backend work."
)
RESUME_MEDIUM = (
    "Java backend engineer with Spring Boot, Kafka, PostgreSQL, and "
    "Docker. Some exposure to AWS. No Python experience."
)


@pytest.fixture(scope="module")
def ranked() -> list[ScoredResume]:
    return rank(
        JD_PYTHON_BACKEND,
        [
            ("Strong Match", RESUME_STRONG),
            ("Weak Match", RESUME_WEAK),
            ("Medium Match", RESUME_MEDIUM),
        ],
    )


def test_rank_returns_all_candidates(ranked: list[ScoredResume]) -> None:
    assert len(ranked) == 3


def test_rank_orders_strong_above_weak(ranked: list[ScoredResume]) -> None:
    names_in_order = [r.candidate_name for r in ranked]
    strong_idx = names_in_order.index("Strong Match")
    weak_idx = names_in_order.index("Weak Match")
    assert strong_idx < weak_idx, f"got order: {names_in_order}"


def test_strong_match_scores_above_medium(ranked: list[ScoredResume]) -> None:
    by_name = {r.candidate_name: r for r in ranked}
    assert by_name["Strong Match"].score > by_name["Medium Match"].score


def test_scores_are_in_valid_range(ranked: list[ScoredResume]) -> None:
    for r in ranked:
        assert 0.0 <= r.score <= 100.0
        assert -1.0 <= r.raw_similarity <= 1.0


def test_strong_match_surfaces_relevant_skills(
    ranked: list[ScoredResume],
) -> None:
    strong = next(r for r in ranked if r.candidate_name == "Strong Match")
    surfaced = set(strong.top_skills_matched)
    # The strong resume contains Python, FastAPI, Docker, AWS, PostgreSQL,
    # REST APIs, CI/CD — top_skills_matched is capped at 3, so we expect
    # exactly 3 entries and all of them should come from the JD vocab.
    assert len(strong.top_skills_matched) == 3, surfaced
    expected_pool = {"python", "fastapi", "docker", "aws", "postgresql",
                     "rest api", "ci/cd"}
    assert surfaced.issubset(expected_pool), (
        f"unexpected items in matched: {surfaced - expected_pool}"
    )


def test_weak_match_has_no_overlap(ranked: list[ScoredResume]) -> None:
    weak = next(r for r in ranked if r.candidate_name == "Weak Match")
    # Frontend resume should match none of the JD's backend skills.
    assert weak.top_skills_matched == []
    # And the gap list should surface JD skills it lacks.
    assert len(weak.top_gaps) == 3


def test_empty_input_returns_empty_list() -> None:
    assert rank(JD_PYTHON_BACKEND, []) == []
