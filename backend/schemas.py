"""Pydantic response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    resume_id: int
    candidate_name: str
    score: float = Field(..., description="Match score, 0–100")
    raw_similarity: float = Field(..., description="Cosine similarity, 0–1")
    projected_score: float = Field(
        0.0, description="Projected score if the top gap skills were added"
    )
    score_uplift: float = Field(
        0.0, description="projected_score − score (potential % gain)"
    )
    ats_score: float = Field(
        0.0, description="ATS keyword-coverage score, 0–100"
    )
    keywords_matched: int = Field(0, description="JD keywords found in resume")
    keywords_total: int = Field(0, description="Distinct keywords in the JD")
    top_skills_matched: list[str]
    top_gaps: list[str]
    all_skills_matched: list[str] = Field(
        default_factory=list, description="Every JD skill present in the resume"
    )
    all_gaps: list[str] = Field(
        default_factory=list, description="Every JD skill missing from the resume"
    )
    explanation: str
    improvements: list[str] = Field(
        default_factory=list,
        description="Actionable resume-improvement suggestions for this role",
    )


class ScoreResponse(BaseModel):
    job_description_preview: str
    candidate_count: int
    results: list[CandidateResult]
