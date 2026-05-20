"""Pydantic response models for the API."""
from __future__ import annotations

from pydantic import BaseModel, Field


class CandidateResult(BaseModel):
    resume_id: int
    candidate_name: str
    score: float = Field(..., description="Match score, 0–100")
    raw_similarity: float = Field(..., description="Cosine similarity, 0–1")
    top_skills_matched: list[str]
    top_gaps: list[str]
    explanation: str


class ScoreResponse(BaseModel):
    job_description_preview: str
    candidate_count: int
    results: list[CandidateResult]
