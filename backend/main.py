"""FastAPI app — single POST /score endpoint."""
from __future__ import annotations

import logging
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from backend.config import TOP_K_EXPLAIN
from backend.explainer import Advice, explain
from backend.pdf_parser import candidate_name_from_filename, extract_resume_text
from backend.schemas import CandidateResult, ScoreResponse
from backend.scorer import rank

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Shangchix AI Resume",
    description=(
        "Shangchix AI Resume — upload N resumes and a job description; "
        "receive ranked candidates with similarity scores and LLM-generated "
        "explanations."
    ),
    version="1.0.0",
)

# Streamlit calls the API directly from a python process, so CORS isn't
# strictly needed — but enabling it makes the API browser-testable too.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", response_model=ScoreResponse)
async def score(
    job_description: str = Form(..., min_length=10),
    resumes: List[UploadFile] = File(...),
) -> ScoreResponse:
    if not resumes:
        raise HTTPException(status_code=400, detail="No resumes uploaded")

    parsed: list[tuple[str, str]] = []
    for upload in resumes:
        raw = await upload.read()
        filename = upload.filename or "candidate.pdf"
        text = extract_resume_text(filename, raw)
        name = candidate_name_from_filename(filename)
        parsed.append((name, text))
        logger.info("parsed %s (%d chars)", name, len(text))

    ranked = rank(job_description, parsed)

    results: list[CandidateResult] = []
    for i, r in enumerate(ranked):
        if i < TOP_K_EXPLAIN:
            # Feed the AI the complete skill picture so its assessment and
            # improvement tips reflect every match and gap, not just three.
            advice = explain(
                job_description,
                r.resume_text,
                r.all_skills_matched,
                r.all_gaps,
            )
        else:
            advice = Advice(
                explanation="Ranked outside top results — explanation skipped."
            )
        results.append(
            CandidateResult(
                resume_id=r.resume_id,
                candidate_name=r.candidate_name,
                score=r.score,
                raw_similarity=r.raw_similarity,
                projected_score=r.projected_score,
                score_uplift=r.score_uplift,
                ats_score=r.ats_score,
                keywords_matched=r.keywords_matched,
                keywords_total=r.keywords_total,
                top_skills_matched=r.top_skills_matched,
                top_gaps=r.top_gaps,
                all_skills_matched=r.all_skills_matched,
                all_gaps=r.all_gaps,
                explanation=advice.explanation,
                improvements=advice.improvements,
            )
        )

    return ScoreResponse(
        job_description_preview=job_description[:200],
        candidate_count=len(results),
        results=results,
    )
