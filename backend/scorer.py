"""Resume ranking pipeline.

For each resume:
  * extract its text
  * embed JD and resume; cosine similarity is the raw match score
    (this is the RAG / semantic-search bullet of the project)
  * detect concrete skill keywords from a baseline vocabulary using
    case-insensitive word-boundary matching so the recruiter-facing
    "top matching skills" / "top gaps" lists are deterministic and
    accurate — single skill words don't embed reliably against full
    documents, so we don't use cosine for this part

Score scaling: cosine on this MiniLM model for related-but-not-identical
text typically sits in 0.3–0.8. We map [0, 1] → [0, 100] linearly and
clip — empirically this gives a recruiter-readable 40–90 range for the
kind of JD/resume pairs this tool sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from backend.embeddings import embed, embed_many
from backend.vector_store import VectorStore

# A baseline tech skill vocabulary. The point of this list is to give
# the explainer concrete strings to surface ("matched: Python, FastAPI,
# Docker"). It is NOT used to compute the overall score — that's pure
# embedding similarity on the full JD vs. full resume.
SKILL_VOCAB: list[str] = [
    "python", "java", "c++", "c#", "javascript", "typescript", "go", "rust",
    "sql", "nosql", "mongodb", "postgresql", "mysql", "redis",
    "react", "angular", "vue", "node.js", "express", "django", "flask",
    "fastapi", "spring boot", "rest api", "graphql", "grpc",
    "docker", "kubernetes", "aws", "azure", "gcp", "terraform", "ansible",
    "ci/cd", "jenkins", "github actions", "gitlab ci",
    "machine learning", "deep learning", "pytorch", "tensorflow",
    "scikit-learn", "pandas", "numpy", "matplotlib",
    "nlp", "computer vision", "transformers", "llm", "rag",
    "data analysis", "data engineering", "etl", "spark", "hadoop", "kafka",
    "linux", "bash", "git", "agile", "scrum", "tdd",
    "html", "css", "tailwind", "next.js",
]

# Pre-compile a case-insensitive word-boundary regex for each vocab
# term. Some skills contain regex metacharacters (c++, c#, node.js,
# next.js) — re.escape handles that. \b doesn't fire next to non-word
# characters like '+', '#', '.', so we anchor with a lookaround that
# accepts start/end/whitespace/punctuation as the boundary.
def _compile_skill(term: str) -> re.Pattern[str]:
    escaped = re.escape(term)
    # Allow an optional trailing 's' so "REST API" matches "REST APIs",
    # "framework" matches "frameworks", etc. The lookarounds prevent
    # matching inside larger words ("rag" won't fire inside "drag").
    return re.compile(
        rf"(?<![A-Za-z0-9]){escaped}s?(?![A-Za-z0-9])", re.IGNORECASE
    )


_SKILL_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (s, _compile_skill(s)) for s in SKILL_VOCAB
]


def _detect_skills(text: str) -> list[tuple[str, int]]:
    """Return (skill, first_offset_in_text) for every vocab skill present.

    Ordered by first appearance in `text` — so callers can rank by
    "what the JD/resume mentioned first" which is the natural recruiter
    priority order.
    """
    if not text:
        return []
    found: list[tuple[str, int]] = []
    for term, pat in _SKILL_PATTERNS:
        m = pat.search(text)
        if m:
            found.append((term, m.start()))
    found.sort(key=lambda pair: pair[1])
    return found


@dataclass
class ScoredResume:
    resume_id: int
    candidate_name: str
    score: float                       # 0–100
    raw_similarity: float              # cosine, 0–1
    top_skills_matched: list[str] = field(default_factory=list)  # headline 3
    top_gaps: list[str] = field(default_factory=list)            # headline 3
    all_skills_matched: list[str] = field(default_factory=list)  # every JD skill present
    all_gaps: list[str] = field(default_factory=list)            # every JD skill missing
    projected_score: float = 0.0       # 0–100, score if top gaps were added
    score_uplift: float = 0.0          # projected_score − score
    ats_score: float = 0.0             # 0–100, JD-keyword coverage (ATS-style)
    keywords_matched: int = 0          # JD skills found in the resume
    keywords_total: int = 0            # distinct skills detected in the JD
    resume_text: str = ""              # kept for the explainer step


def rank(
    jd_text: str,
    resumes: list[tuple[str, str]],
) -> list[ScoredResume]:
    """Rank resumes against a job description.

    `resumes` is a list of (candidate_name, resume_text) tuples. The
    returned list is sorted by descending score.
    """
    if not resumes:
        return []

    jd_vec = embed(jd_text)
    resume_texts = [r[1] for r in resumes]
    resume_vecs = embed_many(resume_texts)

    store = VectorStore()
    store.add(resume_vecs)
    hits = store.search(jd_vec, k=len(resumes))
    hit_by_idx = {h.index: h.score for h in hits}

    jd_skills_ordered = [s for s, _ in _detect_skills(jd_text)]

    # First pass: matched/gap skills, plus an "augmented" copy of each
    # resume with its top missing JD skills appended. Re-embedding that
    # augmented text against the JD gives an honest projected score —
    # "what this resume could score if it credibly added these skills".
    jd_skill_count = len(jd_skills_ordered)

    interim: list[tuple] = []
    augmented_texts: list[str] = []
    for i, (name, text) in enumerate(resumes):
        sim = hit_by_idx.get(i, 0.0)
        resume_skill_set = {s for s, _ in _detect_skills(text)}
        # Preserve JD ordering — earliest-mentioned skills come first.
        covered = [s for s in jd_skills_ordered if s in resume_skill_set]
        gaps_all = [s for s in jd_skills_ordered if s not in resume_skill_set]
        matched, gaps = covered[:3], gaps_all[:3]
        interim.append((i, name, text, sim, matched, gaps, covered, gaps_all))
        augmented_texts.append(
            text + ("\nAdditional skills: " + ", ".join(gaps) if gaps else "")
        )

    aug_vecs = embed_many(augmented_texts)

    results: list[ScoredResume] = []
    for (i, name, text, sim, matched, gaps, covered, gaps_all), aug_vec in zip(
        interim, aug_vecs
    ):
        n_covered = len(covered)
        # Clip negative cosines to 0 (irrelevant), scale [0,1] → [0,100].
        score_100 = max(0.0, min(100.0, sim * 100.0))

        if gaps:
            proj_sim = float(np.dot(jd_vec, aug_vec))
            proj_100 = max(0.0, min(100.0, proj_sim * 100.0))
            # Adding skills should never *lower* the displayed potential.
            proj_100 = max(proj_100, score_100)
        else:
            proj_100 = score_100

        # ATS-style score: fraction of the JD's distinct skill keywords the
        # resume actually contains — this is what keyword-matching applicant
        # tracking systems screen on, distinct from the semantic score above.
        ats = round(100.0 * n_covered / jd_skill_count, 1) if jd_skill_count else 0.0

        results.append(
            ScoredResume(
                resume_id=i,
                candidate_name=name,
                score=round(score_100, 2),
                raw_similarity=round(float(sim), 4),
                top_skills_matched=matched,
                top_gaps=gaps,
                all_skills_matched=covered,
                all_gaps=gaps_all,
                projected_score=round(proj_100, 2),
                score_uplift=round(proj_100 - score_100, 2),
                ats_score=ats,
                keywords_matched=n_covered,
                keywords_total=jd_skill_count,
                resume_text=text,
            )
        )

    results.sort(key=lambda r: r.score, reverse=True)
    return results
