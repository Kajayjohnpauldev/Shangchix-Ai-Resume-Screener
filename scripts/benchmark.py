"""Synthetic benchmark for the Shangchix AI Resume ranker.

Generates a labelled test set of N resumes split across K role profiles.
For each profile's JD, ranks the full set and measures:

  * top-1 accuracy   — top-ranked resume's profile == JD's profile
  * precision@P      — of the top-P ranked, fraction from the right profile,
                       where P = (#resumes per profile)
  * latency          — wall-clock seconds per 100 resumes

Run from the repo root:
    python scripts/benchmark.py
    python scripts/benchmark.py --resumes 500 --profiles 5
"""
from __future__ import annotations

import argparse
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# Ensure repo root is on sys.path when invoked as `python scripts/benchmark.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252, which cannot encode the ✓/✗/→ glyphs we
# print below. Force UTF-8 so the benchmark runs identically on Windows, macOS,
# and Linux. No-op on streams that don't support reconfigure (e.g. some IDEs).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from backend.scorer import rank  # noqa: E402


# --- Synthetic profile pool -------------------------------------------------
# Each profile pairs a JD with templates we can vary slightly to produce
# many surface-different but semantically-equivalent resumes. The variations
# mimic real-world noise (different companies, year ranges, project names).

@dataclass(frozen=True)
class Profile:
    name: str
    jd: str
    headline: str
    summary: str
    skills: str
    role_keywords: tuple[str, ...]


PROFILES: list[Profile] = [
    Profile(
        name="python_backend",
        jd=(
            "We are hiring a Python backend engineer with strong experience "
            "building REST APIs in FastAPI or Flask, PostgreSQL, Docker, and "
            "AWS deployments. CI/CD via GitHub Actions is a plus."
        ),
        headline="Backend Engineer — Python / FastAPI / AWS",
        summary=(
            "Backend engineer building production REST APIs in Python. "
            "Comfortable with FastAPI, PostgreSQL, Docker, and AWS ECS. "
            "Owns CI/CD pipelines on GitHub Actions."
        ),
        skills=(
            "Python, FastAPI, Flask, PostgreSQL, Redis, Docker, Kubernetes, "
            "AWS, CI/CD, GitHub Actions, REST API, Linux, Git"
        ),
        role_keywords=("backend", "API service", "microservice"),
    ),
    Profile(
        name="ml_engineer",
        jd=(
            "Looking for an applied ML engineer to build LLM and RAG "
            "pipelines. Must have hands-on experience with sentence-"
            "transformers, FAISS, PyTorch, and serving models in production."
        ),
        headline="ML Engineer — LLM, RAG, Vector Search",
        summary=(
            "Applied ML engineer focused on LLM and RAG systems. Built "
            "retrieval pipelines with sentence-transformers and FAISS, "
            "served behind FastAPI."
        ),
        skills=(
            "Python, PyTorch, Hugging Face, sentence-transformers, FAISS, "
            "LangChain, LLM, RAG, NLP, FastAPI, Docker, AWS, pandas, numpy"
        ),
        role_keywords=("RAG", "embeddings", "model serving"),
    ),
    Profile(
        name="frontend",
        jd=(
            "We need a senior frontend engineer with deep React + "
            "TypeScript experience. Familiarity with Next.js, Tailwind, "
            "and accessibility-first component design."
        ),
        headline="Frontend Engineer — React / TypeScript / Next.js",
        summary=(
            "Frontend specialist shipping React + TypeScript apps with "
            "Next.js and Tailwind. Strong on design systems and "
            "accessibility (WCAG-AA)."
        ),
        skills=(
            "JavaScript, TypeScript, React, Next.js, HTML, CSS, Tailwind, "
            "Redux, Jest, Cypress, Figma, Git"
        ),
        role_keywords=("design system", "component library", "UI"),
    ),
    Profile(
        name="data_engineer",
        jd=(
            "Hiring a data engineer experienced with Spark, Kafka, and "
            "building ETL pipelines on AWS. SQL fluency required; Airflow "
            "and dbt experience a plus."
        ),
        headline="Data Engineer — Spark / Kafka / AWS",
        summary=(
            "Data engineer building ETL pipelines on Spark and Kafka. "
            "Orchestrates jobs with Airflow; transforms with dbt; deploys "
            "on AWS EMR."
        ),
        skills=(
            "Python, Scala, Spark, Kafka, Airflow, dbt, SQL, PostgreSQL, "
            "Snowflake, AWS, ETL, data engineering, Docker, Linux"
        ),
        role_keywords=("ETL", "data pipeline", "warehouse"),
    ),
    Profile(
        name="devops",
        jd=(
            "Hiring a DevOps engineer to own our Kubernetes platform on "
            "AWS. Terraform for IaC, Prometheus + Grafana for observability, "
            "and a strong CI/CD discipline."
        ),
        headline="DevOps Engineer — Kubernetes / AWS / Terraform",
        summary=(
            "Platform engineer running Kubernetes on AWS EKS. Infrastructure "
            "as code with Terraform; monitoring with Prometheus and Grafana; "
            "GitOps via ArgoCD."
        ),
        skills=(
            "Linux, Bash, Python, Docker, Kubernetes, AWS, Terraform, "
            "Ansible, Prometheus, Grafana, CI/CD, Jenkins, GitHub Actions"
        ),
        role_keywords=("Kubernetes", "platform", "infrastructure"),
    ),
]


COMPANIES = [
    "Acme Cloud", "Globex", "DemoAI Labs", "BrightApps", "StartupCo",
    "PriorCorp", "PixelWorks", "BigBank", "FinTechCo", "DataMesh Inc",
]


def _make_resume(profile: Profile, seed: int) -> tuple[str, str]:
    """Return (candidate_name, full resume text) varied by seed."""
    rng = random.Random(seed)
    name = f"Candidate {seed:04d} {profile.name.replace('_', ' ').title()}"
    company = rng.choice(COMPANIES)
    years = rng.randint(2, 10)
    project = rng.choice(profile.role_keywords)
    body = (
        f"{name}\n"
        f"{profile.headline}\n\n"
        f"Summary: {profile.summary} {years} years of experience.\n\n"
        f"Skills: {profile.skills}\n\n"
        f"Experience: {company} — {profile.headline} "
        f"(20{20 - years:02d}–2025). Owned the {project} workstream. "
        f"Shipped {rng.randint(2, 9)} production releases per quarter.\n\n"
        f"Education: B.E. Computer Science, 20{18 - years:02d}."
    )
    return name, body


def build_dataset(
    total_resumes: int,
    profile_count: int,
    seed: int = 0,
) -> tuple[list[Profile], list[tuple[str, str, str]]]:
    """Build a labelled dataset of (profile_name, candidate_name, resume_text)."""
    profiles = PROFILES[:profile_count]
    per_profile = total_resumes // len(profiles)
    rows: list[tuple[str, str, str]] = []
    s = seed
    for p in profiles:
        for _ in range(per_profile):
            cand_name, text = _make_resume(p, s)
            rows.append((p.name, cand_name, text))
            s += 1
    random.Random(seed).shuffle(rows)
    return profiles, rows


def evaluate(
    profiles: list[Profile],
    rows: list[tuple[str, str, str]],
) -> None:
    print(f"Dataset: {len(rows)} resumes across {len(profiles)} profiles")
    print(f"  per profile: {len(rows) // len(profiles)}")
    print()

    resumes_for_ranker = [(name, text) for _, name, text in rows]
    truth_by_name = {name: prof for prof, name, _ in rows}
    per_profile_count = len(rows) // len(profiles)

    top1_hits = 0
    top10_hits = 0
    precision_at_p_values: list[float] = []
    elapsed_per_jd: list[float] = []

    for prof in profiles:
        t0 = time.perf_counter()
        ranked = rank(prof.jd, resumes_for_ranker)
        elapsed = time.perf_counter() - t0
        elapsed_per_jd.append(elapsed)

        top1_hits += int(truth_by_name[ranked[0].candidate_name] == prof.name)
        top10_correct = sum(
            1 for r in ranked[:10]
            if truth_by_name[r.candidate_name] == prof.name
        )
        top10_hits += int(top10_correct > 0)

        top_p = ranked[:per_profile_count]
        correct_in_p = sum(
            1 for r in top_p
            if truth_by_name[r.candidate_name] == prof.name
        )
        precision_at_p = correct_in_p / per_profile_count
        precision_at_p_values.append(precision_at_p)

        print(
            f"  [{prof.name:14s}] "
            f"top-1 {'✓' if truth_by_name[ranked[0].candidate_name] == prof.name else '✗'}  "
            f"precision@{per_profile_count}: {precision_at_p:.1%}  "
            f"latency: {elapsed:.2f}s"
        )

    n_jds = len(profiles)
    n_resumes = len(rows)
    print()
    print("Aggregate")
    print(f"  top-1 accuracy:                  {top1_hits / n_jds:.1%}")
    print(f"  top-10 hit rate:                 {top10_hits / n_jds:.1%}")
    print(
        f"  mean precision@{per_profile_count}: "
        f"{statistics.mean(precision_at_p_values):.1%}"
    )
    mean_jd_latency = statistics.mean(elapsed_per_jd)
    per_100 = mean_jd_latency * (100 / n_resumes)
    print(f"  mean latency per JD ({n_resumes} resumes): {mean_jd_latency:.2f}s")
    print(f"  → projected latency per 100 resumes:       {per_100:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", type=int, default=500,
                    help="total resumes in the synthetic set")
    ap.add_argument("--profiles", type=int, default=5,
                    help="number of role profiles (max 5)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    profiles, rows = build_dataset(args.resumes, args.profiles, args.seed)
    evaluate(profiles, rows)


if __name__ == "__main__":
    main()
