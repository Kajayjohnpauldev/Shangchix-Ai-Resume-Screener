"""Generates 5 obviously-fake sample resume PDFs into tests/sample_resumes/.

Run from the repo root:  python scripts/generate_sample_resumes.py
"""
from __future__ import annotations

from pathlib import Path

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer


OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "sample_resumes"


CANDIDATES: list[dict] = [
    {
        "name": "Test Candidate One",
        "headline": "Senior Backend Engineer — Python / FastAPI / AWS",
        "summary": (
            "Backend engineer with 6 years building production REST APIs in "
            "Python. Deep experience with FastAPI, PostgreSQL, Docker, and "
            "AWS (ECS, Lambda, S3). Comfortable owning systems from design "
            "through on-call."
        ),
        "skills": (
            "Python, FastAPI, Flask, PostgreSQL, Redis, Docker, Kubernetes, "
            "AWS, CI/CD, GitHub Actions, REST API, gRPC, Linux, Bash, Git"
        ),
        "experience": [
            "Acme Cloud — Senior Backend Engineer (2021–2025). Owned the "
            "billing service: FastAPI on ECS, PostgreSQL, Redis. Reduced p99 "
            "latency from 800ms to 120ms.",
            "Globex — Backend Engineer (2019–2021). Built a multi-tenant "
            "REST API in Flask. Containerized with Docker, CI on GitHub "
            "Actions.",
        ],
        "education": "B.E. Computer Science, Demo Tech University, 2019.",
    },
    {
        "name": "Test Candidate Two",
        "headline": "ML Engineer — NLP, RAG, Vector Search",
        "summary": (
            "Applied ML engineer focused on LLM and RAG pipelines. Strong "
            "with sentence-transformers, FAISS, and LangChain. Has shipped "
            "production retrieval systems at scale."
        ),
        "skills": (
            "Python, PyTorch, Hugging Face Transformers, sentence-transformers, "
            "FAISS, LangChain, LLM, RAG, NLP, FastAPI, Docker, AWS, pandas, "
            "numpy, scikit-learn"
        ),
        "experience": [
            "DemoAI Labs — ML Engineer (2022–2025). Built a RAG-based "
            "support assistant with FAISS over 2M docs. Embeddings via "
            "all-MiniLM-L6-v2; served behind FastAPI.",
            "PriorCorp — Data Scientist (2020–2022). Sentiment analysis on "
            "10M reviews; deployed via Flask + Docker.",
        ],
        "education": "M.S. Data Science, Demo Tech University, 2020.",
    },
    {
        "name": "Test Candidate Three",
        "headline": "Frontend Engineer — React / TypeScript",
        "summary": (
            "Frontend specialist with 5 years in React and TypeScript. "
            "Strong on design systems and accessibility. Ships polished "
            "component libraries used across multiple product teams."
        ),
        "skills": (
            "JavaScript, TypeScript, React, Next.js, HTML, CSS, Tailwind, "
            "Redux, Jest, Cypress, Figma, Git"
        ),
        "experience": [
            "PixelWorks — Senior Frontend Engineer (2022–2025). Led "
            "migration of a legacy Angular app to React + TypeScript. "
            "Shipped a shared design system.",
            "BrightApps — Frontend Engineer (2020–2022). React + Redux "
            "dashboards. WCAG-AA accessibility audits.",
        ],
        "education": "B.Sc. Computer Science, Demo Tech University, 2020.",
    },
    {
        "name": "Test Candidate Four",
        "headline": "Full-Stack Engineer — Python + React",
        "summary": (
            "Full-stack engineer with balanced Python backend and React "
            "frontend experience. Has touched Docker and basic AWS."
        ),
        "skills": (
            "Python, Django, FastAPI, JavaScript, TypeScript, React, "
            "PostgreSQL, Docker, AWS, REST API, Git, Linux"
        ),
        "experience": [
            "StartupCo — Full-Stack Engineer (2021–2025). Owned features "
            "end-to-end: Django backend + React frontend. Wrote integration "
            "tests; deployed on AWS ECS.",
            "FreelanceProjects (2019–2021). Built small web apps for "
            "clients, mostly React + FastAPI.",
        ],
        "education": "B.Tech. Information Technology, Demo Tech University, 2019.",
    },
    {
        "name": "Test Candidate Five",
        "headline": "Java Backend Engineer — Spring Boot",
        "summary": (
            "Java backend engineer with 8 years on Spring Boot / "
            "microservices. Heavy Kafka and SQL background. Owns "
            "production microservices end-to-end."
        ),
        "skills": (
            "Java, Spring Boot, Maven, REST API, gRPC, Kafka, MySQL, "
            "PostgreSQL, Docker, Kubernetes, Jenkins, CI/CD, Linux, Git"
        ),
        "experience": [
            "BigBank — Senior Java Engineer (2018–2025). Microservices in "
            "Spring Boot, Kafka event streams, deployed on Kubernetes.",
            "FinTechCo — Java Developer (2016–2018). REST APIs in Spring; "
            "MySQL backends; Jenkins pipelines.",
        ],
        "education": "M.C.A., Demo Tech University, 2016.",
    },
]


def _build_pdf(out_path: Path, candidate: dict) -> None:
    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h3 = styles["Heading3"]
    body = ParagraphStyle("body", parent=styles["BodyText"], spaceAfter=6)

    doc = SimpleDocTemplate(
        str(out_path),
        pagesize=LETTER,
        title=candidate["name"],
        author="Resume Screener Sample Generator",
    )
    story = [
        Paragraph(candidate["name"], h1),
        Paragraph(candidate["headline"], styles["Italic"]),
        Spacer(1, 12),
        Paragraph("Summary", h3),
        Paragraph(candidate["summary"], body),
        Paragraph("Skills", h3),
        Paragraph(candidate["skills"], body),
        Paragraph("Experience", h3),
    ]
    for line in candidate["experience"]:
        story.append(Paragraph("• " + line, body))
    story.append(Paragraph("Education", h3))
    story.append(Paragraph(candidate["education"], body))

    doc.build(story)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in CANDIDATES:
        filename = c["name"].replace(" ", "_") + ".pdf"
        out = OUT_DIR / filename
        _build_pdf(out, c)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
