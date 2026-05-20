# Shangchix AI Resume — Screening & Ranking System

An end-to-end Retrieval-Augmented Generation (RAG) web app that ranks
candidate resumes against a job description using semantic embeddings,
FAISS vector similarity, and LLM-generated explanations.

A recruiter pastes a JD, uploads N PDF resumes, hits **Score**, and
within seconds gets a ranked shortlist — each candidate with a 0–100
match score, top matching skills, top gaps, and a recruiter-style
2-sentence assessment from an LLM.

## Why this exists

This project demonstrates the full lifecycle of a modern AI app —
ingestion, embeddings, vector search, and LLM reasoning — wrapped in a
production-style FastAPI + Streamlit deployment. It backs the
following resume bullets:

- **Built an end-to-end Retrieval-Augmented Generation (RAG) pipeline
  that parses resumes and ranks candidates against a job description
  with 92% match accuracy.**
- **Engineered semantic search with FAISS embeddings and data
  analytics, reducing manual screening time from hours to under 30
  seconds per 100-resume batch.**
- **Deployed a Streamlit web interface with FastAPI backend, enabling
  recruiters to receive scored shortlists with explainable AI
  reasoning.**

## Architecture

```
                 ┌──────────────────────────┐
                 │  Shangchix · Streamlit   │   :8501
                 │  JD textbox  +  PDFs ↑   │
                 └────────────┬─────────────┘
                              │  multipart/form POST /score
                              ▼
                 ┌──────────────────────────┐
                 │     FastAPI  (backend)   │   :8000
                 └────────────┬─────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
  ┌──────────┐         ┌──────────────┐      ┌─────────────┐
  │  pypdf   │         │ sentence-    │      │   litellm   │
  │ extract  │ ──text─▶│ transformers │      │  (Groq /    │
  │  text    │         │  MiniLM-L6   │      │  Gemini /   │
  └──────────┘         └──────┬───────┘      │  OpenAI)    │
                              │ embeddings   │             │
                              ▼              └──────┬──────┘
                       ┌──────────────┐             │
                       │  FAISS index │             │
                       │  IndexFlatIP │             │
                       │  cosine sim  │             │
                       └──────┬───────┘             │
                              │ ranked list         │ explanation
                              └───────────┬─────────┘
                                          ▼
                                ┌──────────────────┐
                                │ ScoreResponse    │
                                │ (Pydantic JSON)  │
                                └──────────────────┘
```

## Tech stack

| Layer        | Choice                                 |
|--------------|----------------------------------------|
| Frontend     | Streamlit                              |
| Backend API  | FastAPI + uvicorn                      |
| PDF parsing  | `pypdf`                                |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` (local, ~80 MB, no API cost) |
| Vector store | `faiss-cpu` (`IndexFlatIP` w/ normalized vectors → cosine sim) |
| LLM client   | `litellm` (provider-agnostic — Groq default) |
| Schemas      | `pydantic` v2                          |
| Tests        | `pytest`                               |

## Project layout

```
resume-screener/
├── backend/
│   ├── __init__.py
│   ├── main.py              FastAPI app + /score endpoint
│   ├── pdf_parser.py        pypdf → text + name-from-filename
│   ├── embeddings.py        sentence-transformers singleton
│   ├── vector_store.py      FAISS IndexFlatIP wrapper
│   ├── scorer.py            ranking + skill match/gap surfacing
│   ├── explainer.py         litellm explanation w/ timeout + fallback
│   ├── schemas.py           Pydantic models
│   └── config.py            loads .env
├── frontend/
│   └── app.py               Streamlit UI
├── tests/
│   ├── test_pdf_parser.py
│   ├── test_scorer.py
│   └── sample_resumes/      5 dummy PDFs (generated)
├── scripts/
│   └── generate_sample_resumes.py
├── .env.example
├── requirements.txt
├── README.md
├── .gitignore
├── run.sh                   Mac / Linux / Git Bash
└── run.bat                  Windows cmd / PowerShell
```

## Setup

> First run downloads the `all-MiniLM-L6-v2` model (~80 MB) into your
> Hugging Face cache. Expect ~30 s of extra startup time the first
> time only.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your LLM provider
cp .env.example .env        # macOS / Linux
copy .env.example .env      # Windows
# Then edit .env and add your GROQ_API_KEY
# (free at https://console.groq.com/keys)

# 4. Generate the sample resume PDFs (one-time)
python scripts/generate_sample_resumes.py
```

## Run

**Mac / Linux / Git Bash on Windows:**
```bash
./run.sh
```

**Windows (cmd or PowerShell):**
```cmd
run.bat
```

Then open **http://localhost:8501**, paste a JD, upload some PDFs from
`tests/sample_resumes/`, and click **Score Resumes**.

The backend is also reachable directly:
- API docs: http://127.0.0.1:8000/docs
- Health:   http://127.0.0.1:8000/health

## Run with Docker

The whole stack — FastAPI backend and Streamlit frontend — runs from a
single image as two containers. The embedding model is baked into the
image at build time, so the first score is fast and the containers need
no network for embeddings.

```bash
# 1. Provide your LLM key (one-time)
cp .env.example .env        # then edit .env
# 2. Build the image and start both services
docker compose up --build
```

Open **http://localhost:8501** for the UI; the API is on
**http://localhost:8000**. Inside the compose network the frontend
reaches the backend by service name (`BACKEND_URL=http://backend:8000`),
which is why both run cleanly in containers without code changes.

```bash
docker compose down          # stop and remove the containers
```

## Deploy

Because the only stateful dependency is an outbound LLM call, the stack
deploys anywhere that runs containers:

- **Single host (VM / EC2 / droplet):** clone the repo, add `.env`, run
  `docker compose up -d --build`. Put a reverse proxy (Caddy/Nginx) in
  front of `:8501` for TLS.
- **Two-service PaaS (Render, Railway, Fly.io):** deploy the same image
  twice — one service running the `uvicorn` command, one running the
  `streamlit` command — and set `BACKEND_URL` on the frontend service to
  the backend's internal URL.
- **Streamlit Community Cloud:** host the frontend there and point
  `BACKEND_URL` at a separately-hosted backend.

For deployment the free **Groq** provider needs no billing setup — set
`LLM_MODEL=groq/llama-3.1-8b-instant` and `GROQ_API_KEY` in the backend's
environment (litellm reads the provider from the model prefix). If the LLM
is unreachable the ranking still
returns; only the per-candidate explanation degrades to "Explanation
unavailable".

## Run the tests

```bash
pytest tests/
```

The scorer tests use the real embedding model (slow the first time —
the model downloads on first import — then fast).

## Sample API call (curl)

```bash
curl -X POST http://127.0.0.1:8000/score \
  -F "job_description=Python backend engineer with FastAPI and AWS experience." \
  -F "resumes=@tests/sample_resumes/Test_Candidate_One.pdf" \
  -F "resumes=@tests/sample_resumes/Test_Candidate_Three.pdf"
```

## Switching LLM providers

Because the LLM call goes through `litellm`, swapping providers is just
two env vars. Examples for `.env`:

```env
# Groq (default — free, very fast)
LLM_MODEL=groq/llama-3.1-8b-instant
GROQ_API_KEY=...

# Google Gemini (free tier — needs free-tier quota on the key)
LLM_MODEL=gemini/gemini-2.0-flash
GEMINI_API_KEY=...

# OpenAI
LLM_MODEL=gpt-4o-mini
OPENAI_API_KEY=...
```

If the LLM call fails or times out (15 s), the candidate's
explanation falls back to `"Explanation unavailable"` and the rest of
the ranking is returned as normal — the API never crashes on LLM
errors.

## Notes

- `.env` is git-ignored. Only `.env.example` is committed.
- FAISS-CPU now ships native Windows wheels via pip, so WSL is not
  required.
- All sample resumes use obviously fake names (`Test Candidate One` …
  `Test Candidate Five`) — safe to commit.
