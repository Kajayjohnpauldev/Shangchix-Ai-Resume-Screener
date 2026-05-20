# syntax=docker/dockerfile:1
#
# Shangchix AI Resume — one shared image used by BOTH services.
# docker-compose runs it twice (FastAPI backend + Streamlit frontend),
# overriding the command per service. Build once, run two processes.

FROM python:3.13-slim

# Predictable, lean Python in containers. HF_HOME points the embedding
# model cache at a baked-in directory (see the pre-download step below).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf-cache

WORKDIR /app

# Install dependencies first so Docker layer-caches them independently of
# application code changes. faiss-cpu, torch, and pydantic-core all ship
# manylinux wheels for CPython 3.13 — no compiler toolchain required.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Bake the all-MiniLM-L6-v2 weights (~80 MB) into the image so the first
# /score request is fast and the running container needs no network for
# embeddings — only the LLM explanation step calls out.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Application code.
COPY . .

# 8000 = FastAPI, 8501 = Streamlit. compose maps both.
EXPOSE 8000 8501

# Default command = backend. The frontend service overrides this in
# docker-compose.yml.
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
