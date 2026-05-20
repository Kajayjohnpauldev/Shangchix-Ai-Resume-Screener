"""Sentence-transformer embeddings.

Loads `all-MiniLM-L6-v2` lazily on first use, then reuses the singleton
for every subsequent call. Outputs are L2-normalized so that inner
product on the FAISS index equals cosine similarity.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.config import EMBEDDING_MODEL_NAME, EMBEDDING_DIM

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed(text: str) -> np.ndarray:
    """Return a (EMBEDDING_DIM,) float32 unit vector."""
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)
    vec = _get_model().encode(
        text, normalize_embeddings=True, convert_to_numpy=True
    )
    return vec.astype(np.float32)


def embed_many(texts: Iterable[str]) -> np.ndarray:
    """Return a (N, EMBEDDING_DIM) float32 array of unit vectors."""
    texts = list(texts)
    if not texts:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32)
    vecs = _get_model().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True, batch_size=32
    )
    return vecs.astype(np.float32)
