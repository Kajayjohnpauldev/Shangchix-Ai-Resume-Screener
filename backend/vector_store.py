"""Minimal FAISS wrapper around IndexFlatIP.

Vectors must be L2-normalized before being added — inner product on
normalized vectors equals cosine similarity.
"""
from __future__ import annotations

from dataclasses import dataclass

import faiss
import numpy as np

from backend.config import EMBEDDING_DIM


@dataclass
class SearchHit:
    index: int
    score: float


class VectorStore:
    def __init__(self, dim: int = EMBEDDING_DIM) -> None:
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(
                f"expected (N, {self.dim}) float32 array, got {vectors.shape}"
            )
        self.index.add(vectors.astype(np.float32))

    def search(self, query: np.ndarray, k: int) -> list[SearchHit]:
        if query.ndim == 1:
            query = query[None, :]
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
        scores, indices = self.index.search(query.astype(np.float32), k)
        return [
            SearchHit(index=int(i), score=float(s))
            for i, s in zip(indices[0], scores[0])
            if i != -1
        ]

    @property
    def size(self) -> int:
        return int(self.index.ntotal)
