from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

import faiss


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))  # cosine similarity if normalized


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


@lru_cache(maxsize=1)
def _load_index() -> faiss.Index:
    idx_path = INDEX_DIR / "faiss.index"
    if not idx_path.exists():
        raise FileNotFoundError(f"Missing index: {idx_path}")
    return faiss.read_index(str(idx_path))


@lru_cache(maxsize=1)
def _load_chunks() -> List[Dict[str, Any]]:
    chunks_path = INDEX_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks: {chunks_path}")
    return json.loads(chunks_path.read_text(encoding="utf-8"))


def retrieve(query: str, top_k: int | None = None) -> Tuple[float, List[Dict[str, Any]]]:
    if not query.strip():
        return 0.0, []

    k = top_k or TOP_K
    model = _load_model()
    index = _load_index()
    chunks = _load_chunks()

    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    scores, ids = index.search(q, k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict[str, Any]] = []
    best = 0.0

    for score, i in zip(scores, ids):
        if i < 0:
            continue
        best = max(best, float(score))
        c = chunks[i]
        results.append(
            {
                "score": float(score),
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": c.get("source_id"),
                "text": c.get("text"),
            }
        )

    return float(best), results


def fail_closed(best_score: float) -> bool:
    return best_score < RAG_MIN_SCORE
