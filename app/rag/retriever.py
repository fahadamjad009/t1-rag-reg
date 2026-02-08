from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# BM25
from app.rag.bm25 import BM25Index, build_bm25_from_chunks, bm25_search


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

# Vector similarity threshold for fail-closed gate (kept as-is for now)
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))  # cosine similarity if normalized


# -------------------------------------------------------
# Step 2D — Low-quality source filtering (fast + measurable)
# -------------------------------------------------------
EXCLUDE_SOURCE_SUBSTRINGS = [
    "_business_your_industry",          # category hubs
    "_business_new_to_austrac_e_learning",
    "_news_and_media_",
    "_inbrief_",
    "_form",
]


def _is_low_quality_source_id(source_id: str) -> bool:
    s = (source_id or "").lower()
    return any(sub.lower() in s for sub in EXCLUDE_SOURCE_SUBSTRINGS)


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
    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise RuntimeError(f"Expected list in chunks.json, got {type(data)}")
    return data


@lru_cache(maxsize=1)
def _load_chunk_map() -> Dict[str, Dict[str, Any]]:
    """
    chunk_id -> chunk dict for fast lookup (used by BM25 and fusion).
    """
    chunks = _load_chunks()
    m: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        cid = c.get("chunk_id")
        if cid:
            m[str(cid)] = c
    return m


@lru_cache(maxsize=1)
def _load_bm25() -> BM25Index:
    """
    Build BM25 once from the same chunks.json.
    """
    chunks_path = INDEX_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks: {chunks_path}")
    return build_bm25_from_chunks(chunks_path)


def _vector_retrieve(query: str, k: int) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Vector retrieval (cosine similarity if embeddings are normalized).
    Returns (best_score, results).
    """
    model = _load_model()
    index = _load_index()
    chunks = _load_chunks()

    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

    # Over-fetch so filtering doesn't shrink below k too often
    # (cap at corpus size)
    over_k = min(max(k * 3, k), max(1, len(chunks)))

    scores, ids = index.search(q, over_k)
    scores = scores[0].tolist()
    ids = ids[0].tolist()

    results: List[Dict[str, Any]] = []
    best = 0.0

    for score, i in zip(scores, ids):
        if i < 0:
            continue

        c = chunks[i]
        sid = c.get("source_id") or ""

        # hard exclude by source_id patterns
        if _is_low_quality_source_id(str(sid)):
            continue

        best = max(best, float(score))

        results.append(
            {
                "score": float(score),
                "retriever": "vector",
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": c.get("source_id"),
                # ✅ provenance
                "retrieved_at": c.get("retrieved_at"),
                "content_hash": c.get("content_hash"),
                "text": c.get("text"),
            }
        )

        if len(results) >= k:
            break

    return float(best), results


def _bm25_retrieve(query: str, k: int) -> Tuple[float, List[Dict[str, Any]]]:
    """
    BM25 retrieval using app.rag.bm25.
    Returns results in the same schema as vector retrieval.
    'best_score' is the top BM25 score (not comparable to cosine).
    """
    ix = _load_bm25()
    cmap = _load_chunk_map()

    # Over-fetch to survive filtering
    over_k = max(k * 5, k)

    pairs = bm25_search(ix, query, top_k=over_k)  # List[(chunk_id, score)]
    results: List[Dict[str, Any]] = []

    best = 0.0
    for cid, score in pairs:
        c = cmap.get(str(cid))
        if not c:
            continue

        sid = c.get("source_id") or ""
        if _is_low_quality_source_id(str(sid)):
            continue

        best = max(best, float(score))

        results.append(
            {
                "score": float(score),
                "retriever": "bm25",
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": c.get("source_id"),
                # ✅ provenance
                "retrieved_at": c.get("retrieved_at"),
                "content_hash": c.get("content_hash"),
                "text": c.get("text"),
            }
        )

        if len(results) >= k:
            break

    return float(best), results


def _rrf_fuse(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF).
    Score = sum( 1 / (rrf_k + rank) ) across retrievers.
    Produces fused ranking over chunk_id.
    """
    fused: Dict[str, float] = {}

    def add(rr: List[Dict[str, Any]]) -> None:
        for rank, item in enumerate(rr, start=1):
            cid = item.get("chunk_id")
            if not cid:
                continue
            fused[str(cid)] = fused.get(str(cid), 0.0) + (1.0 / (rrf_k + rank))

    add(vector_results)
    add(bm25_results)

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]

    cmap = _load_chunk_map()
    out: List[Dict[str, Any]] = []
    for cid, fscore in ranked:
        c = cmap.get(str(cid))
        if not c:
            continue
        out.append(
            {
                "score": float(fscore),  # fused score (not cosine, not bm25)
                "retriever": "hybrid_rrf",
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": c.get("source_id"),
                # ✅ provenance
                "retrieved_at": c.get("retrieved_at"),
                "content_hash": c.get("content_hash"),
                "text": c.get("text"),
                "debug": {"rrf_score": float(fscore)},
            }
        )
    return out


def retrieve(
    query: str,
    top_k: int | None = None,
    mode: str = "vector",  # "vector" | "bm25" | "hybrid_rrf"
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Main retrieval entrypoint.

    Returns:
      best_score: float (meaning depends on mode; kept for backward compatibility)
      results: list of dicts
    """
    if not (query or "").strip():
        return 0.0, []

    k = top_k or TOP_K
    mode = (mode or "vector").lower().strip()

    if mode == "bm25":
        return _bm25_retrieve(query, k)

    if mode in ("hybrid_rrf", "hybrid", "rrf"):
        # Over-fetch each retriever, fuse a larger pool, then filter + slice to k.
        v_best, v_res = _vector_retrieve(query, k * 3)
        _, b_res = _bm25_retrieve(query, k * 3)

        fused_big = _rrf_fuse(v_res, b_res, k=max(k * 5, k), rrf_k=60)

        fused = [
            h for h in fused_big
            if not _is_low_quality_source_id(h.get("source_id", ""))
        ][:k]

        # best_score remains vector-best for now (threshold is cosine-based)
        return float(v_best), fused

    return _vector_retrieve(query, k)


def fail_closed(best_score: float) -> bool:
    """
    Still uses vector cosine threshold.
    In hybrid mode we return vector-best as best_score, so this remains consistent.
    """
    return float(best_score) < RAG_MIN_SCORE
