from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# BM25
from app.rag.bm25 import BM25Index, build_bm25_from_chunks, bm25_search

# ✅ Query rewriting
from app.rag.query_rewrite import rewrite_query

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
INDEX_DIR = DATA_DIR / "index"

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K", "5"))

# Vector similarity threshold for fail-closed gate
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.25"))

# -------------------------------------------------------
# Low-quality source filtering
# -------------------------------------------------------
EXCLUDE_SOURCE_SUBSTRINGS = [
    "_business_your_industry",
    "_business_new_to_austrac_e_learning",
    "_news_and_media_",
    "_inbrief_",
    "_form",
]


def _is_low_quality_source_id(source_id: str) -> bool:
    s = (source_id or "").lower()
    return any(sub.lower() in s for sub in EXCLUDE_SOURCE_SUBSTRINGS)


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        x = (x or "").strip()
        if not x:
            continue
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _dedupe_hits_by_chunk_keep_order(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for h in hits:
        cid = str(h.get("chunk_id") or "")
        if not cid:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        out.append(h)
    return out


# -------------------------------------------------------
# Cached loaders
# -------------------------------------------------------
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
        raise RuntimeError("chunks.json must contain a list")
    return data


@lru_cache(maxsize=1)
def _load_chunk_map() -> Dict[str, Dict[str, Any]]:
    m: Dict[str, Dict[str, Any]] = {}
    for c in _load_chunks():
        cid = c.get("chunk_id")
        if cid:
            m[str(cid)] = c
    return m


@lru_cache(maxsize=1)
def _load_bm25() -> BM25Index:
    chunks_path = INDEX_DIR / "chunks.json"
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks: {chunks_path}")
    return build_bm25_from_chunks(chunks_path)


# -------------------------------------------------------
# Vector retrieval
# -------------------------------------------------------
def _vector_retrieve(query: str, k: int) -> Tuple[float, List[Dict[str, Any]]]:
    model = _load_model()
    index = _load_index()
    chunks = _load_chunks()

    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype="float32")

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
        sid = str(c.get("source_id") or "")

        if _is_low_quality_source_id(sid):
            continue

        best = max(best, float(score))

        results.append(
            {
                "score": float(score),
                "retriever": "vector",
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": sid,
                "retrieved_at": c.get("retrieved_at"),
                "content_hash": c.get("content_hash"),
                "text": c.get("text"),
            }
        )

        if len(results) >= k:
            break

    return float(best), results


# -------------------------------------------------------
# BM25 retrieval
# -------------------------------------------------------
def _bm25_retrieve(query: str, k: int) -> Tuple[float, List[Dict[str, Any]]]:
    ix = _load_bm25()
    cmap = _load_chunk_map()

    # Note: bm25_search itself is capped by this top_k.
    # We do NOT multiply again here; the caller controls depth.
    pairs = bm25_search(ix, query, top_k=max(1, k))

    results: List[Dict[str, Any]] = []
    best = 0.0

    for cid, score in pairs:
        c = cmap.get(str(cid))
        if not c:
            continue

        sid = str(c.get("source_id") or "")
        if _is_low_quality_source_id(sid):
            continue

        best = max(best, float(score))

        results.append(
            {
                "score": float(score),
                "retriever": "bm25",
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": sid,
                "retrieved_at": c.get("retrieved_at"),
                "content_hash": c.get("content_hash"),
                "text": c.get("text"),
            }
        )

        if len(results) >= k:
            break

    return float(best), results


# -------------------------------------------------------
# RRF fusion (chunk-level)
# -------------------------------------------------------
def _rrf_fuse(
    vector_results: List[Dict[str, Any]],
    bm25_results: List[Dict[str, Any]],
    k: int,
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal Rank Fusion (RRF) at CHUNK level.
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

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: max(1, k)]
    cmap = _load_chunk_map()

    out: List[Dict[str, Any]] = []
    for cid, fscore in ranked:
        c = cmap.get(str(cid))
        if not c:
            continue

        sid = str(c.get("source_id") or "")
        if _is_low_quality_source_id(sid):
            continue

        out.append(
            {
                "score": float(fscore),
                "retriever": "hybrid_rrf",
                "chunk_id": c.get("chunk_id"),
                "title": c.get("title"),
                "url": c.get("url"),
                "source_id": sid,
                "retrieved_at": c.get("retrieved_at"),
                "content_hash": c.get("content_hash"),
                "text": c.get("text"),
                "debug": {"rrf_score": float(fscore)},
            }
        )

    return out


# -------------------------------------------------------
# ✅ Source-level fusion across multiple ranked lists
# -------------------------------------------------------
def _source_rrf_fuse(
    results_lists: List[List[Dict[str, Any]]],
    k_sources: int,
    rrf_k: int = 60,
) -> List[str]:
    """
    Fuse rankings across multiple ranked lists, but score at SOURCE_ID level.
    This prevents many chunks from one page dominating the top-k.

    Returns: ranked list of source_ids (length <= k_sources)
    """
    fused: Dict[str, float] = {}

    for rr in results_lists:
        for rank, item in enumerate(rr, start=1):
            sid = item.get("source_id")
            if not sid:
                continue
            sid = str(sid)
            if _is_low_quality_source_id(sid):
                continue
            fused[sid] = fused.get(sid, 0.0) + (1.0 / (rrf_k + rank))

    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[: max(1, k_sources)]
    return [sid for sid, _ in ranked]


# -------------------------------------------------------
# MAIN RETRIEVE — Phase 3D.2 (rewrite + deeper BM25 pool + source-aware hybrid_rrf)
# -------------------------------------------------------
def retrieve(
    query: str,
    top_k: int | None = None,
    mode: str = "vector",
) -> Tuple[float, List[Dict[str, Any]]]:
    if not (query or "").strip():
        return 0.0, []

    k = int(top_k or TOP_K)
    mode = (mode or "vector").lower().strip()

    # ✅ deterministic query rewrites (safe fallback)
    try:
        rewrites = rewrite_query(query) or [query]
    except Exception:
        rewrites = [query]
    rewrites = _dedupe_keep_order([str(x) for x in rewrites])

    # ---------------------------------------------------
    # BM25 mode (merge across rewrites, dedupe chunks)
    # ---------------------------------------------------
    if mode == "bm25":
        # Phase 3D.2 — deeper BM25 for stability
        bm25_depth = max(200, k * 20)

        best = 0.0
        merged: List[Dict[str, Any]] = []
        for rq in rewrites:
            b, r = _bm25_retrieve(rq, bm25_depth)
            best = max(best, b)
            merged.extend(r)

        merged = _dedupe_hits_by_chunk_keep_order(merged)
        merged = [h for h in merged if not _is_low_quality_source_id(str(h.get("source_id") or ""))]
        return float(best), merged[:k]

    # ---------------------------------------------------
    # ✅ HYBRID RRF mode — rewrite-fused + SOURCE-aware + (Phase 3D.2) deeper BM25 pool
    # ---------------------------------------------------
    if mode in ("hybrid_rrf", "hybrid", "rrf"):
        # 🔑 Phase 3D.2 tuning — deeper candidate pools
        bm25_depth = max(200, k * 20)     # must exceed “rank ~48” type cases
        vector_depth = max(80, k * 8)
        fuse_k = max(200, k * 20)         # how big per-rewrite fused list is

        per_rewrite_fused: List[List[Dict[str, Any]]] = []
        best_vec = 0.0

        # 1) per-rewrite: vector + bm25 -> chunk-level rrf list
        for rq in rewrites:
            v_best, v_res = _vector_retrieve(rq, vector_depth)
            _, b_res = _bm25_retrieve(rq, bm25_depth)
            best_vec = max(best_vec, v_best)

            per = _rrf_fuse(v_res, b_res, k=fuse_k, rrf_k=60)
            per_rewrite_fused.append(per)

        # 2) fuse across rewrites at SOURCE level
        k_sources = max(200, k * 20)
        top_sources = _source_rrf_fuse(per_rewrite_fused, k_sources=k_sources, rrf_k=60)

        # 3) expand back to chunks: pick best chunk per source (first seen = highest rank)
        best_chunk_by_source: Dict[str, Dict[str, Any]] = {}
        for rr in per_rewrite_fused:
            for item in rr:
                sid = str(item.get("source_id") or "")
                if not sid:
                    continue
                if _is_low_quality_source_id(sid):
                    continue
                if sid not in best_chunk_by_source:
                    best_chunk_by_source[sid] = item

        out: List[Dict[str, Any]] = []
        for sid in top_sources:
            h = best_chunk_by_source.get(sid)
            if h:
                out.append(h)
            if len(out) >= k:
                break

        out = _dedupe_hits_by_chunk_keep_order(out)
        out = [h for h in out if not _is_low_quality_source_id(str(h.get("source_id") or ""))]
        return float(best_vec), out[:k]

    # ---------------------------------------------------
    # VECTOR default (merge across rewrites, dedupe chunks)
    # ---------------------------------------------------
    best = 0.0
    merged: List[Dict[str, Any]] = []
    for rq in rewrites:
        b, r = _vector_retrieve(rq, k)
        best = max(best, b)
        merged.extend(r)

    merged = _dedupe_hits_by_chunk_keep_order(merged)
    merged = [h for h in merged if not _is_low_quality_source_id(str(h.get("source_id") or ""))]
    return float(best), merged[:k]


# -------------------------------------------------------
# Fail-closed gate
# -------------------------------------------------------
def fail_closed(best_score: float) -> bool:
    return float(best_score) < RAG_MIN_SCORE
