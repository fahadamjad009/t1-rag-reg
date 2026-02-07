from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import List, Dict, Any, Tuple

from rank_bm25 import BM25Okapi


@dataclass
class BM25Index:
    bm25: BM25Okapi
    doc_ids: List[str]
    tokenized_docs: List[List[str]]


def _simple_tokenize(text: str) -> List[str]:
    """
    Lightweight + deterministic; dependency-free tokenizer.
    Good enough for baseline BM25; we can upgrade later.
    """
    text = (text or "").lower()
    out: List[str] = []
    cur: List[str] = []
    for ch in text:
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                out.append("".join(cur))
                cur = []
    if cur:
        out.append("".join(cur))
    return out


def build_bm25_from_chunks(chunks_path: str | Path) -> BM25Index:
    """
    Build BM25 index from data/index/chunks.json produced by your ingest/index step.

    Your chunks.json is a LIST of dicts like:
      {
        "chunk_id": "...",
        "doc_id": "...",
        "source_id": "...",
        "url": "...",
        "title": "...",
        "text": "..."
      }

    Also supports {"chunks":[...]} just in case.
    """
    chunks_path = Path(chunks_path)
    data = json.loads(chunks_path.read_text(encoding="utf-8"))

    # Accept either {"chunks":[...]} or raw list
    chunks: List[Dict[str, Any]] = data["chunks"] if isinstance(data, dict) and "chunks" in data else data
    if not isinstance(chunks, list):
        raise RuntimeError(f"Expected list (or {{'chunks':[...]}}) in {chunks_path}, got {type(chunks)}")

    doc_ids: List[str] = []
    tokenized_docs: List[List[str]] = []

    for c in chunks:
        # FIX: your file uses chunk_id, not id
        cid = str(c.get("chunk_id") or c.get("id") or c.get("doc_id") or "")
        txt = str(c.get("text", ""))

        if not cid or not txt.strip():
            continue

        doc_ids.append(cid)
        tokenized_docs.append(_simple_tokenize(txt))

    if not doc_ids:
        # helpful debug: show what keys exist in first element (if any)
        sample_keys = list(chunks[0].keys()) if chunks else []
        raise RuntimeError(
            f"No BM25 documents found in {chunks_path}. "
            f"Check that each item has non-empty 'text' and an id field like 'chunk_id'. "
            f"Sample keys={sample_keys}"
        )

    bm25 = BM25Okapi(tokenized_docs)
    return BM25Index(bm25=bm25, doc_ids=doc_ids, tokenized_docs=tokenized_docs)


def bm25_search(index: BM25Index, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
    """
    Returns list of (chunk_id, score) sorted desc.
    """
    q_tokens = _simple_tokenize(query)
    if not q_tokens:
        return []

    scores = index.bm25.get_scores(q_tokens)  # numpy array-like
    ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[: max(1, top_k)]
    return [(index.doc_ids[i], float(scores[i])) for i in ranked]
