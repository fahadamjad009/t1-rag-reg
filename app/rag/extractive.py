from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(sent: str, query_terms: List[str]) -> int:
    s = sent.lower()
    return sum(1 for t in query_terms if t and t in s)


def extractive_answer(query: str, hits: List[Dict[str, Any]], max_sentences: int = 2) -> Dict[str, Any]:
    q = (query or "").strip()
    terms = [t for t in re.findall(r"[a-zA-Z0-9]{3,}", q.lower())]
    if not terms or not hits:
        return {"answer": "", "citations": []}

    candidates: List[Tuple[int, int, str, Dict[str, Any]]] = []
    # (score, hit_rank, sentence, hit)

    for i, h in enumerate(hits, start=1):
        text = (h.get("text") or "")
        for sent in _sentences(text):
            sc = _score_sentence(sent, terms)
            if sc > 0:
                candidates.append((sc, i, sent, h))

    if not candidates:
        h0 = hits[0]
        ev = (h0.get("text") or "")[:250]
        return {
            "answer": ev,
            "citations": [{
                "rank": 1,
                "source_id": h0.get("source_id"),  # ⭐ added
                "doc_id": h0.get("doc_id"),
                "chunk_id": h0.get("chunk_id"),
                "title": h0.get("title"),
                "url": h0.get("url"),
                "evidence": ev
            }]
        }

    candidates.sort(key=lambda x: (-x[0], x[1]))

    chosen: List[Tuple[str, Dict[str, Any], int]] = []
    seen = set()

    for sc, hit_rank, sent, h in candidates:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sent, h, hit_rank))
        if len(chosen) >= max_sentences:
            break

    answer_text = " ".join(s for s, _, _ in chosen).strip()

    citations = []
    for sent, h, hit_rank in chosen:
        citations.append({
            "rank": hit_rank,
            "source_id": h.get("source_id"),  # ⭐ added
            "doc_id": h.get("doc_id"),
            "chunk_id": h.get("chunk_id"),
            "title": h.get("title"),
            "url": h.get("url"),
            "evidence": sent[:300],
        })

    return {"answer": answer_text, "citations": citations}


# -------------------------------------------------------------------
# Compatibility wrapper for Phase 3A wiring (used by run_eval.py)
# -------------------------------------------------------------------
def answer_from_hits(query: str, hits: List[Dict[str, Any]], max_sentences: int = 2) -> Dict[str, Any]:
    """
    run_eval.py expects answer_from_hits(). Keep this thin wrapper so eval wiring
    doesn't require refactoring other modules.
    """
    return extractive_answer(query=query, hits=hits, max_sentences=max_sentences)
