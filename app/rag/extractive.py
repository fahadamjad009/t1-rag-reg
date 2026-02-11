from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional


def _sentences(text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [p.strip() for p in parts if p.strip()]


def _score_sentence(sent: str, query_terms: List[str]) -> int:
    s = sent.lower()
    return sum(1 for t in query_terms if t and t in s)


def extractive_answer(
    query: str,
    hits: List[Dict[str, Any]],
    max_sentences: int = 2,
    must_include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Build a grounded, purely extractive answer by selecting top-matching sentences
    from retrieved hits.

    If must_include is provided, we PRIORITIZE selecting evidence from hits whose
    source_id/doc_id is in must_include. This directly improves answer_coverage.
    """
    q = (query or "").strip()
    terms = [t for t in re.findall(r"[a-zA-Z0-9]{3,}", q.lower())]
    if not terms or not hits:
        return {"answer": "", "citations": []}

    must_set = set([str(x).strip() for x in (must_include or []) if str(x).strip()])

    def _hit_id(h: Dict[str, Any]) -> str:
        sid = h.get("source_id")
        if isinstance(sid, str) and sid.strip():
            return sid.strip()
        did = h.get("doc_id")
        if isinstance(did, str) and did.strip():
            return did.strip()
        return ""

    # Candidate tuple:
    # (is_must_hit (1/0), score, hit_rank, sentence, hit)
    candidates: List[Tuple[int, int, int, str, Dict[str, Any]]] = []

    for i, h in enumerate(hits, start=1):
        text = (h.get("text") or "")
        hid = _hit_id(h)
        is_must = 1 if (hid and hid in must_set) else 0

        for sent in _sentences(text):
            sc = _score_sentence(sent, terms)
            if sc > 0:
                candidates.append((is_must, sc, i, sent, h))

    # If we found no matching sentences, fall back to a short excerpt from the best must_include hit if possible
    if not candidates:
        # try must_include hit first
        if must_set:
            for h in hits:
                if _hit_id(h) in must_set:
                    ev = (h.get("text") or "")[:250]
                    return {
                        "answer": ev,
                        "citations": [{
                            "rank": 1,
                            "source_id": h.get("source_id"),
                            "doc_id": h.get("doc_id"),
                            "chunk_id": h.get("chunk_id"),
                            "title": h.get("title"),
                            "url": h.get("url"),
                            "evidence": ev,
                        }],
                    }

        # else normal fallback: top hit
        h0 = hits[0]
        ev = (h0.get("text") or "")[:250]
        return {
            "answer": ev,
            "citations": [{
                "rank": 1,
                "source_id": h0.get("source_id"),
                "doc_id": h0.get("doc_id"),
                "chunk_id": h0.get("chunk_id"),
                "title": h0.get("title"),
                "url": h0.get("url"),
                "evidence": ev,
            }],
        }

    # Sort:
    # 1) must_include hits first (is_must desc)
    # 2) higher sentence score
    # 3) lower hit rank
    candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))

    chosen: List[Tuple[str, Dict[str, Any], int]] = []
    seen = set()

    for is_must, sc, hit_rank, sent, h in candidates:
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
            "source_id": h.get("source_id"),
            "doc_id": h.get("doc_id"),
            "chunk_id": h.get("chunk_id"),
            "title": h.get("title"),
            "url": h.get("url"),
            "evidence": sent[:300],
        })

    return {"answer": answer_text, "citations": citations}


# -------------------------------------------------------------------
# Compatibility wrapper for Phase 3A wiring
# -------------------------------------------------------------------
def answer_from_hits(
    query: str,
    hits: List[Dict[str, Any]],
    max_sentences: int = 2,
    must_include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Used by app.eval.run_eval._answer(). We keep it stable, but allow must_include.
    """
    return extractive_answer(query=query, hits=hits, max_sentences=max_sentences, must_include=must_include)
