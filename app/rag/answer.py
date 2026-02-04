# app/rag/answer.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass
class ExtractiveAnswer:
    answer: str
    confidence: float
    rationale: str
    evidence: List[Dict[str, Any]]  # normalized evidence objects
    citations: List[Dict[str, Any]]  # sentence-level citations


_WORD_RE = re.compile(r"[A-Za-z0-9]+")
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    sents = _SENT_SPLIT_RE.split(text)
    return [s.strip() for s in sents if s and s.strip()]


def _jaccard(a: List[str], b: List[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def _first_present(d: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in d and d[k] not in (None, "", []):
            return d[k]
    return None


def normalize_hit(hit: Dict[str, Any], i: int) -> Dict[str, Any]:
    """
    Normalizes varying retriever schemas into a stable evidence shape.
    Update key preferences here once you settle your retriever output.
    """
    meta = hit.get("metadata") or {}

    text = _first_present(hit, ["text", "chunk", "content", "passage"]) or _first_present(
        meta, ["text", "chunk", "content", "passage"]
    ) or ""

    score = _first_present(hit, ["score", "similarity", "cosine", "rank_score"]) or _first_present(
        meta, ["score", "similarity", "cosine", "rank_score"]
    )

    source = _first_present(hit, ["source", "doc_id", "document", "file", "url", "path"]) or _first_present(
        meta, ["source", "doc_id", "document", "file", "url", "path"]
    ) or f"doc_{i}"

    title = _first_present(hit, ["title", "doc_title"]) or _first_present(meta, ["title", "doc_title"])
    section = _first_present(hit, ["section", "heading"]) or _first_present(meta, ["section", "heading"])
    page = _first_present(hit, ["page", "page_number"]) or _first_present(meta, ["page", "page_number"])
    chunk_id = _first_present(hit, ["chunk_id", "id"]) or _first_present(meta, ["chunk_id", "id"]) or i

    return {
        "id": str(chunk_id),
        "source": str(source),
        "title": title,
        "section": section,
        "page": page,
        "score": float(score) if score is not None else None,
        "text": str(text),
        "raw": hit,  # keep for debugging; remove later if you want
    }


def answer_extractive(
    query: str,
    hits: List[Dict[str, Any]],
    max_sentences: int = 3,
    min_sentence_score: float = 0.08,
) -> ExtractiveAnswer:
    q_tokens = _tokenize(query)
    if not q_tokens or not hits:
        return ExtractiveAnswer(
            answer="",
            confidence=0.0,
            rationale="no_query_or_hits",
            evidence=[],
            citations=[],
        )

    # Normalize top hits
    norm = [normalize_hit(h, i) for i, h in enumerate(hits[:10])]

    # Build sentence candidates with linkage to evidence index
    candidates: List[Tuple[float, str, int]] = []  # (score, sentence, evidence_idx)
    for ei, ev in enumerate(norm):
        for sent in _sentence_split(ev["text"]):
            s_tokens = _tokenize(sent)
            score = _jaccard(q_tokens, s_tokens)
            if score > 0:
                candidates.append((score, sent, ei))

    if not candidates:
        return ExtractiveAnswer(
            answer="",
            confidence=0.0,
            rationale="no_candidate_sentences",
            evidence=norm[:5],
            citations=[],
        )

    candidates.sort(key=lambda x: x[0], reverse=True)

    selected: List[Tuple[float, str, int]] = []
    for score, sent, ei in candidates:
        if score < min_sentence_score:
            break
        # de-dup sentences
        if any(_jaccard(_tokenize(sent), _tokenize(s2)) > 0.7 for _, s2, _ in selected):
            continue
        selected.append((score, sent, ei))
        if len(selected) >= max_sentences:
            break

    # Allow multi-chunk synthesis if multiple weak but relevant sentences exist
    if not selected:
        unique_chunks = set(ei for _, _, ei in candidates)

        if len(unique_chunks) >= 2:
           # pick top sentence per chunk
           per_chunk = {}
           for score, sent, ei in candidates:
               if ei not in per_chunk:
                per_chunk[ei] = (score, sent)
               if len(per_chunk) >= max_sentences:
                   break

           selected = [(s, sent, ei) for ei, (s, sent) in per_chunk.items()]

        if not selected:
            return ExtractiveAnswer(
            answer="",
            confidence=0.0,
            rationale="multi_chunk_candidates_insufficient",
            evidence=norm[:5],
            citations=[],
        )


    # Confidence: top overlap + evidence spread
    top_score = selected[0][0]
    spread = len(set(ei for _, _, ei in selected)) / max(1, len(norm))
    confidence = max(0.0, min(1.0, 0.75 * top_score + 0.25 * spread))

    # Build answer with sentence-level citations
    citations: List[Dict[str, Any]] = []
    answer_sents: List[str] = []
    for idx, (score, sent, ei) in enumerate(selected, start=1):
        ev = norm[ei]
        cite = {
            "sentence_index": idx,
            "evidence_id": ev["id"],
            "source": ev["source"],
            "title": ev["title"],
            "section": ev["section"],
            "page": ev["page"],
            "score": float(score),
        }
        citations.append(cite)
        # append inline marker [1], [2], ...
        answer_sents.append(f"{sent} [{idx}]")

    # Evidence subset used
    used_ei = sorted(set(ei for _, _, ei in selected))
    used_evidence = [norm[i] for i in used_ei]

    return ExtractiveAnswer(
        answer=" ".join(answer_sents).strip(),
        confidence=confidence,
        rationale="extractive_overlap_with_citations",
        evidence=used_evidence,
        citations=citations,
    )
