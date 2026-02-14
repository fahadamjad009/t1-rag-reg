# app/rag/extractive.py
# FULL REPLACEMENT
# Includes:
# - boilerplate cleaning
# - better sentence split
# - token overlap scoring
# - MUST-INCLUDE enforcement (restrict to must hits when present)
# - MUST-INCLUDE completion (ensure missing must sources get cited if retrievable)
#   ✅ FIX: completion now cites even when token overlap is 0 (fallback excerpt)
# - stronger must_include bias
# - stable citation schema (rank/source_id/doc_id/chunk_id/title/url/evidence/is_must)
# - citation capping that keeps must citations (unique) first

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


_WORD_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)

_BOILERPLATE_PATTERNS = [
    r"\bSkip to main content\b",
    r"\bSecondary navigation\b",
    r"\bSubscribe to InBrief\b",
    r"\bInBrief\b",
    r"\bMenu\b",
    r"\bClose Search\b",
    r"\bSearch Close\b",
    r"\bClose\b",
    r"\bCareers\b",
    r"\bContact us\b",
    r"\bAUSTRAC Online\b",
    r"\bPrint PDF\b",
    r"\bHome\b",
    r"\bBusiness\b",
    r"\bNew to AUSTRAC\b",
    r"\bLatest updates\b",
    r"\bFor journalists\b",
    r"\bAbout us\b",
    r"\bWhat we do\b",
    r"\bAnnual reports\b",
    r"\bReports and accountability\b",
    r"\bCorporate information\b",
    r"\bFreedom of information\b",
    r"\bCorporate plan\b",
    r"\bCheck if you need to enrol or register\b",
    r"\bEnrol or register\b",
    r"\bWho and what we regulate\b",
    r"\bYour obligations\b",
    r"\bE-learning\b",
    r"\bYour industry\b",
]


def _clean_text(text: str) -> str:
    t = (text or "").replace("\u00a0", " ")
    t = re.sub(r"\s+", " ", t).strip()
    for pat in _BOILERPLATE_PATTERNS:
        t = re.sub(pat, " ", t, flags=re.IGNORECASE)
    t = re.sub(r"[•●▪■]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _sentences(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []

    # Primary: punctuation boundaries and a likely new sentence start.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", t)

    # Fallback: blob without punctuation, split on multi-space.
    if len(parts) <= 1:
        parts = re.split(r"\s{2,}", t)

    out: List[str] = []
    for p in parts:
        p = p.strip(" -•\t\r\n")
        if len(p) < 40:
            continue
        if len(p) > 520:
            p = p[:520].rsplit(" ", 1)[0] + "…"
        out.append(p)
    return out


def _query_terms(query: str) -> List[str]:
    q = (query or "").strip().lower()
    return [t.lower() for t in _WORD_RE.findall(q) if t]


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(s or "")]


def _score_sentence(sent: str, query_terms: List[str]) -> float:
    if not sent or not query_terms:
        return 0.0
    s_tokens = set(_tokenize(sent))
    if not s_tokens:
        return 0.0
    q_tokens = set(query_terms)
    overlap = len(s_tokens & q_tokens)
    if overlap <= 0:
        return 0.0
    length_penalty = min(len(sent) / 220.0, 2.0)
    return overlap / length_penalty


def _hit_source_id(hit: Dict[str, Any]) -> str:
    sid = hit.get("source_id")
    if isinstance(sid, str) and sid.strip():
        return sid.strip()
    did = hit.get("doc_id")
    if isinstance(did, str) and did.strip():
        return did.strip()
    return ""


def _cap_answer(text: str, max_chars: int = 600) -> str:
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def _citation_from_sentence(
    sent: str,
    h: Dict[str, Any],
    hit_rank: int,
    must_set: set[str],
) -> Dict[str, Any]:
    sid = _hit_source_id(h)
    return {
        "rank": hit_rank,
        "source_id": h.get("source_id"),
        "doc_id": h.get("doc_id"),
        "chunk_id": h.get("chunk_id"),
        "title": h.get("title"),
        "url": h.get("url"),
        "evidence": (sent or "")[:300],
        "is_must": bool(sid and sid in must_set) if must_set else False,
    }


def extractive_answer(
    query: str,
    hits: List[Dict[str, Any]],
    max_sentences: int = 2,
    must_include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    terms = _query_terms(query)
    if not terms or not hits:
        return {"answer": "", "citations": []}

    must_set: set[str] = set()
    if isinstance(must_include, list):
        must_set = {str(x).strip() for x in must_include if str(x).strip()}

    # MUST-INCLUDE ENFORCEMENT: restrict extraction to must hits if any present
    filtered_hits = hits
    if must_set:
        must_hits: List[Dict[str, Any]] = []
        for h in hits:
            sid = _hit_source_id(h)
            if sid and sid in must_set:
                must_hits.append(h)
        if must_hits:
            filtered_hits = must_hits

    candidates: List[Tuple[float, int, str, Dict[str, Any]]] = []

    for hit_rank, h in enumerate(filtered_hits, start=1):
        text = h.get("text") or ""
        if not isinstance(text, str) or not text.strip():
            continue

        sid = _hit_source_id(h)
        is_must = bool(sid and must_set and sid in must_set)

        text = _clean_text(text)

        for sent in _sentences(text):
            base = _score_sentence(sent, terms)
            if base <= 0:
                continue
            boost = 2.0 if is_must else 0.0
            candidates.append((float(base) + boost, hit_rank, sent, h))

    # Fallback: no matching sentences -> cite excerpt from top hit
    if not candidates:
        h0 = hits[0]
        ev = h0.get("text") or ""
        ev = ev if isinstance(ev, str) else str(ev)
        ev = _clean_text(ev)[:280]
        ev = _cap_answer(ev)
        return {
            "answer": ev,
            "citations": [_citation_from_sentence(ev, h0, 1, must_set)],
        }

    candidates.sort(key=lambda x: (-x[0], x[1]))

    chosen: List[Tuple[str, Dict[str, Any], int]] = []
    seen = set()
    for score, hit_rank, sent, h in candidates:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sent, h, hit_rank))
        if len(chosen) >= max_sentences:
            break

    answer_text = _cap_answer(" ".join(s for s, _, _ in chosen).strip())

    citations: List[Dict[str, Any]] = []
    cited_sources: set[str] = set()

    for sent, h, hit_rank in chosen:
        sid = _hit_source_id(h)
        if sid:
            cited_sources.add(sid)
        citations.append(_citation_from_sentence(sent, h, hit_rank, must_set))

    # MUST-INCLUDE COMPLETION: add missing must citations (do not append to answer)
    # ✅ FIX: completion now cites even when token overlap is 0 (fallback excerpt)
    if must_set:
        missing = [sid for sid in must_set if sid not in cited_sources]
        for miss_sid in missing:
            best: Optional[Tuple[float, int, str, Dict[str, Any]]] = None

            for hit_rank, h in enumerate(hits, start=1):
                sid = _hit_source_id(h)
                if sid != miss_sid:
                    continue

                raw = h.get("text") or ""
                if not isinstance(raw, str) or not raw.strip():
                    continue

                text = _clean_text(raw)

                # 1) Prefer best-overlap sentence; allow 0-overlap selection by "pick first"
                sents = _sentences(text)
                for sent in sents:
                    sc = _score_sentence(sent, terms)  # may be 0.0
                    if best is None or sc > best[0]:
                        best = (float(sc), hit_rank, sent, h)

                # 2) If no usable sentences, fall back to excerpt
                if best is None:
                    excerpt = (text[:280] or "").strip()
                    if excerpt:
                        best = (0.0, hit_rank, excerpt, h)

            if best:
                _, hit_rank, sent, h = best
                citations.append(_citation_from_sentence(sent, h, hit_rank, must_set))
                cited_sources.add(miss_sid)

    return {"answer": answer_text, "citations": citations}


def answer_from_hits(
    query: str,
    hits: List[Dict[str, Any]],
    max_sentences: int = 2,
    must_include: Optional[List[str]] = None,
    max_citations: int = 5,
) -> Dict[str, Any]:
    """
    Cap citations without dropping must_include coverage:
    - de-dupe by source_id/doc_id while preserving order
    - keep must citations first (unique sources)
    - then fill remaining slots with best non-must
    """
    out = extractive_answer(
        query=query,
        hits=hits,
        max_sentences=max_sentences,
        must_include=must_include,
    )

    cits = out.get("citations") or []
    if not isinstance(cits, list) or max_citations is None:
        return out

    # de-dupe by source_id/doc_id while preserving order
    seen_keys: set[str] = set()
    deduped: List[Dict[str, Any]] = []
    for c in cits:
        if not isinstance(c, dict):
            continue
        sid = str(c.get("source_id") or c.get("doc_id") or "").strip()
        key = sid or (str(c.get("evidence") or "")[:40] or "")
        if not key:
            continue
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(c)

    # must-first
    must_first = [c for c in deduped if c.get("is_must") is True]
    non_must = [c for c in deduped if c.get("is_must") is not True]

    capped = must_first[:max_citations]
    if len(capped) < max_citations:
        capped.extend(non_must[: (max_citations - len(capped))])

    out["citations"] = capped
    return out
