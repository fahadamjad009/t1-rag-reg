# app/rag/extractive.py
# FULL REPLACEMENT — FINAL STABLE VERSION (Phase 3A)
#
# Guarantees:
# - Deterministic extractive answering
# - Correct definitional handling for “What is / Define / Meaning of” (GENERAL, not just KYC)
# - Strict fail-closed for weak definitions (no filler)
# - MUST-INCLUDE enforcement + completion (including 0-overlap fallback excerpt)
# - Stable citation schema (unchanged):
#     rank/source_id/doc_id/chunk_id/title/url/evidence/is_must
# - Safe to run in Docker / API / eval pipeline
#
# Notes:
# - Definition logic:
#     * Detect definition-style questions
#     * Extract the term being defined ("What is KYC?" -> "KYC")
#     * Build dynamic definitional patterns for that term (plus acronym expansions)
#     * Score = overlap + must_boost + def_bonus + 0.1*retrieval_score
#     * Fail-closed if no definitional sentence is found
#
# - MUST-INCLUDE:
#     * Enforcement: if any must hits exist, restrict extraction to must hits
#     * Completion: if must sources not cited, add citations from their best sentence or excerpt
#
# - Citation capping:
#     * Deduplicate while preserving order
#     * Keep must citations first
#     * Fill remaining slots with non-must

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

_WORD_RE = re.compile(r"[a-z0-9]{3,}", re.IGNORECASE)


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


def _query_terms(query: str) -> List[str]:
    return _tokenize(query)


# --------------------------------------------------
# Definition query detection (GENERAL)
# --------------------------------------------------

_DEF_QUERY_PAT = re.compile(r"^\s*(what is|define|meaning of)\b", re.IGNORECASE)


def _is_definition_query(q: str) -> bool:
    return bool(_DEF_QUERY_PAT.search(q or ""))


def _extract_def_term(q: str) -> str:
    """
    Pull the term being defined from queries like:
      "What is KYC?" / "Define KYC" / "Meaning of KYC"
    Returns "" if not detected.
    """
    s = (q or "").strip()
    m = re.match(r"^\s*(what is|define|meaning of)\s+(.*)$", s, flags=re.IGNORECASE)
    if not m:
        return ""
    tail = (m.group(2) or "").strip()
    tail = re.sub(r"[?!.]+$", "", tail).strip()
    tail = tail.strip("\"'“”")
    toks = tail.split()
    return " ".join(toks[:6]).strip()  # avoid capturing whole sentences


# Optional domain expansions (AUSTRAC/common compliance acronyms)
_ACRONYM_EXPANSIONS = {
    "KYC": "know your customer",
    "AML": "anti-money laundering",
    "CTF": "counter-terrorism financing",
    "AUSTRAC": "australian transaction reports and analysis centre",
}


def _build_def_patterns(term: str) -> List[Tuple[re.Pattern, float]]:
    """
    Build definitional patterns for the extracted term.
    Includes generic "X is/means/refers to/stands for" and acronym expansion patterns.
    """
    t = (term or "").strip()
    if not t:
        return []

    t_esc = re.escape(t)

    patterns: List[Tuple[re.Pattern, float]] = []

    # Generic: "TERM is/means/refers to/stands for ..."
    patterns.append((re.compile(rf"\b{t_esc}\b\s+(is|means|refers to|stands for)\b", re.IGNORECASE), 0.9))

    # Definition style headers: "TERM: ..." or "TERM - ..."
    patterns.append((re.compile(rf"^\s*\b{t_esc}\b\s*[:\-]\s+", re.IGNORECASE), 0.7))

    # Parenthetical expansions:
    # "Long phrase (TERM)" OR "TERM (long phrase)"
    patterns.append((re.compile(rf"\b{t_esc}\b\s*\(\s*[^){{}}]{{3,80}}\)", re.IGNORECASE), 1.0))
    patterns.append((re.compile(rf"[^(){{}}]{{3,80}}\(\s*\b{t_esc}\b\s*\)", re.IGNORECASE), 1.0))

    # If term looks like an acronym, add known expansions where available
    if t.upper() == t and 2 <= len(t) <= 16:
        exp = _ACRONYM_EXPANSIONS.get(t.upper())
        if exp:
            exp_esc = re.escape(exp)
            patterns.append((re.compile(rf"\b{t_esc}\b.*\b{exp_esc}\b", re.IGNORECASE), 1.0))
            patterns.append((re.compile(rf"\b{exp_esc}\b.*\b{t_esc}\b", re.IGNORECASE), 1.0))
            patterns.append((re.compile(rf"\b{exp_esc}\b\s*\(\s*\b{t_esc}\b\s*\)", re.IGNORECASE), 1.0))
            patterns.append((re.compile(rf"\b{t_esc}\b\s*\(\s*\b{exp_esc}\b\s*\)", re.IGNORECASE), 1.0))

    return patterns


def _definition_bonus(sentence: str, term: str) -> float:
    s = sentence or ""
    for pat, bonus in _build_def_patterns(term):
        if pat.search(s):
            return float(bonus)
    return 0.0


# --------------------------------------------------
# Cleaning + sentence splitting
# --------------------------------------------------

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


# --------------------------------------------------
# Scoring
# --------------------------------------------------

def _score_sentence(sent: str, query_terms: List[str]) -> float:
    """
    Token overlap score.
    Returns 0.0 if there is zero overlap.
    """
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


# --------------------------------------------------
# Citation helpers (stable schema)
# --------------------------------------------------

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


# --------------------------------------------------
# Core extractive logic
# --------------------------------------------------

def extractive_answer(
    query: str,
    hits: List[Dict[str, Any]],
    max_sentences: int = 2,
    must_include: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Returns:
      {
        "answer": str,
        "citations": [ ... ],
        "strict_fail_closed": bool,
        "rationale": str,
      }
    """
    terms = _query_terms(query)
    if not terms or not hits:
        return {"answer": "", "citations": [], "strict_fail_closed": True, "rationale": "no_terms_or_hits"}

    is_def_q = _is_definition_query(query)
    def_term = _extract_def_term(query) if is_def_q else ""

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

    # candidates tuple:
    # (final_score, hit_rank, def_bonus, sent, hit)
    candidates: List[Tuple[float, int, float, str, Dict[str, Any]]] = []

    for hit_rank, h in enumerate(filtered_hits, start=1):
        raw_text = h.get("text") or ""
        if not isinstance(raw_text, str) or not raw_text.strip():
            continue

        text = _clean_text(raw_text)

        sid = _hit_source_id(h)
        is_must = bool(sid and must_set and sid in must_set)

        # retrieval_score: whatever retriever provided (safe default 0.0)
        retrieval_score = 0.0
        try:
            retrieval_score = float(h.get("score") or 0.0)
        except Exception:
            retrieval_score = 0.0

        for sent in _sentences(text):
            overlap_score = _score_sentence(sent, terms)

            # For definition queries, require overlap to avoid random “definitions”
            # from unrelated contexts.
            if overlap_score <= 0:
                continue

            must_boost = 2.0 if is_must else 0.0
            def_bonus = _definition_bonus(sent, def_term) if is_def_q else 0.0

            final_score = float(overlap_score) + float(must_boost) + float(def_bonus) + 0.1 * float(retrieval_score)
            candidates.append((final_score, hit_rank, def_bonus, sent, h))

    # If definition query: do NOT fall back to random excerpt. Fail closed.
    if not candidates:
        if is_def_q:
            # Optionally: include MUST completion citations (citations only) here.
            return {
                "answer": "I couldn't find a direct definition in the retrieved sources.",
                "citations": [],
                "strict_fail_closed": True,
                "rationale": "no_candidate_sentences_for_definition",
            }

        # Non-definition fallback: cite excerpt from top hit
        h0 = hits[0]
        ev = h0.get("text") or ""
        ev = ev if isinstance(ev, str) else str(ev)
        ev = _cap_answer(_clean_text(ev)[:280])
        return {
            "answer": ev,
            "citations": [_citation_from_sentence(ev, h0, 1, must_set)],
            "strict_fail_closed": False,
            "rationale": "fallback_excerpt_no_overlap_sentences",
        }

    candidates.sort(key=lambda x: (-x[0], x[1]))

    # Choose unique sentences
    chosen: List[Tuple[str, Dict[str, Any], int, float]] = []  # (sent, hit, rank, def_bonus)
    seen = set()
    for _score, hit_rank, def_bonus, sent, h in candidates:
        key = sent.lower()
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sent, h, hit_rank, def_bonus))
        if len(chosen) >= max_sentences:
            break

    # Definition fail-closed: require at least one definitional sentence
    # We accept 0.7+ as "definition-like" (is/means/refers/parenthetical expansion).
    if is_def_q:
        has_definition = any((db or 0.0) >= 0.7 for _, _, _, db in chosen)
        if not has_definition:
            # Still do MUST completion citations (citations only), but do not answer with filler.
            cits: List[Dict[str, Any]] = []
            cited_sources: set[str] = set()

            if must_set:
                missing = [sid for sid in must_set if sid not in cited_sources]
                for miss_sid in missing:
                    best: Optional[Tuple[float, int, str, Dict[str, Any]]] = None
                    for hr, hh in enumerate(hits, start=1):
                        sid = _hit_source_id(hh)
                        if sid != miss_sid:
                            continue
                        raw = hh.get("text") or ""
                        if not isinstance(raw, str) or not raw.strip():
                            continue
                        text2 = _clean_text(raw)
                        sents = _sentences(text2)
                        for s in sents:
                            sc = _score_sentence(s, terms)
                            if best is None or sc > best[0]:
                                best = (float(sc), hr, s, hh)
                        if best is None:
                            excerpt = (text2[:280] or "").strip()
                            if excerpt:
                                best = (0.0, hr, excerpt, hh)
                    if best:
                        _, hr, s, hh = best
                        cits.append(_citation_from_sentence(s, hh, hr, must_set))

            return {
                "answer": "I couldn't find a direct definition in the retrieved sources.",
                "citations": cits,
                "strict_fail_closed": True,
                "rationale": "no_definitional_sentence_found",
            }

    answer_text = _cap_answer(" ".join(s for s, _, _, _ in chosen).strip())

    citations: List[Dict[str, Any]] = []
    cited_sources: set[str] = set()

    for sent, h, hit_rank, _db in chosen:
        sid = _hit_source_id(h)
        if sid:
            cited_sources.add(sid)
        citations.append(_citation_from_sentence(sent, h, hit_rank, must_set))

    # MUST-INCLUDE COMPLETION: add missing must citations (do not append to answer)
    # completion cites even when token overlap is 0 (fallback excerpt)
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

                text2 = _clean_text(raw)

                # 1) Prefer best-overlap sentence; allow 0-overlap selection by "pick best seen"
                sents = _sentences(text2)
                for sent in sents:
                    sc = _score_sentence(sent, terms)  # may be 0.0
                    if best is None or sc > best[0]:
                        best = (float(sc), hit_rank, sent, h)

                # 2) If no usable sentences, fall back to excerpt
                if best is None:
                    excerpt = (text2[:280] or "").strip()
                    if excerpt:
                        best = (0.0, hit_rank, excerpt, h)

            if best:
                _, hit_rank, sent, h = best
                citations.append(_citation_from_sentence(sent, h, hit_rank, must_set))
                cited_sources.add(miss_sid)

    return {"answer": answer_text, "citations": citations, "strict_fail_closed": False, "rationale": "ok"}


# --------------------------------------------------
# Public wrapper (citation capping)
# --------------------------------------------------

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
