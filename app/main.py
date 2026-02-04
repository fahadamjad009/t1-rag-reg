from __future__ import annotations

import os
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.rag.retriever import retrieve
from app.rag.answer import answer_extractive  # Step 1C.2: extractive + citations

app = FastAPI(
    title="T1-RAG-REG",
    version="0.1.0",
    description="Regulatory RAG system with evaluation-first design",
)

# ============================================================
# Fail-closed gate configuration
# ============================================================
# Default is strict and safe.
# For demo/dev you can override via env:
#   setx STRICT_FAIL_CLOSED 0
#   setx MIN_SCORE 0.35
#
# NOTE: setx applies to new terminals only.
DEFAULT_MIN_SCORE = 0.55

def _env_bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return str(v).strip().lower() in {"1", "true", "yes", "y", "on"}

def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    if v is None:
        return default
    try:
        return float(v)
    except Exception:
        return default

STRICT_FAIL_CLOSED = _env_bool("STRICT_FAIL_CLOSED", True)
MIN_SCORE = _env_float("MIN_SCORE", DEFAULT_MIN_SCORE)

# Extractive confidence gate
MIN_EXTRACTIVE_CONF = _env_float("MIN_EXTRACTIVE_CONF", 0.15)


def fail_closed(best_score: float | None) -> bool:
    """Returns True if we should refuse due to insufficient retrieval confidence."""
    if not STRICT_FAIL_CLOSED:
        return False
    if best_score is None:
        return True
    try:
        return float(best_score) < MIN_SCORE
    except Exception:
        return True


class QueryReq(BaseModel):
    q: str
    top_k: int = 5


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok", "service": "t1-rag-reg"})


def _truncate_evidence(evidence: list[dict], text_limit: int = 800) -> list[dict]:
    """Keep payload small so PowerShell output stays readable."""
    out: list[dict] = []
    for ev in evidence or []:
        if isinstance(ev, dict):
            ev2 = dict(ev)
            if isinstance(ev2.get("text"), str):
                ev2["text"] = ev2["text"][:text_limit]
            out.append(ev2)
    return out


def _truncate_hits(hits: list[dict], text_limit: int = 400) -> list[dict]:
    """Trim retrieval hits text for debug payload readability."""
    out: list[dict] = []
    for h in hits or []:
        if isinstance(h, dict):
            h2 = dict(h)
            if isinstance(h2.get("text"), str):
                h2["text"] = h2["text"][:text_limit]
            out.append(h2)
    return out


def _refusal(
    *,
    reason: str,
    stage: str,
    best: float | None,
    top_k: int,
    confidence: float,
    rationale: str | None = None,
    citations: list | None = None,
    evidence: list[dict] | None = None,
):
    return JSONResponse(
        status_code=424,
        content={
            "status": "refused",
            "reason": reason,
            "stage": stage,
            "best_score": float(best) if best is not None else None,
            "min_score_gate": MIN_SCORE,
            "strict_fail_closed": STRICT_FAIL_CLOSED,
            "top_k": int(top_k),
            "answer": None,
            "confidence": float(confidence),
            "rationale": rationale,
            "citations": citations or [],
            "evidence": evidence or [],
        },
    )


def _ok(
    *,
    stage: str,
    best: float | None,
    top_k: int,
    answer: str,
    confidence: float,
    rationale: str | None,
    citations: list,
    evidence: list[dict],
):
    return JSONResponse(
        content={
            "status": "ok",
            "stage": stage,
            "best_score": float(best) if best is not None else None,
            "min_score_gate": MIN_SCORE,
            "strict_fail_closed": STRICT_FAIL_CLOSED,
            "top_k": int(top_k),
            "answer": answer,
            "confidence": float(confidence),
            "rationale": rationale,
            "citations": citations,
            "evidence": evidence,
        }
    )


@app.post("/query")
def query(req: QueryReq):
    q = (req.q or "").strip()
    top_k = int(req.top_k or 5)

    best, hits = retrieve(q, top_k)

    # Stage 1: retrieval gate (fail-closed)
    if fail_closed(best):
        return _refusal(
            reason="insufficient_retrieval_confidence",
            stage="retrieval",
            best=float(best) if best is not None else None,
            top_k=top_k,
            confidence=0.0,
        )

    # Stage 2: extractive answering (still no LLM)
    ex = answer_extractive(q, hits)

    if (not ex.answer) or (float(ex.confidence) < float(MIN_EXTRACTIVE_CONF)):
        return _refusal(
            reason="insufficient_extractive_confidence",
            stage="extractive",
            best=float(best) if best is not None else None,
            top_k=top_k,
            confidence=float(ex.confidence),
            rationale=ex.rationale,
            citations=ex.citations,
            evidence=_truncate_evidence((ex.evidence or [])[:5]),
        )

    return _ok(
        stage="extractive",
        best=float(best) if best is not None else None,
        top_k=top_k,
        answer=ex.answer,
        confidence=float(ex.confidence),
        rationale=ex.rationale,
        citations=ex.citations,
        evidence=_truncate_evidence((ex.evidence or [])[:5]),
    )


@app.post("/answer")
def answer_endpoint(payload: dict):
    """
    Step 1C: Consistent extractive answering endpoint (LLM-free).
    """
    q = (payload.get("q", "") or "").strip()
    top_k = int(payload.get("top_k", 5) or 5)

    best, hits = retrieve(q, top_k)

    if fail_closed(best):
        return _refusal(
            reason="insufficient_retrieval_confidence",
            stage="retrieval",
            best=float(best) if best is not None else None,
            top_k=top_k,
            confidence=0.0,
        )

    ex = answer_extractive(q, hits)

    if (not ex.answer) or (float(ex.confidence) < float(MIN_EXTRACTIVE_CONF)):
        return _refusal(
            reason="insufficient_extractive_confidence",
            stage="extractive",
            best=float(best) if best is not None else None,
            top_k=top_k,
            confidence=float(ex.confidence),
            rationale=ex.rationale,
            citations=ex.citations,
            evidence=_truncate_evidence((ex.evidence or [])[:5]),
        )

    return _ok(
        stage="extractive",
        best=float(best) if best is not None else None,
        top_k=top_k,
        answer=ex.answer,
        confidence=float(ex.confidence),
        rationale=ex.rationale,
        citations=ex.citations,
        evidence=_truncate_evidence((ex.evidence or [])[:5]),
    )


@app.post("/debug/retrieve")
def debug_retrieve(payload: dict):
    q = (payload.get("q", "") or "").strip()
    top_k = int(payload.get("top_k", 10) or 10)

    best, hits = retrieve(q, top_k)

    # Do NOT fail-closed here; return everything for inspection
    return JSONResponse(
        content={
            "q": q,
            "top_k": top_k,
            "best_score": float(best) if best is not None else None,
            "min_score_gate": MIN_SCORE,
            "strict_fail_closed": STRICT_FAIL_CLOSED,
            "would_refuse": fail_closed(best),
            "hits": _truncate_hits(hits, text_limit=400),
        }
    )


@app.on_event("startup")
def _debug_print_routes():
    print("=== ROUTES LOADED ===")
    for r in app.router.routes:
        try:
            methods = ",".join(sorted(getattr(r, "methods", []) or []))
        except Exception:
            methods = ""
        print(f"{methods:10s} {getattr(r, 'path', str(r))}")
    print("=====================")
    print(f"CONFIG: STRICT_FAIL_CLOSED={STRICT_FAIL_CLOSED} MIN_SCORE={MIN_SCORE} MIN_EXTRACTIVE_CONF={MIN_EXTRACTIVE_CONF}")
