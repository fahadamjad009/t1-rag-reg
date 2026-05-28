from __future__ import annotations

import json
import os
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

# ✅ SWITCH: use your new extractive engine (Phase 3A)
from app.rag.extractive import answer_from_hits
from app.rag.calibration import calibrate_best, explain_calibration
from app.rag.retriever import retrieve

# Change this string whenever you edit main.py (proves reload)
BUILD_ID = "main_py_build_" + datetime.now().strftime("%Y%m%d_%H%M%S")

app = FastAPI(
    title="T1-RAG-REG",
    version="0.1.0",
    description="Regulatory RAG system with evaluation-first design",
)

DEFAULT_MIN_CALIB_CONF = 0.35


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
MIN_CALIB_CONF = _env_float("MIN_CALIB_CONF", DEFAULT_MIN_CALIB_CONF)
MIN_EXTRACTIVE_CONF = _env_float("MIN_EXTRACTIVE_CONF", 0.15)


def _normalize_mode(mode: str | None) -> str:
    m = (mode or "vector").strip().lower()
    if m in {"vector", "bm25", "hybrid_rrf"}:
        return m
    if m in {"rrf", "hybrid", "hybridrrf"}:
        return "hybrid_rrf"
    return "vector"


def should_refuse(best_score: float | None, mode: str) -> tuple[bool, float]:
    conf = calibrate_best(best_score, mode)
    if not STRICT_FAIL_CLOSED:
        return False, conf
    return conf < MIN_CALIB_CONF, conf


class QueryReq(BaseModel):
    q: str
    top_k: int = 5
    mode: str | None = None
    # ✅ optional knobs (useful for eval + debugging)
    must_include: list[str] | None = None
    max_citations: int | None = 5
    max_sentences: int | None = 2


@app.get("/health")
def health():
    return JSONResponse(content={"status": "ok", "service": "t1-rag-reg"})


@app.get("/debug/build")
def debug_build():
    # Proves which file/code is running
    return JSONResponse(
        content={
            "build_id": BUILD_ID,
            "main_file": __file__,
            "cwd": os.getcwd(),
            "STRICT_FAIL_CLOSED": STRICT_FAIL_CLOSED,
            "MIN_CALIB_CONF": MIN_CALIB_CONF,
            "MIN_EXTRACTIVE_CONF": MIN_EXTRACTIVE_CONF,
            "REFUSAL_HTTP_STATUS": int(os.getenv("REFUSAL_HTTP_STATUS", "200")),
            "calibration_ranges": explain_calibration(),
        }
    )


def _truncate_evidence(evidence: list[dict], text_limit: int = 800) -> list[dict]:
    out: list[dict] = []
    for ev in evidence or []:
        if isinstance(ev, dict):
            ev2 = dict(ev)
            if isinstance(ev2.get("text"), str):
                ev2["text"] = ev2["text"][:text_limit]
            out.append(ev2)
    return out


def _truncate_hits(hits: list[dict], text_limit: int = 400) -> list[dict]:
    out: list[dict] = []
    for h in hits or []:
        if isinstance(h, dict):
            h2 = dict(h)
            if isinstance(h2.get("text"), str):
                h2["text"] = h2["text"][:text_limit]
            out.append(h2)
    return out


def _json_response(*, status_code: int, payload: dict):
    """
    Always return JSON, never empty body.
    """
    try:
        safe = jsonable_encoder(payload)
        json.dumps(safe, ensure_ascii=False)  # validate serializable
        return JSONResponse(status_code=status_code, content=safe)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "reason": "response_serialization_failed",
                "detail": str(e),
                "original_status_code": status_code,
            },
        )


# ---- Global exception handlers: NEVER empty body again ----
@app.exception_handler(StarletteHTTPException)
async def _http_exception_handler(request: Request, exc: StarletteHTTPException):
    return _json_response(
        status_code=exc.status_code,
        payload={
            "status": "error",
            "reason": "http_exception",
            "path": str(request.url),
            "detail": exc.detail,
        },
    )


@app.exception_handler(RequestValidationError)
async def _validation_exception_handler(request: Request, exc: RequestValidationError):
    return _json_response(
        status_code=422,
        payload={
            "status": "error",
            "reason": "validation_error",
            "path": str(request.url),
            "detail": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def _unhandled_exception_handler(request: Request, exc: Exception):
    return _json_response(
        status_code=500,
        payload={
            "status": "error",
            "reason": "unhandled_exception",
            "path": str(request.url),
            "detail": repr(exc),
        },
    )


def _refusal(
    *,
    reason: str,
    stage: str,
    mode: str,
    best: float | None,
    top_k: int,
    confidence: float,
    rationale: str | None = None,
    citations: list | None = None,
    evidence: list[dict] | None = None,
    calibrated_confidence: float | None = None,
    strict_fail_closed: bool | None = None,
):
    """
    PowerShell-friendly refusal:
    - Default HTTP status is 200 so Invoke-RestMethod won't throw.
    - True semantic status is carried as payload["http_status"] = 424.
    - If you want strict HTTP, set: setx REFUSAL_HTTP_STATUS 424 (restart uvicorn).
    """
    refusal_http_status = int(os.getenv("REFUSAL_HTTP_STATUS", "200"))

    payload = {
        "status": "refused",
        "http_status": 424,
        "reason": reason,
        "stage": stage,
        "mode": mode,
        "best_score": float(best) if best is not None else None,
        "strict_fail_closed": bool(strict_fail_closed) if strict_fail_closed is not None else STRICT_FAIL_CLOSED,
        "top_k": int(top_k),
        "answer": None,
        "confidence": float(confidence),
        "rationale": rationale,
        "citations": citations or [],
        "evidence": evidence or [],
        "calibrated_confidence": float(calibrated_confidence) if calibrated_confidence is not None else None,
        "min_calib_conf_gate": float(MIN_CALIB_CONF),
        "calibration_ranges": explain_calibration(),
    }

    return _json_response(status_code=refusal_http_status, payload=payload)


def _ok(
    *,
    stage: str,
    mode: str,
    best: float | None,
    top_k: int,
    answer: str,
    confidence: float,
    rationale: str | None,
    citations: list,
    evidence: list[dict],
    calibrated_confidence: float | None = None,
    strict_fail_closed: bool | None = None,
):
    return _json_response(
        status_code=200,
        payload={
            "status": "ok",
            "stage": stage,
            "mode": mode,
            "best_score": float(best) if best is not None else None,
            "strict_fail_closed": bool(strict_fail_closed) if strict_fail_closed is not None else STRICT_FAIL_CLOSED,
            "top_k": int(top_k),
            "answer": answer,
            "confidence": float(confidence),
            "rationale": rationale,
            "citations": citations,
            "evidence": evidence,
            "calibrated_confidence": float(calibrated_confidence) if calibrated_confidence is not None else None,
            "min_calib_conf_gate": float(MIN_CALIB_CONF),
            "calibration_ranges": explain_calibration(),
        },
    )


def _run_extractive_from_payload(*, q: str, hits: list[dict], payload: dict) -> dict:
    """
    ✅ Centralizes the new extractive call, so /query and /answer behave identically.
    """
    must_include = payload.get("must_include")
    if must_include is not None and not isinstance(must_include, list):
        must_include = None

    max_citations = payload.get("max_citations", 5)
    try:
        max_citations = int(max_citations)
    except Exception:
        max_citations = 5

    max_sentences = payload.get("max_sentences", 2)
    try:
        max_sentences = int(max_sentences)
    except Exception:
        max_sentences = 2

    # answer_from_hits returns: {answer, citations, strict_fail_closed, ...}
    ex = answer_from_hits(
        query=q,
        hits=hits,
        max_sentences=max_sentences,
        must_include=must_include,
        max_citations=max_citations,
    )
    if not isinstance(ex, dict):
        ex = {"answer": "", "citations": [], "strict_fail_closed": True}
    return ex


@app.post("/query")
def query(req: QueryReq):
    q = (req.q or "").strip()
    top_k = int(req.top_k or 5)
    mode = _normalize_mode(req.mode)

    best, hits = retrieve(q, top_k, mode=mode)

    refuse, calib_conf = should_refuse(best, mode)
    if refuse:
        return _refusal(
            reason="insufficient_retrieval_confidence",
            stage="retrieval",
            mode=mode,
            best=best,
            top_k=top_k,
            confidence=0.0,
            calibrated_confidence=calib_conf,
        )

    payload_like = {
        "must_include": req.must_include,
        "max_citations": req.max_citations,
        "max_sentences": req.max_sentences,
    }
    ex = _run_extractive_from_payload(q=q, hits=hits, payload=payload_like)

    ex_answer = ex.get("answer") or ""
    ex_conf = float(ex.get("confidence", 1.0))  # if your extractive doesn't output confidence, default high
    ex_rationale = ex.get("rationale", "extractive_answer_from_hits")
    ex_cits = ex.get("citations") or []
    ex_evidence = ex.get("evidence") or []
    ex_strict = bool(ex.get("strict_fail_closed", False))

    if (not ex_answer) or (ex_conf < float(MIN_EXTRACTIVE_CONF)) or (ex_strict is True):
        # If strict_fail_closed=True, treat as refusal (reg-grade)
        return _refusal(
            reason="insufficient_extractive_confidence" if not ex_strict else "no_grounded_definition_or_fail_closed",
            stage="extractive",
            mode=mode,
            best=best,
            top_k=top_k,
            confidence=float(ex_conf),
            rationale=str(ex_rationale) if ex_rationale else None,
            citations=ex_cits,
            evidence=_truncate_evidence((ex_evidence or [])[:5]),
            calibrated_confidence=calib_conf,
            strict_fail_closed=ex_strict,
        )

    return _ok(
        stage="extractive",
        mode=mode,
        best=best,
        top_k=top_k,
        answer=ex_answer,
        confidence=float(ex_conf),
        rationale=str(ex_rationale) if ex_rationale else None,
        citations=ex_cits,
        evidence=_truncate_evidence((ex_evidence or [])[:5]),
        calibrated_confidence=calib_conf,
        strict_fail_closed=ex_strict,
    )


@app.post("/answer")
def answer_endpoint(payload: dict):
    q = (payload.get("q", "") or "").strip()
    top_k = int(payload.get("top_k", 5) or 5)
    mode = _normalize_mode(payload.get("mode"))

    best, hits = retrieve(q, top_k, mode=mode)

    refuse, calib_conf = should_refuse(best, mode)
    if refuse:
        return _refusal(
            reason="insufficient_retrieval_confidence",
            stage="retrieval",
            mode=mode,
            best=best,
            top_k=top_k,
            confidence=0.0,
            calibrated_confidence=calib_conf,
        )

    ex = _run_extractive_from_payload(q=q, hits=hits, payload=payload)

    ex_answer = ex.get("answer") or ""
    ex_conf = float(ex.get("confidence", 1.0))  # if not provided by extractive, default high
    ex_rationale = ex.get("rationale", "extractive_answer_from_hits")
    ex_cits = ex.get("citations") or []
    ex_evidence = ex.get("evidence") or []
    ex_strict = bool(ex.get("strict_fail_closed", False))

    if (not ex_answer) or (ex_conf < float(MIN_EXTRACTIVE_CONF)) or (ex_strict is True):
        return _refusal(
            reason="insufficient_extractive_confidence" if not ex_strict else "no_grounded_definition_or_fail_closed",
            stage="extractive",
            mode=mode,
            best=best,
            top_k=top_k,
            confidence=float(ex_conf),
            rationale=str(ex_rationale) if ex_rationale else None,
            citations=ex_cits,
            evidence=_truncate_evidence((ex_evidence or [])[:5]),
            calibrated_confidence=calib_conf,
            strict_fail_closed=ex_strict,
        )

    return _ok(
        stage="extractive",
        mode=mode,
        best=best,
        top_k=top_k,
        answer=ex_answer,
        confidence=float(ex_conf),
        rationale=str(ex_rationale) if ex_rationale else None,
        citations=ex_cits,
        evidence=_truncate_evidence((ex_evidence or [])[:5]),
        calibrated_confidence=calib_conf,
        strict_fail_closed=ex_strict,
    )


@app.post("/debug/retrieve")
def debug_retrieve(payload: dict):
    q = (payload.get("q", "") or "").strip()
    top_k = int(payload.get("top_k", 10) or 10)
    mode = _normalize_mode(payload.get("mode"))

    best, hits = retrieve(q, top_k, mode=mode)
    refuse, calib_conf = should_refuse(best, mode)

    return _json_response(
        status_code=200,
        payload={
            "q": q,
            "mode": mode,
            "top_k": top_k,
            "best_score": float(best) if best is not None else None,
            "calibrated_confidence": float(calib_conf),
            "min_calib_conf_gate": float(MIN_CALIB_CONF),
            "strict_fail_closed": STRICT_FAIL_CLOSED,
            "would_refuse": bool(refuse),
            "calibration_ranges": explain_calibration(),
            "hits": _truncate_hits(hits, text_limit=400),
        },
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
    print("MAIN FILE:", __file__)
    print("BUILD_ID:", BUILD_ID)
    print(
        f"CONFIG: STRICT_FAIL_CLOSED={STRICT_FAIL_CLOSED} "
        f"MIN_CALIB_CONF={MIN_CALIB_CONF} MIN_EXTRACTIVE_CONF={MIN_EXTRACTIVE_CONF} "
        f"REFUSAL_HTTP_STATUS={int(os.getenv('REFUSAL_HTTP_STATUS', '200'))}"
    )
