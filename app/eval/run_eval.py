"""
T1-RAG-REG — Evaluation Metrics Runner (Phase 3D.1 FINAL)

Computes:
- Recall@k
- MRR@k
- nDCG@k
- Answer coverage (must_include satisfied via citations)
- failure_reason diagnostics

Outputs:
- reports/eval_report_<mode>_<timestamp>.json
- reports/eval_per_query_<mode>_<timestamp>.csv
"""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# -----------------------------
# Config
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = REPO_ROOT / "app" / "eval" / "eval_dataset.jsonl"
REPORTS_DIR = REPO_ROOT / "reports"

DEFAULT_KS = [1, 3, 5, 10]


# -----------------------------
# Helpers
# -----------------------------

def _now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Eval dataset not found: {path}")

    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} in {path}: {e}") from e
    return rows


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen: set[str] = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _extract_doc_ids_from_hits(hits: List[Dict[str, Any]]) -> List[str]:
    ids: List[str] = []
    for h in hits or []:
        if not isinstance(h, dict):
            continue

        sid = h.get("source_id")
        if isinstance(sid, str) and sid.strip():
            ids.append(sid.strip())
            continue

        did = h.get("doc_id")
        if isinstance(did, str) and did.strip():
            ids.append(did.strip())

    return _dedupe_keep_order(ids)


def _extract_doc_ids_from_citations(answer_payload: Dict[str, Any]) -> List[str]:
    ids: List[str] = []

    if not isinstance(answer_payload, dict):
        return ids

    def _pull(lst: Any) -> None:
        if not isinstance(lst, list):
            return
        for c in lst:
            if isinstance(c, dict):
                sid = c.get("source_id")
                if isinstance(sid, str) and sid.strip():
                    ids.append(sid.strip())
                    continue

                did = c.get("doc_id")
                if isinstance(did, str) and did.strip():
                    ids.append(did.strip())
                    continue

            elif isinstance(c, str) and c.strip():
                ids.append(c.strip())

    _pull(answer_payload.get("citations"))
    _pull(answer_payload.get("evidence"))
    _pull(answer_payload.get("sources"))

    return _dedupe_keep_order(ids)


def _format_citations(citations: Any, max_items: int = 5) -> str:
    if not isinstance(citations, list):
        return ""

    out: List[str] = []
    for i, c in enumerate(citations[:max_items], start=1):
        if not isinstance(c, dict):
            continue

        sid = str(c.get("source_id") or c.get("doc_id") or "").strip()
        ev = str(c.get("evidence") or c.get("span") or "").strip().replace("\n", " ")

        if len(ev) > 160:
            ev = ev[:160] + "…"

        out.append(f"[{i}] {sid} — {ev}")

    return " | ".join(out)


def _normalize_must_include(r: Dict[str, Any]) -> List[str]:
    mi = r.get("must_include") or r.get("relevant") or []
    out: List[str] = []
    if isinstance(mi, list):
        out = [str(x).strip() for x in mi if str(x).strip()]
    elif isinstance(mi, str) and mi.strip():
        # allow single string in dataset by mistake
        out = [mi.strip()]
    return _dedupe_keep_order(out)


# -----------------------------
# Metrics
# -----------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel = set(relevant)
    return sum(1 for d in retrieved[:k] if d in rel) / float(len(rel))


def mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    rel = set(relevant)
    for i, d in enumerate(retrieved[:k], start=1):
        if d in rel:
            return 1.0 / float(i)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0

    rel = set(relevant)
    dcg = sum(
        (1.0 if d in rel else 0.0) / math.log2(i + 1.0)
        for i, d in enumerate(retrieved[:k], start=1)
    )
    ideal = sum(1.0 / math.log2(i + 1.0) for i in range(1, min(len(rel), k) + 1))
    return dcg / ideal if ideal > 0 else 0.0


def coverage(referenced: List[str], must_include: List[str]) -> float:
    if not must_include:
        return 0.0
    mi = set(must_include)
    return len(set(referenced) & mi) / float(len(mi))


# -----------------------------
# Repo hooks
# -----------------------------

def _retrieve(query: str, top_k: int, mode: str) -> List[Dict[str, Any]]:
    from app.rag.retriever import retrieve
    _, hits = retrieve(query=query, top_k=top_k, mode=mode)
    return hits or []


def _allowed_source_ids_from_hits(hits: List[Dict[str, Any]]) -> set[str]:
    return set(_extract_doc_ids_from_hits(hits))


def _answer(
    query: str,
    hits: List[Dict[str, Any]],
    must_include: Optional[List[str]],
    *,
    max_citations: int = 5,
) -> Dict[str, Any]:
    """
    Always returns a dict:
      {
        "answer": str,
        "citations": list[dict],
        "failure_reason": str (optional; runner will use if present)
      }

    Enforces:
    - Extractive grounding (delegated to answer_from_hits)
    - Citations must reference returned hits (source_id/doc_id)
    """
    if not hits:
        return {"answer": "", "citations": [], "failure_reason": "not_retrieved"}

    from app.rag.extractive import answer_from_hits

    # Try multiple calling conventions for compatibility with your extractive module.
    res: Any
    try:
        res = answer_from_hits(query, hits, must_include=must_include, max_citations=max_citations)
    except TypeError:
        try:
            res = answer_from_hits(query, hits, must_include=must_include)
        except TypeError:
            try:
                res = answer_from_hits(query, hits, max_citations=max_citations)
            except TypeError:
                res = answer_from_hits(query, hits)

    # Normalize to (answer, citations)
    answer_text = ""
    citations: Any = []

    if isinstance(res, dict):
        answer_text = str(res.get("answer") or "").strip()
        citations = res.get("citations") or res.get("evidence") or res.get("sources") or []
    elif isinstance(res, tuple) or isinstance(res, list):
        answer_text = str(res[0] if len(res) > 0 else "").strip()
        citations = res[1] if len(res) > 1 else []
    else:
        # plain string or unknown
        answer_text = str(res or "").strip()
        citations = []

    # Normalize citations -> list[dict]
    if isinstance(citations, dict):
        citations_list: List[Dict[str, Any]] = [citations]
    elif isinstance(citations, list):
        citations_list = [c for c in citations if isinstance(c, dict)]
        # if list of strings slipped through, convert to dicts
        if not citations_list and any(isinstance(c, str) for c in citations):
            citations_list = [{"source_id": str(c).strip(), "evidence": ""} for c in citations if str(c).strip()]
    else:
        citations_list = []

    # Enforce citations reference returned hits
    allowed = _allowed_source_ids_from_hits(hits)
    filtered: List[Dict[str, Any]] = []
    for c in citations_list:
        sid = str(c.get("source_id") or c.get("doc_id") or "").strip()
        if not sid:
            continue
        if sid in allowed:
            filtered.append(c)

    filtered = filtered[:max_citations]

    # Failure diagnostics (answer-level)
    if not answer_text and not filtered:
        fr = "no_grounded_span"
    elif answer_text and not filtered:
        fr = "no_citations"
    else:
        fr = "ok"

    return {"answer": answer_text, "citations": filtered, "failure_reason": fr}


# -----------------------------
# Runner
# -----------------------------

def run(mode: str = "vector", ks: Optional[List[int]] = None) -> Dict[str, Any]:
    ks = ks or DEFAULT_KS
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(DATASET_PATH)

    per_query: List[Dict[str, Any]] = []
    agg = {f"{m}@{k}": 0.0 for m in ["recall", "mrr", "ndcg"] for k in ks}
    agg["answer_coverage"] = 0.0

    n = 0

    for r in rows:
        q = str(r.get("q") or "").strip()
        if not q:
            continue

        must_include = _normalize_must_include(r)

        hits = _retrieve(q, top_k=max(ks), mode=mode)
        retrieved_ids = _extract_doc_ids_from_hits(hits)

        # Use more context for answering (as you did)
        ans_payload = _answer(q, hits=hits[:12], must_include=must_include, max_citations=5)

        cited_ids = _extract_doc_ids_from_citations(ans_payload)

        row: Dict[str, Any] = {
            "q": q,
            "mode": mode,
            "relevant_count": len(set(must_include)),
            "retrieved_count": len(retrieved_ids),
        }

        for k in ks:
            r_k = recall_at_k(retrieved_ids, must_include, k)
            m_k = mrr_at_k(retrieved_ids, must_include, k)
            n_k = ndcg_at_k(retrieved_ids, must_include, k)

            row[f"recall@{k}"] = r_k
            row[f"mrr@{k}"] = m_k
            row[f"ndcg@{k}"] = n_k

            agg[f"recall@{k}"] += r_k
            agg[f"mrr@{k}"] += m_k
            agg[f"ndcg@{k}"] += n_k

        cov = coverage(cited_ids, must_include)
        row["answer_coverage"] = cov
        agg["answer_coverage"] += cov

        # failure_reason (runner-level, prioritizing answer-level failures)
        answer_level_reason = str(ans_payload.get("failure_reason") or "").strip()

        if cov > 0:
            reason = "ok"
        elif not must_include:
            reason = "no_must_include"
        elif not hits:
            reason = "not_retrieved"
        elif not (set(retrieved_ids) & set(must_include)):
            reason = "not_retrieved"
        elif answer_level_reason in {"no_citations", "no_grounded_span"}:
            reason = answer_level_reason
        elif not (set(cited_ids) & set(must_include)):
            # retrieved relevant docs but citations didn't hit required sources
            reason = "retrieved_but_not_cited"
        else:
            reason = "other"

        row["failure_reason"] = reason
        row["citations_formatted"] = _format_citations(ans_payload.get("citations"))
        row["answer"] = ans_payload.get("answer", "")

        per_query.append(row)
        n += 1

    if n == 0:
        raise RuntimeError("No valid queries in dataset")

    avg = {k: v / n for k, v in agg.items()}

    stamp = _now_stamp()
    report_path = REPORTS_DIR / f"eval_report_{mode}_{stamp}.json"
    csv_path = REPORTS_DIR / f"eval_per_query_{mode}_{stamp}.csv"

    report = {"meta": {"mode": mode, "n_queries": n, "ks": ks}, "metrics": avg}

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=per_query[0].keys())
        w.writeheader()
        w.writerows(per_query)

    return {"report_path": str(report_path), "csv_path": str(csv_path), "report": report}


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--mode", default="vector", choices=["vector", "bm25", "hybrid_rrf"])
    p.add_argument("--ks", default=None)

    args = p.parse_args()
    ks = [int(x) for x in args.ks.split(",")] if args.ks else None

    out = run(args.mode, ks)

    print("Wrote:", out["report_path"])
    print("Wrote:", out["csv_path"])
    print("answer_coverage:", out["report"]["metrics"]["answer_coverage"])
