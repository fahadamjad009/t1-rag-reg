"""
T1-RAG-REG — Step 1D-3: Evaluation Metrics Runner

Computes:
- Recall@k
- MRR@k
- nDCG@k
- Answer coverage for must_include sources (based on citations)  [stubbed until answer module path is wired]

Outputs:
- reports/eval_report_<mode>_<timestamp>.json
- reports/eval_per_query_<mode>_<timestamp>.csv
"""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


# -----------------------------
# Config
# -----------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../app/eval/run_eval.py -> repo root
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
    """
    IMPORTANT:
    For evaluation right now, prefer chunk-level IDs first (chunk_id),
    then fall back to source_id/doc_id/source/id.

    This lets you get meaningful retrieval metrics even if your corpus currently
    has only 1 source_id for many chunks.
    """
    ids: List[str] = []
    for h in hits or []:
        if not isinstance(h, dict):
            continue
        for key in ("chunk_id", "source_id", "doc_id", "source", "id"):
            v = h.get(key)
            if isinstance(v, str) and v.strip():
                ids.append(v.strip())
                break
    return _dedupe_keep_order(ids)


def _extract_doc_ids_from_citations(answer_payload: Dict[str, Any]) -> List[str]:
    """
    For answer coverage: use citations/evidence if present.
    (Will be empty while _answer() is stubbed.)
    """
    ids: List[str] = []

    cits = answer_payload.get("citations")
    if isinstance(cits, list):
        for c in cits:
            if isinstance(c, dict):
                for key in ("chunk_id", "source_id", "doc_id", "source", "id"):
                    v = c.get(key)
                    if isinstance(v, str) and v.strip():
                        ids.append(v.strip())
                        break

    ev = answer_payload.get("evidence")
    if isinstance(ev, list):
        for e in ev:
            if isinstance(e, dict):
                for key in ("chunk_id", "source_id", "doc_id", "source", "id"):
                    v = e.get(key)
                    if isinstance(v, str) and v.strip():
                        ids.append(v.strip())
                        break

    return _dedupe_keep_order(ids)


# -----------------------------
# Metrics
# -----------------------------

def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    topk = retrieved[:k]
    hits = sum(1 for d in topk if d in rel_set)
    return hits / float(len(rel_set))


def mrr_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    rel_set = set(relevant)
    for i, d in enumerate(retrieved[:k], start=1):
        if d in rel_set:
            return 1.0 / float(i)
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    """
    Binary relevance:
      DCG = sum rel_i / log2(i+1)
      IDCG = ideal DCG with all relevant at top (up to k)
    """
    if not relevant:
        return 0.0

    rel_set = set(relevant)

    dcg = 0.0
    for i, d in enumerate(retrieved[:k], start=1):
        rel_i = 1.0 if d in rel_set else 0.0
        dcg += rel_i / math.log2(i + 1.0)

    ideal_hits = min(len(rel_set), k)
    idcg = 0.0
    for i in range(1, ideal_hits + 1):
        idcg += 1.0 / math.log2(i + 1.0)

    return (dcg / idcg) if idcg > 0 else 0.0


def coverage(referenced: List[str], must_include: List[str]) -> float:
    """
    Answer coverage: fraction of must_include IDs present in answer citations/evidence.
    """
    if not must_include:
        return 0.0
    ref_set = set(referenced)
    target_set = set(must_include)
    hits = sum(1 for d in target_set if d in ref_set)
    return hits / float(len(target_set))


# -----------------------------
# Hooks into your repo
# -----------------------------

def _retrieve(query: str, top_k: int = 10, mode: str = "vector") -> List[Dict[str, Any]]:
    """
    Your repo: app.rag.retriever.retrieve(query, top_k, mode) -> (threshold, hits)
    """
    from app.rag.retriever import retrieve  # type: ignore

    threshold, hits = retrieve(query=query, top_k=top_k, mode=mode)
    _ = threshold  # kept for future reporting/debug
    return hits or []


def _answer(query: str, top_k: int = 5, mode: str = "vector") -> Dict[str, Any]:
    """
    Stub: your repo does not have app/rag/answerer.py.
    Retrieval eval (Recall/MRR/nDCG) is the core of Step 1D-3.
    We'll wire this later once we locate the real answer module.
    """
    _ = (query, top_k, mode)
    return {}


# -----------------------------
# Runner
# -----------------------------

def run(mode: str = "vector", ks: Optional[List[int]] = None) -> Dict[str, Any]:
    ks = ks or DEFAULT_KS
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = _load_jsonl(DATASET_PATH)
    per_query: List[Dict[str, Any]] = []

    # Aggregate sums
    agg: Dict[str, float] = {}
    for k in ks:
        agg[f"recall@{k}"] = 0.0
        agg[f"mrr@{k}"] = 0.0
        agg[f"ndcg@{k}"] = 0.0
    agg["answer_coverage"] = 0.0

    n = 0

    for r in rows:
        q = (r.get("q") or "").strip()
        if not q:
            continue

        must_include = r.get("must_include") or r.get("relevant") or []
        if not isinstance(must_include, list):
            must_include = [str(must_include)]
        must_include = [str(x).strip() for x in must_include if str(x).strip()]

        # Retrieval
        hits = _retrieve(q, top_k=max(ks), mode=mode)
        retrieved_ids = _extract_doc_ids_from_hits(hits)

        # Answer coverage (stubbed for now)
        ans_payload = _answer(q, top_k=5, mode=mode)
        cited_ids = _extract_doc_ids_from_citations(ans_payload)

        row_out: Dict[str, Any] = {
            "q": q,
            "mode": mode,
            "relevant_count": len(set(must_include)),
            "retrieved_count": len(retrieved_ids),
        }

        for k in ks:
            r_at_k = recall_at_k(retrieved_ids, must_include, k)
            m_at_k = mrr_at_k(retrieved_ids, must_include, k)
            n_at_k = ndcg_at_k(retrieved_ids, must_include, k)

            row_out[f"recall@{k}"] = r_at_k
            row_out[f"mrr@{k}"] = m_at_k
            row_out[f"ndcg@{k}"] = n_at_k

            agg[f"recall@{k}"] += r_at_k
            agg[f"mrr@{k}"] += m_at_k
            agg[f"ndcg@{k}"] += n_at_k

        cov = coverage(cited_ids, must_include)
        row_out["answer_coverage"] = cov
        agg["answer_coverage"] += cov

        # Debug fields
        row_out["retrieved_ids_top10"] = "|".join(retrieved_ids[:10])
        row_out["must_include"] = "|".join(must_include)
        row_out["cited_ids"] = "|".join(cited_ids[:10])

        per_query.append(row_out)
        n += 1

    if n == 0:
        raise RuntimeError(f"No valid queries found in dataset: {DATASET_PATH}")

    # Averages
    avg = {k: (v / float(n)) for k, v in agg.items()}

    stamp = _now_stamp()
    report_path = REPORTS_DIR / f"eval_report_{mode}_{stamp}.json"
    csv_path = REPORTS_DIR / f"eval_per_query_{mode}_{stamp}.csv"

    report = {
        "meta": {
            "mode": mode,
            "dataset": str(DATASET_PATH.relative_to(REPO_ROOT)),
            "n_queries": n,
            "ks": ks,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        "metrics": avg,
        "notes": {
            "definition": {
                "recall@k": "fraction of relevant items retrieved in top-k (binary relevance)",
                "mrr@k": "reciprocal rank of first relevant item in top-k",
                "ndcg@k": "discounted cumulative gain normalized by ideal (binary relevance)",
                "answer_coverage": "fraction of must_include items present in answer citations/evidence (stubbed until answer module wired)",
            }
        },
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # CSV breakdown
    if per_query:
        fieldnames = list(per_query[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(per_query)

    return {"report_path": str(report_path), "csv_path": str(csv_path), "report": report}


if __name__ == "__main__":
    out = run(mode=os.environ.get("T1_EVAL_MODE", "vector"))
    print("Wrote:", out["report_path"])
    print("Wrote:", out["csv_path"])
    print(json.dumps(out["report"]["metrics"], indent=2))
