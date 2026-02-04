from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import urllib.request

API_URL = "http://127.0.0.1:8000/query"
IN_PATH = Path("data/eval/questions.jsonl")
OUT_DIR = Path("reports")

TOP_K_DEFAULT = 5
TIMEOUT_SEC = 30


def post_json(url: str, payload: Dict[str, Any], timeout: int = TIMEOUT_SEC) -> Tuple[int, Dict[str, Any]]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8") if e.fp else ""
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"error": body}
    except Exception as e:
        return 0, {"error": str(e)}


def contains_all(answer: str, expected: List[str]) -> bool:
    a = (answer or "").lower()
    return all(e.lower() in a for e in expected)


def main() -> int:
    if not IN_PATH.exists():
        print(f"Missing eval file: {IN_PATH}")
        return 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = OUT_DIR / f"eval_run_{ts}.json"

    rows = [json.loads(line) for line in IN_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]

    results = []
    totals = {
        "n": 0,
        "ok": 0,
        "refused": 0,
        "retrieval_refused": 0,
        "extractive_refused": 0,
        "contains_pass": 0,
        "avg_confidence_ok": 0.0,
    }

    conf_sum = 0.0
    conf_count = 0

    for r in rows:
        q = r["q"]
        expected = r.get("expected_contains", [])
        min_conf = float(r.get("min_confidence", 0.15))
        top_k = int(r.get("top_k", TOP_K_DEFAULT))

        status, body = post_json(API_URL, {"q": q, "top_k": top_k})
        totals["n"] += 1

        ok = status == 200 and body.get("status") == "ok"
        refused = status == 424

        answer = body.get("answer", "")
        conf = float(body.get("confidence") or 0.0)
        stage = body.get("stage")

        contains_pass = contains_all(answer, expected) if ok else False

        if ok:
            totals["ok"] += 1
            conf_sum += conf
            conf_count += 1
        if refused:
            totals["refused"] += 1
            if stage == "retrieval":
                totals["retrieval_refused"] += 1
            elif stage == "extractive":
                totals["extractive_refused"] += 1

        if contains_pass:
            totals["contains_pass"] += 1

        results.append(
            {
                "q": q,
                "status": body.get("status"),
                "stage": stage,
                "confidence": conf,
                "contains_pass": contains_pass,
                "citations": body.get("citations", []),
                "evidence_count": len(body.get("evidence") or []),
            }
        )

    if conf_count:
        totals["avg_confidence_ok"] = conf_sum / conf_count

    report = {
        "timestamp": ts,
        "api_url": API_URL,
        "totals": totals,
        "results": results,
    }

    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote report: {out_path}\n")
    print(json.dumps(totals, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
