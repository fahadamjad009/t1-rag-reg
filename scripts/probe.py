import json, csv, urllib.request
from pathlib import Path

# --- 1. Rebuild leaderboard ---
reports = Path("reports")
rows = []
for mode in ["vector", "bm25", "hybrid_rrf"]:
    files = sorted(reports.glob(f"eval_report_{mode}_*.json"))
    if not files:
        continue
    r = json.loads(files[-1].read_text())
    m = r["metrics"]
    row = {"mode": mode}
    for k in ["recall@1","recall@3","recall@5","recall@10","mrr@10","ndcg@10"]:
        row[k] = m.get(k, 0)
    rows.append(row)
    print(f"{mode:12s}  R@1={row['recall@1']:.2f}  R@3={row['recall@3']:.2f}  R@5={row['recall@5']:.2f}  R@10={row['recall@10']:.2f}  MRR={row['mrr@10']:.2f}  nDCG={row['ndcg@10']:.2f}")

out = reports / "eval_leaderboard_latest.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)
print(f"\nLeaderboard written to {out}")

# --- 2. Probe for a success example ---
queries = [
    "Who must enrol with AUSTRAC?",
    "What triggers a suspicious matter report?",
    "What are the penalties for non-compliance with the AML/CTF Act?",
    "What is customer due diligence under the AML/CTF Act?",
    "What must a reporting entity include in its AML/CTF program?",
    "What are the customer identification procedures?",
]
print("\n--- Probing for a successful answer ---")
for q in queries:
    body = json.dumps({"q": q, "mode": "hybrid_rrf", "top_k": 5}).encode()
    req = urllib.request.Request("http://localhost:8000/query", data=body, headers={"Content-Type": "application/json"})
    resp = json.loads(urllib.request.urlopen(req).read())
    status = resp.get("status", "?")
    conf = resp.get("calibrated_confidence", 0)
    ans = (resp.get("answer") or "")[:80]
    print(f"  [{status:7s}] conf={conf:.2f}  q={q[:50]}")
    if status == "ok":
        Path("examples/query_success.json").write_text(json.dumps(resp, indent=2))
        print(f"  ^ SAVED to examples/query_success.json")
        break
else:
    print("  No success found - all queries refused. Saving best refusal instead.")
