from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"


def _latest_report_path(mode: str = "vector") -> Path:
    # Picks the newest eval_report_<mode>_*.json
    candidates = sorted(REPORTS_DIR.glob(f"eval_report_{mode}_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError(f"No eval reports found in {REPORTS_DIR} for mode={mode}")
    return candidates[0]


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_series(metrics: Dict[str, Any]) -> Tuple[List[int], List[float], List[float], List[float]]:
    # Pull k values from keys like recall@1, mrr@3, ndcg@10
    ks = []
    for k in metrics.keys():
        if isinstance(k, str) and k.startswith("recall@"):
            try:
                ks.append(int(k.split("@", 1)[1]))
            except Exception:
                pass
    ks = sorted(set(ks))

    recall = [float(metrics.get(f"recall@{k}", 0.0)) for k in ks]
    mrr = [float(metrics.get(f"mrr@{k}", 0.0)) for k in ks]
    ndcg = [float(metrics.get(f"ndcg@{k}", 0.0)) for k in ks]
    return ks, recall, mrr, ndcg


def _save_bar_chart(x: List[int], y: List[float], title: str, out_path: Path) -> None:
    plt.figure()
    plt.bar([str(v) for v in x], y)
    plt.title(title)
    plt.xlabel("k")
    plt.ylabel("score")
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def main(mode: str = "vector") -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    report_path = _latest_report_path(mode=mode)
    report = _load_json(report_path)

    meta = report.get("meta", {})
    metrics = report.get("metrics", {})

    ks, recall, mrr, ndcg = _extract_series(metrics)

    stamp = report_path.stem.replace("eval_report_", "")
    out_recall = REPORTS_DIR / f"plot_recall_{stamp}.png"
    out_mrr = REPORTS_DIR / f"plot_mrr_{stamp}.png"
    out_ndcg = REPORTS_DIR / f"plot_ndcg_{stamp}.png"
    out_md = REPORTS_DIR / f"eval_summary_{stamp}.md"

    _save_bar_chart(ks, recall, f"Recall@k ({mode})", out_recall)
    _save_bar_chart(ks, mrr, f"MRR@k ({mode})", out_mrr)
    _save_bar_chart(ks, ndcg, f"nDCG@k ({mode})", out_ndcg)

    md = []
    md.append(f"# T1-RAG-REG — Evaluation Summary ({mode})")
    md.append("")
    md.append(f"- Report: `{report_path.name}`")
    md.append(f"- Created: `{meta.get('created_at')}`")
    md.append(f"- Queries: `{meta.get('n_queries')}`")
    md.append(f"- ks: `{meta.get('ks')}`")
    md.append("")
    md.append("## Metrics (averages)")
    md.append("")
    md.append("| Metric | Value |")
    md.append("|---|---:|")
    for k in ks:
        md.append(f"| recall@{k} | {metrics.get(f'recall@{k}', 0.0):.4f} |")
    for k in ks:
        md.append(f"| mrr@{k} | {metrics.get(f'mrr@{k}', 0.0):.4f} |")
    for k in ks:
        md.append(f"| ndcg@{k} | {metrics.get(f'ndcg@{k}', 0.0):.4f} |")
    md.append(f"| answer_coverage | {metrics.get('answer_coverage', 0.0):.4f} |")
    md.append("")
    md.append("## Plots")
    md.append("")
    md.append(f"- `{out_recall.name}`")
    md.append(f"- `{out_mrr.name}`")
    md.append(f"- `{out_ndcg.name}`")
    md.append("")

    out_md.write_text("\n".join(md), encoding="utf-8")

    print("Using report:", report_path)
    print("Wrote:", out_recall)
    print("Wrote:", out_mrr)
    print("Wrote:", out_ndcg)
    print("Wrote:", out_md)


if __name__ == "__main__":
    import os
    main(mode=os.environ.get("T1_EVAL_MODE", "vector"))
