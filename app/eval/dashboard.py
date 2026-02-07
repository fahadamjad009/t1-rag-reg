from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"


def list_runs(mode: str) -> list[Path]:
    return sorted(
        REPORTS_DIR.glob(f"eval_report_{mode}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def find_matching_csv(report_path: Path) -> Path | None:
    # eval_report_vector_YYYYMMDD_HHMMSS.json -> eval_per_query_vector_YYYYMMDD_HHMMSS.csv
    name = report_path.name.replace("eval_report_", "eval_per_query_").replace(".json", ".csv")
    p = REPORTS_DIR / name
    return p if p.exists() else None


def find_plot(report_path: Path, kind: str) -> Path | None:
    # eval_report_vector_YYYYMMDD_HHMMSS.json -> plot_<kind>_vector_YYYYMMDD_HHMMSS.png
    # where kind in {"recall","mrr","ndcg"}
    stamp = report_path.name.replace("eval_report_", "").replace(".json", "")
    p = REPORTS_DIR / f"plot_{kind}_{stamp}.png"
    return p if p.exists() else None


st.set_page_config(page_title="T1-RAG-REG • Eval Dashboard", layout="wide")
st.title("T1-RAG-REG — Evaluation Dashboard")

mode = st.sidebar.selectbox("Mode", ["vector", "bm25", "hybrid_rrf"], index=0)

runs = list_runs(mode)
if not runs:
    st.warning(f"No reports found in {REPORTS_DIR} for mode={mode}. Run: python -m app.eval.run_eval")
    st.stop()

run_path = st.sidebar.selectbox("Run", runs, format_func=lambda p: p.name)
report = load_report(run_path)

meta = report.get("meta", {})
metrics = report.get("metrics", {})

st.subheader("Run metadata")
st.write(
    {
        "mode": meta.get("mode"),
        "created_at": meta.get("created_at"),
        "n_queries": meta.get("n_queries"),
        "ks": meta.get("ks"),
        "dataset": meta.get("dataset"),
        "report_file": run_path.name,
    }
)

st.subheader("Metrics (averages)")

# Build a tidy metrics table
rows = [{"metric": k, "value": float(v)} for k, v in metrics.items()]
df_metrics = pd.DataFrame(rows).sort_values(by="metric")
st.dataframe(df_metrics, use_container_width=True)

# Quick charts for @k
def metric_series(prefix: str) -> pd.DataFrame:
    series = []
    for k in meta.get("ks", []) or []:
        series.append({"k": int(k), "value": float(metrics.get(f"{prefix}@{k}", 0.0))})
    return pd.DataFrame(series)


c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Recall@k")
    st.line_chart(metric_series("recall"), x="k", y="value")
with c2:
    st.caption("MRR@k")
    st.line_chart(metric_series("mrr"), x="k", y="value")
with c3:
    st.caption("nDCG@k")
    st.line_chart(metric_series("ndcg"), x="k", y="value")

st.caption("Answer coverage (currently stubbed if answer module not wired)")
st.metric("answer_coverage", f"{float(metrics.get('answer_coverage', 0.0)):.4f}")

st.subheader("Saved plots (from reports/)")
p_recall = find_plot(run_path, "recall")
p_mrr = find_plot(run_path, "mrr")
p_ndcg = find_plot(run_path, "ndcg")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    st.caption("Recall@k (PNG)")
    if p_recall:
        st.image(str(p_recall), use_container_width=True)
    else:
        st.info("PNG not found for this run. Run: python -m app.eval.report_plots")
with pc2:
    st.caption("MRR@k (PNG)")
    if p_mrr:
        st.image(str(p_mrr), use_container_width=True)
    else:
        st.info("PNG not found for this run. Run: python -m app.eval.report_plots")
with pc3:
    st.caption("nDCG@k (PNG)")
    if p_ndcg:
        st.image(str(p_ndcg), use_container_width=True)
    else:
        st.info("PNG not found for this run. Run: python -m app.eval.report_plots")

st.divider()

st.subheader("Per-query breakdown")

csv_path = find_matching_csv(run_path)
if not csv_path:
    st.info("Per-query CSV not found for this run.")
    st.stop()

df = pd.read_csv(csv_path)

# Filters
q_filter = st.text_input("Search query text contains", "")
if q_filter.strip():
    df = df[df["q"].astype(str).str.contains(q_filter, case=False, na=False)]

sort_choices = [c for c in ["recall@1", "mrr@1", "ndcg@1", "recall@3", "mrr@3", "ndcg@3"] if c in df.columns]
if not sort_choices:
    sort_choices = list(df.columns)

sort_col = st.selectbox("Sort by", sort_choices, index=0)
df = df.sort_values(by=sort_col, ascending=True)

st.dataframe(df, use_container_width=True)

st.divider()
st.subheader("Worst queries (quick view)")

cols = [c for c in ["q", sort_col, "retrieved_ids_top10", "must_include"] if c in df.columns]
st.write(df[cols].head(10))
