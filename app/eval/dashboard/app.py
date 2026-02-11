from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORTS_DIR = REPO_ROOT / "reports"


def _latest(pattern: str) -> Optional[str]:
    files = sorted(glob.glob(str(REPORTS_DIR / pattern)), key=lambda p: Path(p).stat().st_mtime, reverse=True)
    return files[0] if files else None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


st.set_page_config(page_title="T1-RAG-REG Eval Dashboard", layout="wide")
st.title("T1-RAG-REG — Evaluation Dashboard")

# Sidebar controls
st.sidebar.header("Inputs")

mode = st.sidebar.selectbox("Mode", ["hybrid_rrf", "vector", "bm25"], index=0)

default_report = _latest(f"eval_report_{mode}_*.json")
default_csv = _latest(f"eval_per_query_{mode}_*.csv")

report_path = st.sidebar.text_input("Report JSON path", value=default_report or "")
csv_path = st.sidebar.text_input("Per-query CSV path", value=default_csv or "")

min_cov = st.sidebar.slider("Min answer_coverage", 0.0, 1.0, 0.0, 0.1)
show_only_failures = st.sidebar.checkbox("Only show failures (coverage == 0)", value=False)

st.sidebar.markdown("---")
st.sidebar.write("Tip: run eval again to refresh reports:")
st.sidebar.code("python -m app.eval.run_eval --mode " + mode)

if not report_path or not Path(report_path).exists():
    st.warning("Report JSON not found. Check path in sidebar.")
    st.stop()

if not csv_path or not Path(csv_path).exists():
    st.warning("Per-query CSV not found. Check path in sidebar.")
    st.stop()

report = _load_json(report_path)
df = _load_csv(csv_path)

# Normalize columns we expect
for col in ["answer_coverage", "answer", "citations_formatted", "answer_preview", "q", "mode"]:
    if col not in df.columns:
        df[col] = ""

# Top metrics
m = report.get("metrics", {})
meta = report.get("meta", {})

st.subheader("Run summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Mode", str(meta.get("mode", mode)))
c2.metric("Queries", str(meta.get("n_queries", len(df))))
c3.metric("Answer coverage", f"{m.get('answer_coverage', 0.0):.2f}")
c4.metric("Report time", str(meta.get("created_at", "")))

# Retrieval metrics table
st.subheader("Aggregate metrics")
metrics_rows = []
for k in meta.get("ks", [1, 3, 5, 10]):
    metrics_rows.append(
        {
            "k": k,
            "recall@k": float(m.get(f"recall@{k}", 0.0)),
            "mrr@k": float(m.get(f"mrr@{k}", 0.0)),
            "ndcg@k": float(m.get(f"ndcg@{k}", 0.0)),
        }
    )
st.dataframe(pd.DataFrame(metrics_rows), use_container_width=True)

# Filter
st.subheader("Per-query analysis")

df2 = df.copy()
df2["answer_coverage"] = pd.to_numeric(df2["answer_coverage"], errors="coerce").fillna(0.0)

if show_only_failures:
    df2 = df2[df2["answer_coverage"] == 0.0]
else:
    df2 = df2[df2["answer_coverage"] >= min_cov]

# Sort: failures first, then lowest coverage
df2 = df2.sort_values(by=["answer_coverage"], ascending=True)

st.write(f"Showing **{len(df2)}** / {len(df)} queries")

# Show table
show_cols = [
    "q",
    "answer_coverage",
    "recall@10",
    "mrr@10",
    "ndcg@10",
    "must_include",
    "cited_ids",
    "retrieved_ids_top10",
]
show_cols = [c for c in show_cols if c in df2.columns]
st.dataframe(df2[show_cols], use_container_width=True, height=320)

# Drill-down
st.subheader("Drill-down")
if len(df2) == 0:
    st.info("No rows match current filters.")
    st.stop()

idx = st.selectbox("Select row", list(df2.index), format_func=lambda i: df2.loc[i, "q"])
row = df.loc[idx].to_dict()

st.markdown("### Question")
st.write(row.get("q", ""))

st.markdown("### Answer")
st.write(row.get("answer", row.get("answer_preview", "")))

st.markdown("### Citations (formatted)")
st.code(row.get("citations_formatted", ""), language="text")

st.markdown("### IDs (debug)")
st.write(
    {
        "must_include": row.get("must_include", ""),
        "cited_ids": row.get("cited_ids", ""),
        "retrieved_ids_top10": row.get("retrieved_ids_top10", ""),
        "coverage": row.get("answer_coverage", ""),
    }
)
