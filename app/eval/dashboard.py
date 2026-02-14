from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
INDEX_CHUNKS = REPO_ROOT / "data" / "index" / "chunks.json"


# -----------------------------
# File discovery helpers
# -----------------------------
def list_runs(mode: str) -> List[Path]:
    return sorted(
        REPORTS_DIR.glob(f"eval_report_{mode}_*.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )


def load_report(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_matching_csv(report_path: Path) -> Optional[Path]:
    # eval_report_hybrid_rrf_YYYYMMDD_HHMMSS.json -> eval_per_query_hybrid_rrf_YYYYMMDD_HHMMSS.csv
    name = report_path.name.replace("eval_report_", "eval_per_query_").replace(".json", ".csv")
    p = REPORTS_DIR / name
    return p if p.exists() else None


def find_plot(report_path: Path, kind: str) -> Optional[Path]:
    # eval_report_hybrid_rrf_YYYYMMDD_HHMMSS.json -> plot_<kind>_hybrid_rrf_YYYYMMDD_HHMMSS.png
    stamp = report_path.name.replace("eval_report_", "").replace(".json", "")
    p = REPORTS_DIR / f"plot_{kind}_{stamp}.png"
    return p if p.exists() else None


@st.cache_data(show_spinner=False)
def load_chunks_map(chunks_path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Map source_id/doc_id -> best-effort metadata (url/title).
    Used to create clickable source links.
    """
    if not chunks_path.exists():
        return {}

    data = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data.get("chunks") if isinstance(data, dict) else data
    if not isinstance(chunks, list):
        return {}

    out: Dict[str, Dict[str, Any]] = {}
    for ch in chunks:
        if not isinstance(ch, dict):
            continue
        sid = (ch.get("source_id") or ch.get("doc_id") or "").strip()
        if not sid:
            continue
        if sid not in out:
            out[sid] = {
                "title": (ch.get("title") or "").strip(),
                "url": (ch.get("url") or "").strip(),
                "source_id": sid,
            }
    return out


def _safe_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _metric_series(metrics: Dict[str, Any], ks: List[int], prefix: str) -> pd.DataFrame:
    rows = []
    for k in ks:
        rows.append({"k": int(k), "value": _safe_float(metrics.get(f"{prefix}@{k}", 0.0))})
    return pd.DataFrame(rows)


def _mk_link(sid: str, meta: Dict[str, Dict[str, Any]]) -> str:
    m = meta.get(sid) or {}
    url = (m.get("url") or "").strip()
    title = (m.get("title") or "").strip()
    label = title if title else sid
    if url:
        # show title but keep sid visible for traceability
        return f"[{label}]({url})  \n`{sid}`"
    return f"`{sid}`"


def _parse_sids_from_citations_formatted(text: str) -> List[str]:
    """
    citations_formatted looks like:
      [1] source_id — evidence | [2] source_id — evidence
    We'll extract source_id segments best-effort.
    """
    out: List[str] = []
    t = (text or "").strip()
    if not t:
        return out

    for part in t.split("|"):
        part = part.strip()
        if not part or "]" not in part:
            continue
        try:
            after = part.split("]", 1)[1].strip()
            sid = after.split("—", 1)[0].strip()
            if sid:
                out.append(sid)
        except Exception:
            continue

    # dedupe preserve order
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _parse_must_include(value: Any) -> List[str]:
    """
    must_include may be:
      - list (ideal)
      - string representation like "['a','b']"
      - JSON-ish string like '["a","b"]'
      - empty/NaN
    Best-effort parse.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    if isinstance(value, float) and pd.isna(value):
        return []
    s = str(value).strip()
    if not s:
        return []
    # try json
    try:
        v = json.loads(s)
        if isinstance(v, list):
            return [str(x).strip() for x in v if str(x).strip()]
    except Exception:
        pass
    # try a simple bracketed list "['a','b']"
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        parts = [p.strip().strip("'").strip('"') for p in inner.split(",")]
        return [p for p in parts if p]
    return [s]


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="T1-RAG-REG • Eval Dashboard", layout="wide")
st.title("T1-RAG-REG — Evaluation Dashboard")

with st.sidebar:
    st.header("Load")
    mode = st.selectbox("Mode", ["hybrid_rrf", "vector", "bm25"], index=0)

    runs = list_runs(mode)
    if not runs:
        st.warning(f"No reports found in {REPORTS_DIR} for mode={mode}. Run: python -m app.eval.run_eval --mode {mode}")
        st.stop()

    run_path = st.selectbox("Run", runs, format_func=lambda p: p.name)
    csv_path = find_matching_csv(run_path)

    st.caption("Artifacts")
    st.write(f"Report: `{run_path.name}`")
    if csv_path:
        st.write(f"CSV: `{csv_path.name}`")
    else:
        st.write("CSV: (missing)")

    st.divider()
    st.header("Filters")

    show_only_failures = st.checkbox("Show only failures (non-ok)", value=False)

    # If show_only_failures is ON, default to non-ok values (user can still pick)
    if show_only_failures:
        default_failure = ["no_must_include", "not_retrieved", "retrieved_but_not_cited", "other"]
    else:
        default_failure = ["ok", "no_must_include", "not_retrieved", "retrieved_but_not_cited", "other"]

    failure_filter = st.multiselect(
        "failure_reason",
        ["ok", "no_must_include", "not_retrieved", "retrieved_but_not_cited", "other"],
        default=default_failure,
    )

    search = st.text_input("Search q contains", value="").strip()
    min_cov = st.slider("Min answer_coverage", 0.0, 1.0, 0.0, 0.05)


report = load_report(run_path)
meta = report.get("meta", {}) or {}
metrics = report.get("metrics", {}) or {}

# load CSV if present
df: Optional[pd.DataFrame] = None
if csv_path and csv_path.exists():
    df = pd.read_csv(csv_path)

chunks_meta = load_chunks_map(INDEX_CHUNKS)

# -----------------------------
# Run metadata + metrics
# -----------------------------
st.subheader("Run metadata")
st.write(
    {
        "mode": meta.get("mode") or mode,
        "created_at": meta.get("created_at"),
        "n_queries": meta.get("n_queries"),
        "ks": meta.get("ks"),
        "dataset": meta.get("dataset"),
        "report_file": run_path.name,
        "csv_file": csv_path.name if csv_path else None,
    }
)

st.subheader("Metrics (averages)")
if metrics:
    rows = [{"metric": k, "value": _safe_float(v)} for k, v in metrics.items()]
    df_metrics = pd.DataFrame(rows).sort_values(by="metric")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
else:
    st.info("No 'metrics' found in report JSON. (Your run_eval should be writing them.)")

# KPIs across top
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("answer_coverage", f"{_safe_float(metrics.get('answer_coverage', 0.0)):.3f}")
with k2:
    st.metric("recall@10", f"{_safe_float(metrics.get('recall@10', 0.0)):.3f}")
with k3:
    st.metric("mrr@10", f"{_safe_float(metrics.get('mrr@10', 0.0)):.3f}")
with k4:
    st.metric("ndcg@10", f"{_safe_float(metrics.get('ndcg@10', 0.0)):.3f}")

# @k charts
ks = meta.get("ks") or [1, 3, 5, 10]
try:
    ks = [int(x) for x in ks]
except Exception:
    ks = [1, 3, 5, 10]

c1, c2, c3 = st.columns(3)
with c1:
    st.caption("Recall@k")
    st.line_chart(_metric_series(metrics, ks, "recall"), x="k", y="value")
with c2:
    st.caption("MRR@k")
    st.line_chart(_metric_series(metrics, ks, "mrr"), x="k", y="value")
with c3:
    st.caption("nDCG@k")
    st.line_chart(_metric_series(metrics, ks, "ndcg"), x="k", y="value")

# Optional saved plots
st.subheader("Saved plots (optional PNGs)")
p_recall = find_plot(run_path, "recall")
p_mrr = find_plot(run_path, "mrr")
p_ndcg = find_plot(run_path, "ndcg")

pc1, pc2, pc3 = st.columns(3)
with pc1:
    st.caption("Recall@k (PNG)")
    if p_recall:
        st.image(str(p_recall), use_container_width=True)
    else:
        st.info("PNG not found for this run. (Optional) create later.")
with pc2:
    st.caption("MRR@k (PNG)")
    if p_mrr:
        st.image(str(p_mrr), use_container_width=True)
    else:
        st.info("PNG not found for this run. (Optional) create later.")
with pc3:
    st.caption("nDCG@k (PNG)")
    if p_ndcg:
        st.image(str(p_ndcg), use_container_width=True)
    else:
        st.info("PNG not found for this run. (Optional) create later.")

st.divider()

# -----------------------------
# Per-query table + drilldown
# -----------------------------
st.subheader("Per-query breakdown")

if df is None:
    st.info("Per-query CSV not found for this run.")
    st.stop()

work = df.copy()

# apply filters only if columns exist
if "failure_reason" in work.columns:
    if show_only_failures:
        work = work[work["failure_reason"].astype(str) != "ok"]
    if failure_filter:
        work = work[work["failure_reason"].astype(str).isin([str(x) for x in failure_filter])]

if search and "q" in work.columns:
    work = work[work["q"].astype(str).str.contains(search, case=False, na=False)]

if "answer_coverage" in work.columns:
    work = work[work["answer_coverage"] >= float(min_cov)]

# sorting strategy:
# - if viewing failures: worst-first (lowest coverage, then lowest ndcg)
# - else: best-first (highest coverage, then highest ndcg)
if len(work) > 0:
    if show_only_failures:
        sort_cols = [c for c in ["answer_coverage", "ndcg@10", "mrr@10", "recall@10"] if c in work.columns]
        if sort_cols:
            work = work.sort_values(by=sort_cols, ascending=[True] * len(sort_cols))
    else:
        sort_cols = [c for c in ["answer_coverage", "ndcg@10", "mrr@10", "recall@10"] if c in work.columns]
        if sort_cols:
            work = work.sort_values(by=sort_cols, ascending=[False] * len(sort_cols))

# Show compact dataframe
cols_to_show = [c for c in ["q", "answer_coverage", "failure_reason", "recall@10", "mrr@10", "ndcg@10"] if c in work.columns]
st.dataframe(work[cols_to_show] if cols_to_show else work, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Inspect a query (answer + citations)")

if len(work) == 0:
    st.info("No rows match your filters.")
    st.stop()

q_list = work["q"].astype(str).tolist() if "q" in work.columns else []
if not q_list:
    st.info("No 'q' column found in CSV.")
    st.stop()

picked_q = st.selectbox("Select query", q_list, index=0)
row = work[work["q"].astype(str) == str(picked_q)].iloc[0].to_dict()

cA, cB, cC = st.columns(3)
with cA:
    st.metric("answer_coverage", _safe_float(row.get("answer_coverage", 0.0)))
with cB:
    st.write({"failure_reason": row.get("failure_reason", "")})
with cC:
    st.write({"mode": meta.get("mode") or mode, "run": run_path.name})

left, right = st.columns([2, 1])

with left:
    st.markdown("### Answer")
    st.write(row.get("answer", ""))

    st.markdown("### Citations (formatted)")
    st.code(str(row.get("citations_formatted", "")), language="text")

with right:
    st.markdown("### Retrieval + constraints")
    st.write(
        {
            "must_include": row.get("must_include", ""),
            "retrieved_ids_top10": row.get("retrieved_ids_top10", ""),
        }
    )

    st.markdown("### Source links (best-effort)")
    ctext = str(row.get("citations_formatted") or "")
    cited_sids = _parse_sids_from_citations_formatted(ctext)
    must_sids = _parse_must_include(row.get("must_include"))
    all_sids = _dedupe_keep_order((must_sids + cited_sids))[:12]

    if not all_sids:
        st.caption("No source ids detected to link.")
    else:
        for sid in all_sids:
            st.markdown(f"- {_mk_link(sid, chunks_meta)}")

st.divider()
st.subheader("Worst queries (quick view)")

quick_cols = [c for c in ["q", "answer_coverage", "failure_reason"] if c in work.columns]
if quick_cols:
    st.write(work[quick_cols].head(10))
else:
    st.write(work.head(10))
