# T1-RAG-REG — Evaluation-First Regulatory RAG

> A retrieval-augmented question-answering system over public Australian financial-regulation guidance (APRA + AUSTRAC), built around **measured retrieval quality** and a **fail-closed** answering policy: it refuses rather than guesses when the evidence is weak.

![Python](https://img.shields.io/badge/python-3.11-blue)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Retrieval](https://img.shields.io/badge/retrieval-BM25%20%2B%20dense%20%2B%20RRF-555)
![Answering](https://img.shields.io/badge/answering-extractive%20(no%20LLM%20API)-555)
![License](https://img.shields.io/badge/license-MIT-green)

Most RAG demos wire an LLM to a vector store and stop there. The harder — and more *regulated-industry-relevant* — problems are: did retrieval actually surface the right source, and what does the system do when it didn't? T1-RAG-REG is built around those two questions. It ships an information-retrieval evaluation harness (Recall / MRR / nDCG), a hybrid retriever with reciprocal-rank fusion, and an answer layer that returns cited extracts or an explicit refusal — never an unsupported claim.

---

## Why this design

In a compliance setting, a confidently-wrong answer is worse than no answer. The system is therefore built to be:

- **Inspectable** — every answer carries its source citations and a calibrated confidence score; a `/debug/retrieve` endpoint exposes the raw ranking.
- **Reproducible** — the corpus is public, the index build is deterministic, and the evaluation is a single command that writes a timestamped report.
- **Fail-closed** — if the best retrieved chunk scores below a configurable gate, the system refuses with a machine-readable reason instead of answering.
- **Dependency-light** — answers are produced **extractively** (sentence selection from retrieved chunks), so the core path needs **no external LLM API**, no API keys, and no per-query cost. Embeddings run locally via `sentence-transformers`.

**Non-goals:** this is not legal advice, uses only public guidance (no internal/restricted data), and makes no "zero hallucination" claims.

---

## Architecture

```mermaid
flowchart LR
    A[Public guidance<br/>APRA · AUSTRAC · legislation] -->|seed_corpus| B[Raw corpus<br/>+ metadata]
    B -->|build_index| C[Chunks 900/150<br/>MiniLM-L6-v2 · FAISS 384-d]
    Q[Query] --> R{Retriever}
    C --> R
    R -->|vector| V[FAISS cosine]
    R -->|bm25| K[BM25Okapi]
    V --> F[RRF fusion<br/>chunk + source level]
    K --> F
    F --> G[Fail-closed gate<br/>calibrated confidence]
    G -->|pass| X[Extractive answer<br/>+ citations]
    G -->|fail| Z[Refusal<br/>+ reason]
```

**Pipeline:** fetch public pages → extract main text → chunk with overlap → embed locally → index in FAISS → retrieve → gate → answer or refuse. Evaluation runs against the same retrieve path and gates CI.

### Retrieval modes
| Mode | What it does |
|---|---|
| `vector` | Dense FAISS search over normalized MiniLM embeddings (cosine). |
| `bm25` | Lexical `BM25Okapi` over a deterministic tokenizer. |
| `hybrid_rrf` | Reciprocal Rank Fusion of dense + lexical, fused **at source level** so one long page can't dominate top-k. |

All modes apply **deterministic query rewriting** (regulatory-term expansion, no LLM), **low-quality-source filtering**, and feed the same **per-mode confidence calibration** that maps raw scores into a comparable `[0,1]` band before the fail-closed gate.

---

## Quickstart

### Docker (recommended)
```bash
cp .env.example .env
docker compose up --build        # serves on http://localhost:8000
```

### Local (Python 3.11)
```bash
python -m venv .venv && . .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env

# 1. seed the public corpus (APRA + AUSTRAC + AML/CTF Act)
python -m scripts.seed_corpus

# 2. build the FAISS index + chunks
python -m scripts.build_index_from_corpus

# 3. serve
uvicorn app.main:app --reload
```

> The committed index lets you skip steps 1–2 and query immediately. Re-seed only to refresh the corpus.

### Try it
```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/query \
  -H "content-type: application/json" \
  -d '{"q":"What is an AML/CTF program and who needs one?","mode":"hybrid_rrf","top_k":5}'
```

A successful response carries the extracted `answer`, `citations`, `evidence` (the supporting chunks), `confidence`, and `calibrated_confidence`. A weak-evidence query returns `{"status":"refused","reason":"...","calibrated_confidence":...}` — the fail-closed path in action.

### API
| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check. |
| `POST` | `/query` | Retrieve + extractive answer (typed request). |
| `POST` | `/answer` | Same, accepts a free-form JSON body. |
| `POST` | `/debug/retrieve` | Raw ranked hits + gate decision, no answering. |
| `GET` | `/debug/build` | Active config + calibration bands (proves which build is running). |

---

## Evaluation

The point of the project. Retrieval is scored against a goldset of regulatory questions, each labelled with the source document(s) that *should* be retrieved.

```bash
python -m app.eval.run_eval --mode hybrid_rrf --ks 1,3,5,10
python -m app.eval.report_plots         # writes Recall/MRR/nDCG plots + summary
python -m app.eval.smoke_eval           # CI gate: fails if answer_coverage < 0.95
```

Each run writes a timestamped `eval_report_<mode>_<ts>.json` and a per-query CSV to `reports/`, plus an updated leaderboard.

**Current results — `hybrid_rrf`, goldset n = 5** *(small by design at this stage; goldset expansion is in progress, see Roadmap):*

| Metric | @1 | @3 | @5 | @10 |
|---|---:|---:|---:|---:|
| Recall | 0.00 | 0.30 | 0.40 | 1.00 |
| MRR | 0.00 | 0.20 | 0.24 | 0.29 |
| nDCG | 0.00 | 0.22 | 0.26 | 0.47 |

`answer_coverage = 1.00` (every gold "must-include" source was cited within top-10).

**How to read this honestly:** with n = 5, these numbers are directional, not statistical. The signal worth taking is the *shape*: recall reaches 1.0 by k=10 (the right sources are in the candidate pool), while MRR/nDCG stay modest — meaning the **ranking** has headroom even though **recall** is solved. Improving rank position (not coverage) is the next measurable target. Reporting that distinction is the skill the harness exists to demonstrate.

---

## Corpus

Public Australian regulatory guidance only, crawled from official domains:

- **APRA** — prudential policy.
- **AUSTRAC** — AML/CTF programs, customer identification & verification (KYC), reporting (SMR), enrolment/registration obligations.
- **legislation.gov.au** — AML/CTF Act series.

Indexed snapshot: **132 source documents → 4,207 chunks** (900-char chunks, 150-char overlap), embedded with `all-MiniLM-L6-v2` (384-d), stored in FAISS. See `data/index/stats.json` for the exact build manifest.

---

## Project structure
```
app/
  main.py            FastAPI app: /query /answer /debug/* + global JSON error handling
  rag/
    retriever.py     vector / bm25 / hybrid_rrf + RRF + source-level fusion
    bm25.py          BM25Okapi index + tokenizer
    extractive.py    citation-grounded extractive answering
    answer.py        answer assembly helpers
    calibration.py   per-mode raw-score -> confidence bands
    query_rewrite.py deterministic regulatory query expansion
    ingest.py        fetch + clean + chunk
  eval/
    run_eval.py      Recall / MRR / nDCG / answer_coverage runner
    report_plots.py  plots + markdown summary
    smoke_eval.py    CI quality gate
    dashboard.py     Streamlit eval explorer
    eval_dataset.jsonl   retrieval goldset (source-labelled)
scripts/
  seed_corpus.py             crawl public guidance
  build_index_from_corpus.py chunk + embed + FAISS
  run_eval.py                end-to-end eval against a live API
data/eval/questions.jsonl    answer-content questions
reports/                     evaluation outputs (leaderboard + summaries)
```

---

## Tech stack
FastAPI · Uvicorn · Pydantic · FAISS (`faiss-cpu`) · `sentence-transformers` (MiniLM-L6-v2) · `rank-bm25` · trafilatura/BeautifulSoup (ingestion) · pandas + matplotlib (eval/plots) · Streamlit (eval dashboard) · Docker. Lint/format/test: ruff · black · pytest.

## Roadmap
- Expand the retrieval goldset beyond n = 5 for statistically meaningful metrics.
- Add a cross-encoder re-ranking pass to lift MRR/nDCG (recall is already saturated at k=10).
- Scheduled corpus refresh with index-diff reporting.

## Disclaimer
Informational only; **not legal or compliance advice.** Built on public guidance that may change — always verify against the primary source.

## License
MIT — see `LICENSE`.
