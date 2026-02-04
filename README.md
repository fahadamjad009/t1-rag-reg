# T1-RAG-REG + embedded T1-EVAL-GENAI

Evaluation-first Regulatory RAG over public Australian guidance
(APRA + AUSTRAC).

This project is built to be:
- Inspectable
- Reproducible
- Auditable
- Fail-closed (no confident answers without evidence)

## Non-goals
- Not legal advice
- No internal or restricted data
- No “zero hallucinations” claims

## Week-1 Definition of Done
A stranger can run:
- `docker compose up`
- seed the corpus
- run an evaluation gate

And then:
- call `/search`
- call `/answer`
- receive cited outputs
- inspect an evaluation report

## Corpus v0 (locked)
- APRA prudential policy landing page
- AUSTRAC AML/CTF programs guidance

## High-level architecture
1. Fetch public guidance pages
2. Extract main text
3. Chunk with overlap
4. Embed using local sentence-transformers
5. Index with FAISS (cosine-style via normalized vectors)
6. Retrieve + fail-closed answer
7. Enforce quality with eval gate in CI
