# app/rag/ingest.py
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# CPU embeddings
from sentence_transformers import SentenceTransformer

try:
    import faiss
except Exception as e:
    raise RuntimeError("faiss not installed in container") from e


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "corpus" / "raw"
META_DIR = DATA_DIR / "corpus" / "meta"
INDEX_DIR = DATA_DIR / "index"

CHUNK_MAX_WORDS = int(os.getenv("CHUNK_MAX_WORDS", "220"))
CHUNK_OVERLAP_WORDS = int(os.getenv("CHUNK_OVERLAP_WORDS", "40"))

# embeddings model (fast + good baseline)
EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")


def _utc_now_iso() -> str:
    # Example: 2026-02-07T09:15:30Z
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_text(text: str) -> str:
    b = (text or "").encode("utf-8", errors="ignore")
    return hashlib.sha256(b).hexdigest()


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    source_id: str
    url: str
    title: str
    text: str
    # New provenance anchors
    retrieved_at: Optional[str] = None
    content_hash: Optional[str] = None


def _words(s: str) -> List[str]:
    return re.findall(r"\S+", s)


def chunk_text(text: str, max_words: int, overlap_words: int) -> List[str]:
    w = _words(text)
    if not w:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(w):
        end = min(start + max_words, len(w))
        chunk = " ".join(w[start:end])
        chunks.append(chunk)
        if end == len(w):
            break
        start = max(0, end - overlap_words)
    return chunks


def load_docs() -> List[Dict]:
    """
    Loads (meta + raw text) into a single doc object.
    We also attach:
      - retrieved_at: when this pipeline ran (fallback)
      - content_hash: sha256 of the doc text
    If your meta files already include retrieved_at, we keep it.
    """
    docs: List[Dict] = []
    fallback_retrieved_at = _utc_now_iso()

    for meta_path in META_DIR.glob("*.json"):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        doc_id = meta["doc_id"]
        raw_path = RAW_DIR / f"{doc_id}.txt"
        if not raw_path.exists():
            continue

        text = raw_path.read_text(encoding="utf-8", errors="ignore").strip()
        if len(text) < 200:
            continue

        # Prefer meta timestamps if present; otherwise fallback to "now"
        retrieved_at = meta.get("retrieved_at") or fallback_retrieved_at
        content_hash = meta.get("content_hash") or _sha256_text(text)

        docs.append({**meta, "text": text, "retrieved_at": retrieved_at, "content_hash": content_hash})

    return docs


def main() -> None:
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    docs = load_docs()
    print(f"[INFO] docs={len(docs)}")

    chunks: List[Chunk] = []
    for d in docs:
        pieces = chunk_text(d["text"], CHUNK_MAX_WORDS, CHUNK_OVERLAP_WORDS)
        for i, p in enumerate(pieces):
            chunk_id = f"{d['doc_id']}_{i:04d}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    doc_id=d["doc_id"],
                    source_id=d.get("source_id", ""),
                    url=d.get("url", ""),
                    title=d.get("title", ""),
                    text=p,
                    retrieved_at=d.get("retrieved_at"),
                    content_hash=d.get("content_hash"),
                )
            )

    print(f"[INFO] chunks={len(chunks)}")

    # embed
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c.text for c in chunks]
    emb = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)
    emb = np.asarray(emb, dtype="float32")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine if normalized
    index.add(emb)

    # save index + chunk store
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    chunk_store = [
        {
            "chunk_id": c.chunk_id,
            "doc_id": c.doc_id,
            "source_id": c.source_id,
            "url": c.url,
            "title": c.title,
            "text": c.text,
            # New per-chunk provenance (doc-derived but stored on every chunk row)
            "retrieved_at": c.retrieved_at,
            "content_hash": c.content_hash,
        }
        for c in chunks
    ]
    (INDEX_DIR / "chunks.json").write_text(
        json.dumps(chunk_store, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    stats = {
        "docs": len(docs),
        "chunks": len(chunks),
        "embed_model": EMBED_MODEL_NAME,
        "chunk_max_words": CHUNK_MAX_WORDS,
        "chunk_overlap_words": CHUNK_OVERLAP_WORDS,
        "dim": int(dim),
        # Optional: stamp the build itself
        "built_at": _utc_now_iso(),
    }
    (INDEX_DIR / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"[DONE] index_dir={INDEX_DIR.resolve()}")
    print("[DONE] wrote: faiss.index, chunks.json, stats.json")


if __name__ == "__main__":
    main()
