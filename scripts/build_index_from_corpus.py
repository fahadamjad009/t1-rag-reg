# scripts/build_index_from_corpus.py
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CORPUS_DIR = DATA_DIR / "corpus"
RAW_DIR = CORPUS_DIR / "raw"
META_DIR = CORPUS_DIR / "meta"

INDEX_DIR = DATA_DIR / "index"
INDEX_DIR.mkdir(parents=True, exist_ok=True)

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))          # chars
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))    # chars


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _chunk_text(text: str, size: int, overlap: int) -> List[str]:
    t = (text or "").strip()
    if not t:
        return []
    if size <= 0:
        return [t]
    if overlap < 0:
        overlap = 0
    if overlap >= size:
        overlap = max(0, size // 4)

    chunks: List[str] = []
    start = 0
    n = len(t)
    while start < n:
        end = min(n, start + size)
        chunk = t[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def _load_docs() -> List[Tuple[Dict[str, Any], str]]:
    """
    Returns list of (meta, text) pairs.
    meta files are keyed by doc_id; raw is doc_id.txt.
    """
    if not RAW_DIR.exists() or not META_DIR.exists():
        raise FileNotFoundError(f"Missing corpus dirs: {RAW_DIR} / {META_DIR}")

    meta_files = sorted(META_DIR.glob("*.json"))
    docs: List[Tuple[Dict[str, Any], str]] = []

    for mp in meta_files:
        meta = _read_json(mp)
        doc_id = meta.get("doc_id") or mp.stem
        raw_path = RAW_DIR / f"{doc_id}.txt"
        if not raw_path.exists():
            continue
        text = _read_text(raw_path)
        if len(text.strip()) < 200:
            continue
        docs.append((meta, text))

    return docs


def build_chunks() -> List[Dict[str, Any]]:
    docs = _load_docs()
    out: List[Dict[str, Any]] = []

    for meta, text in docs:
        doc_id = str(meta.get("doc_id") or "")
        url = str(meta.get("url") or "")
        title = str(meta.get("title") or "")
        source_id = str(meta.get("source_id") or "")
        source_group = str(meta.get("source_group") or "")
        fetched_at = str(meta.get("fetched_at") or "")

        parts = _chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
        for i, part in enumerate(parts):
            chunk_local = f"{i:04d}"
            chunk_id = f"{doc_id}_{chunk_local}"
            out.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_id,
                    "source_group": source_group,
                    "source_id": source_id,
                    "url": url,
                    "title": title,
                    "retrieved_at": fetched_at,
                    "content_hash": _sha256(part),
                    "text": part,
                }
            )

    return out


def embed_chunks(chunks: List[Dict[str, Any]]) -> np.ndarray:
    model = SentenceTransformer(EMBED_MODEL_NAME)
    texts = [c["text"] for c in chunks]
    emb = model.encode(texts, normalize_embeddings=True, show_progress_bar=True)
    emb = np.asarray(emb, dtype="float32")
    return emb


def write_index(chunks: List[Dict[str, Any]], emb: np.ndarray) -> None:
    if len(chunks) == 0:
        raise RuntimeError("No chunks produced; corpus may be empty.")
    if emb.shape[0] != len(chunks):
        raise RuntimeError(f"Embedding count mismatch: emb={emb.shape[0]} chunks={len(chunks)}")

    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine via normalized embeddings
    index.add(emb)

    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    stats = {
        "built_at": _utc_now(),
        "embed_model": EMBED_MODEL_NAME,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "n_docs_meta": len(list(META_DIR.glob("*.json"))),
        "n_chunks": len(chunks),
        "faiss_dim": int(dim),
    }
    (INDEX_DIR / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print("[INFO] Building chunks from corpus...")
    chunks = build_chunks()
    print(f"[INFO] chunks={len(chunks)} -> embedding...")

    emb = embed_chunks(chunks)
    print("[INFO] writing index...")
    write_index(chunks, emb)

    print("[DONE] Wrote:")
    print(f"  {INDEX_DIR / 'chunks.json'}")
    print(f"  {INDEX_DIR / 'faiss.index'}")
    print(f"  {INDEX_DIR / 'stats.json'}")


if __name__ == "__main__":
    main()
