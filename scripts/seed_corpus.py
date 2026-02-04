# scripts/seed_corpus.py
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Corpus v0 (locked sources)
# ----------------------------
SOURCES = [
    {
        "source_id": "apra_prudential_policy",
        "base_url": "https://www.apra.gov.au/prudential-policy",
        "allow_domains": ["www.apra.gov.au", "apra.gov.au"],
        "max_pages": 25,
    },
    {
        "source_id": "austrac_aml_ctf_programs",
        "base_url": "https://www.austrac.gov.au/business/core-guidance/amlctf-programs",
        "allow_domains": ["www.austrac.gov.au", "austrac.gov.au"],
        "max_pages": 25,
    },
]

# polite crawling
REQUEST_TIMEOUT = 30
SLEEP_SECONDS = 1.0

USER_AGENT = "t1-rag-reg/0.1 (corpus seeder; educational)"
HEADERS = {"User-Agent": USER_AGENT}

DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "corpus" / "raw"
META_DIR = DATA_DIR / "corpus" / "meta"


@dataclass
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


def _is_allowed(url: str, allow_domains: List[str]) -> bool:
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return False
    return any(host == d or host.endswith("." + d) for d in allow_domains)


def _canonicalize(url: str) -> str:
    # remove fragments, keep query (some sites use it meaningfully)
    u = urlparse(url)
    return u._replace(fragment="").geturl()


def _sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _clean_text(html: str) -> Tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    # remove noisy tags
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()

    # get visible text
    text = soup.get_text(" ", strip=True)

    # normalize whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return title, text


def _extract_links(base_url: str, html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links = []
    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        links.append(_canonicalize(abs_url))
    return links


def fetch_url(url: str) -> Optional[Tuple[str, str]]:
    """Return (final_url, html) or None."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        r.raise_for_status()
        final_url = _canonicalize(r.url)
        return final_url, r.text
    except Exception as e:
        print(f"[WARN] fetch failed: {url} -> {e}")
        return None


def crawl_source(source_id: str, base_url: str, allow_domains: List[str], max_pages: int) -> List[Page]:
    seen = set()
    queue = [ _canonicalize(base_url) ]
    pages: List[Page] = []

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        if not _is_allowed(url, allow_domains):
            continue

        res = fetch_url(url)
        if not res:
            continue
        final_url, html = res

        title, text = _clean_text(html)

        # skip ultra-short pages
        if len(text) < 400:
            # still allow collecting links from it
            links = _extract_links(final_url, html)
            for link in links:
                if link not in seen and _is_allowed(link, allow_domains):
                    queue.append(link)
            continue

        fetched_at = datetime.now(timezone.utc).isoformat()
        pages.append(Page(url=final_url, title=title, text=text, fetched_at=fetched_at))
        print(f"[OK] ({len(pages)}/{max_pages}) {source_id}: {final_url}")

        # enqueue more links
        links = _extract_links(final_url, html)
        for link in links:
            if link not in seen and _is_allowed(link, allow_domains):
                queue.append(link)

        time.sleep(SLEEP_SECONDS)

    return pages


def write_pages(source_id: str, pages: List[Page]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    for p in pages:
        doc_id = _sha256(p.url)

        raw_path = RAW_DIR / f"{doc_id}.txt"
        meta_path = META_DIR / f"{doc_id}.json"

        raw_path.write_text(p.text, encoding="utf-8")

        meta = {
            "doc_id": doc_id,
            "source_id": source_id,
            "url": p.url,
            "title": p.title,
            "fetched_at": p.fetched_at,
            "char_len": len(p.text),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print(f"[INFO] Writing corpus to: {DATA_DIR.resolve()}")
    total_pages = 0

    for s in SOURCES:
        pages = crawl_source(
            source_id=s["source_id"],
            base_url=s["base_url"],
            allow_domains=s["allow_domains"],
            max_pages=int(s["max_pages"]),
        )
        write_pages(s["source_id"], pages)
        total_pages += len(pages)

    print(f"[DONE] total_pages={total_pages}")
    print(f"[DONE] raw_dir={RAW_DIR.resolve()}")
    print(f"[DONE] meta_dir={META_DIR.resolve()}")


if __name__ == "__main__":
    main()
