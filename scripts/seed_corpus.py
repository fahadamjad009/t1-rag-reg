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
from typing import List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Corpus v0 (locked sources)
# ----------------------------
SOURCES = [
    {
        "source_group": "apra",
        "start_urls": [
            "https://www.apra.gov.au/prudential-policy",
        ],
        "allow_domains": ["www.apra.gov.au", "apra.gov.au"],
        "max_pages": 25,
    },
    {
        "source_group": "austrac",
        "start_urls": [
            # ✅ seed the exact pages your eval questions reference
            "https://www.austrac.gov.au/business/new-to-austrac/your-obligations",
            "https://www.austrac.gov.au/business/new-to-austrac/who-and-what-we-regulate",
            "https://www.austrac.gov.au/business/core-guidance/amlctf-programs",
            "https://www.austrac.gov.au/work-with-austrac",
        ],
        "allow_domains": ["www.austrac.gov.au", "austrac.gov.au"],
        "max_pages": 40,  # bump a bit since we added multiple start pages
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
    source_id: str          # ✅ per-page source_id
    source_group: str       # ✅ group label: apra/austrac


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
    links: List[str] = []
    for a in soup.select("a[href]"):
        href = (a.get("href") or "").strip()
        if not href:
            continue
        abs_url = urljoin(base_url, href)
        links.append(_canonicalize(abs_url))
    return links


def _slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s


def url_to_source_id(url: str, prefix: str) -> str:
    """
    Build a stable source_id like:
      austrac_www_austrac_gov_au_business_new_to_austrac_your_obligations
    """
    u = urlparse(url)
    host = _slugify(u.netloc)
    path = _slugify(u.path)
    if not path:
        path = "root"
    return f"{prefix}_{host}_{path}"


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


def crawl_source(
    source_group: str,
    start_urls: List[str],
    allow_domains: List[str],
    max_pages: int,
) -> List[Page]:
    seen = set()
    queue = [_canonicalize(u) for u in start_urls]
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

        # always enqueue links (even if we skip text)
        links = _extract_links(final_url, html)
        for link in links:
            if link not in seen and _is_allowed(link, allow_domains):
                queue.append(link)

        # skip ultra-short pages (often nav/landing)
        if len(text) < 400:
            time.sleep(SLEEP_SECONDS)
            continue

        fetched_at = datetime.now(timezone.utc).isoformat()

        # ✅ per-page source_id derived from URL
        prefix = source_group  # "austrac" or "apra"
        sid = url_to_source_id(final_url, prefix=prefix)

        pages.append(Page(url=final_url, title=title, text=text, fetched_at=fetched_at,
                          source_id=sid, source_group=source_group))
        print(f"[OK] ({len(pages)}/{max_pages}) {source_group}: {final_url} -> {sid}")

        time.sleep(SLEEP_SECONDS)

    return pages


def write_pages(pages: List[Page]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    for p in pages:
        doc_id = _sha256(p.url)

        raw_path = RAW_DIR / f"{doc_id}.txt"
        meta_path = META_DIR / f"{doc_id}.json"

        raw_path.write_text(p.text, encoding="utf-8")

        meta = {
            "doc_id": doc_id,
            "source_group": p.source_group,  # ✅ group label
            "source_id": p.source_id,         # ✅ per-page id (used by retriever/eval)
            "url": p.url,
            "title": p.title,
            "fetched_at": p.fetched_at,
            "char_len": len(p.text),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    print(f"[INFO] Writing corpus to: {DATA_DIR.resolve()}")
    total_pages = 0
    all_pages: List[Page] = []

    for s in SOURCES:
        pages = crawl_source(
            source_group=s["source_group"],
            start_urls=s["start_urls"],
            allow_domains=s["allow_domains"],
            max_pages=int(s["max_pages"]),
        )
        all_pages.extend(pages)
        total_pages += len(pages)

    write_pages(all_pages)

    print(f"[DONE] total_pages={total_pages}")
    print(f"[DONE] raw_dir={RAW_DIR.resolve()}")
    print(f"[DONE] meta_dir={META_DIR.resolve()}")


if __name__ == "__main__":
    main()
