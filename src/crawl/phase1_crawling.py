"""
Phase 1: Web Crawling & Cleaning
Lab Session 1 - From the Unstructured Web to Structured Entities
Course: Web Mining & Semantics — ESILV A4

Strategy: Use the MediaWiki API (MarioWiki + Wikipedia) to obtain clean plain-text
content for Mario-universe entities.  Direct API access avoids HTML boilerplate
without relying on trafilatura (still used as fallback for generic HTML pages).

Deliverable: outputs/crawler_output.jsonl
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False
    logger.warning("trafilatura not found — will use httpx fallback only")


# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

MARIO_WIKI_API = "https://www.mariowiki.com/api.php"
WIKI_API       = "https://en.wikipedia.org/w/api.php"

MARIO_SEED_PAGES = [
    "Mario",
    "Luigi",
    "Princess Peach",
    "Bowser",
    "Yoshi",
    "Toad (Nintendo)",
    "Wario",
    "Donkey Kong",
    "Rosalina",
    "Daisy (Nintendo)",
    "Super Mario Bros.",
    "Super Mario World",
    "Super Mario Odyssey",
    "Mario Kart",
    "Luigi's Mansion",
    "Mushroom Kingdom",
    "Koopa Troopa",
    "Goomba",
    "Shy Guy",
    "Fire Flower",
]

MIN_WORD_COUNT = 200
REQUEST_DELAY  = 1.0
TIMEOUT        = 15


# ──────────────────────────────────────────────────────────────────────────────
# Crawler
# ──────────────────────────────────────────────────────────────────────────────

class MarioWikiAPICrawler:
    """
    Fetches article text via the MediaWiki Action API.
    Tries MarioWiki first, then English Wikipedia as fallback.
    The ?action=query&prop=extracts&explaintext=1 endpoint returns clean
    plain text — ideal for downstream NLP.
    """

    def __init__(self,
                 api_url: str = MARIO_WIKI_API,
                 fallback_api: str = WIKI_API,
                 delay: float = REQUEST_DELAY,
                 timeout: int = TIMEOUT):
        self.api_url  = api_url
        self.fallback = fallback_api
        self.delay    = delay
        self.session  = httpx.Client(
            headers={"User-Agent": "MarioKnowledgeBot/1.0 (ESILV academic project)"},
            timeout=timeout,
            follow_redirects=True,
        )

    def _query_mediawiki(self, api: str, title: str) -> Optional[Dict]:
        """Call the MediaWiki API and return the extract dict, or None."""
        params = {
            "action":          "query",
            "titles":          title,
            "prop":            "extracts|info",
            "explaintext":     "1",
            "exsectionformat": "plain",
            "inprop":          "url",
            "format":          "json",
            "redirects":       "1",
        }
        try:
            resp = self.session.get(api, params=params)
            resp.raise_for_status()
            data  = resp.json()
            pages = data.get("query", {}).get("pages", {})
            for pid, page in pages.items():
                if pid == "-1":
                    return None
                return {
                    "url":        page.get("fullurl", f"{api}?title={title}"),
                    "title":      page.get("title", title),
                    "text":       page.get("extract", ""),
                    "date":       datetime.now().isoformat(),
                    "author":     "",
                    "method":     "mediawiki_api",
                    "source_api": api,
                }
        except Exception as exc:
            logger.warning(f"API error for '{title}' on {api}: {exc}")
        return None

    def _fetch_html_fallback(self, url: str) -> Optional[Dict]:
        """Generic HTTP fetch + trafilatura extraction for non-API pages."""
        if not HAS_TRAFILATURA:
            return None
        try:
            resp = self.session.get(url)
            resp.raise_for_status()
            result = trafilatura.extract(resp.text, include_comments=False, output_format="python")
            if result:
                return {
                    "url":    url,
                    "title":  result.get("title", ""),
                    "text":   result.get("text", ""),
                    "date":   result.get("date", ""),
                    "author": result.get("author", ""),
                    "method": "trafilatura",
                }
        except Exception as exc:
            logger.warning(f"HTML fallback failed for {url}: {exc}")
        return None

    def clean_text(self, text: str) -> str:
        """Remove section headers, empty lines, and very short fragments."""
        lines   = text.split("\n")
        cleaned = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if line.startswith("==") and line.endswith("=="):
                continue  # skip MediaWiki section headers
            if len(line) < 15:
                continue  # skip too-short lines (captions, lone numbers…)
            cleaned.append(line)
        return " ".join(cleaned)

    def is_useful(self, text: str) -> bool:
        return len(text.split()) >= MIN_WORD_COUNT

    def crawl(self, pages: List[str]) -> List[Dict]:
        """
        Crawl a list of page titles.
        MarioWiki is tried first; Wikipedia serves as fallback.

        Challenge (TD1.4): Pages with fewer than MIN_WORD_COUNT words are
        discarded — ensuring only content-rich articles feed the NLP pipeline.
        """
        results: List[Dict] = []
        visited: set        = set()

        for title in pages:
            if title in visited:
                continue
            visited.add(title)

            logger.info(f"[{len(results)+1}/{len(pages)}] Fetching: {title}")

            doc = self._query_mediawiki(self.api_url, title)

            if not doc or not doc["text"].strip():
                logger.info(f"  ↳ MarioWiki empty — trying Wikipedia for '{title}'")
                doc = self._query_mediawiki(self.fallback, title)

            if not doc:
                logger.warning(f"  ✗ No content for '{title}' — skipping")
                time.sleep(self.delay)
                continue

            doc["text"]       = self.clean_text(doc["text"])
            doc["domain"]     = "mario-universe"
            doc["crawled_at"] = datetime.now().isoformat()
            doc["word_count"] = len(doc["text"].split())

            if not self.is_useful(doc["text"]):
                logger.warning(
                    f"  ✗ '{title}' too short ({doc['word_count']} words) — skipping"
                )
                time.sleep(self.delay)
                continue

            logger.info(f"  ✓ {doc['word_count']} words  [{doc['source_api']}]")
            results.append(doc)
            time.sleep(self.delay)

        logger.info(f"\nCrawl complete — {len(results)}/{len(pages)} documents collected")
        return results

    def close(self):
        self.session.close()


# ──────────────────────────────────────────────────────────────────────────────
# Storage helpers
# ──────────────────────────────────────────────────────────────────────────────

def save_jsonl(docs: List[Dict], path: str = "outputs/crawler_output.jsonl") -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for doc in docs:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(docs)} documents → {out}")
    return out


def load_jsonl(path: str = "outputs/crawler_output.jsonl") -> List[Dict]:
    docs = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def run_phase1(output_path: str = "outputs/crawler_output.jsonl",
               max_pages: int = 20) -> Path:
    print("\n" + "=" * 60)
    print("PHASE 1: WEB CRAWLING & CLEANING")
    print("=" * 60)
    print(f"Target  : Mario Universe  (MarioWiki + Wikipedia)")
    print(f"Pages   : {max_pages}  |  Min words/page: {MIN_WORD_COUNT}")
    print(f"Output  : {output_path}\n")

    crawler = MarioWikiAPICrawler()
    try:
        pages = MARIO_SEED_PAGES[:max_pages]
        docs  = crawler.crawl(pages)
    finally:
        crawler.close()

    if not docs:
        raise RuntimeError("No documents collected — check network access.")

    out        = save_jsonl(docs, output_path)
    total_w    = sum(d["word_count"] for d in docs)

    print(f"\n{'─'*60}")
    print("PHASE 1 SUMMARY")
    print(f"{'─'*60}")
    print(f"  Pages crawled  : {len(docs)}")
    print(f"  Total words    : {total_w:,}")
    print(f"  Avg words/page : {total_w // len(docs):,}")
    print(f"  Output         : {out}")
    return out


if __name__ == "__main__":
    run_phase1()
