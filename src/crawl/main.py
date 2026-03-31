"""
Lab Session 1 — Main Orchestrator
Course: Web Mining & Semantics — ESILV A4
Domain: Mario Universe

Runs Phase 1 (crawling) then Phase 2 (extraction) and generates the lab report.
Usage:
  python main.py              # run everything
  python main.py --phase 1    # crawl only
  python main.py --phase 2    # extract only (needs existing crawler_output.jsonl)
  python main.py --report     # regenerate report from existing outputs
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from phase1_crawling  import run_phase1
from phase2_extraction import run_phase2


OUTPUTS = Path("outputs")


def generate_lab_report():
    """Write a 2-page markdown lab report as required by TD1."""
    report = OUTPUTS / "LAB_REPORT.md"
    OUTPUTS.mkdir(parents=True, exist_ok=True)

    entity_count = rel_count = doc_count = total_words = 0

    try:
        import json
        docs = [json.loads(l) for l in open(OUTPUTS / "crawler_output.jsonl", encoding="utf-8") if l.strip()]
        doc_count   = len(docs)
        total_words = sum(d.get("word_count", 0) for d in docs)
    except Exception:
        pass

    try:
        import csv
        with open(OUTPUTS / "extracted_knowledge.csv", encoding="utf-8") as f:
            entity_count = sum(1 for _ in csv.reader(f)) - 1
        with open(OUTPUTS / "extracted_relationships.csv", encoding="utf-8") as f:
            rel_count = sum(1 for _ in csv.reader(f)) - 1
    except Exception:
        pass

    content = f"""# Lab Session 1 — Lab Report
**Course:** Web Mining & Semantics | **Domain:** Mario Universe  
**Date:** {datetime.now().strftime("%Y-%m-%d")} | **Team:** The Mario Bros 9

---

## 1. Methodology

### 1.1 Domain & Source Selection
We crawl the **Mario Universe** domain — a rich fictional corpus covering characters,
games, locations and items from the Nintendo franchise.

**Primary source:** MarioWiki MediaWiki API (`www.mariowiki.com/api.php`)  
**Fallback source:** English Wikipedia API (`en.wikipedia.org/w/api.php`)

Seed pages ({doc_count} collected, total ≈ {total_words:,} words):
Mario, Luigi, Princess Peach, Bowser, Yoshi, Toad, Wario, Donkey Kong,
Rosalina, Daisy, Super Mario Bros., Super Mario World, Super Mario Odyssey,
Mario Kart, Luigi's Mansion, Mushroom Kingdom, Koopa Troopa, Goomba, Shy Guy,
Fire Flower.

### 1.2 Web-Crawling Pipeline (Phase 1)
| Step | Tool | Role |
|------|------|------|
| Fetch  | `httpx` | HTTP client with retries & `User-Agent` header |
| Extract | MediaWiki `?prop=extracts&explaintext=1` | Returns clean plain-text, no HTML |
| Fallback | `trafilatura` | Boilerpplate removal on raw HTML pages |
| Filter | word-count ≥ 200 | Discards stubs and redirect pages |
| Store | JSONL | One JSON object per line; preserves URL↔text mapping |

The `?explaintext=1` API flag replaces the need for a full boilerplate-removal
step — the API itself returns readable prose without navigation bars,
cookie banners or ad markers.

### 1.3 Information Extraction Pipeline (Phase 2)

> **Note:** The TD recommends `spaCy en_core_web_trf`.  spaCy 3.x is
> incompatible with Python 3.14 (pydantic v1 type inference fails).
> We use **NLTK 3.9** as a functionally equivalent replacement.

| Step | Tool | Role |
|------|------|------|
| Sentence split | `nltk.sent_tokenize` | Produces sentence contexts for relation extraction |
| POS tagging    | `nltk.pos_tag`       | Labels tokens (NN, VB…) for SVO parsing |
| NE chunking    | `nltk.ne_chunk`      | Extracts PERSON, ORG, GPE chunks |
| Domain NER     | Regex patterns       | 50+ Mario-specific patterns for CHARACTER, GAME, LOCATION, ITEM |
| Relation       | Verb-anchored SVO    | Finds nearest left/right entity around relation verbs |
| Fallback rel.  | Co-occurrence        | Entity pairs within the same sentence |

---

## 2. Entity Ambiguity Analysis

**Example 1 — "Toad"**  
- Extracted as: CHARACTER  
- Ambiguity: "toad" is also a common noun (amphibian) and a verb ("to toad").  
- Misclassification scenario: the sentence *"The child picked up a toad"*
  would incorrectly match the Mario character pattern `\bToad\b`.  
- Resolution applied: sentence-level context check; the pattern requires
  capitalisation (`\bToad\b`, not `\btoad\b`).

**Example 2 — "Bowser"**  
- Extracted as: CHARACTER  
- Ambiguity: "Bowser" is also slang for "dog" in Irish English, and a type of
  fuel tanker in British English.  
- Misclassification scenario: an article about petrol storage could mention
  a "bowser" and be labelled as a Mario character.  
- Resolution: domain restriction — only texts from Mario-universe sources are
  processed, so off-domain "bowser" occurrences are structurally excluded.

**Example 3 — "Mario" (character vs. franchise)**  
- Extracted as: CHARACTER  
- Ambiguity: "Mario" refers to *the character* in most sentences, but also to
  *the entire franchise* ("the Mario series", "the Mario universe") or to
  *real people* named Mario (e.g., Mario Draghi).  
- Misclassification scenario: *"The Mario Bros. franchise earned $35 billion"*
  — here "Mario" should be ORG/FRANCHISE, not CHARACTER.  
- Resolution: multi-word patterns (`Super Mario Bros.`, `the Mario series`)
  capture franchise references; isolated `\bMario\b` defaults to CHARACTER.

---

## 3. Scaling Reflection — 10 vs 10,000 pages

| Concern | 10 pages (current) | 10,000 pages |
|---------|-------------------|--------------|
| **HTTP** | Sequential `httpx` | Async `httpx.AsyncClient` with bounded semaphore |
| **Storage** | JSONL files | PostgreSQL + full-text index |
| **Deduplication** | In-memory `set` | Bloom filter or Redis set |
| **NLP** | Sequential NLTK | Batched multiprocess `Pool` (or Ray) |
| **NER quality** | Regex + ne_chunk | Fine-tuned NER model on Mario corpus |
| **URL frontier** | Static list | Priority queue (BFS / PageRank-weighted) |
| **Rate limiting** | `time.sleep(1)` | Token-bucket per-domain limiter |
| **Error handling** | Log & skip | Retry queue with exponential back-off |

The biggest bottleneck at scale is NLP throughput: `nltk.ne_chunk` does not
parallelise natively. A batched pipeline using Python `multiprocessing.Pool`
with chunk sizes of ~500 documents per worker would reduce wall-clock time
roughly proportionally to the number of CPU cores.

---

## 4. Results Summary

| Metric | Value |
|--------|-------|
| Documents crawled | {doc_count} |
| Total words | {total_words:,} |
| Unique entities | {entity_count} |
| Unique relationships | {rel_count} |
| Entity types | CHARACTER, ENEMY, GAME, LOCATION, ITEM, ORG, PERSON, GPE |
| Relation types | saves, kidnaps, defeats, appears_in, rules, located_in, co_occurs_with |

### Output files
| File | Description |
|------|-------------|
| `outputs/crawler_output.jsonl` | Crawled & cleaned documents |
| `outputs/extracted_knowledge.csv` | Named entities with type, source, confidence |
| `outputs/extracted_relationships.csv` | Triples (subject, predicate, object) |

---

*This lab is the foundation of the Mario Knowledge Graph (TD4) and RAG agent (Project).*
"""

    with open(report, "w", encoding="utf-8") as fh:
        fh.write(content)
    print(f"\n✓ Lab report → {report}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="Lab Session 1 — Mario Universe Knowledge Extraction"
    )
    parser.add_argument("--phase", choices=["1", "2", "all"], default="all")
    parser.add_argument("--report", action="store_true", help="Regenerate report only")
    parser.add_argument("--max-pages", type=int, default=20,
                        help="Max pages to crawl (default 20)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("LAB SESSION 1 — MARIO UNIVERSE KNOWLEDGE EXTRACTION")
    print("=" * 60)

    try:
        if args.report:
            generate_lab_report()
            return

        if args.phase in ("1", "all"):
            run_phase1(max_pages=args.max_pages)

        if args.phase in ("2", "all"):
            run_phase2()

        generate_lab_report()

        print("\n" + "=" * 60)
        print("✓  LAB SESSION 1 COMPLETE")
        print("=" * 60)
        print("\nOutputs in outputs/:")
        for f in sorted(OUTPUTS.glob("*")):
            size = f.stat().st_size
            print(f"  {f.name:<40} {size:>8,} bytes")

    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(1)
    except Exception as exc:
        import traceback
        print(f"\n✗ Error: {exc}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
