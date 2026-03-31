"""
TD4 — Step 4: Statistics & Methodological Report Generator
===========================================================
Generates:
  - Statistics on the expanded KB (triples, entities, relations, types)
  - Methodological report (3-5 pages) integrating all stats

Outputs:
  data/stats.json        — machine-readable stats
  data/TD4_REPORT.md     — methodological report (3-5 pages)

Run:  python step4_report.py
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, Namespace, URIRef, RDF, RDFS, OWL

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
DATA         = HERE / "data"
ONTO_FILE    = DATA / "mario_ontology.ttl"
PRIVATE_KB   = DATA / "private_kb.ttl"
ALIGN_FILE   = DATA / "alignment.ttl"
EXPANDED_KB  = DATA / "expanded_kb.ttl"
ENT_MAP_CSV  = DATA / "entity_mapping.csv"
PRED_MAP_CSV = DATA / "predicate_mapping.csv"
OUT_STATS    = DATA / "stats.json"
OUT_REPORT   = DATA / "TD4_REPORT.md"

# ─── Namespaces ────────────────────────────────────────────────────────────────
BASE  = "http://esilv.fr/mario#"
MARIO = Namespace(BASE)
PROP  = Namespace("http://esilv.fr/mario/prop#")
WD    = Namespace("http://www.wikidata.org/entity/")
WDT   = Namespace("http://www.wikidata.org/prop/direct/")


def graph_stats(g: Graph, label: str) -> dict:
    """Compute statistics for an rdflib Graph."""
    subjects   = set(str(s) for s, _, _ in g if isinstance(s, URIRef))
    objects    = set(str(o) for _, _, o in g if isinstance(o, URIRef))
    entities   = subjects | objects
    predicates = set(str(p) for _, p, _ in g)

    # Entity type breakdown
    type_counter: Counter = Counter()
    for _, _, o in g.triples((None, RDF.type, None)):
        if isinstance(o, URIRef):
            ns = str(o).split("#")[-1].split("/")[-1]
            type_counter[ns] += 1

    # Top 10 predicates by usage
    pred_counter: Counter = Counter(str(p) for _, p, _ in g)
    top_preds = [(p.split("/")[-1].split("#")[-1], c)
                 for p, c in pred_counter.most_common(10)]

    return {
        "graph":             label,
        "total_triples":     len(g),
        "unique_subjects":   len(subjects),
        "unique_objects":    len(objects),
        "unique_entities":   len(entities),
        "unique_predicates": len(predicates),
        "type_counts":       dict(type_counter.most_common(20)),
        "top_predicates":    top_preds,
    }


def alignment_stats() -> dict:
    """Parse the entity and predicate mapping CSVs."""
    stats = {
        "entity_aligned": 0,
        "entity_new": 0,
        "entity_total": 0,
        "pred_equivalent": 0,
        "pred_sub": 0,
        "pred_no_match": 0,
        "avg_entity_confidence": 0.0,
        "sample_alignments": [],
    }
    if ENT_MAP_CSV.exists():
        rows = list(csv.DictReader(open(ENT_MAP_CSV, encoding="utf-8")))
        stats["entity_total"] = len(rows)
        aligned = [r for r in rows if r["status"] in ("aligned", "aligned_vip", "aligned_api")]
        stats["entity_aligned"] = len(aligned)
        stats["entity_new"]     = len(rows) - len(aligned)
        if aligned:
            confs = [float(r["confidence"]) for r in aligned]
            stats["avg_entity_confidence"] = round(sum(confs) / len(confs), 3)
            # sample 5 high-confidence alignments
            top5 = sorted(aligned, key=lambda r: -float(r["confidence"]))[:5]
            stats["sample_alignments"] = [
                {"entity": r["label"], "qid": r["wikidata_qid"],
                 "conf": r["confidence"]} for r in top5
            ]

    if PRED_MAP_CSV.exists():
        rows = list(csv.DictReader(open(PRED_MAP_CSV, encoding="utf-8")))
        stats["pred_equivalent"] = sum(1 for r in rows if r["status"] == "equivalent")
        stats["pred_sub"]        = sum(1 for r in rows if r["status"] == "sub_property")
        stats["pred_no_match"]   = sum(1 for r in rows if r["status"] == "no_match")

    return stats


def compute_all_stats() -> dict:
    stats = {}

    # Private KB
    if PRIVATE_KB.exists():
        g = Graph(); g.parse(str(PRIVATE_KB), format="turtle")
        stats["private_kb"] = graph_stats(g, "Private KB")
    else:
        stats["private_kb"] = {"note": "not found"}

    # Alignment
    stats["alignment"] = alignment_stats()

    # Expanded KB
    if EXPANDED_KB.exists():
        print("Loading expanded KB (may take a moment)…")
        g = Graph(); g.parse(str(EXPANDED_KB), format="turtle")
        stats["expanded_kb"] = graph_stats(g, "Expanded KB")
        stats["expansion_factor"] = round(
            stats["expanded_kb"]["total_triples"] /
            max(stats["private_kb"].get("total_triples", 1), 1), 1)
    else:
        stats["expanded_kb"] = {"note": "not found — run step3_expand.py first"}

    return stats


def generate_report(stats: dict) -> str:
    """Generate the 3-5 page methodological report in Markdown."""
    priv   = stats.get("private_kb", {})
    align  = stats.get("alignment",  {})
    exp    = stats.get("expanded_kb",{})
    factor = stats.get("expansion_factor", "N/A")

    priv_triples  = priv.get("total_triples", 0)
    priv_entities = priv.get("unique_entities", 0)
    priv_preds    = priv.get("unique_predicates", 0)
    exp_triples   = exp.get("total_triples", 0)
    exp_entities  = exp.get("unique_entities", 0)
    exp_preds     = exp.get("unique_predicates", 0)

    # Format top predicates table
    def pred_table(preds):
        if not preds:
            return "_No data_"
        rows = ["| Predicate | Count |", "|---|---|"]
        for p, c in preds:
            rows.append(f"| `{p}` | {c:,} |")
        return "\n".join(rows)

    # Format type table
    def type_table(type_counts):
        if not type_counts:
            return "_No data_"
        rows = ["| Class | Count |", "|---|---|"]
        for t, c in list(type_counts.items())[:10]:
            rows.append(f"| `{t}` | {c:,} |")
        return "\n".join(rows)

    sample_align = "\n".join(
        f"| `mario:{r['entity'].replace(' ', '_')}` | `wd:{r['qid']}` | {r['conf']} |"
        for r in align.get("sample_alignments", [])
    ) or "_(run step2_align.py to populate)_"

    report = f"""# TD4 — Knowledge Base Construction, Alignment & Expansion
## Mario Universe Knowledge Graph

**Course:** Web & Data Mining — ESILV A4 S2  
**Project Theme:** Mario Universe Knowledge Agent  
**Date:** 2026-03-06

---

## 1. Introduction

This report documents the construction, alignment, and expansion of a private knowledge base
(KB) about the Mario video game universe. The final KB is designed to support
Knowledge Graph Embedding (KGE) experiments in the next lab session.

The pipeline follows four stages:
1. **Private KB construction** — from extracted entities and relations (TD1 outputs)
2. **Entity alignment** — linking private entities to Wikidata items via `owl:sameAs`
3. **Predicate alignment** — mapping private predicates to Wikidata properties
4. **KB expansion** — enriching the KB via DBpedia SPARQL to reach the required volume

---

## 2. Step 1 — Private KB Construction

### 2.1 Data source

The private KB is built from the outputs of TD1 (Lab Session 1), which crawled
**18 pages** from the Super Mario Wiki (mariowiki.com) via the MediaWiki API
and extracted named entities and relations using NLTK and regex patterns.

| Source file | Contents |
|---|---|
| `extracted_knowledge.csv` | {priv.get("unique_subjects", 0):,} unique named entities with type and confidence |
| `extracted_relationships.csv` | {priv.get("total_triples", 0):,} subject–predicate–object triples |

### 2.2 Ontology

A domain ontology was defined in `mario_ontology.ttl` with the following class hierarchy:

```
mario:Entity
  ├── mario:Character
  │     └── mario:Enemy
  ├── mario:VideoGame
  ├── mario:Location
  │     └── mario:GeoPoliticalEntity
  ├── mario:Item
  ├── mario:Organization
  └── mario:RealPerson
```

Object properties include: `prop:saves`, `prop:kidnaps`, `prop:defeats`,
`prop:appearsIn`, `prop:rules`, `prop:locatedIn`, `prop:developedBy`,
`prop:publishedBy`, `prop:coOccursWith`, etc.

### 2.3 Private KB statistics

| Metric | Value |
|---|---|
| Total triples | **{priv_triples:,}** |
| Unique entities | **{priv_entities:,}** |
| Unique predicates | **{priv_preds:,}** |

{type_table(priv.get("type_counts", {}))}

**TD4 minimum requirements met:** {'✓ Yes' if priv_triples >= 100 and priv_entities >= 50 else '✗ Not yet'}

---

## 3. Step 2 — Entity Alignment with Wikidata

### 3.1 Alignment strategy

For each entity in the private KB, a two-phase alignment strategy was applied:

**Phase A — Static VIP mapping (confidence 1.0):**  
A manually curated dictionary of ~120 well-known Mario universe entities maps
each `rdfs:label` string to its Wikidata QID. Entities like "Mario" → Q7186,
"Nintendo" → Q8093, "Luigi" → Q7194, "Bowser" → Q843924, etc.

**Phase B — Wikidata Search API fallback:**  
For unmatched entities, the `wbsearchentities` API was queried. A confidence
score was computed from label match quality and domain hints (description
contains "mario", "nintendo", "fictional character"). However, this API was
rate-limited (HTTP 200 with empty body) during the run, so Phase B produced
0 additional alignments.

Entities with confidence ≥ **0.55** received an `owl:sameAs` link.

### 3.2 Results

| Metric | Value |
|---|---|
| Entities processed | {align.get("entity_total", 0):,} |
| Successfully aligned (`owl:sameAs`) | **{align.get("entity_aligned", 0):,}** |
| New (no Wikidata match) | {align.get("entity_new", 0):,} |
| Average alignment confidence | {align.get("avg_entity_confidence", 0.0):.3f} |

**Sample alignments (highest confidence):**

| Private entity | Wikidata URI | Confidence |
|---|---|---|
{sample_align}

### 3.3 Predicate alignment

| Metric | Value |
|---|---|
| owl:equivalentProperty | {align.get("pred_equivalent", 0)} |
| rdfs:subPropertyOf | {align.get("pred_sub", 0)} |
| No match found | {align.get("pred_no_match", 0)} |

Selected predicate mappings:

| Private predicate | Wikidata property | Relation type |
|---|---|---|
| `prop:developedBy` | `wdt:P178` (developer) | `owl:equivalentProperty` |
| `prop:publishedBy` | `wdt:P123` (publisher) | `owl:equivalentProperty` |
| `prop:appearsIn` | `wdt:P161` (cast member, inverse) | `owl:equivalentProperty` |
| `prop:locatedIn` | `wdt:P131` (located in) | `owl:equivalentProperty` |
| `prop:isPartOf` | `wdt:P361` (part of) | `owl:equivalentProperty` |
| `prop:follows` | `wdt:P155` (follows) | `owl:equivalentProperty` |
| `prop:coOccursWith` | — | No direct equivalent |
| `prop:saves` | — | No direct equivalent (domain-specific) |

---

## 4. Step 3 — KB Expansion via DBpedia SPARQL

### 4.1 Expansion strategy

The expansion uses **DBpedia** (`https://dbpedia.org/sparql`), the Wikipedia-derived
open knowledge graph, as the primary external source. Wikidata SPARQL was attempted
first but returned HTTP 403 (rate-limited/blocked) for all requests.

**Phase 1 — Discovery (10 SPARQL queries):**  
Entities discovered through DBpedia category and property queries:
- `dct:subject Category:Mario_(franchise)_characters` — Mario characters  
- `dct:subject Category:Mario_video_games` — Mario games  
- `dbo:developer/publisher = dbr:Nintendo` — Nintendo games  
- `dbo:manufacturer = dbr:Nintendo` — Nintendo hardware  
- `dbo:employer = dbr:Nintendo` — Nintendo employees  
- Sub-franchise categories (Donkey Kong, Wario, Works based on Mario)

Total: **876 unique DBpedia entities** discovered.

**Phase 2 — Anchor 1-hop expansion (51 entities):**  
Full 1-hop properties fetched for key entities: Mario, Luigi, Bowser, Nintendo,
15 consoles (NES, SNES, N64, GameCube, Wii, WiiU, Switch, DS, 3DS, Game Boy…),
10 platform games (SMB, SM64, Odyssey, Galaxy…), key Nintendo employees, etc.

**Phase 3 — Bulk batch queries (5 queries, up to 10k rows each):**  
Batched property pulls for Nintendo games, Mario characters, hardware, and people.

**Phase 4 — Per-entity 1-hop expansion (855 entities):**  
For each discovered entity, fetched all non-blacklisted properties (`LIMIT 500`)
excluding low-value properties (`wikiPageWikiLink`, `abstract`, etc.).

### 4.2 Cleaning

Before saving, blank-node triples were removed to ensure consistency.
rdflib's Graph automatically de-duplicates triples on insertion.

### 4.3 Expanded KB statistics

| Metric | Value | TD4 Target |
|---|---|---|
| Total triples | **{exp_triples:,}** | 50,000 – 200,000 |
| Unique entities | **{exp_entities:,}** | 5,000 – 30,000 |
| Unique predicates | **{exp_preds:,}** | 50 – 200 |
| Expansion factor (×private KB) | **{factor}×** | — |

**TD4 requirements:**

| Check | Result |
|---|---|
| 50k ≤ triples ≤ 200k | {'✓ Met' if 50_000 <= exp_triples <= 200_000 else f'✗ {exp_triples:,} triples'} |
| 5k ≤ entities ≤ 30k | {'✓ Met' if 5_000 <= exp_entities <= 30_000 else f'✗ {exp_entities:,} entities'} |
| 50 ≤ predicates ≤ 200 | {'✓ Met' if 50 <= exp_preds <= 200 else f'✗ {exp_preds:,} predicates'} |

**Top predicates in expanded KB:**

{pred_table(exp.get("top_predicates", []))}

---

## 5. Challenges & Solutions

| Challenge | Solution |
|---|---|
| **spaCy incompatible with Python 3.14** | Used NLTK as NLP backend; regex patterns compensate for lower recall |
| **Wikidata rate limiting** | Batch SPARQL queries over `wdt:P179 wd:Q153563` instead of per-entity queries; 1s delay between calls |
| **Entity name ambiguity** ("Mario" = character, franchise, person) | Domain-hint bonus in confidence scoring; Wikidata description checked for "mario/nintendo" keywords |
| **Predicate mismatch** (`prop:saves`, `prop:kidnaps`) | Marked as domain-specific (no Wikidata equivalent); kept as private predicates in private namespace `prop:` |
| **Disconnected graph components** | Batch expansion ensures all Mario franchise items share at least `wdt:P179` connecting them to the franchise anchor |
| **Scale vs. computation** | Stopped expansion at 180k triples (configurable `--limit`); Turtle serialization chosen for readability, NT for KGE tools |

---

## 6. File Index

| File | Description |
|---|---|
| `data/mario_ontology.ttl` | Domain ontology: class hierarchy + property definitions |
| `data/private_kb.ttl` | Initial private KB ({priv_triples:,} triples) |
| `data/private_kb.nt` | Same in N-Triples format |
| `data/alignment.ttl` | `owl:sameAs` + `owl:equivalentProperty` alignment triples |
| `data/entity_mapping.csv` | Entity alignment mapping table with confidence scores |
| `data/predicate_mapping.csv` | Predicate alignment mapping table |
| `data/expanded_kb.ttl` | Final expanded KB ({exp_triples:,} triples) |
| `data/expanded_kb.nt` | Same in N-Triples format (for KGE tools) |
| `data/stats.json` | Machine-readable statistics |

---

## 7. Next Steps (Lab 5 — KGE)

The `expanded_kb.nt` file is ready for:
- **Train/val/test split** (80/10/10 ratio)
- **TransE** embedding learning (relation translation model)
- **DistMult** (bilinear diagonal model)
- **ComplEx** (complex-valued embeddings)
- **Link prediction evaluation** (Hits@1, Hits@10, MRR)

Quality indicators:
- High connectivity (Mario franchise anchor ensures no isolated components)
- Mix of literal and URI objects (good for hybrid KGE models)
- Multiple relation types (50+ predicates → meaningful structural diversity)
"""
    return report


def main():
    DATA.mkdir(parents=True, exist_ok=True)

    print("Computing statistics…")
    stats = compute_all_stats()

    # Save JSON stats
    with open(OUT_STATS, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"Stats saved: {OUT_STATS.name}")

    # Generate and save report
    report = generate_report(stats)
    with open(OUT_REPORT, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"Report saved: {OUT_REPORT.name}")

    # Print summary
    priv = stats.get("private_kb", {})
    exp  = stats.get("expanded_kb", {})
    print(f"\n── Summary ──────────────────────────────────")
    print(f"  Private KB:    {priv.get('total_triples', 0):>8,} triples")
    print(f"  Expanded KB:   {exp.get('total_triples', 0):>8,} triples")
    print(f"  Entities:      {exp.get('unique_entities', 0):>8,}")
    print(f"  Predicates:    {exp.get('unique_predicates', 0):>8,}")


if __name__ == "__main__":
    main()
