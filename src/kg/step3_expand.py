"""
TD4 — Step 3: KB Expansion via DBpedia SPARQL
=====================================================
Expands the private KB by pulling DBpedia facts for the Mario universe.

Sources:
  1. Mario franchise characters  (dct:subject dbc:Mario_(franchise)_characters)
  2. Mario video games            (dct:subject dbc:Mario_video_games)
  3. Nintendo games               (dbo:developer/publisher = dbr:Nintendo)
  4. Nintendo characters & hardware
  5. Anchor entity 1-hop facts    (Mario, Luigi, Bowser, Nintendo, …)
  6. Per-entity 1-hop facts for all discovered entities

Target: 50,000 – 200,000 triples after expansion.

Outputs:
  data/expanded_kb.ttl    — full expanded KB (Turtle)
  data/expanded_kb.nt     — same in N-Triples (for KGE tools)
  data/expansion_log.csv  — per-query expansion counts

Run:  python step3_expand.py  [--fast] [--limit N]
      --fast  skip deep entity-level expansion
      --limit N  max triples to collect (default 200000)
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import httpx
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE       = Path(__file__).parent
DATA       = HERE / "data"
KB_FILE    = DATA / "private_kb.ttl"
ALIGN_FILE = DATA / "alignment.ttl"
OUT_TTL    = DATA / "expanded_kb.ttl"
OUT_NT     = DATA / "expanded_kb.nt"
OUT_LOG    = DATA / "expansion_log.csv"

# ─── Namespaces ────────────────────────────────────────────────────────────────
MARIO = Namespace("http://esilv.fr/mario#")
PROP  = Namespace("http://esilv.fr/mario/prop#")
WD    = Namespace("http://www.wikidata.org/entity/")
DBO   = Namespace("http://dbpedia.org/ontology/")
DBR   = Namespace("http://dbpedia.org/resource/")

# ─── Endpoints & config ───────────────────────────────────────────────────────
DBPEDIA   = "https://dbpedia.org/sparql"
HEADERS   = {
    "Accept":     "application/sparql-results+json",
    "User-Agent": "MarioKB-TD4/1.0 (ESILV student research project; educational use)",
}
SPARQL_PREFIX = """
PREFIX dbo:  <http://dbpedia.org/ontology/>
PREFIX dbr:  <http://dbpedia.org/resource/>
PREFIX dbc:  <http://dbpedia.org/resource/Category:>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX owl:  <http://www.w3.org/2002/07/owl#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX foaf: <http://xmlns.com/foaf/0.1/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
"""
REQUEST_DELAY = 1.5   # seconds between queries


# =============================================================================
# SPARQL helper
# =============================================================================

def dbq(query: str, client: httpx.Client, timeout: int = 35) -> list[dict]:
    """Execute a DBpedia SPARQL query. Returns result bindings or []."""
    full = SPARQL_PREFIX + query
    try:
        r = client.get(DBPEDIA,
                       params={"query": full, "format": "application/sparql-results+json"},
                       headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        return r.json()["results"]["bindings"]
    except Exception as exc:
        print(f"    [WARN] DBpedia query failed: {exc}")
        return []


# =============================================================================
# Properties to exclude (low-value / too verbose)
# =============================================================================

PROPERTY_BLACKLIST = {
    "http://dbpedia.org/ontology/wikiPageWikiLink",
    "http://dbpedia.org/ontology/wikiPageRevisionID",
    "http://dbpedia.org/ontology/wikiPageID",
    "http://dbpedia.org/ontology/wikiPageExternalLink",
    "http://dbpedia.org/property/wikiPageUsesTemplate",
    "http://www.w3.org/ns/prov#wasDerivedFrom",
    "http://dbpedia.org/ontology/abstract",
    "http://dbpedia.org/ontology/wikiPageLength",
    "http://dbpedia.org/ontology/wikiPageOutDegree",
    "http://dbpedia.org/ontology/wikiPageInDegree",
    "http://dbpedia.org/ontology/wikiPageWikiLinkText",
    "http://dbpedia.org/ontology/wikiPageDisambiguates",
    "http://dbpedia.org/ontology/wikiPageRedirects",
}


# =============================================================================
# Discovery queries
# =============================================================================

DISCOVERY_QUERIES: list[tuple[str, str]] = [
    ("Mario franchise characters",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dct:subject <http://dbpedia.org/resource/Category:Mario_(franchise)_characters> .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 2000"""),

    ("Mario video games",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dct:subject dbc:Mario_video_games .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 2000"""),

    ("Nintendo video games (developer)",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item a dbo:VideoGame .
          ?item dbo:developer dbr:Nintendo .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 2000"""),

    ("Nintendo video games (publisher)",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item a dbo:VideoGame .
          ?item dbo:publisher dbr:Nintendo .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 2000"""),

    ("Nintendo hardware",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dbo:manufacturer dbr:Nintendo .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 500"""),

    ("Nintendo people (employees)",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dbo:employer dbr:Nintendo .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 500"""),

    ("Donkey Kong items",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dct:subject dbc:Donkey_Kong .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 500"""),

    ("Wario series items",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dct:subject <http://dbpedia.org/resource/Category:Wario_(series)> .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 500"""),

    ("Works based on Mario",
     """SELECT DISTINCT ?item ?label WHERE {
          ?item dct:subject dbc:Works_based_on_Mario .
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 500"""),

    ("Nintendo subsidiaries & companies",
     """SELECT DISTINCT ?item ?label WHERE {
          { ?item dbo:parentCompany dbr:Nintendo . }
          UNION
          { dbr:Nintendo dbo:subsidiary ?item . }
          OPTIONAL { ?item rdfs:label ?label . FILTER(LANG(?label)="en") }
        } LIMIT 200"""),
]


# =============================================================================
# Anchor entities (fetch full 1-hop regardless of discovery)
# =============================================================================

ANCHORS: list[str] = [
    "http://dbpedia.org/resource/Mario",
    "http://dbpedia.org/resource/Nintendo",
    "http://dbpedia.org/resource/Luigi",
    "http://dbpedia.org/resource/Bowser_(character)",
    "http://dbpedia.org/resource/Princess_Peach",
    "http://dbpedia.org/resource/Yoshi",
    "http://dbpedia.org/resource/Donkey_Kong",
    "http://dbpedia.org/resource/Toad_(Nintendo)",
    "http://dbpedia.org/resource/Wario",
    "http://dbpedia.org/resource/Waluigi",
    "http://dbpedia.org/resource/Goomba",
    "http://dbpedia.org/resource/Koopa_Troopa",
    "http://dbpedia.org/resource/Birdo",
    "http://dbpedia.org/resource/Rosalina_(Nintendo)",
    "http://dbpedia.org/resource/Princess_Daisy",
    "http://dbpedia.org/resource/Kamek",
    "http://dbpedia.org/resource/Lakitu",
    "http://dbpedia.org/resource/Boo_(Nintendo)",
    "http://dbpedia.org/resource/Shy_Guy",
    "http://dbpedia.org/resource/Diddy_Kong",
    "http://dbpedia.org/resource/Bowser_Jr.",
    "http://dbpedia.org/resource/Shigeru_Miyamoto",
    "http://dbpedia.org/resource/Gunpei_Yokoi",
    "http://dbpedia.org/resource/Hiroshi_Yamauchi",
    "http://dbpedia.org/resource/Satoru_Iwata",
    "http://dbpedia.org/resource/Nintendo_Switch",
    "http://dbpedia.org/resource/Nintendo_Entertainment_System",
    "http://dbpedia.org/resource/Super_Nintendo_Entertainment_System",
    "http://dbpedia.org/resource/Nintendo_64",
    "http://dbpedia.org/resource/Nintendo_GameCube",
    "http://dbpedia.org/resource/Wii",
    "http://dbpedia.org/resource/Wii_U",
    "http://dbpedia.org/resource/Game_Boy",
    "http://dbpedia.org/resource/Nintendo_DS",
    "http://dbpedia.org/resource/Nintendo_3DS",
    "http://dbpedia.org/resource/Super_Mario_Bros.",
    "http://dbpedia.org/resource/Super_Mario_World",
    "http://dbpedia.org/resource/Super_Mario_64",
    "http://dbpedia.org/resource/Super_Mario_Odyssey",
    "http://dbpedia.org/resource/Super_Mario_Galaxy",
    "http://dbpedia.org/resource/Super_Mario_Galaxy_2",
    "http://dbpedia.org/resource/Super_Mario_3D_World",
    "http://dbpedia.org/resource/Mario_Kart",
    "http://dbpedia.org/resource/Mario_Kart_8",
    "http://dbpedia.org/resource/Mario_Party",
    "http://dbpedia.org/resource/Paper_Mario",
    "http://dbpedia.org/resource/Mario_%26_Luigi:_Superstar_Saga",
    "http://dbpedia.org/resource/Donkey_Kong_(video_game)",
    "http://dbpedia.org/resource/Yoshi%27s_Island",
    "http://dbpedia.org/resource/The_Legend_of_Zelda",
    "http://dbpedia.org/resource/Metroid",
    "http://dbpedia.org/resource/Pokemon",
]


def fetch_entity_facts(uri: str, client: httpx.Client) -> list[tuple]:
    """Fetch all non-blank, non-blacklisted properties for a DBpedia entity."""
    q = f"""SELECT ?p ?o WHERE {{
  <{uri}> ?p ?o .
  FILTER(!isBlank(?o))
}} LIMIT 500"""
    rows = dbq(q, client)
    facts = []
    for r in rows:
        p_uri = r["p"]["value"]
        if p_uri in PROPERTY_BLACKLIST:
            continue
        o_val    = r["o"]["value"]
        o_is_uri = r["o"]["type"] == "uri" and o_val.startswith("http")
        facts.append((uri, p_uri, o_val, o_is_uri))
    return facts


def add_facts(g: Graph, facts: list[tuple]) -> int:
    """Add facts as triples to the graph. Returns count of NEW triples added."""
    added = 0
    for subj_str, pred_str, obj_str, obj_is_uri in facts:
        subj = URIRef(subj_str)
        pred = URIRef(pred_str)
        obj  = URIRef(obj_str) if obj_is_uri else Literal(obj_str)
        if (subj, pred, obj) not in g:
            g.add((subj, pred, obj))
            added += 1
    return added


# =============================================================================
# Bulk property batch queries
# =============================================================================

BULK_QUERIES: list[tuple[str, str]] = [
    ("Mario game bulk props (developer side)",
     """SELECT ?item ?p ?o WHERE {
          ?item a dbo:VideoGame .
          ?item dbo:developer dbr:Nintendo .
          ?item ?p ?o .
          FILTER(?p IN (dbo:publisher, dbo:developer, dbo:platform, dbo:genre,
                        dbo:series, rdfs:label, rdf:type, dbo:releaseDate,
                        dbo:numberOfPlayers, foaf:name, dbo:composer,
                        dbo:director, dbo:designer))
          FILTER(!isBlank(?o))
        } LIMIT 10000"""),

    ("Mario game bulk props (publisher side)",
     """SELECT ?item ?p ?o WHERE {
          ?item a dbo:VideoGame .
          ?item dbo:publisher dbr:Nintendo .
          ?item ?p ?o .
          FILTER(?p IN (dbo:publisher, dbo:developer, dbo:platform, dbo:genre,
                        dbo:series, rdfs:label, rdf:type,
                        dbo:distributor, dbo:gameEngine, dbo:media))
          FILTER(!isBlank(?o))
        } LIMIT 10000"""),

    ("Mario characters bulk props",
     """SELECT ?item ?p ?o WHERE {
          ?item dct:subject <http://dbpedia.org/resource/Category:Mario_(franchise)_characters> .
          ?item ?p ?o .
          FILTER(?p IN (rdf:type, dbo:creator, rdfs:label, foaf:name,
                        dbo:firstAppearance, dbo:species, dbo:voiceActor,
                        dbo:occupation, dbo:relative, owl:sameAs,
                        dbo:alias, dbo:gender))
          FILTER(!isBlank(?o))
        } LIMIT 10000"""),

    ("Nintendo hardware bulk props",
     """SELECT ?item ?p ?o WHERE {
          ?item dbo:manufacturer dbr:Nintendo .
          ?item ?p ?o .
          FILTER(?p IN (rdf:type, rdfs:label, dbo:introductionYear,
                        dbo:discontinued, dbo:cpu, dbo:media, dbo:unitsSold,
                        dbo:successor, dbo:predecessor, dbo:manufacturer,
                        foaf:name, dbo:operatingSystem))
          FILTER(!isBlank(?o))
        } LIMIT 3000"""),

    ("Nintendo people bulk props",
     """SELECT ?item ?p ?o WHERE {
          ?item dbo:employer dbr:Nintendo .
          ?item ?p ?o .
          FILTER(?p IN (rdf:type, rdfs:label, foaf:name, dbo:birthDate,
                        dbo:birthPlace, dbo:employer, dbo:occupation,
                        dbo:award, dbo:nationality, dbo:notableWork))
          FILTER(!isBlank(?o))
        } LIMIT 3000"""),
]


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="TD4 Step 3 — KB Expansion via DBpedia")
    parser.add_argument("--fast",  action="store_true",
                        help="Skip per-entity 1-hop expansion")
    parser.add_argument("--limit", type=int, default=200_000,
                        help="Max triples (default: 200000)")
    args = parser.parse_args()

    DATA.mkdir(parents=True, exist_ok=True)

    # ── Load private KB + alignment ──────────────────────────────────────────
    print("Loading private KB…")
    g = Graph()
    g.bind("mario", MARIO)
    g.bind("prop",  PROP)
    g.bind("dbo",   DBO)
    g.bind("dbr",   DBR)
    g.bind("owl",   OWL)
    g.bind("rdfs",  RDFS)
    g.parse(str(KB_FILE), format="turtle")
    base_count = len(g)
    print(f"  Private KB: {base_count} triples")

    if ALIGN_FILE.exists():
        g.parse(str(ALIGN_FILE), format="turtle")
        print(f"  + alignment: {len(g)} triples total")

    log_entries: list[dict] = []
    anchor_set = set(ANCHORS)

    with httpx.Client(timeout=35) as client:

        # ── Phase 1: Discovery ────────────────────────────────────────────────
        print("\n=== Phase 1: Entity Discovery ===")
        discovered: set[str] = set()

        for desc, q in DISCOVERY_QUERIES:
            if len(g) >= args.limit:
                print("  [LIMIT] Triple limit reached.")
                break
            rows = dbq(q, client)
            time.sleep(REQUEST_DELAY)
            n = 0
            for row in rows:
                uri = row.get("item", {}).get("value", "")
                if uri and uri.startswith("http://dbpedia.org/resource/") \
                        and "Category:" not in uri:
                    discovered.add(uri)
                    n += 1
            print(f"  {desc}: {n} entities  (total: {len(discovered)})")
            log_entries.append({"phase": "discovery", "description": desc,
                                 "items": n, "triples_added": 0})

        print(f"\n  Discovered {len(discovered)} unique DBpedia entities")

        # ── Phase 2: Anchor 1-hop facts ───────────────────────────────────────
        print("\n=== Phase 2: Anchor Entity Properties ===")
        for uri in ANCHORS:
            if len(g) >= args.limit:
                break
            facts = fetch_entity_facts(uri, client)
            added = add_facts(g, facts)
            name  = uri.split("/")[-1][:40]
            print(f"  {name:<42} +{added:5d} triples  total: {len(g):,}")
            log_entries.append({"phase": "anchor", "description": name,
                                 "items": 1, "triples_added": added})
            time.sleep(REQUEST_DELAY)

        # ── Phase 3: Bulk batch queries ───────────────────────────────────────
        print("\n=== Phase 3: Bulk Batch Queries ===")
        for desc, q in BULK_QUERIES:
            if len(g) >= args.limit:
                break
            rows = dbq(q, client, timeout=50)
            time.sleep(REQUEST_DELAY)
            added = 0
            for row in rows:
                item  = row.get("item", {}).get("value", "")
                p_uri = row.get("p",    {}).get("value", "")
                o_val = row.get("o",    {}).get("value", "")
                if not (item and p_uri and o_val):
                    continue
                if p_uri in PROPERTY_BLACKLIST:
                    continue
                o_is_uri = row["o"]["type"] == "uri" and o_val.startswith("http")
                triple = (URIRef(item), URIRef(p_uri),
                          URIRef(o_val) if o_is_uri else Literal(o_val))
                if triple not in g:
                    g.add(triple)
                    added += 1
            print(f"  {desc[:55]:57s} +{added:6d}  total: {len(g):,}")
            log_entries.append({"phase": "bulk", "description": desc,
                                 "items": len(rows), "triples_added": added})

        # ── Phase 4: Per-entity expansion ─────────────────────────────────────
        if not args.fast:
            leftover = [u for u in discovered if u not in anchor_set]
            print(f"\n=== Phase 4: Per-entity Expansion ({len(leftover)} entities) ===")
            done = 0
            for uri in leftover:
                if len(g) >= args.limit:
                    print(f"  [LIMIT] Reached {args.limit:,} triples — stopping")
                    break
                facts = fetch_entity_facts(uri, client)
                added = add_facts(g, facts)
                done += 1
                if done % 100 == 0:
                    print(f"  … {done}/{len(leftover)} entities  total: {len(g):,}")
                log_entries.append({"phase": "expansion",
                                     "description": uri.split("/")[-1],
                                     "items": 1, "triples_added": added})
                time.sleep(REQUEST_DELAY)
            print(f"  Finished per-entity expansion ({done} entities processed)")
        else:
            print("\n=== Phase 4: Skipped (--fast) ===")

    # ── Cleaning ─────────────────────────────────────────────────────────────
    print("\n=== Cleaning ===")
    before = len(g)
    bad = [(s, p, o) for s, p, o in g if not isinstance(s, URIRef)]
    for t in bad:
        g.remove(t)
    print(f"  Removed {before - len(g)} non-URI-subject triples")

    predicates = set(str(p) for _, p, _ in g)
    subjects   = set(str(s) for s, _, _ in g)
    print(f"  Final triples:     {len(g):>8,}")
    print(f"  Unique subjects:   {len(subjects):>8,}")
    print(f"  Unique predicates: {len(predicates):>8,}")

    # ── Requirements check ────────────────────────────────────────────────────
    print("\n=== TD4 Requirements Check ===")
    t_ok = 50_000  <= len(g)         <= 200_000
    s_ok = 5_000   <= len(subjects)  <= 30_000
    p_ok = 50      <= len(predicates)<= 200
    print(f"  Triples:    {len(g):>8,}  (required: 50,000–200,000)  {'✓' if t_ok else '✗'}")
    print(f"  Subjects:   {len(subjects):>8,}  (required:  5,000– 30,000)  {'✓' if s_ok else '✗'}")
    print(f"  Predicates: {len(predicates):>8,}  (required:     50–    200)  {'✓' if p_ok else '✗'}")

    # ── Save ─────────────────────────────────────────────────────────────────
    print("\nSaving…")
    g.serialize(str(OUT_TTL), format="turtle")
    g.serialize(str(OUT_NT),  format="nt")
    print(f"  {OUT_TTL.name}: {OUT_TTL.stat().st_size:,} bytes")
    print(f"  {OUT_NT.name}:  {OUT_NT.stat().st_size:,} bytes")

    with open(OUT_LOG, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["phase", "description", "items", "triples_added"])
        w.writeheader()
        w.writerows(log_entries)
    print(f"  Expansion log: {OUT_LOG.name} ({len(log_entries)} entries)")


if __name__ == "__main__":
    main()
