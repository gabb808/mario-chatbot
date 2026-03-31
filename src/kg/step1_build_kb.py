"""
TD4 — Step 1: Build Initial Private Knowledge Base
===================================================
Reads TD1 outputs (extracted_knowledge.csv + extracted_relationships.csv)
and constructs a typed RDF graph with a proper Mario ontology.

Outputs:
  data/private_kb.ttl      — initial private KB (Turtle)
  data/private_kb.nt       — same in N-Triples (for KGE tools)
  data/mario_ontology.ttl  — ontology (class hierarchy + property definitions)

Run:  python step1_build_kb.py
"""

import csv
import re
from pathlib import Path
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# ─── Paths ────────────────────────────────────────────────────────────────────
HERE         = Path(__file__).parent
TD1_ENTITIES = HERE / "../TP1/outputs/extracted_knowledge.csv"
TD1_RELS     = HERE / "../TP1/outputs/extracted_relationships.csv"
OUT_DIR      = HERE / "data"
OUT_TTL      = OUT_DIR / "private_kb.ttl"
OUT_NT       = OUT_DIR / "private_kb.nt"
OUT_ONTO     = OUT_DIR / "mario_ontology.ttl"

# ─── Namespaces ────────────────────────────────────────────────────────────────
BASE  = "http://esilv.fr/mario#"
MARIO = Namespace(BASE)
PROP  = Namespace("http://esilv.fr/mario/prop#")
SCHEMA = Namespace("http://schema.org/")


def slugify(s: str) -> str:
    """Convert entity text to safe URI slug."""
    s = s.strip()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s or "Unknown"


# ─── TYPE MAP: CSV type string → ontology class URI ───────────────────────────
TYPE_MAP = {
    "CHARACTER":    MARIO.Character,
    "ENEMY":        MARIO.Enemy,
    "GAME":         MARIO.VideoGame,
    "LOCATION":     MARIO.Location,
    "ITEM":         MARIO.Item,
    "ORG":          MARIO.Organization,
    "ORGANIZATION": MARIO.Organization,
    "PERSON":       MARIO.RealPerson,
    "GPE":          MARIO.GeoPoliticalEntity,
}

# ─── PREDICATE MAP: CSV predicate → property URI ──────────────────────────────
PRED_MAP = {
    "co_occurs_with":  PROP.coOccursWith,
    "saves":           PROP.saves,
    "kidnaps":         PROP.kidnaps,
    "defeats":         PROP.defeats,
    "appears_in":      PROP.appearsIn,
    "rules":           PROP.rules,
    "located_in":      PROP.locatedIn,
    "contains":        PROP.contains,
    "is_part_of":      PROP.isPartOf,
    "created_by":      PROP.createdBy,
    "developed_by":    PROP.developedBy,
    "developer_of":    PROP.developedBy,
    "published_by":    PROP.publishedBy,
    "features":        PROP.features,
    "fights":          PROP.fights,
    "helps":           PROP.helps,
    "opposes":         PROP.opposes,
    "enemy_of":        PROP.enemyOf,
    "follows":         PROP.follows,
    "precedes":        PROP.precedes,
    "uses":            PROP.uses,
    "is_used_in":      PROP.isUsedIn,
    "transforms_to":   PROP.transformsTo,
    "belongs_to":      PROP.belongsTo,
    "is_related_to":   PROP.isRelatedTo,
    "part_of":         PROP.isPartOf,
    "owned_by":        PROP.ownedBy,
    "works_for":       PROP.worksFor,
    "collaborated_with": PROP.collaboratedWith,
}


def build_ontology() -> Graph:
    """Construct the Mario domain ontology."""
    g = Graph()
    g.bind("mario", MARIO)
    g.bind("prop",  PROP)
    g.bind("owl",   OWL)
    g.bind("rdfs",  RDFS)
    g.bind("xsd",   XSD)

    ONT = MARIO.MarioOntology
    g.add((ONT, RDF.type, OWL.Ontology))
    g.add((ONT, RDFS.label, Literal("Mario Universe Ontology")))

    # ── Classes ──────────────────────────────────────────────────────────────
    classes = {
        MARIO.Entity:             ("Entity",             None),
        MARIO.Character:          ("Character",          MARIO.Entity),
        MARIO.Enemy:              ("Enemy",              MARIO.Character),
        MARIO.VideoGame:          ("VideoGame",          MARIO.Entity),
        MARIO.Location:           ("Location",           MARIO.Entity),
        MARIO.Item:               ("Item",               MARIO.Entity),
        MARIO.Organization:       ("Organization",       MARIO.Entity),
        MARIO.RealPerson:         ("RealPerson",         MARIO.Entity),
        MARIO.GeoPoliticalEntity: ("GeoPoliticalEntity", MARIO.Location),
    }
    for uri, (label, parent) in classes.items():
        g.add((uri, RDF.type, OWL.Class))
        g.add((uri, RDFS.label, Literal(label)))
        if parent:
            g.add((uri, RDFS.subClassOf, parent))

    # Disjoint
    g.add((MARIO.Character, OWL.disjointWith, MARIO.VideoGame))
    g.add((MARIO.Character, OWL.disjointWith, MARIO.Location))
    g.add((MARIO.Character, OWL.disjointWith, MARIO.Item))

    # ── Object Properties ─────────────────────────────────────────────────────
    obj_props = [
        (PROP.coOccursWith,  "coOccursWith",  MARIO.Entity,   MARIO.Entity,   OWL.SymmetricProperty),
        (PROP.saves,         "saves",         MARIO.Character, MARIO.Character, None),
        (PROP.kidnaps,       "kidnaps",       MARIO.Character, MARIO.Character, None),
        (PROP.defeats,       "defeats",       MARIO.Character, MARIO.Character, None),
        (PROP.appearsIn,     "appearsIn",     MARIO.Character, MARIO.VideoGame, None),
        (PROP.rules,         "rules",         MARIO.Character, MARIO.Location,  None),
        (PROP.locatedIn,     "locatedIn",     MARIO.Entity,    MARIO.Location,  None),
        (PROP.contains,      "contains",      MARIO.Location,  MARIO.Entity,    None),
        (PROP.isPartOf,      "isPartOf",      MARIO.Entity,    MARIO.Entity,    None),
        (PROP.createdBy,     "createdBy",     MARIO.Entity,    MARIO.RealPerson, None),
        (PROP.developedBy,   "developedBy",   MARIO.VideoGame, MARIO.Organization, None),
        (PROP.publishedBy,   "publishedBy",   MARIO.VideoGame, MARIO.Organization, None),
        (PROP.features,      "features",      MARIO.VideoGame, MARIO.Character, None),
        (PROP.fights,        "fights",        MARIO.Character, MARIO.Character, OWL.SymmetricProperty),
        (PROP.helps,         "helps",         MARIO.Character, MARIO.Character, None),
        (PROP.opposes,       "opposes",       MARIO.Character, MARIO.Character, None),
        (PROP.follows,       "follows",       MARIO.VideoGame, MARIO.VideoGame, None),
        (PROP.isRelatedTo,   "isRelatedTo",   MARIO.Entity,    MARIO.Entity,    OWL.SymmetricProperty),
    ]
    for uri, label, dom, rng, char in obj_props:
        g.add((uri, RDF.type, OWL.ObjectProperty))
        g.add((uri, RDFS.label, Literal(label)))
        g.add((uri, RDFS.domain, dom))
        g.add((uri, RDFS.range, rng))
        if char:
            g.add((uri, RDF.type, char))

    # ── Datatype Properties ────────────────────────────────────────────────────
    for uri, label, rng in [
        (PROP.confidence,    "confidence",    XSD.decimal),
        (PROP.sourceURL,     "sourceURL",     XSD.anyURI),
        (PROP.extractMethod, "extractMethod", XSD.string),
    ]:
        g.add((uri, RDF.type, OWL.DatatypeProperty))
        g.add((uri, RDFS.label, Literal(label)))
        g.add((uri, RDFS.range, rng))

    return g


def build_private_kb() -> Graph:
    """Build the private Mario KB from TD1 CSV outputs."""
    g = Graph()
    g.bind("mario", MARIO)
    g.bind("prop",  PROP)
    g.bind("rdfs",  RDFS)

    entity_uris: dict[str, URIRef] = {}  # text → URI cache
    type_counts: dict[str, int] = {}

    # ── Load entities ─────────────────────────────────────────────────────────
    print(f"Loading entities from {TD1_ENTITIES.name}…")
    with open(TD1_ENTITIES, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text       = row["text"].strip()
            etype      = row["type"].strip().upper()
            confidence = float(row.get("confidence", 0.5))
            source_url = row.get("source_url", "")
            method     = row.get("method", "")

            slug = slugify(text)
            uri  = MARIO[slug]
            entity_uris[text] = uri

            cls = TYPE_MAP.get(etype, MARIO.Entity)
            g.add((uri, RDF.type, OWL.NamedIndividual))
            g.add((uri, RDF.type, cls))
            g.add((uri, RDFS.label, Literal(text)))
            g.add((uri, PROP.confidence, Literal(confidence, datatype=XSD.decimal)))
            if source_url:
                g.add((uri, PROP.sourceURL, Literal(source_url, datatype=XSD.anyURI)))
            if method:
                g.add((uri, PROP.extractMethod, Literal(method)))

            type_counts[etype] = type_counts.get(etype, 0) + 1

    print(f"  Entities loaded: {sum(type_counts.values())}")
    for t, c in sorted(type_counts.items()):
        print(f"    {t}: {c}")

    # ── Load relationships ─────────────────────────────────────────────────────
    print(f"\nLoading relationships from {TD1_RELS.name}…")
    rel_counts: dict[str, int] = {}
    skipped = 0
    with open(TD1_RELS, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            subj_text = row["subject"].strip()
            pred_text = row["predicate"].strip()
            obj_text  = row["object"].strip()

            # Ensure entities exist (might not have been extracted explicitly)
            if subj_text not in entity_uris:
                slug = slugify(subj_text)
                uri  = MARIO[slug]
                entity_uris[subj_text] = uri
                g.add((uri, RDF.type, OWL.NamedIndividual))
                g.add((uri, RDF.type, MARIO.Entity))
                g.add((uri, RDFS.label, Literal(subj_text)))

            if obj_text not in entity_uris:
                slug = slugify(obj_text)
                uri  = MARIO[slug]
                entity_uris[obj_text] = uri
                g.add((uri, RDF.type, OWL.NamedIndividual))
                g.add((uri, RDF.type, MARIO.Entity))
                g.add((uri, RDFS.label, Literal(obj_text)))

            prop = PRED_MAP.get(pred_text, PROP[slugify(pred_text)])
            g.add((entity_uris[subj_text], prop, entity_uris[obj_text]))
            rel_counts[pred_text] = rel_counts.get(pred_text, 0) + 1

    total_rels = sum(rel_counts.values())
    print(f"  Relationships loaded: {total_rels}")
    for p, c in sorted(rel_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {p}: {c}")
    if skipped:
        print(f"  Skipped (missing endpoint): {skipped}")

    print(f"\nPrivate KB: {len(g)} triples, {len(entity_uris)} entities")
    return g, entity_uris


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build & save ontology
    print("=== Building ontology ===")
    onto = build_ontology()
    onto.serialize(str(OUT_ONTO), format="turtle")
    print(f"Ontology saved: {OUT_ONTO.name} ({len(onto)} triples)")

    # Build & save private KB
    print("\n=== Building private KB ===")
    kb, _ = build_private_kb()
    kb.serialize(str(OUT_TTL), format="turtle")
    kb.serialize(str(OUT_NT),  format="ntriples")
    print(f"\nPrivate KB saved:")
    print(f"  {OUT_TTL.name} — {OUT_TTL.stat().st_size:,} bytes")
    print(f"  {OUT_NT.name}  — {OUT_NT.stat().st_size:,} bytes")

    # Quick stats
    from collections import Counter
    from rdflib.namespace import RDF as RDF2
    type_counts = Counter(str(o).split("#")[-1]
                          for _, _, o in kb.triples((None, RDF2.type, None))
                          if "esilv.fr" in str(o) and "Individual" not in str(o))
    print("\nEntity types in KB:")
    for cls, cnt in type_counts.most_common():
        print(f"  {cls}: {cnt}")


if __name__ == "__main__":
    main()
