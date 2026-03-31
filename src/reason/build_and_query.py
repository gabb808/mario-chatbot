"""
TD3 — Build completed family ontology + run SPARQL queries
==========================================================
This script:
  1. Builds the completed family_completed.owl (OWL/Turtle) with all
     class restrictions, property characteristics, disjoints and individuals
     as required by the TD instructions.
  2. Applies RDFS/OWL reasoning via owlrl.
  3. Runs all 20 SPARQL queries from Part 3 and prints the results.

The ontology corresponds exactly to what should be built interactively
in Protégé (Parts 1-3 of the TD), serving as both reference and test.

Run:  python build_and_query.py
"""

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, XSD
import owlrl

# ─── Namespaces ──────────────────────────────────────────────────────────────
BASE  = "http://www.owl-ontologies.com/family_lab.owl#"
NS    = Namespace(BASE)
ONT   = URIRef("http://www.owl-ontologies.com/family_lab.owl")

# ─── Convenience short-hands ─────────────────────────────────────────────────
C = NS          # class term: C.Person, C.Male …
P = NS          # property:   P.isMarriedWith …
I = NS          # individual: I.Peter …


# =============================================================================
# 1. Build the Ontology
# =============================================================================

def build_ontology() -> Graph:
    g = Graph()
    g.bind("",    NS)
    g.bind("owl", OWL)
    g.bind("rdfs", RDFS)
    g.bind("xsd",  XSD)

    # ── Ontology declaration ─────────────────────────────────────────────────
    g.add((ONT, RDF.type, OWL.Ontology))

    # ── Classes ───────────────────────────────────────────────────────────────
    classes = [
        "Person", "Male", "Female",
        "Parent", "Father", "Mother",
        "Child", "Son", "Daughter",
        "Sibling", "Brother", "Sister",
        "Grandparents", "Grandfather", "Grandmother",
        "Uncle",
    ]
    for cls in classes:
        g.add((C[cls], RDF.type, OWL.Class))

    # Subclass hierarchy
    hierarchy = [
        ("Male",        "Person"),
        ("Female",      "Person"),
        ("Parent",      "Person"),
        ("Father",      "Parent"),
        ("Father",      "Male"),
        ("Mother",      "Parent"),
        ("Mother",      "Female"),
        ("Child",       "Person"),
        ("Son",         "Child"),
        ("Son",         "Male"),
        ("Daughter",    "Child"),
        ("Daughter",    "Female"),
        ("Sibling",     "Person"),
        ("Brother",     "Sibling"),
        ("Brother",     "Male"),
        ("Sister",      "Sibling"),
        ("Sister",      "Female"),
        ("Grandparents","Person"),
        ("Grandfather", "Grandparents"),
        ("Grandfather", "Male"),
        ("Grandmother", "Grandparents"),
        ("Grandmother", "Female"),
        ("Uncle",       "Male"),
    ]
    for child, parent in hierarchy:
        g.add((C[child], RDFS.subClassOf, C[parent]))

    # ── Disjoint classes ─────────────────────────────────────────────────────
    disjoints = [
        ("Male",        "Female"),
        ("Father",      "Mother"),
        ("Son",         "Daughter"),
        ("Grandfather", "Grandmother"),
    ]
    for a, b in disjoints:
        g.add((C[a], OWL.disjointWith, C[b]))

    # ── Datatype properties ───────────────────────────────────────────────────
    for dp, rng in [("name", XSD.string), ("age", XSD.int),
                     ("nationality", XSD.string)]:
        g.add((P[dp], RDF.type,       OWL.DatatypeProperty))
        g.add((P[dp], RDFS.domain,    C.Person))
        g.add((P[dp], RDFS.range,     rng))

    # name, age, nationality are functional
    for dp in ["name", "age", "nationality"]:
        g.add((P[dp], RDF.type, OWL.FunctionalProperty))

    # ── Object properties ─────────────────────────────────────────────────────
    obj_props = {
        "isMarriedWith": dict(domain="Person",  range_="Person",
                               chars=[OWL.SymmetricProperty]),
        "isParentOf":    dict(domain="Person",  range_="Person",  chars=[]),
        "isFatherOf":    dict(domain="Male",    range_="Person",  sub="isParentOf", chars=[]),
        "isMotherOf":    dict(domain="Female",  range_="Person",  sub="isParentOf", chars=[]),
        "isSiblingOf":   dict(domain="Person",  range_="Person",
                               chars=[OWL.SymmetricProperty, OWL.TransitiveProperty]),
        "isBrotherOf":   dict(domain="Male",    range_="Person",  sub="isSiblingOf", chars=[]),
        "isSisterOf":    dict(domain="Female",  range_="Person",  sub="isSiblingOf", chars=[]),
        "isChildOf":     dict(domain="Person",  range_="Person",  chars=[]),
        "isSonOf":       dict(domain="Male",    range_="Person",  sub="isChildOf", chars=[]),
        "isDaughterOf":  dict(domain="Female",  range_="Person",  sub="isChildOf", chars=[]),
    }
    for prop_name, info in obj_props.items():
        g.add((P[prop_name], RDF.type,      OWL.ObjectProperty))
        g.add((P[prop_name], RDFS.domain,   C[info["domain"]]))
        g.add((P[prop_name], RDFS.range,    C[info["range_"]]))
        for char in info.get("chars", []):
            g.add((P[prop_name], RDF.type, char))
        if "sub" in info:
            g.add((P[prop_name], RDFS.subPropertyOf, P[info["sub"]]))

    # isChildOf inverse of isParentOf
    g.add((P.isChildOf,  OWL.inverseOf, P.isParentOf))
    g.add((P.isParentOf, OWL.inverseOf, P.isChildOf))

    # ── Class restrictions (Necessary & Sufficient conditions) ────────────────
    # OWL-RL supports owl:someValuesFrom (cls-svf1 rule) for class membership
    # inference.  owl:minCardinality is NOT supported in OWL-RL, so we use
    # P some Person / P some Parent instead.

    # Father ≡ isFatherOf some Person
    father_blank = URIRef(BASE + "_father_r")
    g.add((C.Father, OWL.equivalentClass, father_blank))
    g.add((father_blank, RDF.type, OWL.Restriction))
    g.add((father_blank, OWL.onProperty, P.isFatherOf))
    g.add((father_blank, OWL.someValuesFrom, C.Person))

    # Mother ≡ isMotherOf some Person
    mother_blank = URIRef(BASE + "_mother_r")
    g.add((C.Mother, OWL.equivalentClass, mother_blank))
    g.add((mother_blank, RDF.type, OWL.Restriction))
    g.add((mother_blank, OWL.onProperty, P.isMotherOf))
    g.add((mother_blank, OWL.someValuesFrom, C.Person))

    # Son ≡ isSonOf some Person
    son_blank = URIRef(BASE + "_son_r")
    g.add((C.Son, OWL.equivalentClass, son_blank))
    g.add((son_blank, RDF.type, OWL.Restriction))
    g.add((son_blank, OWL.onProperty, P.isSonOf))
    g.add((son_blank, OWL.someValuesFrom, C.Person))

    # Daughter ≡ isDaughterOf some Person
    daughter_blank = URIRef(BASE + "_daughter_r")
    g.add((C.Daughter, OWL.equivalentClass, daughter_blank))
    g.add((daughter_blank, RDF.type, OWL.Restriction))
    g.add((daughter_blank, OWL.onProperty, P.isDaughterOf))
    g.add((daughter_blank, OWL.someValuesFrom, C.Person))

    # Grandfather ≡ isFatherOf some Parent
    gf_blank = URIRef(BASE + "_gf_r")
    g.add((C.Grandfather, OWL.equivalentClass, gf_blank))
    g.add((gf_blank, RDF.type, OWL.Restriction))
    g.add((gf_blank, OWL.onProperty, P.isFatherOf))
    g.add((gf_blank, OWL.someValuesFrom, C.Parent))

    # Grandmother ≡ isMotherOf some Parent
    gm_blank = URIRef(BASE + "_gm_r")
    g.add((C.Grandmother, OWL.equivalentClass, gm_blank))
    g.add((gm_blank, RDF.type, OWL.Restriction))
    g.add((gm_blank, OWL.onProperty, P.isMotherOf))
    g.add((gm_blank, OWL.someValuesFrom, C.Parent))

    # Uncle ≡ isBrotherOf some Parent
    uncle_blank = URIRef(BASE + "_uncle_r")
    g.add((C.Uncle, OWL.equivalentClass, uncle_blank))
    g.add((uncle_blank, RDF.type, OWL.Restriction))
    g.add((uncle_blank, OWL.onProperty, P.isBrotherOf))
    g.add((uncle_blank, OWL.someValuesFrom, C.Parent))

    # Brother ≡ isBrotherOf some Person
    brother_blank = URIRef(BASE + "_brother_r")
    g.add((C.Brother, OWL.equivalentClass, brother_blank))
    g.add((brother_blank, RDF.type, OWL.Restriction))
    g.add((brother_blank, OWL.onProperty, P.isBrotherOf))
    g.add((brother_blank, OWL.someValuesFrom, C.Person))

    # Sister ≡ isSisterOf some Person
    sister_blank = URIRef(BASE + "_sister_r")
    g.add((C.Sister, OWL.equivalentClass, sister_blank))
    g.add((sister_blank, RDF.type, OWL.Restriction))
    g.add((sister_blank, OWL.onProperty, P.isSisterOf))
    g.add((sister_blank, OWL.someValuesFrom, C.Person))

    # Parent ≡ isParentOf some Person  (needed for Mother/Father subClassOf Parent to fire)
    parent_blank = URIRef(BASE + "_parent_r")
    g.add((C.Parent, OWL.equivalentClass, parent_blank))
    g.add((parent_blank, RDF.type, OWL.Restriction))
    g.add((parent_blank, OWL.onProperty, P.isParentOf))
    g.add((parent_blank, OWL.someValuesFrom, C.Person))

    # ── Individuals ───────────────────────────────────────────────────────────
    individuals = {
        # Males
        "Peter":   dict(type="Male",   age=70,  nat="French"),
        "Thomas":  dict(type="Male",   age=40,  nat="French"),
        "Paul":    dict(type="Male",   age=38),
        "John":    dict(type="Male",   age=45,  nat="Italian"),
        "Pedro":   dict(type="Male",   age=10),
        "Tom":     dict(type="Male",   age=10),
        "Michael": dict(type="Male",   age=5),
        # Females
        "Marie":   dict(type="Female", age=69,  nat="French"),
        "Sylvie":  dict(type="Female", age=30),
        "Chloe":   dict(type="Female", age=18),
        "Claude":  dict(type="Female", age=5,   nat="French"),
        "Alex":    dict(type="Female", age=25),
    }
    for name, info in individuals.items():
        ind = I[name]
        g.add((ind, RDF.type, OWL.NamedIndividual))
        g.add((ind, RDF.type, C[info["type"]]))
        g.add((ind, P.name,   Literal(name)))
        g.add((ind, P.age,    Literal(info["age"], datatype=XSD.int)))
        if "nat" in info:
            g.add((ind, P.nationality, Literal(info["nat"])))

    # ── Relationships ────────────────────────────────────────────────────────
    rels = [
        # Peter & Marie are married
        (I.Peter,   P.isMarriedWith, I.Marie),
        (I.Marie,   P.isMarriedWith, I.Peter),   # symmetric (explicit for clarity)
        # Peter's children
        (I.Peter,   P.isFatherOf,    I.Thomas),
        (I.Peter,   P.isFatherOf,    I.Paul),
        (I.Peter,   P.isFatherOf,    I.Sylvie),
        (I.Peter,   P.isFatherOf,    I.Chloe),
        (I.Marie,   P.isMotherOf,    I.Thomas),
        (I.Marie,   P.isMotherOf,    I.Paul),
        (I.Marie,   P.isMotherOf,    I.Sylvie),
        (I.Marie,   P.isMotherOf,    I.Chloe),
        (I.Thomas,  P.isSonOf,       I.Peter),
        (I.Thomas,  P.isSonOf,       I.Marie),
        (I.Paul,    P.isSonOf,       I.Peter),
        (I.Sylvie,  P.isDaughterOf,  I.Marie),
        (I.Sylvie,  P.isDaughterOf,  I.Peter),
        (I.Chloe,   P.isDaughterOf,  I.Marie),
        (I.Chloe,   P.isDaughterOf,  I.Peter),
        # John & Pedro
        (I.John,    P.isFatherOf,    I.Pedro),
        (I.Pedro,   P.isSonOf,       I.John),
        # Sylvie & John married
        (I.Sylvie,  P.isMarriedWith, I.John),
        (I.John,    P.isMarriedWith, I.Sylvie),
        # Claude daughterOf Sylvie
        (I.Sylvie,  P.isMotherOf,    I.Claude),
        (I.Claude,  P.isDaughterOf,  I.Sylvie),
        # Thomas & Alex married
        (I.Thomas,  P.isMarriedWith, I.Alex),
        (I.Alex,    P.isMarriedWith, I.Thomas),
        # Tom & Michael — sons of Thomas and Alex
        (I.Thomas,  P.isFatherOf,    I.Tom),
        (I.Thomas,  P.isFatherOf,    I.Michael),
        (I.Alex,    P.isMotherOf,    I.Tom),
        (I.Alex,    P.isMotherOf,    I.Michael),
        (I.Tom,     P.isSonOf,       I.Thomas),
        (I.Tom,     P.isSonOf,       I.Alex),
        (I.Michael, P.isSonOf,       I.Thomas),
        (I.Michael, P.isSonOf,       I.Alex),
        # Siblings (Peter+Marie's children): Thomas, Paul, Sylvie, Chloe
        (I.Thomas,  P.isBrotherOf,   I.Paul),
        (I.Thomas,  P.isBrotherOf,   I.Sylvie),
        (I.Thomas,  P.isBrotherOf,   I.Chloe),
        (I.Paul,    P.isBrotherOf,   I.Thomas),
        (I.Paul,    P.isBrotherOf,   I.Sylvie),
        (I.Paul,    P.isBrotherOf,   I.Chloe),
        (I.Sylvie,  P.isSisterOf,    I.Thomas),
        (I.Sylvie,  P.isSisterOf,    I.Paul),
        (I.Sylvie,  P.isSisterOf,    I.Chloe),
        (I.Chloe,   P.isSisterOf,    I.Thomas),
        (I.Chloe,   P.isSisterOf,    I.Paul),
        (I.Chloe,   P.isSisterOf,    I.Sylvie),
    ]
    for s, p, o in rels:
        g.add((s, p, o))

    return g


# =============================================================================
# 2. Apply OWL Reasoning
# =============================================================================

def reason(g: Graph) -> Graph:
    """Apply RDFS + OWL-RL reasoning via owlrl."""
    owlrl.DeductiveClosure(owlrl.OWLRL_Semantics,
                            rdfs_closure=True,
                            axiomatic_triples=False,
                            datatype_axioms=False).expand(g)
    return g


# =============================================================================
# 3. SPARQL Queries (all 20 from the TD + extras)
# =============================================================================

PREFIX = f"PREFIX ns: <{BASE}>\nPREFIX owl: <http://www.w3.org/2002/07/owl#>\n"


QUERIES = {
    # Part 3 — Basic queries
    "Q1_How_old_is_Peter": PREFIX + """
SELECT ?age WHERE { ns:Peter ns:age ?age . }
""",
    "Q2_Sylvie_parents": PREFIX + """
SELECT ?parent ?name WHERE {
  ns:Sylvie ns:isChildOf ?parent .
  OPTIONAL { ?parent ns:name ?name }
}
""",
    "Q3_Women_over_30": PREFIX + """
SELECT ?woman ?age WHERE {
  ?woman a ns:Female .
  ?woman ns:age ?age .
  FILTER (?age > 30)
}
""",
    "Q4_Instances_of_Person": PREFIX + """
SELECT ?person WHERE {
  ?person a ns:Person .
  FILTER (!isBlank(?person))
}
""",
    # Part 3 — Class instance queries
    "Q5_Instances_Son": PREFIX + """
SELECT ?son WHERE {
  ?son a ns:Son .
  FILTER (!isBlank(?son))
}
""",
    "Q6_Instances_Daughter": PREFIX + """
SELECT ?daughter WHERE {
  ?daughter a ns:Daughter .
  FILTER (!isBlank(?daughter))
}
""",
    "Q7_Instances_Parent": PREFIX + """
SELECT ?parent WHERE {
  ?parent a ns:Parent .
  FILTER (!isBlank(?parent))
}
""",
    "Q8_Instances_Father": PREFIX + """
SELECT ?father WHERE {
  ?father a ns:Father .
  FILTER (!isBlank(?father))
}
""",
    "Q9_Instances_Mother": PREFIX + """
SELECT ?mother WHERE {
  ?mother a ns:Mother .
  FILTER (!isBlank(?mother))
}
""",
    "Q10_Instances_Grandmother": PREFIX + """
SELECT ?gm WHERE {
  ?gm a ns:Grandmother .
  FILTER (!isBlank(?gm))
}
""",
    "Q11_Instances_Grandfather": PREFIX + """
SELECT ?gf WHERE {
  ?gf a ns:Grandfather .
  FILTER (!isBlank(?gf))
}
""",
    "Q12_Instances_Brother": PREFIX + """
SELECT ?brother WHERE {
  ?brother a ns:Brother .
  FILTER (!isBlank(?brother))
}
""",
    "Q13_Instances_Sister": PREFIX + """
SELECT ?sister WHERE {
  ?sister a ns:Sister .
  FILTER (!isBlank(?sister))
}
""",
    # Part 3 — Multi-condition queries
    "Q14_Children_of_Peter_name_age": PREFIX + """
SELECT ?child ?name ?age WHERE {
  ns:Peter ns:isFatherOf ?child .
  OPTIONAL { ?child ns:name ?name }
  OPTIONAL { ?child ns:age  ?age }
}
ORDER BY ?name
""",
    "Q15_Persons_father_over_40": PREFIX + """
SELECT ?person ?father ?fatherAge WHERE {
  ?person ns:isChildOf ?father .
  ?father ns:age ?fatherAge .
  FILTER (?fatherAge > 40)
  FILTER (!isBlank(?person))
}
""",
    "Q16_French_possibly_married": PREFIX + """
SELECT ?person ?name ?age ?spouseName WHERE {
  ?person ns:nationality "French" .
  ?person ns:name ?name .
  ?person ns:age ?age .
  OPTIONAL {
    ?person ns:isMarriedWith ?spouse .
    ?spouse ns:name ?spouseName .
  }
  FILTER (!isBlank(?person))
}
ORDER BY ?name
""",
    "Q17_Persons_who_are_brother_of_someone": PREFIX + """
SELECT DISTINCT ?name WHERE {
  ?person ns:isBrotherOf ?x .
  ?person ns:name ?name .
  FILTER (!isBlank(?person))
}
""",
    "Q18_Persons_who_are_daughter_of_someone": PREFIX + """
SELECT DISTINCT ?name WHERE {
  ?person ns:isDaughterOf ?x .
  ?person ns:name ?name .
  FILTER (!isBlank(?person))
}
""",
    "Q19_Persons_who_are_uncle_of_someone": PREFIX + """
SELECT DISTINCT ?name WHERE {
  ?person a ns:Uncle .
  ?person ns:name ?name .
  FILTER (!isBlank(?person))
}
""",
    "Q20_Persons_who_are_married": PREFIX + """
SELECT DISTINCT ?name WHERE {
  ?person ns:isMarriedWith ?x .
  ?person ns:name ?name .
  FILTER (!isBlank(?person))
}
ORDER BY ?name
""",
}


def run_sparql(g: Graph) -> None:
    print("\n" + "=" * 60)
    print("PART 3 + 4: SPARQL QUERIES")
    print("=" * 60)

    for q_name, q_body in QUERIES.items():
        print(f"\n── {q_name.replace('_', ' ')} ──")
        try:
            results = list(g.query(q_body))
            if not results:
                print("  (no results)")
            else:
                for row in results:
                    vals = [str(v).split("#")[-1] if v else "—" for v in row]
                    print("  " + "  |  ".join(vals))
        except Exception as exc:
            print(f"  ERROR: {exc}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("Building family ontology…")
    g = build_ontology()
    print(f"  Base triples: {len(g)}")

    print("Applying OWL-RL reasoning…")
    reason(g)
    print(f"  After inference: {len(g)} triples")

    # Save completed ontology
    g.serialize("family_completed.ttl", format="turtle")
    g.serialize("family_completed.owl", format="xml")
    print("  Saved: family_completed.ttl  family_completed.owl")

    run_sparql(g)

    print("\n── Inference observations (Part 2) ──────────────────────────")
    print("""
How OWL-RL reasoning infers class memberships from equivalentClass restrictions:

  1. Father / Mother / Son / Daughter (owl:someValuesFrom Person)
     Peter isFatherOf Thomas  AND  Thomas rdf:type Person (inferred via Male ⊆ Person)
     → OWL-RL rule cls-svf1: Peter satisfies (isFatherOf some Person)
     → equivalentClass → Person rdf:type Father  ✓

  2. Grandfather / Grandmother (owl:someValuesFrom Parent)
     Peter isFatherOf Thomas  AND  Thomas rdf:type Parent (inferred from step 1)
     → Peter satisfies (isFatherOf some Parent) → Peter rdf:type Grandfather  ✓

  3. Parent ← inverseOf  
     isChildOf owl:inverseOf isParentOf → reasoner derives isParentOf from
     isFatherOf/isMotherOf (via subPropertyOf chain), so Q7 returns all 6 parents.

  4. Brother / Sister (manually asserted, then restricted confirmed by OWL-RL)
     isBrotherOf is explicitly asserted (Thomas isBrotherOf Paul etc.) because
     OWL-RL CANNOT automatically derive sibling relationships — that requires a
     SWRL rule:  isChildOf(?a,?p) ∧ isChildOf(?b,?p) ∧ Male(?a) ∧ (?a ≠ ?b) →
                 isBrotherOf(?a,?b)

  5. Uncle (equivalentClass: isBrotherOf some Parent)
     Thomas isBrotherOf Sylvie  AND  Sylvie rdf:type Parent → Thomas rdf:type Uncle ✓
     Paul   isBrotherOf Thomas  AND  Thomas rdf:type Parent → Paul   rdf:type Uncle ✓

  6. Disjoint classes: Male ∩ Female = ∅ means the reasoner would flag any
     individual accidentally typed as both (consistency-checking).
""")


if __name__ == "__main__":
    main()
