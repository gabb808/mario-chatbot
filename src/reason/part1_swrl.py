"""
TD5 - Part 1: Knowledge Reasoning with SWRL
=============================================
Load family ontology, add SWRL rule:
  "A person who is older than 60 years old is an OldPerson"
Run Pellet reasoner via owlready2 and report results.
"""
import os
import sys
import json
import subprocess

def check_java():
    """Check if Java is available (needed for Pellet reasoner)."""
    try:
        r = subprocess.run(["java", "-version"], capture_output=True, timeout=10)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.SubprocessError):
        return False


def main():
    print("=" * 60)
    print("Part 1: SWRL Reasoning on Family Ontology")
    print("=" * 60)

    if not check_java():
        print("WARNING: Java not found on PATH. Pellet reasoner requires Java.")
        print("Install Java (JRE/JDK) and ensure 'java' is on PATH.")

    from owlready2 import (
        get_ontology, Thing, ObjectProperty, DataProperty,
        FunctionalProperty, SymmetricProperty, Imp,
        sync_reasoner_pellet,
    )

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    os.makedirs(out_dir, exist_ok=True)

    # ── Create ontology ─────────────────────────────────────────
    onto = get_ontology("http://www.owl-ontologies.com/family_lab.owl#")

    with onto:
        # ── Classes ──
        class Person(Thing): pass
        class Male(Person): pass
        class Female(Person): pass
        class Parent(Person): pass
        class Child(Person): pass
        class Sibling(Person): pass
        class Father(Male, Parent): pass
        class Mother(Female, Parent): pass
        class Son(Male, Child): pass
        class Daughter(Female, Child): pass
        class Brother(Male, Sibling): pass
        class Sister(Female, Sibling): pass
        class Grandparents(Person): pass
        class Grandfather(Male, Grandparents): pass
        class Grandmother(Female, Grandparents): pass
        class Uncle(Male): pass

        # TARGET CLASS for SWRL inference
        class OldPerson(Person): pass

        # ── Object Properties ──
        class isParentOf(ObjectProperty):
            domain = [Person]; range = [Person]
        class isChildOf(ObjectProperty):
            domain = [Person]; range = [Person]
            inverse_property = isParentOf
        class isFatherOf(isParentOf):
            domain = [Male]
        class isMotherOf(isParentOf):
            domain = [Female]
        class isSonOf(isChildOf):
            domain = [Male]
        class isDaughterOf(isChildOf):
            domain = [Female]
        class isSiblingOf(ObjectProperty, SymmetricProperty):
            domain = [Person]; range = [Person]
        class isBrotherOf(isSiblingOf):
            domain = [Male]
        class isSisterOf(isSiblingOf):
            domain = [Female]
        class isMarriedWith(ObjectProperty, SymmetricProperty):
            domain = [Person]; range = [Person]

        # ── Data Properties ──
        class age(DataProperty, FunctionalProperty):
            domain = [Person]; range = [int]
        class nationality(DataProperty, FunctionalProperty):
            domain = [Person]; range = [str]

        # ── Individuals (from family.owl Moodle data) ──
        peter   = Male("Peter");   peter.age = 70;   peter.nationality = "French"
        marie   = Female("Marie"); marie.age = 69;   marie.nationality = "French"
        thomas  = Male("Thomas");  thomas.age = 40;  thomas.nationality = "French"
        paul    = Male("Paul");    paul.age = 38
        sylvie  = Female("Sylvie"); sylvie.age = 30
        alex    = Male("Alex");    alex.age = 25
        chloe   = Female("Chloe"); chloe.age = 18
        john    = Male("John");    john.age = 45;    john.nationality = "Italian"
        pedro   = Male("Pedro");   pedro.age = 10
        tom     = Male("Tom");     tom.age = 10
        michael = Male("Michael"); michael.age = 5
        claude  = Male("Claude");  claude.age = 5;   claude.nationality = "French"

        # Family relationships
        peter.isMarriedWith  = [marie]
        peter.isFatherOf     = [thomas, paul, sylvie, chloe]
        marie.isMotherOf     = [thomas, paul, sylvie, chloe]
        thomas.isSiblingOf   = [paul, sylvie, chloe]
        paul.isSiblingOf     = [thomas, sylvie, chloe]
        sylvie.isSiblingOf   = [thomas, paul, chloe]
        chloe.isSiblingOf    = [thomas, paul, sylvie]
        sylvie.isMarriedWith = [john]
        john.isFatherOf      = [pedro, tom]
        sylvie.isMotherOf    = [pedro, tom]
        thomas.isFatherOf    = [alex, michael, claude]

        # ── SWRL Rule ──
        rule = Imp()
        rule.set_as_rule(
            "Person(?p), age(?p, ?a), greaterThan(?a, 60) -> OldPerson(?p)"
        )

    # ── Report before reasoning ──
    print("\n--- SWRL Rule ---")
    print("  Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> OldPerson(?p)")

    print("\n--- All persons before reasoning ---")
    for p in Person.instances():
        a = p.age if p.age else "?"
        print(f"  {p.name:>10}  age={a}")

    print(f"\n  OldPerson instances before: {list(OldPerson.instances())}")

    # ── Run Pellet reasoner ──
    print("\n--- Running Pellet Reasoner ---")
    reasoner_ok = False
    try:
        with onto:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True,
            )
        reasoner_ok = True
        print("  Pellet reasoning completed successfully.")
    except Exception as e:
        print(f"  Pellet FAILED: {e}")
        print("  (Requires Java 17+. Falling back to Python-based SWRL evaluation.)")

    # ── Fallback: apply the SWRL rule manually in Python ──
    if not reasoner_ok:
        print("\n--- Python Fallback: Applying SWRL Rule ---")
        with onto:
            for p in Person.instances():
                a = p.age
                if a is not None and a > 60:
                    p.is_a.append(OldPerson)
                    print(f"  Inferred: {p.name} (age={a}) -> OldPerson")

    # ── Report after reasoning ──
    old_persons = list(OldPerson.instances())
    print(f"\n--- After Reasoning ---")
    print(f"  OldPerson instances ({len(old_persons)}):")
    results = []
    for p in old_persons:
        a = p.age if p.age else "?"
        print(f"    -> {p.name} (age={a})")
        results.append({"name": p.name, "age": a})

    # ── Save ──
    owl_path = os.path.join(out_dir, "family_reasoned.owl")
    onto.save(owl_path, format="rdfxml")
    print(f"\n  Saved reasoned ontology -> {owl_path}")

    result = {
        "swrl_rule": "Person(?p) ^ age(?p, ?a) ^ swrlb:greaterThan(?a, 60) -> OldPerson(?p)",
        "old_persons": results,
        "total_persons": len(list(Person.instances())),
    }
    json_path = os.path.join(out_dir, "part1_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  Saved results -> {json_path}")

    return result


if __name__ == "__main__":
    main()
