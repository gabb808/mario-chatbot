"""
TD5 - Part 2, Step 1: Data Preparation for KGE
================================================
Load expanded KB from TD4, clean for embedding, and split
into train / validation / test sets (80/10/10).
"""
import os
import re
import json
import numpy as np
from collections import Counter

# ── Paths ────────────────────────────────────────────────────
BASE      = os.path.dirname(os.path.abspath(__file__))
TD4_DATA  = os.path.join(BASE, "..", "TD4", "data")
OUT_DIR   = os.path.join(BASE, "data")
NT_PATH   = os.path.join(TD4_DATA, "expanded_kb.nt")


# ── NT Parser ────────────────────────────────────────────────
_NT_RE = re.compile(
    r'^<([^>]+)>\s+<([^>]+)>\s+<([^>]+)>\s*\.\s*$'
)

def parse_nt(path):
    """Fast NT parser: yield only (s, p, o) where o is a URI."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _NT_RE.match(line)
            if m:
                yield m.group(1), m.group(2), m.group(3)


# ── Cleaning ─────────────────────────────────────────────────
# Predicates that are literal-heavy or noisy for KGE
SKIP_PREDICATES = {
    "http://dbpedia.org/ontology/abstract",
    "http://www.w3.org/2000/01/rdf-schema#comment",
    "http://xmlns.com/foaf/0.1/name",
}

def load_and_clean():
    """
    Load expanded KB, keep only entity-entity triples,
    remove duplicates and noisy predicates.
    """
    print(f"Loading {NT_PATH} ...")
    triples = []
    seen = set()
    stats = Counter()

    for s, p, o in parse_nt(NT_PATH):
        stats["total_parsed"] += 1

        if p in SKIP_PREDICATES:
            stats["skipped_predicate"] += 1
            continue

        t = (s, p, o)
        if t in seen:
            stats["duplicates"] += 1
            continue
        seen.add(t)
        triples.append(t)

    stats["kept"] = len(triples)
    print(f"  Parsed URI-URI triples: {stats['total_parsed']}")
    print(f"  Duplicates removed:     {stats['duplicates']}")
    print(f"  Noisy preds removed:    {stats['skipped_predicate']}")
    print(f"  Triples kept:           {stats['kept']}")

    # Entity and relation stats
    entities  = set()
    relations = set()
    for s, p, o in triples:
        entities.add(s); entities.add(o); relations.add(p)
    stats["entities"]  = len(entities)
    stats["relations"] = len(relations)
    print(f"  Unique entities:        {stats['entities']}")
    print(f"  Unique relations:       {stats['relations']}")
    return triples, dict(stats)


# ── Splitting ────────────────────────────────────────────────
def split_triples(triples, ratios=(0.8, 0.1, 0.1), seed=42):
    """
    Split triples 80/10/10 ensuring every entity in valid/test
    also appears in training (no leakage).
    """
    rng = np.random.RandomState(seed)
    idx = np.arange(len(triples))
    rng.shuffle(idx)

    n_train = int(len(triples) * ratios[0])
    n_valid = int(len(triples) * ratios[1])

    train = [triples[i] for i in idx[:n_train]]
    valid = [triples[i] for i in idx[n_train:n_train + n_valid]]
    test  = [triples[i] for i in idx[n_train + n_valid:]]

    # Iteratively fix entity leakage
    for iteration in range(10):
        train_ents = set()
        for s, _, o in train:
            train_ents.add(s); train_ents.add(o)

        new_valid, new_test = [], []
        moved = 0
        for t in valid:
            if t[0] not in train_ents or t[2] not in train_ents:
                train.append(t)
                train_ents.add(t[0]); train_ents.add(t[2])
                moved += 1
            else:
                new_valid.append(t)
        for t in test:
            if t[0] not in train_ents or t[2] not in train_ents:
                train.append(t)
                train_ents.add(t[0]); train_ents.add(t[2])
                moved += 1
            else:
                new_test.append(t)
        valid, test = new_valid, new_test
        if moved == 0:
            break
        print(f"  Leakage fix iter {iteration+1}: moved {moved} triples to train")

    return train, valid, test


# ── Save ─────────────────────────────────────────────────────
def save_split(triples_list, path):
    """Save triples as tab-separated file (PyKEEN compatible)."""
    with open(path, "w", encoding="utf-8") as f:
        for s, p, o in triples_list:
            f.write(f"{s}\t{p}\t{o}\n")


def save_entity_type_map(triples, path):
    """Extract rdf:type mappings for later t-SNE coloring."""
    RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
    type_map = {}
    for s, p, o in triples:
        if p == RDF_TYPE:
            # Keep the most specific type (last fragment)
            type_label = o.rsplit("/", 1)[-1].rsplit("#", 1)[-1]
            type_map[s] = type_label
    with open(path, "w", encoding="utf-8") as f:
        json.dump(type_map, f, indent=1)
    return type_map


# ── Main ─────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Part 2.1: Data Preparation for Knowledge Graph Embedding")
    print("=" * 60)

    os.makedirs(OUT_DIR, exist_ok=True)

    # 1. Load & clean
    triples, clean_stats = load_and_clean()

    # 2. Split
    print("\nSplitting 80/10/10 ...")
    train, valid, test = split_triples(triples)
    print(f"  Train: {len(train):>7}")
    print(f"  Valid: {len(valid):>7}")
    print(f"  Test:  {len(test):>7}")

    # 3. Save
    save_split(train, os.path.join(OUT_DIR, "train.txt"))
    save_split(valid, os.path.join(OUT_DIR, "valid.txt"))
    save_split(test,  os.path.join(OUT_DIR, "test.txt"))
    save_split(triples, os.path.join(OUT_DIR, "cleaned_triples.tsv"))
    print("\n  Saved: train.txt, valid.txt, test.txt, cleaned_triples.tsv")

    # 4. Entity type map (for t-SNE coloring later)
    type_map = save_entity_type_map(triples, os.path.join(OUT_DIR, "entity_types.json"))
    print(f"  Entity type map: {len(type_map)} entities with types")

    # 5. Stats
    split_stats = {
        "cleaning": clean_stats,
        "train_size": len(train),
        "valid_size": len(valid),
        "test_size":  len(test),
        "typed_entities": len(type_map),
    }
    with open(os.path.join(OUT_DIR, "prepare_stats.json"), "w") as f:
        json.dump(split_stats, f, indent=2)
    print(f"  Stats saved -> prepare_stats.json")

    return triples, train, valid, test, split_stats


if __name__ == "__main__":
    main()
