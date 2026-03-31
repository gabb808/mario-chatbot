"""
TD5 KGE integration — uses pre-computed nearest-neighbour results from PyKEEN
(TransE / ComplEx trained on expanded_kb.nt) to surface related entities.

The model weights were not persisted, but the nearest_neighbours.json produced
by TD5/part2_train_evaluate.py stores the top-K similar entities for 10 key
Mario entities.  This module loads that file, filters noisy results, and looks
up KG facts for the useful neighbours via the existing KGRetriever.
"""

import json
import logging
import os
import re

from rdflib import URIRef

logger = logging.getLogger(__name__)

# Regex: Wikidata QID (Q + digits only)
_QID_RE = re.compile(r'^Q\d+$')
# Regex: Freebase / opaque machine IDs
_OPAQUE_RE = re.compile(r'^(m\.|g\/)', re.IGNORECASE)
# Image/file extensions
_FILE_EXTS = {'.jpg', '.jpeg', '.png', '.svg', '.gif', '.pdf', '.ogg', '.wav'}

# Keyword → list of covered entity URIs in nearest_neighbors.json
_ENTITY_KEYWORDS: dict[str, list[str]] = {
    "mario":     ["http://dbpedia.org/resource/Mario",
                  "http://esilv.fr/mario#Mario"],
    "luigi":     ["http://dbpedia.org/resource/Luigi",
                  "http://esilv.fr/mario#Luigi"],
    "nintendo":  ["http://dbpedia.org/resource/Nintendo"],
    "miyamoto":  ["http://dbpedia.org/resource/Shigeru_Miyamoto"],
    "bowser":    ["http://dbpedia.org/resource/Bowser_(character)"],
    "zelda":     ["http://dbpedia.org/resource/The_Legend_of_Zelda"],
    "super mario bros": ["http://dbpedia.org/resource/Super_Mario_Bros."],
    "game boy":  ["http://dbpedia.org/resource/Game_Boy"],
}


def _is_useful(uri: str, label: str) -> bool:
    """Return True only for English DBpedia URIs that look like real entities."""
    # Must be English DBpedia
    if not uri.startswith("http://dbpedia.org/resource/"):
        return False
    # No image/file paths
    _, ext = os.path.splitext(label.lower())
    if ext in _FILE_EXTS:
        return False
    # No opaque Wikidata QIDs or Freebase IDs
    local = uri.split("/")[-1]
    if _QID_RE.match(local) or _OPAQUE_RE.match(local):
        return False
    # Must have at least 3 ASCII letters
    ascii_letters = sum(1 for c in label if c.isascii() and c.isalpha())
    if ascii_letters < 3:
        return False
    # No Category: URIs (too broad)
    if local.startswith("Category:"):
        return False
    return True


class KGENeighbourRetriever:
    """
    Looks up pre-computed KGE nearest neighbours for key Mario entities,
    then fetches their KG facts via the existing KGRetriever.
    """

    def __init__(self, neighbors_path: str):
        self._data: dict[str, list[dict]] = {}
        if not os.path.exists(neighbors_path):
            logger.warning(f"KGE neighbours file not found: {neighbors_path}")
            return
        with open(neighbors_path, encoding="utf-8") as f:
            raw = json.load(f)

        # Filter once at load time — keep only useful English DBpedia neighbours
        for entity_uri, neighbours in raw.items():
            useful = [
                n for n in neighbours
                if _is_useful(n.get("entity", ""), n.get("label", ""))
                and n.get("sim", 0) >= 0.30
            ]
            if useful:
                self._data[entity_uri] = useful

        covered = sum(len(v) for v in self._data.values())
        logger.info(f"KGE neighbours loaded: {len(self._data)} entities, "
                    f"{covered} useful neighbours total")

    # ── Public API ────────────────────────────────────────────────────

    def get_neighbour_facts(self, question: str, kg_retriever) -> list[str]:
        """
        Given a natural-language question and a KGRetriever instance,
        return a list of formatted fact blocks for KGE-suggested related entities.
        """
        if not self._data:
            return []

        question_lower = question.lower()
        matched_uris: list[str] = []

        # Match question keywords to covered entities
        for keyword, uris in _ENTITY_KEYWORDS.items():
            if keyword in question_lower:
                for uri in uris:
                    if uri in self._data and uri not in matched_uris:
                        matched_uris.append(uri)

        if not matched_uris:
            return []

        facts_blocks: list[str] = []
        seen_neighbours: set[str] = set()

        for entity_uri in matched_uris[:2]:          # at most 2 seed entities
            neighbours = self._data[entity_uri]
            for n in neighbours[:3]:                 # top-3 neighbours each
                nb_uri = n["entity"]
                if nb_uri in seen_neighbours:
                    continue
                seen_neighbours.add(nb_uri)

                nb_ref = URIRef(nb_uri)
                facts = kg_retriever.get_facts(nb_ref, limit=5)
                if facts:
                    label = n["label"].replace("_", " ")
                    block = (f"[KGE-related: {label} (sim={n['sim']:.2f})]\n"
                             + "\n".join(f"  - {f}" for f in facts))
                    facts_blocks.append(block)

        return facts_blocks

    def stats(self) -> dict:
        return {
            "covered_entities": len(self._data),
            "total_useful_neighbours": sum(len(v) for v in self._data.values()),
        }
