"""KG retriever — loads TD4 RDF graph, provides SPARQL-based fact retrieval."""

import logging
from rdflib import Graph, Namespace, Literal, URIRef
from rdflib.namespace import RDF, RDFS, OWL, FOAF, XSD

logger = logging.getLogger(__name__)

# Namespaces used in the TD4 expanded KB
MARIO = Namespace("http://esilv.fr/mario#")
PROP = Namespace("http://esilv.fr/mario/prop#")
DBO = Namespace("http://dbpedia.org/ontology/")
DBR = Namespace("http://dbpedia.org/resource/")
DBP = Namespace("http://dbpedia.org/property/")
WD = Namespace("http://www.wikidata.org/entity/")
WDT = Namespace("http://www.wikidata.org/prop/direct/")
DCT = Namespace("http://purl.org/dc/terms/")
SCHEMA = Namespace("http://schema.org/")


class KGRetriever:
    """SPARQL-based fact retrieval on the Mario Knowledge Graph."""

    def __init__(self, *graph_paths: str):
        """Load one or more TTL files into a single rdflib Graph."""
        self.graph = Graph()
        for path in graph_paths:
            logger.info(f"Loading RDF graph: {path}")
            self.graph.parse(path, format="turtle")

        # bind prefixes for cleaner SPARQL
        self.graph.bind("mario", MARIO)
        self.graph.bind("prop", PROP)
        self.graph.bind("dbo", DBO)
        self.graph.bind("dbr", DBR)
        self.graph.bind("dbp", DBP)
        self.graph.bind("wd", WD)
        self.graph.bind("wdt", WDT)
        self.graph.bind("dct", DCT)
        self.graph.bind("schema", SCHEMA)
        self.graph.bind("foaf", FOAF)

        logger.info(f"KG loaded: {len(self.graph)} triples")

        # build label index for entity lookup
        self._label_index = {}  # lowercase label -> URI
        self._build_label_index()

    # ── Label index ───────────────────────────────────────────────────

    def _build_label_index(self):
        """Index rdfs:label and foaf:name for entity lookup by name."""
        for s, p, o in self.graph.triples((None, RDFS.label, None)):
            if isinstance(o, Literal):
                lang = o.language
                if lang is None or lang == "en":
                    self._label_index[str(o).lower()] = s

        for s, p, o in self.graph.triples((None, FOAF.name, None)):
            if isinstance(o, Literal):
                self._label_index[str(o).lower()] = s

        # also index mario: URIs by their local name
        for s, p, o in self.graph.triples((None, RDF.type, None)):
            uri = str(s)
            if uri.startswith(str(MARIO)):
                local = uri.split("#")[-1].replace("_", " ").lower()
                self._label_index[local] = s

        logger.info(f"Label index: {len(self._label_index)} entries")

    # ── Entity search ─────────────────────────────────────────────────

    def search_entity(self, name: str):
        """Find entity URI by name. Tries exact match, then substring."""
        key = name.lower().strip()

        # exact match
        if key in self._label_index:
            return self._label_index[key]

        # try mario: namespace directly
        slug = name.replace(" ", "_")
        mario_uri = MARIO[slug]
        if (mario_uri, None, None) in self.graph:
            return mario_uri

        # try dbr: namespace
        dbr_uri = DBR[slug]
        if (dbr_uri, None, None) in self.graph:
            return dbr_uri

        # substring match
        for label, uri in self._label_index.items():
            if key in label or label in key:
                return uri

        return None

    # ── Get facts ─────────────────────────────────────────────────────

    def get_facts(self, entity_uri, limit: int = 20):
        """Get human-readable facts about an entity."""
        facts = []
        noisy = {"sameAs", "wikiPageWikiLink", "wikiPageExternalLink",
                 "wikiPageID", "wikiPageRevisionID", "wikiPageLength",
                 "depiction", "thumbnail", "isPrimaryTopicOf", "wasDerivedFrom",
                 "wikiPageUsesTemplate", "wikiPageDisambiguates"}

        for s, p, o in self.graph.triples((entity_uri, None, None)):
            pred_name = self._short_name(p)
            if pred_name in noisy:
                continue

            # filter non-English literals
            if isinstance(o, Literal) and o.language and o.language != "en":
                continue

            obj_str = self._format_value(o)
            if obj_str:
                facts.append(f"{pred_name}: {obj_str}")
                if len(facts) >= limit:
                    break

        return facts

    def get_related(self, entity_uri, limit: int = 10):
        """Get entities related to this one (1-hop)."""
        related = []
        for s, p, o in self.graph.triples((entity_uri, None, None)):
            if isinstance(o, URIRef) and str(p) != str(OWL.sameAs):
                name = self._short_name(o)
                if name and len(name) > 1:
                    related.append(name)
        return list(set(related))[:limit]

    # ── Question answering ────────────────────────────────────────────

    def answer_question(self, question: str):
        """Extract entities from question, retrieve their facts."""
        stop_words = {"who", "what", "where", "when", "how", "why", "is", "are",
                      "was", "were", "the", "a", "an", "in", "of", "and", "or",
                      "to", "for", "does", "did", "do", "can", "tell", "me",
                      "about", "which", "that", "this", "it", "has", "have",
                      "be", "been", "with", "from", "on", "at", "by", "not", "no"}
        words = [w for w in question.lower().split() if w.strip("?,!.") not in stop_words]

        found_entities = []

        # try multi-word matches first (longest match)
        for n in range(min(4, len(words)), 0, -1):
            for i in range(len(words) - n + 1):
                phrase = " ".join(words[i:i + n])
                if len(phrase) < 2:
                    continue
                uri = self.search_entity(phrase)
                if uri and uri not in found_entities:
                    found_entities.append(uri)

        if not found_entities:
            return []

        all_facts = []
        for uri in found_entities[:3]:  # limit to 3 entities
            label = self._short_name(uri)
            facts = self.get_facts(uri, limit=15)
            if facts:
                all_facts.append(f"[{label}]\n" + "\n".join(f"  - {f}" for f in facts))

        return all_facts

    # ── Custom SPARQL ─────────────────────────────────────────────────

    def sparql(self, query_str: str):
        """Execute an arbitrary SPARQL query."""
        try:
            results = self.graph.query(query_str)
            return [dict(row.asdict()) for row in results]
        except Exception as e:
            logger.error(f"SPARQL error: {e}")
            return []

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        subjects = set(s for s, _, _ in self.graph)
        predicates = set(p for _, p, _ in self.graph)
        return {
            "triples": len(self.graph),
            "entities": len(subjects),
            "predicates": len(predicates),
            "labels_indexed": len(self._label_index),
        }

    # ── Helpers ───────────────────────────────────────────────────────

    def _short_name(self, uri) -> str:
        """Extract human-readable short name from a URI."""
        s = str(uri)
        for sep in ["#", "/"]:
            if sep in s:
                s = s.rsplit(sep, 1)[-1]
        return s.replace("_", " ")

    def _format_value(self, obj) -> str:
        """Format an RDF object for display."""
        if isinstance(obj, Literal):
            return str(obj)
        if isinstance(obj, URIRef):
            name = self._short_name(obj)
            # skip opaque IDs
            if name.startswith("Q") and name[1:].isdigit():
                return None
            if name.startswith("Category:"):
                return name[9:]
            return name
        return str(obj)
