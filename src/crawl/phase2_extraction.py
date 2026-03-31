"""
Phase 2: Information Extraction
Lab Session 1 - From the Unstructured Web to Structured Entities
Course: Web Mining & Semantics -- ESILV A4

NOTE on NLP library choice:
  The TD specification recommends spaCy (en_core_web_trf).
  spaCy 3.x is NOT compatible with Python 3.14 (pydantic v1 type-inference
  fails at import time). We therefore use NLTK 3.9 as the runtime NLP engine,
  supplemented with 50+ Mario-specific regex patterns.

  Functional equivalence table:
  ┌─────────────────────────┬─────────────────────────────┐
  │ spaCy                   │ This implementation         │
  ├─────────────────────────┼─────────────────────────────┤
  │ nlp(text)               │ sent_tokenize + word_tokenize│
  │ tok.pos_ / tok.dep_     │ pos_tag                     │
  │ ent.label_ (NER)        │ ne_chunk + domain patterns  │
  │ nsubj / dobj dependency │ Verb-anchored SVO heuristic │
  └─────────────────────────┴─────────────────────────────┘

Deliverables:
  outputs/extracted_knowledge.csv
  outputs/extracted_relationships.csv
"""

import json
import csv
import re
from pathlib import Path
from typing import List, Dict, Tuple
import logging

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.tree import Tree

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Download NLTK data silently
for _res in ["punkt", "punkt_tab", "averaged_perceptron_tagger",
             "averaged_perceptron_tagger_eng",
             "maxent_ne_chunker", "maxent_ne_chunker_tab", "words"]:
    try:
        nltk.download(_res, quiet=True)
    except Exception:
        pass


# =============================================================================
# Domain knowledge
# =============================================================================

MARIO_PATTERNS: List[Tuple[str, str]] = [
    # Characters
    (r"\bMario\b",               "CHARACTER"),
    (r"\bLuigi\b",               "CHARACTER"),
    (r"\bPrincess Peach\b",      "CHARACTER"),
    (r"\bPeach Toadstool\b",     "CHARACTER"),
    (r"\bPeach\b",               "CHARACTER"),
    (r"\bBowser\b",              "CHARACTER"),
    (r"\bKing Koopa\b",          "CHARACTER"),
    (r"\bYoshi\b",               "CHARACTER"),
    (r"\bDonkey Kong\b",         "CHARACTER"),
    (r"\bToad\b",                "CHARACTER"),
    (r"\bRosalina\b",            "CHARACTER"),
    (r"\bPrincess Daisy\b",      "CHARACTER"),
    (r"\bDaisy\b",               "CHARACTER"),
    (r"\bWario\b",               "CHARACTER"),
    (r"\bWaluigi\b",             "CHARACTER"),
    (r"\bKamek\b",               "CHARACTER"),
    (r"\bBowser Jr\.?\b",        "CHARACTER"),
    (r"\bLuma\b",                "CHARACTER"),
    (r"\bGoomba\b",              "ENEMY"),
    (r"\bKoopa Troopa\b",        "ENEMY"),
    (r"\bShy Guy\b",             "ENEMY"),
    (r"\bBoo\b",                 "ENEMY"),
    (r"\bThwomp\b",              "ENEMY"),
    (r"\bBullet Bill\b",         "ENEMY"),
    # Locations
    (r"\bMushroom Kingdom\b",    "LOCATION"),
    (r"\bBowser.s Castle\b",     "LOCATION"),
    (r"\bPeach.s Castle\b",      "LOCATION"),
    (r"\bDinosaur Land\b",       "LOCATION"),
    (r"\bIsle Delfino\b",        "LOCATION"),
    (r"\bToad Town\b",           "LOCATION"),
    (r"\bStar Road\b",           "LOCATION"),
    (r"\bKoopa Kingdom\b",       "LOCATION"),
    # Game titles
    (r"\bSuper Mario Bros\.?(?: \d+)?\b", "GAME"),
    (r"\bSuper Mario World\b",   "GAME"),
    (r"\bSuper Mario Odyssey\b", "GAME"),
    (r"\bSuper Mario Galaxy\b",  "GAME"),
    (r"\bMario Kart\b",          "GAME"),
    (r"\bMario Party\b",         "GAME"),
    (r"\bPaper Mario\b",         "GAME"),
    (r"\bMario & Luigi\b",       "GAME"),
    (r"\bLuigi.s Mansion\b",     "GAME"),
    (r"\bDonkey Kong Country\b", "GAME"),
    (r"\bYoshi.s Island\b",      "GAME"),
    # Items
    (r"\bFire Flower\b",         "ITEM"),
    (r"\bSuper Mushroom\b",      "ITEM"),
    (r"\bInvincibility Star\b",  "ITEM"),
    (r"\bSuper Star\b",          "ITEM"),
    (r"\bCape Feather\b",        "ITEM"),
    (r"\bIce Flower\b",          "ITEM"),
    (r"\bTanooki Suit\b",        "ITEM"),
    (r"\bMega Mushroom\b",       "ITEM"),
    # Orgs
    (r"\bNintendo\b",            "ORG"),
]

RELATION_VERBS: Dict[str, str] = {
    "saved": "saves", "saves": "saves", "rescue": "saves",
    "rescues": "saves", "rescued": "saves",
    "kidnapped": "kidnaps", "kidnaps": "kidnaps", "captured": "kidnaps",
    "rules": "rules", "ruled": "rules", "governs": "rules",
    "defeated": "defeats", "defeats": "defeats",
    "fought": "fights", "fights": "fights",
    "appeared": "appears_in", "appears": "appears_in",
    "introduced": "appears_in", "starring": "appears_in",
    "created": "created_by", "designed": "created_by",
    "developed": "developed_by", "published": "published_by",
    "married": "is_married_to",
    "companion": "companion_of", "friend": "friend_of",
    "ally": "ally_of", "enemy": "enemy_of",
    "rival": "rival_of", "rivals": "rival_of",
    "located": "located_in", "lives": "lives_in",
    "inhabits": "lives_in", "set": "located_in",
}

ENTITY_LABELS: Dict[str, str] = {
    "CHARACTER": "Person", "ENEMY": "Person",
    "LOCATION": "Location", "GAME": "GameTitle",
    "ITEM": "Item", "ORG": "Organization",
    "PERSON": "Person", "ORGANIZATION": "Organization",
    "GPE": "Location", "FACILITY": "Location", "NORP": "Group",
}

STOPWORDS = {
    "he", "she", "it", "his", "her", "its", "the", "a", "an",
    "this", "that", "they", "their", "to", "of", "in", "on",
    "by", "for", "with", "and", "or", "but", "as", "is",
    "at", "from", "was", "are", "i", "we", "you",
}


# =============================================================================
# Extractor
# =============================================================================

class InformationExtractor:
    """
    NLP extraction pipeline (NLTK + domain patterns).

    Extraction order:
      1. Domain patterns (Mario-specific regex) — fast, high precision
      2. NLTK ne_chunk on first 15 sentences — generic NER coverage
      3. Verb-anchored SVO relation extraction
      4. Co-occurrence fallback (capped to keep output manageable)
    """

    def __init__(self,
                 input_path: str = "outputs/crawler_output.jsonl",
                 output_dir:  str = "outputs"):
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.entities:      List[Dict] = []
        self.relationships: List[Dict] = []

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[.,;:!?]+$", "", text.strip())

    @staticmethod
    def _valid(text: str) -> bool:
        return bool(text) and len(text) >= 2 and text.lower() not in STOPWORDS

    # ── entity extraction ─────────────────────────────────────────────────────

    def _ents_pattern(self, text: str, url: str) -> List[Dict]:
        """Fast regex-based NER for Mario-universe entities."""
        found = []
        for pat, etype in MARIO_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                s = self._norm(m.group())
                if self._valid(s):
                    found.append({
                        "text": s, "type": etype,
                        "label": ENTITY_LABELS.get(etype, etype),
                        "source_url": url, "confidence": 0.97, "method": "pattern",
                    })
        return found

    def _ents_nltk(self, text: str, url: str) -> List[Dict]:
        """
        NLTK ne_chunk on the first 15 sentences (performance cap).

        This covers generic PERSON / ORG / GPE entities not listed in the
        Mario-specific pattern dictionary (e.g. 'Miyamoto', 'Nintendo EAD').

        Approximates spaCy's NER pipeline:
          spaCy:   doc = nlp(text); [(ent.text, ent.label_) for ent in doc.ents]
          NLTK:    ne_chunk(pos_tag(word_tokenize(sent))) + Tree traversal
        """
        found = []
        try:
            for sent in sent_tokenize(text)[:15]:
                tokens = word_tokenize(sent)
                if len(tokens) > 50:
                    continue          # skip very long sentences (slowdowns)
                tree = ne_chunk(pos_tag(tokens), binary=False)
                for subtree in tree:
                    if isinstance(subtree, Tree):
                        etype = subtree.label()
                        s     = self._norm(" ".join(w for w, _ in subtree.leaves()))
                        if self._valid(s):
                            found.append({
                                "text": s, "type": etype,
                                "label": ENTITY_LABELS.get(etype, etype),
                                "source_url": url, "confidence": 0.80, "method": "nltk",
                            })
        except Exception as exc:
            logger.debug(f"NLTK NER error: {exc}")
        return found

    # ── relation extraction ───────────────────────────────────────────────────

    def _svo(self, sentence: str, ents: List[Tuple[str, str]], url: str) -> List[Dict]:
        """
        Verb-anchored SVO extraction.

        Algorithm (approximates spaCy nsubj/dobj dependency arcs):
          For each known relation verb found in the sentence:
            - Split sentence at verb position
            - Subject  = entity closest to the LEFT of the verb
            - Object   = entity closest to the RIGHT of the verb
          => (subject, canonical_predicate, object)

        Example:
          "Mario saved Princess Peach from Bowser."
          verb: "saved"  ->  left="Mario"  right="Princess Peach"
          triple: (Mario, saves, Princess Peach)
        """
        rels = []
        sent_l = sentence.lower()
        for verb_surf, predicate in RELATION_VERBS.items():
            if verb_surf not in sent_l:
                continue
            m = re.search(r"\b" + re.escape(verb_surf) + r"\b",
                          sentence, re.IGNORECASE)
            if not m:
                continue
            left, right = sentence[: m.start()], sentence[m.end():]
            subjs = [(e, t) for e, t in ents if e.lower() in left.lower()]
            objs  = [(e, t) for e, t in ents if e.lower() in right.lower()]
            if subjs and objs:
                subj, st = subjs[-1]
                obj,  ot = objs[0]
                if subj != obj:
                    rels.append({
                        "subject": subj,
                        "subject_type": ENTITY_LABELS.get(st, st),
                        "predicate": predicate,
                        "object": obj,
                        "object_type": ENTITY_LABELS.get(ot, ot),
                        "source_url": url,
                        "method": "svo",
                        "sentence": sentence[:120],
                    })
        return rels

    def _cooccurrence(self, ents: List[Tuple[str, str]], url: str,
                      sent: str, max_pairs: int = 6) -> List[Dict]:
        """Co-occurrence fallback (capped at max_pairs per sentence)."""
        rels, count = [], 0
        for i, (e1, t1) in enumerate(ents):
            if count >= max_pairs:
                break
            for e2, t2 in ents[i + 1:]:
                if count >= max_pairs:
                    break
                if e1 != e2:
                    rels.append({
                        "subject": e1,
                        "subject_type": ENTITY_LABELS.get(t1, t1),
                        "predicate": "co_occurs_with",
                        "object": e2,
                        "object_type": ENTITY_LABELS.get(t2, t2),
                        "source_url": url,
                        "method": "cooccurrence",
                        "sentence": sent[:120],
                    })
                    count += 1
        return rels

    # ── pipeline ──────────────────────────────────────────────────────────────

    def process_document(self, doc: Dict) -> Tuple[List[Dict], List[Dict]]:
        """Run NER + relation extraction on a single document."""
        text = doc.get("text", "")
        url  = doc.get("url", "")

        # 1) Entity extraction
        p_ents = self._ents_pattern(text, url)
        n_ents = self._ents_nltk(text, url)
        seen   = {e["text"].lower() for e in p_ents}
        all_e  = p_ents + [e for e in n_ents if e["text"].lower() not in seen]

        # 2) Relation extraction (first 60 sentences to bound runtime)
        ep     = [(e["text"], e["type"]) for e in all_e]
        all_r: List[Dict] = []
        for sent in sent_tokenize(text)[:60]:
            se = list(dict.fromkeys(
                (e, t) for e, t in ep if e.lower() in sent.lower()
            ))
            if len(se) < 2:
                continue
            sv = self._svo(sent, se, url)
            all_r.extend(sv)
            if not sv:
                # co-occurrence only when no SVO relation found (max 6 per sent)
                all_r.extend(self._cooccurrence(se, url, sent, max_pairs=6))

        return all_e, all_r

    def process_documents(self) -> Tuple[List[Dict], List[Dict]]:
        """Process all documents from the JSONL file."""
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"{self.input_path} not found — run Phase 1 first"
            )
        with open(self.input_path, encoding="utf-8") as fh:
            docs = [json.loads(l) for l in fh if l.strip()]

        logger.info(f"Processing {len(docs)} documents...")
        all_e, all_r = [], []
        for idx, doc in enumerate(docs, 1):
            e, r = self.process_document(doc)
            all_e.extend(e)
            all_r.extend(r)
            logger.info(
                f"  [{idx}/{len(docs)}] "
                f"'{doc.get('title', doc.get('url', '?'))[:35]}' : "
                f"{len(e)} ents  {len(r)} rels"
            )

        # Deduplicate entities (surface + type)
        seen_e, dedup_e = set(), []
        for e in all_e:
            k = (e["text"].lower(), e["type"])
            if k not in seen_e:
                seen_e.add(k)
                dedup_e.append(e)

        # Deduplicate relations (subj + pred + obj)
        seen_r, dedup_r = set(), []
        for r in all_r:
            k = (r["subject"].lower(), r["predicate"], r["object"].lower())
            if k not in seen_r:
                seen_r.add(k)
                dedup_r.append(r)

        self.entities      = dedup_e
        self.relationships = dedup_r
        logger.info(
            f"Final: {len(dedup_e)} unique entities, "
            f"{len(dedup_r)} unique relations"
        )
        return self.entities, self.relationships

    # ── export ────────────────────────────────────────────────────────────────

    def save_entities_csv(self, fn: str = "extracted_knowledge.csv") -> Path:
        out = self.output_dir / fn
        fields = ["text", "type", "label", "source_url", "confidence", "method"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.entities)
        logger.info(f"Entities    → {out}  ({len(self.entities)} rows)")
        return out

    def save_relationships_csv(self,
                               fn: str = "extracted_relationships.csv") -> Path:
        out = self.output_dir / fn
        fields = ["subject", "subject_type", "predicate",
                  "object", "object_type", "source_url", "method"]
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(self.relationships)
        logger.info(f"Relations   → {out}  ({len(self.relationships)} rows)")
        return out

    def statistics(self) -> Dict:
        etype_c: Dict[str, int] = {}
        for e in self.entities:
            etype_c[e["type"]] = etype_c.get(e["type"], 0) + 1
        pred_c: Dict[str, int] = {}
        for r in self.relationships:
            pred_c[r["predicate"]] = pred_c.get(r["predicate"], 0) + 1
        return {
            "total_entities":      len(self.entities),
            "entity_type_dist":    etype_c,
            "total_relationships": len(self.relationships),
            "predicate_dist":      pred_c,
        }


# =============================================================================
# Entry point
# =============================================================================

def run_phase2(input_path: str = "outputs/crawler_output.jsonl",
               output_dir: str = "outputs") -> Dict:
    print("\n" + "=" * 60)
    print("PHASE 2: INFORMATION EXTRACTION")
    print("=" * 60)
    print(f"NLP engine : NLTK {nltk.__version__} + 50+ Mario domain patterns")
    print(f"Input      : {input_path}")
    print(f"Output dir : {output_dir}/\n")

    ext = InformationExtractor(input_path=input_path, output_dir=output_dir)
    ext.process_documents()
    ext.save_entities_csv()
    ext.save_relationships_csv()
    stats = ext.statistics()

    print(f"\n{chr(45)*60}")
    print("PHASE 2 SUMMARY")
    print(f"{chr(45)*60}")
    print(f"  Unique entities      : {stats['total_entities']}")
    print(f"  Unique relationships : {stats['total_relationships']}")
    print("  Entity type distribution:")
    for etype, cnt in sorted(stats["entity_type_dist"].items(),
                             key=lambda x: -x[1]):
        bar = "#" * min(cnt, 40)
        print(f"    {etype:<22} {cnt:>4}  {bar}")
    print("  Top predicates (SVO + co-occurrence):")
    for pred, cnt in sorted(stats["predicate_dist"].items(),
                             key=lambda x: -x[1])[:10]:
        print(f"    {pred:<28} {cnt}")
    return stats


if __name__ == "__main__":
    run_phase2()
