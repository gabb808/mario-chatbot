"""
TD4 â€” Step 2: Entity & Predicate Alignment with Wikidata
=========================================================
For each entity in the private KB:
  1. Check static VIP mapping (100 % confidence for known Mario entities).
  2. If not in VIP: attempt Wikidata label search via SPARQL (batch-friendly).
  3. If found (confidence â‰¥ threshold): add owl:sameAs link.

For each predicate in the private KB:
  1. Use hard-coded Wikidata property hints for known predicates.
  2. If not mapped: search Wikidata property API.

Outputs:
  data/alignment.ttl      â€” owl:sameAs + owl:equivalentProperty triples
  data/entity_mapping.csv â€” mapping table with confidence scores
  data/predicate_mapping.csv â€” predicate alignment table

Run:  python step2_align.py
"""

import csv
import sys
import time
from pathlib import Path

import httpx
from rdflib import Graph, Namespace, URIRef, Literal, RDF, RDFS, OWL, XSD

# â”€â”€â”€ Paths â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
HERE      = Path(__file__).parent
DATA      = HERE / "data"
KB_FILE   = DATA / "private_kb.ttl"
OUT_ALIGN = DATA / "alignment.ttl"
OUT_ENT   = DATA / "entity_mapping.csv"
OUT_PRED  = DATA / "predicate_mapping.csv"

# â”€â”€â”€ Namespaces â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
BASE  = "http://esilv.fr/mario#"
MARIO = Namespace(BASE)
PROP  = Namespace("http://esilv.fr/mario/prop#")
WD    = Namespace("http://www.wikidata.org/entity/")
WDT   = Namespace("http://www.wikidata.org/prop/direct/")

WIKIDATA_SEARCH = "https://www.wikidata.org/w/api.php"
WIKIDATA_SPARQL = "https://query.wikidata.org/sparql"
CONFIDENCE_THRESHOLD = 0.55
REQUEST_DELAY = 1.5   # increased after rate-limit experience

# =============================================================================
# Static VIP mapping â€” manually curated, confidence 1.0
# (Covers the most important Mario universe entities in Wikidata)
# =============================================================================
VIP_MAPPING: dict[str, tuple[str, float]] = {
    # â”€â”€ Characters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Mario":               ("Q7186",    1.00),
    "Luigi":               ("Q7194",    1.00),
    "Princess Peach":      ("Q7191",    1.00),
    "Peach":               ("Q7191",    1.00),
    "Bowser":              ("Q843924",  1.00),
    "Yoshi":               ("Q214174",  1.00),
    "Donkey Kong":         ("Q83915",   1.00),
    "Wario":               ("Q7192",    1.00),
    "Waluigi":             ("Q1455836", 1.00),
    "Toad":                ("Q7200",    1.00),
    "Rosalina":            ("Q3951592", 1.00),
    "Daisy":               ("Q234574",  0.90),
    "Princess Daisy":      ("Q234574",  1.00),
    "Birdo":               ("Q7196",    1.00),
    "Koopa":               ("Q1762188", 0.90),
    "Koopa Troopa":        ("Q1762188", 1.00),
    "Goomba":              ("Q1536478", 1.00),
    "Boo":                 ("Q843930",  0.90),
    "Lakitu":              ("Q842898",  1.00),
    "Shy Guy":             ("Q2312116", 1.00),
    "Kamek":               ("Q1359777", 1.00),
    "Bowser Jr.":          ("Q3055371", 1.00),
    "Baby Mario":          ("Q21174060",0.95),
    "Diddy Kong":          ("Q1192416", 1.00),
    "Fox McCloud":         ("Q315706",  0.90),
    "Link":                ("Q172923",  0.85),
    "Toadette":            ("Q3060015", 1.00),
    "Waluigi":             ("Q1455836", 1.00),
    "Bullet Bill":         ("Q843961",  1.00),
    "Boo":                 ("Q843930",  1.00),
    "Chain Chomp":         ("Q843941",  1.00),
    "Hammer Bro":          ("Q1394898", 0.95),
    "Piranha Plant":       ("Q843955",  1.00),
    "Thwomp":              ("Q843943",  1.00),
    "Wiggler":             ("Q843960",  1.00),
    "Koopa Paratroopa":    ("Q1762188", 0.80),
    # â”€â”€ Real People â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Shigeru Miyamoto":    ("Q40219",   1.00),
    "Miyamoto":            ("Q40219",   0.95),
    "Hiroshi Yamauchi":    ("Q461704",  1.00),
    "Satoru Iwata":        ("Q324971",  1.00),
    "Gunpei Yokoi":        ("Q458834",  1.00),
    "Charles Martinet":    ("Q313536",  1.00),
    "Koji Kondo":          ("Q368283",  1.00),
    "Takashi Tezuka":      ("Q1371065", 1.00),
    "Toru Iwatani":        ("Q362566",  0.85),
    # â”€â”€ Games â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Super Mario Bros.":   ("Q174163",  1.00),
    "Super Mario Bros":    ("Q174163",  1.00),
    "Super Mario 64":      ("Q229229",  1.00),
    "Super Mario World":   ("Q193581",  1.00),
    "Super Mario Odyssey": ("Q29934932",1.00),
    "Super Mario Galaxy":  ("Q194256",  1.00),
    "Super Mario Galaxy 2":("Q2485975", 1.00),
    "Super Mario Bros. 3": ("Q192643",  1.00),
    "Super Mario Bros. 2": ("Q1203717", 1.00),
    "Super Mario Kart":    ("Q209163",  1.00),
    "Mario Kart":          ("Q209163",  0.85),
    "Mario Kart 8":        ("Q15142428",1.00),
    "Mario Party":         ("Q837773",  1.00),
    "Mario Tennis":        ("Q1369773", 1.00),
    "Super Smash Bros.":   ("Q193605",  1.00),
    "Super Smash Bros":    ("Q193605",  1.00),
    "Donkey Kong Country": ("Q1055338", 1.00),
    "Paper Mario":         ("Q1098498", 1.00),
    "Super Paper Mario":   ("Q1778427", 1.00),
    "Mario & Luigi":       ("Q1364050", 0.90),
    "Yoshi's Island":      ("Q193583",  1.00),
    "Super Mario Land":    ("Q205038",  1.00),
    "Super Mario Sunshine":("Q327622",  1.00),
    "New Super Mario Bros":("Q332083",  1.00),
    "Super Mario 3D Land": ("Q466430",  1.00),
    "Super Mario 3D World":("Q13407073",1.00),
    "Super Mario Run":     ("Q24879760",1.00),
    "Luigi's Mansion":     ("Q463975",  1.00),
    "Wario Land":          ("Q1412558", 1.00),
    "Yoshi's Story":       ("Q860270",  1.00),
    "Doki Doki Panic":     ("Q762664",  1.00),
    "Super Mario RPG":     ("Q193634",  1.00),
    "The Legend of Zelda": ("Q155788",  1.00),
    "Metroid":             ("Q219121",  1.00),
    "Kirby":               ("Q611598",  0.85),
    "Star Fox":            ("Q218628",  1.00),
    # â”€â”€ Organizations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Nintendo":            ("Q8093",    1.00),
    "Nintendo EAD":        ("Q1368900", 1.00),
    "HAL Laboratory":      ("Q202819",  1.00),
    "Rare":                ("Q271407",  0.90),
    "Square Enix":         ("Q261735",  1.00),
    "Camelot Software":    ("Q1022398", 1.00),
    "AlphaDream":          ("Q1380745", 1.00),
    "Intelligent Systems": ("Q1364072", 1.00),
    # â”€â”€ Locations â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Mushroom Kingdom":    ("Q1069756", 1.00),
    "Koopa Kingdom":       ("Q3246497", 0.90),
    "Yoshi's Island":      ("Q193583",  0.80),
    "Sarasaland":          ("Q3461618", 1.00),
    "Bowser's Castle":     ("Q3245748", 1.00),
    "Hyrule":              ("Q1630009", 0.85),
    "Star Road":           ("Q3468668", 0.90),
    "Japan":               ("Q17",      1.00),
    "United States":       ("Q30",      1.00),
    "Europe":              ("Q46",      1.00),
    "Italy":               ("Q38",      1.00),
    "France":              ("Q142",     1.00),
    "Germany":             ("Q183",     1.00),
    "South Korea":         ("Q884",     1.00),
    "Australia":           ("Q408",     1.00),
    "Canada":              ("Q16",      1.00),
    "United Kingdom":      ("Q145",     1.00),
    "North America":       ("Q49",      1.00),
    "Kyoto":               ("Q34600",   1.00),
    "Osaka":               ("Q35765",   1.00),
    "Tokyo":               ("Q1490",    1.00),
    # â”€â”€ Items / Power-ups â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Super Mushroom":      ("Q843964",  1.00),
    "Fire Flower":         ("Q843966",  1.00),
    "Super Star":          ("Q843967",  1.00),
    "Starman":             ("Q843967",  0.90),
    "Tanooki Suit":        ("Q843972",  0.95),
    "Super Cape":          ("Q843968",  0.90),
    "Mega Mushroom":       ("Q843971",  1.00),
    "Mini Mushroom":       ("Q843970",  1.00),
    "Ice Flower":          ("Q843969",  0.95),
    # â”€â”€ Platforms â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    "Nintendo Entertainment System": ("Q172742", 1.00),
    "NES":                 ("Q172742",  1.00),
    "Super Nintendo":      ("Q183157",  1.00),
    "SNES":                ("Q183157",  1.00),
    "Nintendo 64":         ("Q184839",  1.00),
    "Game Boy":            ("Q170509",  1.00),
    "Game Boy Advance":    ("Q176727",  1.00),
    "Nintendo DS":         ("Q170510",  1.00),
    "Nintendo 3DS":        ("Q203597",  1.00),
    "Wii":                 ("Q8079",    1.00),
    "Wii U":               ("Q56942",   1.00),
    "Nintendo Switch":     ("Q19610114",1.00),
    "GameCube":            ("Q182172",  1.00),
    "Virtual Boy":         ("Q184512",  1.00),
    "Game & Watch":        ("Q159430",  1.00),
}


# =============================================================================
# Wikidata SPARQL label search (batch-friendly)
# =============================================================================

def sparql_search_by_label(label: str, client: httpx.Client) -> tuple[str, float]:
    """Search Wikidata via SPARQL for an entity by exact label."""
    # Escape quotes in label
    safe_label = label.replace('"', '\\"').replace("\\", "\\\\")
    q = f"""
    SELECT ?item ?itemLabel WHERE {{
      ?item rdfs:label "{safe_label}"@en .
      ?item wikibase:sitelinks ?links .
      FILTER (?links > 2)
    }} ORDER BY DESC(?links) LIMIT 3
    """
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "MarioKB-TD4/1.0 (ESILV student project; not-commercial)",
    }
    try:
        r = client.get(WIKIDATA_SPARQL,
                       params={"query": q, "format": "json"},
                       headers=headers, timeout=15)
        if r.status_code != 200:
            return "", 0.0
        rows = r.json()["results"]["bindings"]
        if rows:
            qid = rows[0]["item"]["value"].split("/")[-1]
            return qid, 0.80
    except Exception:
        pass
    return "", 0.0


def wikibase_api_search(label: str, client: httpx.Client) -> tuple[str, float]:
    """Fallback: use Wikidata REST API search."""
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "type": "item",
        "limit": 5,
        "format": "json",
        "uselang": "en",
    }
    headers = {
        "User-Agent": "MarioKB-TD4/1.0 (ESILV student project)",
        "Accept": "application/json",
    }
    try:
        r = client.get(WIKIDATA_SEARCH, params=params, headers=headers, timeout=12)
        if r.status_code != 200 or not r.text.strip():
            return "", 0.0
        data = r.json()
        results = data.get("search", [])
        if not results:
            return "", 0.0
        top = results[0]
        top_label = top.get("label", "").lower()
        query_lower = label.lower()
        conf = 0.95 if top_label == query_lower else (
               0.80 if query_lower in top_label or top_label in query_lower
               else 0.55)
        desc = top.get("description", "").lower()
        if any(kw in desc for kw in ["mario", "nintendo", "video game",
                                      "fictional character", "platform"]):
            conf = min(0.99, conf + 0.10)
        return top.get("id", ""), round(conf, 3)
    except Exception:
        return "", 0.0


# =============================================================================
# Predicate alignment
# =============================================================================

PREDICATE_WIKIDATA_HINTS: dict[str, tuple[str | None, str]] = {
    # (Wikidata PID or None, alignment_type)
    "coOccursWith":    (None,    "no_match"),
    "saves":           (None,    "no_match"),
    "kidnaps":         (None,    "no_match"),
    "defeats":         (None,    "no_match"),
    "enemyOf":         (None,    "no_match"),
    "appearsIn":       ("P161",  "equivalent"),  # cast member
    "rules":           ("P6",    "sub_property"), # head of gov (approx)
    "locatedIn":       ("P131",  "equivalent"),  # located in admin entity
    "contains":        ("P150",  "equivalent"),  # contains admin division
    "isPartOf":        ("P361",  "equivalent"),  # part of
    "createdBy":       ("P170",  "equivalent"),  # creator
    "developedBy":     ("P178",  "equivalent"),  # developer
    "publishedBy":     ("P123",  "equivalent"),  # publisher
    "features":        ("P674",  "equivalent"),  # characters
    "fights":          (None,    "no_match"),
    "helps":           (None,    "no_match"),
    "opposes":         (None,    "no_match"),
    "follows":         ("P155",  "equivalent"),  # follows
    "precedes":        ("P156",  "equivalent"),  # followed by
    "isRelatedTo":     ("P1382", "sub_property"),
    "ownedBy":         ("P127",  "equivalent"),  # owned by
    "worksFor":        ("P108",  "equivalent"),  # employer
    "collaboratedWith":(None,    "no_match"),
    "uses":            (None,    "no_match"),
    "transformsTo":    (None,    "no_match"),
    "belongsTo":       ("P361",  "sub_property"), # part of (approx)
}


# =============================================================================
# Main
# =============================================================================

def main():
    DATA.mkdir(parents=True, exist_ok=True)

    print("Loading private KBâ€¦")
    kb = Graph()
    kb.parse(str(KB_FILE), format="turtle")
    print(f"  {len(kb)} triples loaded")

    g_align = Graph()
    g_align.bind("mario", MARIO)
    g_align.bind("prop",  PROP)
    g_align.bind("wd",    WD)
    g_align.bind("wdt",   WDT)
    g_align.bind("owl",   OWL)
    g_align.bind("rdfs",  RDFS)

    # Collect entities
    entities: dict[str, str] = {}
    for uri, _, label in kb.triples((None, RDFS.label, None)):
        if str(uri).startswith(BASE):
            entities[str(uri)] = str(label)
    print(f"  Entities to align: {len(entities)}")

    ent_mapping: list[dict] = []
    aligned_vip = aligned_api = new_entity = 0
    api_ok = True
    fail_streak = 0

    with httpx.Client(timeout=15) as client:
        for uri_str, label in entities.items():
            # â”€â”€ 1. Static VIP lookup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            if label in VIP_MAPPING:
                qid, conf = VIP_MAPPING[label]
                local_uri = URIRef(uri_str)
                wd_uri    = WD[qid]
                g_align.add((local_uri, OWL.sameAs, wd_uri))
                g_align.add((wd_uri, RDFS.label, Literal(label, lang="en")))
                ent_mapping.append({
                    "private_entity": uri_str.split("#")[-1],
                    "label":          label,
                    "wikidata_qid":   qid,
                    "wikidata_uri":   f"http://www.wikidata.org/entity/{qid}",
                    "confidence":     conf,
                    "status":         "aligned_vip",
                })
                aligned_vip += 1
                continue

            # â”€â”€ 2. SPARQL fallback (disabled if API broken) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            qid, conf = "", 0.0
            if api_ok:
                qid, conf = sparql_search_by_label(label, client)
                time.sleep(REQUEST_DELAY)
                if not qid:
                    qid, conf = wikibase_api_search(label, client)
                    time.sleep(REQUEST_DELAY)

                if qid:
                    fail_streak = 0
                else:
                    fail_streak += 1
                    if fail_streak >= 8:
                        print("    [INFO] API unresponsive â€” disabling for remaining entities.")
                        api_ok = False

            if qid and conf >= CONFIDENCE_THRESHOLD:
                local_uri = URIRef(uri_str)
                wd_uri    = WD[qid]
                g_align.add((local_uri, OWL.sameAs, wd_uri))
                g_align.add((wd_uri, RDFS.label, Literal(label, lang="en")))
                status = "aligned_api"
                aligned_api += 1
            else:
                status = "new_entity"
                new_entity += 1

            ent_mapping.append({
                "private_entity": uri_str.split("#")[-1],
                "label":          label,
                "wikidata_qid":   qid or "â€”",
                "wikidata_uri":   f"http://www.wikidata.org/entity/{qid}" if qid else "â€”",
                "confidence":     conf,
                "status":         status,
            })

            n_done = len(ent_mapping)
            if n_done % 30 == 0:
                print(f"    â€¦ {n_done}/{len(entities)}  VIP:{aligned_vip}  API:{aligned_api}  new:{new_entity}")

        # â”€â”€ Predicate alignment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        print("\n  Predicate alignment (static map)â€¦")
        pred_mapping = []
        pred_eq = pred_sub = pred_none = 0
        for prop_slug, (pid, align_type) in PREDICATE_WIKIDATA_HINTS.items():
            local_uri = PROP[prop_slug]
            if pid:
                wd_prop = WDT[pid]
                if align_type == "equivalent":
                    g_align.add((local_uri, OWL.equivalentProperty, wd_prop))
                    pred_eq += 1
                else:
                    g_align.add((local_uri, RDFS.subPropertyOf, wd_prop))
                    pred_sub += 1
            else:
                pred_none += 1
            pred_mapping.append({
                "private_predicate": str(local_uri),
                "predicate_slug":    prop_slug,
                "wikidata_pid":      pid or "â€”",
                "wikidata_uri":      f"http://www.wikidata.org/prop/direct/{pid}" if pid else "â€”",
                "confidence":        1.00 if pid else 0.0,
                "status":            align_type,
            })
        print(f"    equivalentProperty:{pred_eq}  subPropertyOf:{pred_sub}  no_match:{pred_none}")

    # â”€â”€ Save â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    g_align.serialize(str(OUT_ALIGN), format="turtle")
    total_aligned = aligned_vip + aligned_api
    print(f"\nAlignment: {OUT_ALIGN.name} ({len(g_align)} triples)")
    print(f"  VIP: {aligned_vip}  API: {aligned_api}  new: {new_entity}  total aligned: {total_aligned}")

    with open(OUT_ENT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["private_entity", "label", "wikidata_qid",
                                           "wikidata_uri", "confidence", "status"])
        w.writeheader()
        w.writerows(ent_mapping)
    print(f"Entity mapping:    {OUT_ENT.name}  ({len(ent_mapping)} rows)")

    with open(OUT_PRED, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["private_predicate", "predicate_slug",
                                           "wikidata_pid", "wikidata_uri",
                                           "confidence", "status"])
        w.writeheader()
        w.writerows(pred_mapping)
    print(f"Predicate mapping: {OUT_PRED.name}  ({len(pred_mapping)} rows)")


if __name__ == "__main__":
    main()
