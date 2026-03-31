"""Configuration — paths relative to project root."""

import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Crawled data (TD1 / TP1) ─────────────────────────────────────────────────
TD1_CRAWL         = os.path.join(PROJECT_ROOT, "data", "crawled", "crawler_output.jsonl")
TD1_ENTITIES      = os.path.join(PROJECT_ROOT, "data", "crawled", "extracted_knowledge.csv")
TD1_RELATIONSHIPS = os.path.join(PROJECT_ROOT, "data", "crawled", "extracted_relationships.csv")

# ── Knowledge Graph artifacts (TD4) ──────────────────────────────────────────
TD4_ONTOLOGY  = os.path.join(PROJECT_ROOT, "kg_artifacts", "mario_ontology.ttl")
TD4_PRIVATE   = os.path.join(PROJECT_ROOT, "kg_artifacts", "private_kb.ttl")
TD4_ALIGNMENT = os.path.join(PROJECT_ROOT, "kg_artifacts", "alignment.ttl")
TD4_EXPANDED  = os.path.join(PROJECT_ROOT, "kg_artifacts", "expanded_kb.ttl")

# ── KGE results (TD5) ────────────────────────────────────────────────────────
TD5_KGE_NEIGHBORS = os.path.join(PROJECT_ROOT, "data", "kge", "results", "nearest_neighbors.json")

# ── Embeddings cache ──────────────────────────────────────────────────────────
CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")

# ── Text retriever settings ───────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE      = 500   # characters per chunk
CHUNK_OVERLAP   = 50    # overlap between chunks
TOP_K           = 5     # results per retrieval

# ── LLM settings ─────────────────────────────────────────────────────────────
LLM_PROVIDER    = "ollama"
LLM_MODEL       = "phi3"
OLLAMA_ENDPOINT = "http://localhost:11434"
TEMPERATURE     = 0.3
