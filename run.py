#!/usr/bin/env python3
"""
Mario Universe RAG Assistant — powered by TD1-TD5 data.

Usage:
    python run.py                  # Interactive chat (needs Ollama)
    python run.py --search "Mario" # Search without LLM
    python run.py --demo           # Run demo questions
    python run.py --stats          # Show KG and text stats
"""

import argparse
import logging
import sys
import os
import io
import time

# Fix Windows console encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# add project root to path so config is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.text_retriever import TextRetriever
from src.kg_retriever import KGRetriever
from src.kge_retriever import KGENeighbourRetriever
from src.llm import get_llm
from src.rag import MarioRAG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

DEMO_QUESTIONS = [
    "Who is Mario?",
    "What games did Nintendo develop?",
    "Tell me about Bowser.",
    "What is the Mushroom Kingdom?",
    "Who created Mario?",
]


def check_data_files():
    """Verify that TD data files exist."""
    files = {
        "TD1 crawl": config.TD1_CRAWL,
        "TD4 expanded KB": config.TD4_EXPANDED,
    }
    ok = True
    for name, path in files.items():
        if not os.path.exists(path):
            logger.error(f"Missing: {name} at {path}")
            ok = False
    return ok


def build_system(skip_llm=False):
    """Initialize all components."""
    t0 = time.time()

    # Text retriever
    logger.info("--- Initializing Text Retriever ---")
    text_ret = TextRetriever(model_name=config.EMBEDDING_MODEL, cache_dir=config.CACHE_DIR)
    text_ret.load_and_chunk(config.TD1_CRAWL, chunk_size=config.CHUNK_SIZE, overlap=config.CHUNK_OVERLAP)
    text_ret.build_index()

    # KG retriever — load expanded + private KBs
    logger.info("--- Initializing KG Retriever ---")
    graph_paths = [config.TD4_EXPANDED]
    if os.path.exists(config.TD4_PRIVATE):
        graph_paths.append(config.TD4_PRIVATE)
    if os.path.exists(config.TD4_ALIGNMENT):
        graph_paths.append(config.TD4_ALIGNMENT)
    kg_ret = KGRetriever(*graph_paths)

    # LLM (optional)
    llm = None
    if not skip_llm:
        logger.info("--- Initializing LLM ---")
        llm = get_llm(
            provider=config.LLM_PROVIDER,
            model=config.LLM_MODEL,
            endpoint=config.OLLAMA_ENDPOINT,
            temperature=config.TEMPERATURE,
        )
        if llm is None:
            logger.warning("No LLM available — running in retrieval-only mode")

    # KGE retriever — pre-computed TD5 nearest neighbours
    logger.info("--- Initializing KGE Retriever (TD5) ---")
    kge_ret = KGENeighbourRetriever(config.TD5_KGE_NEIGHBORS)

    rag = MarioRAG(text_retriever=text_ret, kg_retriever=kg_ret, llm=llm, kge_retriever=kge_ret)
    logger.info(f"System ready in {time.time() - t0:.1f}s")
    return rag, text_ret, kg_ret


def cmd_stats(text_ret, kg_ret):
    """Print system statistics."""
    ts = text_ret.stats()
    ks = kg_ret.stats()
    print(f"\n{'='*50}")
    print(f"  Mario RAG — System Statistics")
    print(f"{'='*50}")
    print(f"  Text corpus:")
    print(f"    Documents:     {ts['documents']}")
    print(f"    Chunks:        {ts['chunks']}")
    print(f"    Embedding dim: {ts['embedding_dim']}")
    print(f"  Knowledge Graph:")
    print(f"    Triples:       {ks['triples']:,}")
    print(f"    Entities:      {ks['entities']:,}")
    print(f"    Predicates:    {ks['predicates']:,}")
    print(f"    Labels indexed:{ks['labels_indexed']:,}")
    print(f"{'='*50}\n")


def cmd_search(rag, query):
    """Search without LLM — show raw retrieval results."""
    print(f"\nSearching: '{query}'\n")

    text_results, kg_facts = rag.retrieve(query)

    print("--- KG Facts ---")
    if kg_facts:
        for fact_block in kg_facts:
            print(fact_block)
    else:
        print("  (no KG facts found)")

    print("\n--- Text Passages ---")
    for r in text_results:
        print(f"  [{r['title']}] (score: {r['score']:.3f})")
        print(f"  {r['text'][:200]}...")
        print()


def cmd_demo(rag):
    """Run demo questions."""
    print(f"\n{'='*50}")
    print("  Mario RAG — Demo")
    print(f"{'='*50}\n")

    for i, q in enumerate(DEMO_QUESTIONS, 1):
        print(f"[{i}/{len(DEMO_QUESTIONS)}] Q: {q}")
        result = rag.answer(q)
        print(f"A: {result['answer']}")
        print(f"   ({len(result['kg_facts'])} KG facts, {len(result['text_passages'])} text passages)")
        print()


def main():
    parser = argparse.ArgumentParser(description="Mario Universe RAG Assistant")
    parser.add_argument("--search", type=str, help="Search query (no LLM needed)")
    parser.add_argument("--demo", action="store_true", help="Run demo questions")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM initialization")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    if not check_data_files():
        print("ERROR: TD data files not found. Make sure the TDs/ folder is at the correct location.")
        sys.exit(1)

    skip_llm = args.no_llm or args.search is not None or args.stats
    rag, text_ret, kg_ret = build_system(skip_llm=skip_llm)

    if args.stats:
        cmd_stats(text_ret, kg_ret)
    elif args.search:
        cmd_search(rag, args.search)
    elif args.demo:
        cmd_demo(rag)
    else:
        rag.chat()


if __name__ == "__main__":
    main()
