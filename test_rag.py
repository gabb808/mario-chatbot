#!/usr/bin/env python3
"""
Comprehensive RAG system tests — validates all 3 layers:
1. Text Retriever (TD1 data)
2. KG Retriever (TD4 data)
3. RAG Integration (both combined)

Run with: python test_rag.py
"""

import sys
import os
import json
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.text_retriever import TextRetriever
from src.kg_retriever import KGRetriever
from src.rag import MarioRAG

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# Test queries covering different aspects
TEST_QUERIES = {
    "Simple entity": "Mario",
    "Question": "Who is Mario?",
    "Multiple entities": "Nintendo Mario Luigi",
    "Relationship": "What games did Nintendo develop?",
    "Location": "Mushroom Kingdom",
    "Character property": "Tell me about Bowser",
    "Game title": "Super Mario Bros",
}


class RAGTester:
    """Validate RAG system in isolation."""

    def __init__(self):
        """Initialize RAG components."""
        logger.info("="*60)
        logger.info("RAG TEST SUITE")
        logger.info("="*60)
        
        # Check data files exist
        if not self._check_data_files():
            sys.exit(1)
        
        # Initialize components
        self.text_ret = None
        self.kg_ret = None
        self.rag = None

    # ── Data validation ────────────────────────────────────────────

    def _check_data_files(self) -> bool:
        """Verify TD1 and TD4 data exist."""
        files = {
            "TD1 crawl": config.TD1_CRAWL,
            "TD4 KB": config.TD4_EXPANDED,
        }
        all_ok = True
        for name, path in files.items():
            if os.path.exists(path):
                logger.info(f"✓ {name}: {path}")
            else:
                logger.error(f"✗ {name} MISSING: {path}")
                all_ok = False
        return all_ok

    # ── Layer 1: Text Retriever ────────────────────────────────────

    def test_text_retriever(self):
        """Test 1: Validate TextRetriever on TD1 data."""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: TEXT RETRIEVER (TD1 data)")
        logger.info("="*60)
        
        try:
            logger.info("Loading TextRetriever...")
            self.text_ret = TextRetriever(
                model_name=config.EMBEDDING_MODEL,
                cache_dir=config.CACHE_DIR
            )
            self.text_ret.load_and_chunk(
                config.TD1_CRAWL,
                chunk_size=config.CHUNK_SIZE,
                overlap=config.CHUNK_OVERLAP
            )
            self.text_ret.build_index()
            
            stats = self.text_ret.stats()
            logger.info(f"✓ TextRetriever loaded")
            logger.info(f"  Documents: {stats['documents']}")
            logger.info(f"  Chunks: {stats['chunks']}")
            logger.info(f"  Embedding dim: {stats['embedding_dim']}")
            
            # Test a few searches
            logger.info("\nTesting semantic search...")
            for query_name, query in list(TEST_QUERIES.items())[:3]:
                results = self.text_ret.search(query, top_k=2)
                logger.info(f"  '{query}' → {len(results)} results")
                if results:
                    logger.info(f"    Top match: {results[0]['title']} (score: {results[0]['score']:.3f})")
            
            return True
        except Exception as e:
            logger.error(f"✗ TextRetriever failed: {e}")
            return False

    # ── Layer 2: KG Retriever ──────────────────────────────────────

    def test_kg_retriever(self):
        """Test 2: Validate KGRetriever on TD4 data."""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: KG RETRIEVER (TD4 data)")
        logger.info("="*60)
        
        try:
            logger.info("Loading KGRetriever...")
            self.kg_ret = KGRetriever(config.TD4_EXPANDED)
            
            stats = self.kg_ret.stats()
            logger.info(f"✓ KGRetriever loaded")
            logger.info(f"  Triples: {stats['triples']:,}")
            logger.info(f"  Entities: {stats['entities']:,}")
            logger.info(f"  Predicates: {stats['predicates']:,}")
            
            # Test entity search
            logger.info("\nTesting entity lookup...")
            test_entities = ["Mario", "Nintendo", "Luigi", "Bowser"]
            for entity in test_entities:
                result = self.kg_ret.search_entity(entity)
                logger.info(f"  '{entity}' → found: {result is not None}")
            
            # Test question answering
            logger.info("\nTesting SPARQL question answering...")
            for query_name, query in list(TEST_QUERIES.items())[:3]:
                facts = self.kg_ret.answer_question(query)
                logger.info(f"  '{query}' → {len(facts)} facts")
            
            return True
        except Exception as e:
            logger.error(f"✗ KGRetriever failed: {e}")
            return False

    # ── Layer 3: Full RAG system ────────────────────────────────────

    def test_rag_integration(self):
        """Test 3: Validate full RAG (text + KG + optional LLM)."""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: RAG INTEGRATION (Dual Retrieval)")
        logger.info("="*60)
        
        if not self.text_ret or not self.kg_ret:
            logger.error("✗ Dependencies missing (run tests 1 & 2 first)")
            return False
        
        try:
            from src.llm import get_llm
            
            # Try to initialize LLM (optional)
            llm = get_llm(
                provider=config.LLM_PROVIDER,
                model=config.LLM_MODEL,
                endpoint=config.OLLAMA_ENDPOINT,
                temperature=config.TEMPERATURE
            )
            mode = "with LLM" if llm else "without LLM (retrieval-only)"
            logger.info(f"Initializing RAG {mode}...")
            
            # Build RAG
            self.rag = MarioRAG(
                text_retriever=self.text_ret,
                kg_retriever=self.kg_ret,
                llm=llm
            )
            logger.info(f"✓ RAG initialized {mode}")
            
            # Test dual retrieval
            logger.info("\nTesting dual retrieval...")
            for query_name, query in TEST_QUERIES.items():
                text_results, kg_facts = self.rag.retrieve(query)
                logger.info(f"  '{query}':")
                logger.info(f"    Text: {len(text_results)} passages")
                logger.info(f"    KG: {len(kg_facts)} facts")
            
            # Test full answer generation
            logger.info("\nTesting full answer generation...")
            test_q = "Who is Mario?"
            result = self.rag.answer(test_q)
            logger.info(f"  Q: {test_q}")
            logger.info(f"  Answer length: {len(result['answer'])} chars")
            logger.info(f"  Sources: {len(result['kg_facts'])} KG, {len(result['text_passages'])} text")
            
            return True
        except Exception as e:
            logger.error(f"✗ RAG integration failed: {e}")
            return False

    # ── Coverage analysis ──────────────────────────────────────────

    def test_coverage(self):
        """Test 4: Measure retrieval coverage on Mario domain."""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: DOMAIN COVERAGE")
        logger.info("="*60)
        
        if not self.rag:
            logger.error("✗ RAG not initialized (run test 3 first)")
            return False
        
        coverage = {
            "entities_found": 0,
            "predicates_found": 0,
            "zero_fact_queries": 0,
            "zero_text_queries": 0,
        }
        
        mario_terms = [
            "Mario", "Luigi", "Peach", "Bowser", "Toad",
            "Nintendo", "GameBoy", "NES",
            "Mushroom Kingdom", "Castles"
        ]
        
        logger.info("Testing coverage on Mario domain terms...\n")
        for term in mario_terms:
            text, facts = self.rag.retrieve(term)
            has_kg = len(facts) > 0
            has_text = len(text) > 0
            
            coverage["entities_found"] += has_kg
            if has_kg:
                coverage["predicates_found"] += 1
            if not has_kg:
                coverage["zero_fact_queries"] += 1
            if not has_text:
                coverage["zero_text_queries"] += 1
            
            status = "✓" if (has_kg or has_text) else "⚠"
            logger.info(f"  {status} {term:20} → KG:{has_kg}, Text:{has_text}")
        
        coverage_pct = (coverage["entities_found"] / len(mario_terms)) * 100
        logger.info(f"\n  Coverage: {coverage['entities_found']}/{len(mario_terms)} ({coverage_pct:.0f}%)")
        logger.info(f"  Zero results: {coverage['zero_fact_queries']} KG, {coverage['zero_text_queries']} Text")
        
        return coverage_pct >= 50  # Pass if ≥ 50% coverage

    # ── Run all tests ──────────────────────────────────────────────

    def run_all(self):
        """Execute all tests and return summary."""
        results = {
            "test_1_text_retriever": self.test_text_retriever(),
            "test_2_kg_retriever": self.test_kg_retriever(),
            "test_3_rag_integration": self.test_rag_integration(),
            "test_4_coverage": self.test_coverage(),
        }
        
        # Summary
        logger.info("\n" + "="*60)
        logger.info("TEST SUMMARY")
        logger.info("="*60)
        passed = sum(1 for v in results.values() if v)
        total = len(results)
        logger.info(f"Passed: {passed}/{total} tests\n")
        
        for test_name, passed in results.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            logger.info(f"  {status}: {test_name}")
        
        logger.info("="*60)
        all_passed = (passed == total)
        logger.info(f"\nOverall: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}\n")
        return all_passed


if __name__ == "__main__":
    try:
        tester = RAGTester()
        success = tester.run_all()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test suite crashed: {e}")
        sys.exit(1)
