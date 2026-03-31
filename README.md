# Mario Chatbot 🍄

**Web & Data Mining — ESILV A4 S2**  
**Team:** Gabriel Beziou
**Domain:** Mario Universe (Nintendo franchise)

A full Knowledge Graph pipeline over the Mario Universe, from web crawling to a RAG chatbot powered by a local LLM.

---

## Pipeline Overview

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/crawl/` | Web crawler (MarioWiki API) + NER (NLTK) |
| 2 | `src/kg/` | RDF ontology, private KB, Wikidata alignment, DBpedia expansion |
| 3 | `src/reason/` | OWL ontology + SWRL reasoning (OWLReady2 / Pellet) |
| 4 | `src/kge/` | Knowledge Graph Embeddings — TransE & ComplEx (PyKEEN) |
| 5 | `src/rag/` | RAG chatbot — NL→SPARQL, self-repair, text + KG + KGE retrieval |

---

## Key Results

| Module | Metric | Value |
|--------|--------|-------|
| Crawling | Documents / Words | 18 pages · 132,207 words |
| NER | Entities / Relations | 257 entities · 623 triples |
| KB Expansion | Total triples | **106,272** (via DBpedia) |
| KGE — TransE | MRR | 0.0388 |
| KGE — ComplEx | MRR | **0.1121** (best model) |
| RAG | Questions grounded | 5/5 with SPARQL grounding |

---

## Installation

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download NLTK data (first run only)
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger'); nltk.download('maxent_ne_chunker'); nltk.download('words')"

# 3. Install Ollama — https://ollama.ai
ollama pull phi3
```

---

## How to Run Each Module

### Step 1 — Crawling + NER
```bash
python src/crawl/main.py
# Output: data/crawled/crawler_output.jsonl
#         data/crawled/extracted_knowledge.csv
#         data/crawled/extracted_relationships.csv
```

### Step 2 — KB Construction, Alignment & Expansion
```bash
python src/kg/pipeline.py
# Or step by step:
python src/kg/step1_build_kb.py    # Build private KB
python src/kg/step2_align.py       # Wikidata alignment
python src/kg/step3_expand.py      # DBpedia SPARQL expansion
python src/kg/step4_report.py      # Generate statistics
```

### Step 3 — SWRL Reasoning
```bash
python src/reason/part1_swrl.py
# Applies SWRL rules on family_lab.owl and Mario KB
```

### Step 4 — Knowledge Graph Embeddings
```bash
# Requires PyKEEN + PyTorch
python src/kge/part2_prepare.py           # Clean & split triples
python src/kge/part2_train_evaluate.py    # Train TransE + ComplEx
```

### Step 5 — RAG Demo (TD6 — SPARQL generation)

```bash
# Single question
python run_td6.py --question "Who created Mario?"

# 5-question evaluation set
python run_td6.py --eval

# Interactive mode
python run_td6.py
```

### Step 6 — Full RAG Chatbot (text + KG + KGE + LLM)

```bash
# Requires Ollama running: ollama serve

# Interactive chat
python run.py

# Demo (5 preset questions)
python run.py --demo

# Statistics (no LLM needed)
python run.py --stats

# Search without LLM
python run.py --search "Bowser"
```

---

## Repository Structure

```
mario-chatbot/
├── src/
│   ├── crawl/          ← Web crawler + NER (TD1)
│   ├── kg/             ← KB construction, alignment, expansion (TD4)
│   ├── reason/         ← OWL + SWRL reasoning (TD3, TD5)
│   ├── kge/            ← KGE training & evaluation (TD5)
│   └── rag/            ← RAG chatbot modules (TD6)
├── data/
│   ├── crawled/        ← Crawled documents + extracted entities
│   └── kge/            ← train.txt, valid.txt, test.txt + results
├── kg_artifacts/
│   ├── mario_ontology.ttl   ← Domain ontology
│   ├── private_kb.ttl       ← Private KB (2,085 triples)
│   ├── alignment.ttl        ← Wikidata owl:sameAs alignment
│   ├── expanded_kb.ttl      ← Expanded KB (106,272 triples)
│   └── stats.json           ← KB statistics
├── ontology/
│   └── family_lab.owl       ← Family ontology for SWRL demo
├── reports/
│   └── FINAL_REPORT.docx    ← Final project report
├── run.py                   ← Full RAG chatbot CLI
├── run_td6.py               ← SPARQL-generation RAG CLI
├── config.py                ← Paths & LLM settings
├── requirements.txt
└── .gitignore
```

---

## Screenshot

```
============================================================
  Mario Universe RAG Assistant
  Mode: RAG (with LLM)
  Type 'exit' to quit, 'history' to see past Q&A
============================================================

You: Who created Mario?
Assistant: Shigeru Miyamoto created Mario while working at Nintendo.
  (4 KG facts, 2 KGE facts, 3 text passages)

You: exit
Bye!
```

---
