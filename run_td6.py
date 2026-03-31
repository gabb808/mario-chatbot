#!/usr/bin/env python3
"""TD6 CLI - baseline vs SPARQL-generation RAG on Mario knowledge graph."""

import argparse
import os
import sys
from datetime import datetime

import config
from src.sparql_gen import SparqlGenerationRAG, load_graph, write_json


DEFAULT_QUESTIONS = [
    "Who is Mario?",
    "Who created Mario?",
    "What games did Nintendo develop?",
    "What is the Mushroom Kingdom?",
    "Tell me about Bowser.",
]


def pretty_print_result(result: dict) -> None:
    if result.get("error"):
        print(f"\n[Execution Error] {result['error']}")

    print("\n[SPARQL Query Used]")
    print(result["query"])
    print(f"\n[Repaired?] {result['repaired']}")

    vars_ = result.get("vars", [])
    rows = result.get("rows", [])
    if not rows:
        print("\n[No rows returned]")
        return

    print("\n[Results]")
    print(" | ".join(vars_))
    for row in rows[:20]:
        print(" | ".join(row))
    if len(rows) > 20:
        print(f"... (showing 20 of {len(rows)})")


def parse_args():
    parser = argparse.ArgumentParser(description="TD6 SPARQL-generation RAG")
    parser.add_argument("--question", type=str, help="Single question to run once")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline LLM answer")
    parser.add_argument("--no-repair", action="store_true", help="Disable SPARQL self-repair loop")
    parser.add_argument("--eval", action="store_true", help="Run the 5-question evaluation set")
    parser.add_argument("--model", type=str, default=config.LLM_MODEL, help="Ollama model tag")
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join("TDs", "TP6", "td6_eval_results.json"),
        help="Path for JSON results in --eval mode",
    )
    return parser.parse_args()


def build_graph_paths():
    paths = [config.TD4_EXPANDED]
    if os.path.exists(config.TD4_PRIVATE):
        paths.append(config.TD4_PRIVATE)
    if os.path.exists(config.TD4_ALIGNMENT):
        paths.append(config.TD4_ALIGNMENT)
    return paths


def run_single(system: SparqlGenerationRAG, schema_summary: str, question: str, no_baseline: bool, no_repair: bool):
    print(f"\nQuestion: {question}")

    if not no_baseline:
        print("\n--- Baseline (No KG RAG) ---")
        baseline = system.answer_no_rag(question)
        print(baseline)
    else:
        baseline = None

    print("\n--- SPARQL-generation RAG ---")
    result = system.answer_with_sparql_generation(
        question=question,
        schema_summary=schema_summary,
        try_repair=not no_repair,
    )
    pretty_print_result(result)
    return baseline, result


def run_eval(system: SparqlGenerationRAG, schema_summary: str, output: str, no_baseline: bool, no_repair: bool) -> None:
    records = []
    print("\nRunning TD6 evaluation set (5 questions)...")
    for index, question in enumerate(DEFAULT_QUESTIONS, start=1):
        print(f"\n[{index}/{len(DEFAULT_QUESTIONS)}]")
        baseline, rag_result = run_single(
            system=system,
            schema_summary=schema_summary,
            question=question,
            no_baseline=no_baseline,
            no_repair=no_repair,
        )
        records.append(
            {
                "question": question,
                "baseline_answer": baseline,
                "sparql_query": rag_result.get("query"),
                "repaired": rag_result.get("repaired"),
                "error": rag_result.get("error"),
                "rows": rag_result.get("rows", []),
                "vars": rag_result.get("vars", []),
            }
        )

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model": system.model,
        "questions": records,
    }
    os.makedirs(os.path.dirname(output), exist_ok=True)
    write_json(output, payload)
    print(f"\nSaved evaluation JSON to: {output}")


def main():
    args = parse_args()

    graph_paths = build_graph_paths()
    for path in graph_paths:
        if not os.path.exists(path):
            print(f"Missing required graph file: {path}")
            sys.exit(1)

    graph = load_graph(*graph_paths)
    system = SparqlGenerationRAG(
        graph=graph,
        ollama_url=f"{config.OLLAMA_ENDPOINT}/api/generate",
        model=args.model,
    )
    schema_summary = system.build_schema_summary()

    if args.eval:
        run_eval(
            system=system,
            schema_summary=schema_summary,
            output=args.output,
            no_baseline=args.no_baseline,
            no_repair=args.no_repair,
        )
        return

    if args.question:
        run_single(
            system=system,
            schema_summary=schema_summary,
            question=args.question,
            no_baseline=args.no_baseline,
            no_repair=args.no_repair,
        )
        return

    while True:
        try:
            question = input("\nQuestion (or 'quit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        run_single(
            system=system,
            schema_summary=schema_summary,
            question=question,
            no_baseline=args.no_baseline,
            no_repair=args.no_repair,
        )


if __name__ == "__main__":
    main()