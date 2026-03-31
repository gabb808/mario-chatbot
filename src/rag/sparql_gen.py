"""TD6 SPARQL-generation RAG helpers (NL -> SPARQL -> execution)."""

from __future__ import annotations

import json
import re
from typing import List, Tuple, Dict, Any

import requests
from rdflib import Graph


MAX_PREDICATES = 80
MAX_CLASSES = 40
SAMPLE_TRIPLES = 20

CODE_BLOCK_RE = re.compile(r"```(?:sparql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

SPARQL_INSTRUCTIONS = """
You are a SPARQL generator. Convert the user QUESTION into a valid SPARQL 1.1 SELECT query
for the given LOCAL RDF graph. Follow these rules strictly:

- Use ONLY the prefixes and IRIs listed in the SCHEMA SUMMARY below.
- Write only simple SELECT queries with WHERE { ?s ?p ?o } patterns.
- Do NOT use SERVICE, FEDERATED queries, or any external endpoint.
- Do NOT invent predicates or classes not present in the schema.
- Do NOT add explanations or text outside the code block.
- Return ONLY the SPARQL query inside a single ```sparql ... ``` code block.

Example of a valid answer:
```sparql
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX mario: <http://esilv.fr/mario#>
SELECT ?label WHERE {
  mario:Mario rdfs:label ?label .
}
```
""".strip()

REPAIR_INSTRUCTIONS = """
The previous SPARQL query failed. Write a corrected, simpler SPARQL 1.1 SELECT query.

Rules:
- Use ONLY prefixes/IRIs from the SCHEMA SUMMARY.
- No SERVICE clauses, no external endpoints, no invented predicates.
- Keep the query as simple as possible (basic triple patterns only).
- Return ONLY the corrected query in a ```sparql ... ``` code block.
""".strip()


class SparqlGenerationRAG:
    """Generate and execute SPARQL from natural language with optional self-repair."""

    def __init__(self, graph: Graph, ollama_url: str, model: str):
        self.graph = graph
        self.ollama_url = ollama_url
        self.model = model

    def ask_local_llm(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
        }
        response = requests.post(self.ollama_url, json=payload, timeout=120)
        if response.status_code != 200:
            raise RuntimeError(f"Ollama API error {response.status_code}: {response.text}")
        data = response.json()
        return data.get("response", "")

    def answer_no_rag(self, question: str) -> str:
        prompt = f"Answer the following question as best as you can:\n\n{question}"
        return self.ask_local_llm(prompt)

    @staticmethod
    def get_prefix_block(graph: Graph) -> str:
        defaults = {
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
            "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
            "xsd": "http://www.w3.org/2001/XMLSchema#",
            "owl": "http://www.w3.org/2002/07/owl#",
        }
        ns_map = {p: str(ns) for p, ns in graph.namespace_manager.namespaces()}
        for key, value in defaults.items():
            ns_map.setdefault(key, value)
        lines = [f"PREFIX {prefix}: <{namespace}>" for prefix, namespace in ns_map.items()]
        return "\n".join(sorted(lines))

    @staticmethod
    def list_distinct_predicates(graph: Graph, limit: int = MAX_PREDICATES) -> List[str]:
        query = f"""
        SELECT DISTINCT ?p WHERE {{
          ?s ?p ?o .
        }} LIMIT {limit}
        """
        return [str(row.p) for row in graph.query(query)]

    @staticmethod
    def list_distinct_classes(graph: Graph, limit: int = MAX_CLASSES) -> List[str]:
        query = f"""
        SELECT DISTINCT ?cls WHERE {{
          ?s a ?cls .
        }} LIMIT {limit}
        """
        return [str(row.cls) for row in graph.query(query)]

    @staticmethod
    def sample_triples(graph: Graph, limit: int = SAMPLE_TRIPLES) -> List[Tuple[str, str, str]]:
        query = f"""
        SELECT ?s ?p ?o WHERE {{
          ?s ?p ?o .
        }} LIMIT {limit}
        """
        return [(str(row.s), str(row.p), str(row.o)) for row in graph.query(query)]

    def build_schema_summary(self) -> str:
        prefixes = self.get_prefix_block(self.graph)
        predicates = self.list_distinct_predicates(self.graph)
        classes = self.list_distinct_classes(self.graph)
        samples = self.sample_triples(self.graph)

        predicate_lines = "\n".join(f"- {predicate}" for predicate in predicates)
        class_lines = "\n".join(f"- {klass}" for klass in classes)
        sample_lines = "\n".join(f"- {s} {p} {o}" for s, p, o in samples)

        return f"""
{prefixes}

# Predicates (sampled, unique up to {MAX_PREDICATES})
{predicate_lines}

# Classes / rdf:type (sampled, unique up to {MAX_CLASSES})
{class_lines}

# Sample triples (up to {SAMPLE_TRIPLES})
{sample_lines}
""".strip()

    def make_sparql_prompt(self, schema_summary: str, question: str) -> str:
        return f"""{SPARQL_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

QUESTION:
{question}

Return only the SPARQL query in a code block.
"""

    @staticmethod
    def extract_sparql_from_text(text: str) -> str:
        # Try fenced code block first (```sparql ... ``` or ``` ... ```)
        match = CODE_BLOCK_RE.search(text)
        if match:
            return match.group(1).strip()

        # Fallback: find first line that looks like the start of a SPARQL query
        sparql_keywords = ("SELECT", "PREFIX", "CONSTRUCT", "ASK", "DESCRIBE")
        lines = text.splitlines()
        for i, line in enumerate(lines):
            if line.strip().upper().startswith(sparql_keywords):
                # strip any trailing backtick lines
                sparql_lines = [
                    l for l in lines[i:] if not l.strip().startswith("```")
                ]
                return "\n".join(sparql_lines).strip()

        return text.strip()

    def generate_sparql(self, question: str, schema_summary: str) -> str:
        raw = self.ask_local_llm(self.make_sparql_prompt(schema_summary, question))
        return self.extract_sparql_from_text(raw)

    def run_sparql(self, query: str) -> Tuple[List[str], List[Tuple[str, ...]]]:
        results = self.graph.query(query)
        variables = [str(v) for v in results.vars]
        rows = [tuple(str(cell) for cell in row) for row in results]
        return variables, rows

    def repair_sparql(self, schema_summary: str, question: str, bad_query: str, error_msg: str) -> str:
        prompt = f"""{REPAIR_INSTRUCTIONS}

SCHEMA SUMMARY:
{schema_summary}

ORIGINAL QUESTION:
{question}

BAD SPARQL:
{bad_query}

ERROR MESSAGE:
{error_msg}

Return only the corrected SPARQL in a code block.
"""
        raw = self.ask_local_llm(prompt)
        return self.extract_sparql_from_text(raw)

    def answer_with_sparql_generation(
        self,
        question: str,
        schema_summary: str,
        try_repair: bool = True,
    ) -> Dict[str, Any]:
        sparql = self.generate_sparql(question, schema_summary)
        try:
            variables, rows = self.run_sparql(sparql)
            return {
                "query": sparql,
                "vars": variables,
                "rows": rows,
                "repaired": False,
                "error": None,
            }
        except Exception as exc:
            if not try_repair:
                return {
                    "query": sparql,
                    "vars": [],
                    "rows": [],
                    "repaired": False,
                    "error": str(exc),
                }

            repaired_query = self.repair_sparql(
                schema_summary=schema_summary,
                question=question,
                bad_query=sparql,
                error_msg=str(exc),
            )
            try:
                variables, rows = self.run_sparql(repaired_query)
                return {
                    "query": repaired_query,
                    "vars": variables,
                    "rows": rows,
                    "repaired": True,
                    "error": None,
                }
            except Exception as second_exc:
                return {
                    "query": repaired_query,
                    "vars": [],
                    "rows": [],
                    "repaired": True,
                    "error": str(second_exc),
                }


def load_graph(*ttl_paths: str) -> Graph:
    """Load one or more Turtle files into a single graph."""
    graph = Graph()
    for path in ttl_paths:
        graph.parse(path, format="turtle")
    return graph


def write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)