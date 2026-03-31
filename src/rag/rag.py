"""RAG agent — dual retrieval (text + KG) with optional LLM generation."""

import logging

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are a Mario Universe expert assistant.
Answer the question using ONLY the provided context.
If the context doesn't contain the answer, say "I don't have enough information."
Be concise and factual.

=== Knowledge Graph Facts ===
{kg_facts}

=== KGE-Related Entities (TD5 embeddings) ===
{kge_facts}

=== Text Passages ===
{text_passages}

Question: {question}
Answer:"""


class MarioRAG:
    """Dual-retrieval RAG: semantic text search + SPARQL KG lookup + KGE neighbours + LLM."""

    def __init__(self, text_retriever, kg_retriever, llm=None, kge_retriever=None):
        self.text = text_retriever
        self.kg = kg_retriever
        self.llm = llm
        self.kge = kge_retriever
        self.history = []

    # ── Retrieve ──────────────────────────────────────────────────────

    def retrieve(self, question: str, top_k: int = 3):
        """Dual retrieval: text chunks + KG facts."""
        text_results = self.text.search(question, top_k=top_k)
        kg_facts = self.kg.answer_question(question)
        return text_results, kg_facts

    # ── Answer ────────────────────────────────────────────────────────

    def answer(self, question: str, top_k: int = 3) -> dict:
        """Answer a question using retrieval + optional LLM generation."""
        text_results, kg_facts = self.retrieve(question, top_k)

        # TD5 KGE neighbours
        kge_facts = []
        if self.kge:
            kge_facts = self.kge.get_neighbour_facts(question, self.kg)

        # format context
        kg_section = "\n".join(kg_facts) if kg_facts else "(no structured facts found)"
        kge_section = "\n".join(kge_facts) if kge_facts else "(no KGE-related entities found)"
        text_section = ""
        for r in text_results:
            text_section += f"[{r['title']}] (score: {r['score']:.3f})\n{r['text']}\n\n"
        text_section = text_section.strip() or "(no text passages found)"

        result = {
            "question": question,
            "kg_facts": kg_facts,
            "kge_facts": kge_facts,
            "text_passages": text_results,
            "answer": None,
        }

        if self.llm:
            prompt = PROMPT_TEMPLATE.format(
                kg_facts=kg_section,
                kge_facts=kge_section,
                text_passages=text_section,
                question=question,
            )
            try:
                result["answer"] = self.llm.invoke(prompt).strip()
            except Exception as e:
                logger.error(f"LLM error: {e}")
                result["answer"] = f"[LLM error: {e}]\n\nRetrieved context:\n{kg_section}\n\n{text_section}"
        else:
            result["answer"] = (
                f"[No LLM — showing raw retrieval]\n\n"
                f"--- KG Facts ---\n{kg_section}\n\n"
                f"--- KGE-Related ---\n{kge_section}\n\n"
                f"--- Text ---\n{text_section}"
            )

        self.history.append((question, result["answer"]))
        return result

    # ── Interactive chat ──────────────────────────────────────────────

    def chat(self):
        """Interactive Q&A loop."""
        mode = "RAG (with LLM)" if self.llm else "Retrieval only (no LLM)"
        print(f"\n{'='*60}")
        print(f"  Mario Universe RAG Assistant")
        print(f"  Mode: {mode}")
        print(f"  Type 'exit' to quit, 'history' to see past Q&A")
        print(f"{'='*60}\n")

        while True:
            try:
                question = input("You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nBye!")
                break

            if not question:
                continue
            if question.lower() in ("exit", "quit", "q"):
                print("Bye!")
                break
            if question.lower() == "history":
                for i, (q, a) in enumerate(self.history):
                    print(f"\n[{i+1}] Q: {q}\n    A: {a[:200]}...")
                continue

            result = self.answer(question)
            print(f"\nAssistant: {result['answer']}")
            print(f"  ({len(result['kg_facts'])} KG facts, "
                  f"{len(result.get('kge_facts', []))} KGE facts, "
                  f"{len(result['text_passages'])} text passages)\n")

    # ── Batch mode ────────────────────────────────────────────────────

    def batch(self, questions: list) -> list:
        """Answer a list of questions."""
        results = []
        for q in questions:
            r = self.answer(q)
            results.append(r)
        return results
