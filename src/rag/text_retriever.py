"""Text retriever — chunks TD1 crawled text, embeds with Sentence Transformers, semantic search."""

import json
import os
import pickle
import logging
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class TextRetriever:
    """Semantic search over MarioWiki text chunks."""

    def __init__(self, model_name: str, cache_dir: str):
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

        self.chunks = []        # list of {"text": ..., "title": ..., "url": ...}
        self.embeddings = None  # np.ndarray (n_chunks, dim)

    # ── Load & chunk ──────────────────────────────────────────────────

    def load_and_chunk(self, jsonl_path: str, chunk_size: int = 500, overlap: int = 50):
        """Read TD1 JSONL and split each document into overlapping text chunks."""
        logger.info(f"Loading text from {jsonl_path}")
        self.chunks = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                doc = json.loads(line)
                title = doc.get("title", "Unknown")
                url = doc.get("url", "")
                text = doc.get("text", "")

                # split into overlapping chunks
                start = 0
                while start < len(text):
                    end = start + chunk_size
                    chunk_text = text[start:end]
                    if len(chunk_text.strip()) > 50:  # skip tiny fragments
                        self.chunks.append({
                            "text": chunk_text,
                            "title": title,
                            "url": url,
                        })
                    start += chunk_size - overlap

        logger.info(f"Created {len(self.chunks)} chunks from {jsonl_path}")

    # ── Build embeddings ──────────────────────────────────────────────

    def build_index(self):
        """Encode all chunks into embeddings. Uses pickle cache."""
        cache_file = os.path.join(self.cache_dir, "text_embeddings.pkl")

        if os.path.exists(cache_file):
            logger.info("Loading cached text embeddings...")
            with open(cache_file, "rb") as f:
                cached = pickle.load(f)
            if cached.get("n_chunks") == len(self.chunks):
                self.embeddings = cached["embeddings"]
                logger.info(f"Loaded {self.embeddings.shape[0]} cached embeddings")
                return

        logger.info(f"Encoding {len(self.chunks)} chunks...")
        texts = [c["text"] for c in self.chunks]
        self.embeddings = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

        # cache
        with open(cache_file, "wb") as f:
            pickle.dump({"n_chunks": len(self.chunks), "embeddings": self.embeddings}, f)
        logger.info(f"Embeddings cached to {cache_file}")

    # ── Search ────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 5):
        """Semantic search: return top_k chunks most similar to query."""
        if self.embeddings is None or len(self.embeddings) == 0:
            return []

        query_vec = self.model.encode(query, convert_to_numpy=True)
        # cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec) + 1e-10
        scores = self.embeddings @ query_vec / norms
        top_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for i in top_idx:
            results.append({
                "text": self.chunks[i]["text"],
                "title": self.chunks[i]["title"],
                "url": self.chunks[i]["url"],
                "score": float(scores[i]),
            })
        return results

    # ── Stats ─────────────────────────────────────────────────────────

    def stats(self) -> dict:
        titles = set(c["title"] for c in self.chunks)
        return {
            "documents": len(titles),
            "chunks": len(self.chunks),
            "embedding_dim": self.model.get_sentence_embedding_dimension(),
        }
