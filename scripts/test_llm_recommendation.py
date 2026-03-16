#!/usr/bin/env python3
"""Test script: load model and vector store, retrieve restaurants, get LLM recommendation, print it."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import get_embedding_model
from src.retrieval.vector_store import load_vector_store
from src.llm.recommender import generate_recommendation

VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
SAMPLE_QUERY = "cheap biryani place for friends"


def main() -> None:
    if not VECTOR_STORE_PATH.exists():
        print(f"Error: Vector store not found at {VECTOR_STORE_PATH}")
        print("Run scripts/build_vector_store.py first.")
        sys.exit(1)

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Loading FAISS vector store...")
    vector_store = load_vector_store(VECTOR_STORE_PATH, embedding_model)

    print(f"Retrieving restaurants for query: {SAMPLE_QUERY!r}")
    retrieved_docs = vector_store.similarity_search(SAMPLE_QUERY, k=5)

    print("Generating recommendation with LLM...")
    recommendation = generate_recommendation(SAMPLE_QUERY, retrieved_docs)

    print("\n--- Final recommendation ---")
    print(recommendation)
    print("--- End ---")


if __name__ == "__main__":
    main()
