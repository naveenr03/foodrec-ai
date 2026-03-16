#!/usr/bin/env python3
"""Test script: load embedding model and vector store, run a sample query, print retrieved restaurants."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.embeddings.embedder import get_embedding_model
from src.retrieval.vector_store import load_vector_store

VECTOR_STORE_PATH = PROJECT_ROOT / "data" / "vector_store"
SAMPLE_QUERY = "Best Burger Places in town"


def main() -> None:
    if not VECTOR_STORE_PATH.exists():
        print(f"Error: Vector store not found at {VECTOR_STORE_PATH}")
        print("Run scripts/build_vector_store.py first.")
        sys.exit(1)

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Loading FAISS vector store...")
    vector_store = load_vector_store(VECTOR_STORE_PATH, embedding_model)

    print(f"Query: {SAMPLE_QUERY!r}\n")
    documents = vector_store.similarity_search(SAMPLE_QUERY, k=5)

    print("Retrieved restaurants:\n")
    for i, doc in enumerate(documents, 1):
        print(f"--- {i} ---")
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()
