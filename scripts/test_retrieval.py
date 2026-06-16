#!/usr/bin/env python3
"""Test script: load embedding model and vector store, run a sample query, print retrieved restaurants."""

import sys

import bootstrap

bootstrap.ensure_project_root()

from src.embeddings.embedder import get_embedding_model
from src.paths import vector_store_dir
from src.retrieval.retriever import SCORE_DESCRIPTION, search_restaurants
from src.retrieval.vector_store import load_vector_store

SAMPLE_QUERY = "Best Burger Places in town"


def main() -> None:
    vector_path = vector_store_dir()
    if not vector_path.exists():
        print(f"Error: Vector store not found at {vector_path}")
        print("Run scripts/build_vector_store.py first.")
        sys.exit(1)

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Loading FAISS vector store...")
    vector_store = load_vector_store(vector_path, embedding_model)

    print(f"Query: {SAMPLE_QUERY!r}\n")
    print(f"({SCORE_DESCRIPTION})\n")
    hits = search_restaurants(vector_store, SAMPLE_QUERY, k=5)

    print("Retrieved restaurants:\n")
    for i, hit in enumerate(hits, 1):
        print(f"--- {i} (score={hit.score:.6g}) ---")
        print(hit.document.page_content)
        print()


if __name__ == "__main__":
    main()
