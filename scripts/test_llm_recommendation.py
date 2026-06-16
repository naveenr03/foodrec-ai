#!/usr/bin/env python3
"""Test script: load model and vector store, retrieve restaurants, get LLM recommendation, print it."""

import sys

import bootstrap

bootstrap.ensure_project_root()

from src.embeddings.embedder import get_embedding_model
from src.llm.recommender import generate_recommendation
from src.paths import vector_store_dir
from src.retrieval.retriever import hits_to_documents, search_restaurants
from src.retrieval.vector_store import load_vector_store

SAMPLE_QUERY = "cheap biryani place for friends"


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

    print(f"Retrieving restaurants for query: {SAMPLE_QUERY!r}")
    hits = search_restaurants(vector_store, SAMPLE_QUERY, k=5)
    retrieved_docs = hits_to_documents(hits)
    for i, h in enumerate(hits, 1):
        print(f"  {i}. score={h.score:.6g}")

    print("Generating recommendation with LLM...")
    recommendation = generate_recommendation(SAMPLE_QUERY, retrieved_docs)

    print("\n--- Final recommendation ---")
    print(recommendation)
    print("--- End ---")


if __name__ == "__main__":
    main()
