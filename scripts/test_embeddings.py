#!/usr/bin/env python3
"""Test script: load data, build documents, load embedding model, print doc count and embedding dimension."""

import sys

import bootstrap

bootstrap.ensure_project_root()

from src.data_processing.document_builder import build_documents
from src.data_processing.load_dataset import load_cleaned_dataset
from src.embeddings.embedder import get_embedding_model
from src.paths import processed_restaurants_csv


def main() -> None:
    processed_path = processed_restaurants_csv()

    if not processed_path.exists():
        print(f"Error: Cleaned dataset not found at {processed_path}")
        print("Run scripts/run_data_pipeline.py first.")
        sys.exit(1)

    print("Loading cleaned dataset...")
    df = load_cleaned_dataset(processed_path)

    print("Building documents...")
    documents = build_documents(df=df)
    print(f"Number of documents: {len(documents)}")

    print("Loading embedding model...")
    embedding_model = get_embedding_model()

    print("Generating embedding for first document...")
    first_embedding = embedding_model.embed_query(documents[0].page_content)
    print(f"Embedding vector length: {len(first_embedding)}")


if __name__ == "__main__":
    main()
