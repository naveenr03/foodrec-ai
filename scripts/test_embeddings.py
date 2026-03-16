#!/usr/bin/env python3
"""Test script: load data, build documents, load embedding model, print doc count and embedding dimension."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_processing.load_dataset import load_cleaned_dataset
from src.data_processing.document_builder import build_documents
from src.embeddings.embedder import get_embedding_model


def main() -> None:
    processed_path = PROJECT_ROOT / "data" / "processed" / "restaurants_clean.csv"

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
